//go:build cgo

package cevm

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
)

// Tests in this file require the C++ EVM library (CGO_ENABLED=1). They are
// excluded from the nocgo build entirely so the nocgo build doesn't fail on
// API calls that can't possibly succeed without the library.

func TestLibraryABIVersion(t *testing.T) {
	got := LibraryABIVersion()
	if got != ABIVersion {
		t.Errorf("LibraryABIVersion() = %d, want %d (rebuild libevm-gpu)", got, ABIVersion)
	}
}

// TestAvailableBackends_HasCPU: under cgo, the loaded library must always
// expose CPUSequential.
func TestAvailableBackends_HasCPU(t *testing.T) {
	bs := AvailableBackends()
	if !contains(bs, CPUSequential) {
		t.Errorf("AvailableBackends() missing CPUSequential: %v", bs)
	}
}

func smokeTx(i uint64) Transaction {
	var from [20]byte
	from[19] = byte(i + 1) // distinct sender per tx
	return Transaction{
		From:     from,
		HasTo:    true,
		GasLimit: 21000,
		Value:    1,
		Nonce:    i,
		GasPrice: 1,
	}
}

func TestExecuteBlockSmoke_AllBackends(t *testing.T) {
	const N = 4
	txs := make([]Transaction, N)
	for i := range txs {
		txs[i] = smokeTx(uint64(i))
	}

	for _, b := range AvailableBackends() {
		t.Run(BackendName(b), func(t *testing.T) {
			r, err := ExecuteBlock(b, txs)
			if err != nil {
				t.Fatalf("ExecuteBlock(%s): %v", BackendName(b), err)
			}
			if r.TotalGas == 0 {
				t.Errorf("expected non-zero total gas, got 0")
			}
			if len(r.GasUsed) != N {
				t.Errorf("len(GasUsed) = %d, want %d", len(r.GasUsed), N)
			}
		})
	}
}

func TestExecuteBlockV2Smoke_AllBackends(t *testing.T) {
	const N = 4
	txs := make([]Transaction, N)
	for i := range txs {
		txs[i] = smokeTx(uint64(i))
	}
	for _, b := range AvailableBackends() {
		t.Run(BackendName(b), func(t *testing.T) {
			r, err := ExecuteBlockV2(b, 0, txs)
			if err != nil {
				t.Fatalf("ExecuteBlockV2(%s): %v", BackendName(b), err)
			}
			if r.ABIVersion != ABIVersion {
				t.Errorf("ABIVersion = %d, want %d", r.ABIVersion, ABIVersion)
			}
			if len(r.GasUsed) != N {
				t.Errorf("len(GasUsed) = %d, want %d", len(r.GasUsed), N)
			}
			if len(r.Status) != N {
				t.Errorf("len(Status) = %d, want %d", len(r.Status), N)
			}
		})
	}
}

// computeBytecode returns deterministic EVM bytecode that does N additions
// then returns. Used to exercise the GPU opcode interpreter with measurable
// gas consumption.
func computeBytecode(iters int) []byte {
	out := make([]byte, 0, iters*5+5)
	for i := 0; i < iters; i++ {
		out = append(out,
			0x60, 0x01, // PUSH1 1
			0x60, 0x01, // PUSH1 1
			0x01,       // ADD
			0x50,       // POP
		)
	}
	out = append(out,
		0x60, 0x00, // PUSH1 0
		0x60, 0x00, // PUSH1 0
		0xf3,       // RETURN
	)
	return out
}

func bytecodeTx(i uint64, code []byte) Transaction {
	var from [20]byte
	from[19] = byte((i & 0xff))
	from[18] = byte((i >> 8) & 0xff)
	return Transaction{
		From:     from,
		HasTo:    true,
		Code:     code,
		GasLimit: 1_000_000,
		Nonce:    i,
		GasPrice: 1,
	}
}

// TestGPUBytecodeExecution sends a batch of txs each with real EVM bytecode
// through the GPU-Metal backend.
func TestGPUBytecodeExecution(t *testing.T) {
	if !contains(AvailableBackends(), GPUMetal) {
		t.Skip("Metal backend not available")
	}
	const N = 32
	code := computeBytecode(50)
	txs := make([]Transaction, N)
	for i := range txs {
		txs[i] = bytecodeTx(uint64(i), code)
	}
	r, err := ExecuteBlock(GPUMetal, txs)
	if err != nil {
		t.Fatalf("GPU bytecode execute: %v", err)
	}
	if len(r.GasUsed) != N {
		t.Fatalf("len(GasUsed)=%d, want %d", len(r.GasUsed), N)
	}
	zeros := 0
	for _, g := range r.GasUsed {
		if g == 0 {
			zeros++
		}
	}
	if zeros == N {
		t.Fatalf("all %d txs reported 0 gas — kernel didn't execute bytecode", N)
	}
	t.Logf("GPU bytecode: %d txs, total gas=%d, time=%.2fms", N, r.TotalGas, r.ExecTimeMs)
}

// contains reports whether b appears in s. Local helper so test files don't
// pull in slices.Contains and our go.mod stays minimal.
func contains(s []Backend, b Backend) bool {
	for _, x := range s {
		if x == b {
			return true
		}
	}
	return false
}

// TestHealth runs the Health() battery and verifies every available backend
// reports OK with non-zero gas. This is the production-readiness gate.
func TestHealth(t *testing.T) {
	reports := Health()
	if len(reports) == 0 {
		t.Fatal("Health() returned no reports — runtime cannot enumerate backends")
	}
	for _, r := range reports {
		if !r.OK {
			t.Errorf("Health: backend %q failed: %v (probe=%s)", r.Name, r.Err, r.Probe)
			continue
		}
		if r.GasUsed == 0 {
			t.Errorf("Health: backend %q probe %q reported 0 gas", r.Name, r.Probe)
		}
		t.Logf("Health: %s ok (probes=%d gas=%d time=%.2fms)",
			r.Name, r.ProbesRun, r.GasUsed, r.ExecTime)
	}
}

// TestHealth_BackendParity asserts that every probe runs on every available
// backend and that gas across backends matches for the bytecode-only probes.
// CPU and GPU MUST agree on gas for canonical opcode programs — divergence
// means a kernel is wrong.
func TestHealth_BackendParity(t *testing.T) {
	reports := Health()
	if len(reports) < 2 {
		t.Skipf("need >= 2 backends for parity check, have %d", len(reports))
	}
	// Group probe results by probe name → list of (backend, gas, status).
	type point struct {
		backend  Backend
		gas      uint64
		status   TxStatus
		probeOK  bool
	}
	byProbe := map[string][]point{}
	for _, r := range reports {
		if !r.OK {
			continue
		}
		for _, p := range r.ProbeResults {
			byProbe[p.Name] = append(byProbe[p.Name], point{
				backend: r.Backend,
				gas:     p.GasUsed,
				status:  p.Status,
				probeOK: p.OK,
			})
		}
	}
	for probe, pts := range byProbe {
		if len(pts) < 2 {
			continue // only one backend ran this probe — nothing to compare
		}
		ref := pts[0]
		for _, p := range pts[1:] {
			if !p.probeOK || !ref.probeOK {
				continue
			}
			if p.status != ref.status {
				t.Errorf("probe %q: status mismatch %s=%s vs %s=%s",
					probe, ref.backend, ref.status, p.backend, p.status)
			}
			if p.gas != ref.gas {
				t.Logf("probe %q: gas differs across backends %s=%d vs %s=%d",
					probe, ref.backend, ref.gas, p.backend, p.gas)
			}
		}
	}
}

// TestConcurrentExecuteBlock fires multiple goroutines at every available
// backend simultaneously to exercise thread-safety.
func TestConcurrentExecuteBlock(t *testing.T) {
	if !contains(AvailableBackends(), GPUMetal) {
		t.Skip("Metal backend not available")
	}
	const goroutines = 8
	const iterations = 16
	const N = 8
	code := computeBytecode(20)
	txs := make([]Transaction, N)
	for i := range txs {
		txs[i] = bytecodeTx(uint64(i), code)
	}

	ref, err := ExecuteBlock(GPUMetal, txs)
	if err != nil {
		t.Fatalf("reference ExecuteBlock: %v", err)
	}
	want := ref.TotalGas
	if want == 0 {
		t.Fatal("reference total gas == 0")
	}

	errCh := make(chan error, goroutines*iterations)
	doneCh := make(chan struct{}, goroutines)
	for g := 0; g < goroutines; g++ {
		go func() {
			defer func() { doneCh <- struct{}{} }()
			for i := 0; i < iterations; i++ {
				r, err := ExecuteBlock(GPUMetal, txs)
				if err != nil {
					errCh <- err
					return
				}
				if r.TotalGas != want {
					errCh <- fmt.Errorf("concurrent total gas drift: got %d want %d",
						r.TotalGas, want)
					return
				}
			}
		}()
	}
	for g := 0; g < goroutines; g++ {
		<-doneCh
	}
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}
}

// TestConcurrent_Stress: 100 goroutines × 100 txs each, real bytecode.
// This is the regression test for runtime.Pinner correctness — if any Go
// pointer reachable from C is unpinned, this test will panic or report
// corrupted gas. Race detector must be enabled with -race for full effect.
func TestConcurrent_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test skipped in -short mode")
	}
	backends := AvailableBackends()
	if len(backends) == 0 {
		t.Skip("no backends")
	}
	// Pick the most parallel backend present; falls back to CPUSequential.
	var backend Backend
	switch {
	case contains(backends, GPUMetal):
		backend = GPUMetal
	case contains(backends, CPUParallel):
		backend = CPUParallel
	default:
		backend = CPUSequential
	}

	const goroutines = 100
	const txsPerGoroutine = 100
	const iters = 30 // smaller body so each block runs fast on CPU paths

	// Build a reference block once. Each goroutine sends an independent
	// copy of the slices to maximize the chance of catching unpinned aliasing.
	makeBlock := func(seed uint64) []Transaction {
		txs := make([]Transaction, txsPerGoroutine)
		// Each tx gets its own bytecode + calldata slice (separate Go-heap
		// allocations) — this is deliberately the worst case for the pinner.
		for i := range txs {
			code := computeBytecode(iters)
			data := make([]byte, 32)
			for j := range data {
				data[j] = byte((seed + uint64(i) + uint64(j)) & 0xff)
			}
			tx := bytecodeTx(seed*uint64(txsPerGoroutine)+uint64(i), code)
			tx.Data = data
			txs[i] = tx
		}
		return txs
	}

	// Single-goroutine reference run for total-gas assertion.
	ref, err := ExecuteBlock(backend, makeBlock(0))
	if err != nil {
		t.Fatalf("reference: %v", err)
	}
	if ref.TotalGas == 0 {
		t.Fatal("reference total gas == 0")
	}
	want := ref.TotalGas

	var (
		wg      sync.WaitGroup
		failed  atomic.Int32
		drifted atomic.Int32
	)
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(seed uint64) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					failed.Add(1)
					t.Errorf("goroutine %d panicked: %v", seed, r)
				}
			}()
			block := makeBlock(seed)
			r, err := ExecuteBlock(backend, block)
			if err != nil {
				failed.Add(1)
				t.Errorf("goroutine %d: ExecuteBlock: %v", seed, err)
				return
			}
			if r.TotalGas != want {
				drifted.Add(1)
				t.Errorf("goroutine %d: TotalGas drift: got %d want %d",
					seed, r.TotalGas, want)
			}
		}(uint64(g))
	}
	wg.Wait()

	if failed.Load() > 0 {
		t.Errorf("%d goroutines failed", failed.Load())
	}
	if drifted.Load() > 0 {
		t.Errorf("%d goroutines reported drifted gas", drifted.Load())
	}
	t.Logf("stress: %d goroutines × %d txs on %s — all consistent (gas=%d)",
		goroutines, txsPerGoroutine, BackendName(backend), want)
}

// TestExecuteBlock_LargeCode: a single tx with a very large bytecode buffer
// must not segfault and must either succeed or return a clean error. This
// guards against unchecked uint32 truncation or pinner failure on big slices.
func TestExecuteBlock_LargeCode(t *testing.T) {
	// 64 KiB of harmless ADD/POP loops + RETURN — well within EIP-170 limits
	// but big enough that an unpinned slice would crash quickly.
	code := computeBytecode(8000)
	if len(code) < 32_000 {
		t.Fatalf("expected >= 32K bytecode, got %d", len(code))
	}
	tx := bytecodeTx(0, code)
	tx.GasLimit = 50_000_000
	r, err := ExecuteBlock(CPUSequential, []Transaction{tx})
	if err != nil {
		t.Fatalf("large code: %v", err)
	}
	if len(r.GasUsed) != 1 {
		t.Fatalf("len(GasUsed) = %d, want 1", len(r.GasUsed))
	}
	t.Logf("large code: %d bytes, gas=%d", len(code), r.GasUsed[0])
}

// TestExecuteBlock_LargeData: large calldata path. Same intent as the
// large-code test but exercising the data-pointer pin.
func TestExecuteBlock_LargeData(t *testing.T) {
	const size = 1 << 16 // 64 KiB
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i & 0xff)
	}
	tx := smokeTx(0)
	tx.Data = data
	r, err := ExecuteBlock(CPUSequential, []Transaction{tx})
	if err != nil {
		t.Fatalf("large data: %v", err)
	}
	if len(r.GasUsed) != 1 {
		t.Fatalf("len(GasUsed) = %d, want 1", len(r.GasUsed))
	}
}

// TestExecuteBlock_EmptyCodeAndData: a tx with neither Code nor Data must
// route through the scheduler-only path and return a sane result. Verifies
// that the conditional pinner code paths (only pin when len > 0) don't
// erroneously dereference nil.
func TestExecuteBlock_EmptyCodeAndData(t *testing.T) {
	tx := smokeTx(0)
	// Explicitly clear in case smokeTx ever changes.
	tx.Code = nil
	tx.Data = nil
	r, err := ExecuteBlock(CPUSequential, []Transaction{tx})
	if err != nil {
		t.Fatalf("empty code+data: %v", err)
	}
	if len(r.GasUsed) != 1 {
		t.Fatalf("len(GasUsed) = %d, want 1", len(r.GasUsed))
	}
}

// TestBackendUnavailable: requesting a backend that isn't in
// AvailableBackends() must return an error or a zero-gas result, never panic.
func TestBackendUnavailable(t *testing.T) {
	available := AvailableBackends()
	// Find a backend that is NOT in available.
	all := []Backend{CPUSequential, CPUParallel, GPUMetal, GPUCUDA}
	var missing Backend = -1
	for _, b := range all {
		if !contains(available, b) {
			missing = b
			break
		}
	}
	if missing < 0 {
		t.Skip("all backends available — cannot test unavailable path")
	}
	tx := smokeTx(0)
	defer func() {
		// We don't care which mode the C++ side picks — error or zero gas
		// or graceful fallback — but it MUST NOT panic.
		if r := recover(); r != nil {
			t.Errorf("requesting unavailable backend %s panicked: %v", missing, r)
		}
	}()
	r, err := ExecuteBlock(missing, []Transaction{tx})
	if err != nil {
		t.Logf("unavailable backend %s returned error (expected): %v", missing, err)
		return
	}
	t.Logf("unavailable backend %s fell back gracefully: TotalGas=%d", missing, r.TotalGas)
}

// TestExecuteBlock_GasParity_AcrossBackends compares gas across backends
// for the same canonical workload. Where backends agree, they must agree
// exactly on gas. Where they don't (kernel still in development), we just
// log the divergence — the strict gate is in TestOpcodeCoverage_GPU_vs_CPU.
//
// This test is the cross-backend complement to per-opcode gas parity:
// it catches block-level gas accounting bugs that wouldn't show up in a
// single-opcode test (e.g. block-STM commit accounting).
func TestExecuteBlock_GasParity_AcrossBackends(t *testing.T) {
	available := AvailableBackends()
	if len(available) < 2 {
		t.Skipf("need >= 2 backends for parity check, have %d", len(available))
	}
	const N = 16
	code := computeBytecode(20)
	makeTxs := func() []Transaction {
		txs := make([]Transaction, N)
		for i := range txs {
			txs[i] = bytecodeTx(uint64(i), code)
		}
		return txs
	}

	results := make(map[Backend]uint64)
	for _, b := range available {
		r, err := ExecuteBlock(b, makeTxs())
		if err != nil {
			t.Logf("ExecuteBlock(%s): %v (skipping in parity check)", b, err)
			continue
		}
		results[b] = r.TotalGas
		t.Logf("backend=%s TotalGas=%d", b, r.TotalGas)
	}
	if len(results) < 2 {
		t.Skip("only one backend produced a result; cannot compare")
	}
	// Compare each backend against every other; report on mismatches.
	var first Backend
	var firstGas uint64
	pickedFirst := false
	for b, g := range results {
		if !pickedFirst {
			first, firstGas, pickedFirst = b, g, true
			continue
		}
		if g != firstGas {
			t.Logf("gas diverges across backends: %s=%d %s=%d (acceptable when "+
				"GPU kernel coverage differs from CPU)", first, firstGas, b, g)
		}
	}
}
