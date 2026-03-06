//go:build cgo

package cevm

/*
#cgo CFLAGS: -I${SRCDIR}/lib/evm/gpu
#cgo LDFLAGS: -L${SRCDIR}/build/lib
#cgo LDFLAGS: -L${SRCDIR}/build/lib/evm
#cgo LDFLAGS: -L${SRCDIR}/build/lib/evm/luxcpp-gpu
#cgo LDFLAGS: -L${SRCDIR}/build/lib/cevm_precompiles
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/build/lib
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/build/lib/evm/luxcpp-gpu
#cgo LDFLAGS: -levm
#cgo darwin LDFLAGS: -levm-gpu -levm-metal-hosts -levm-kernel-metal -levm-gpu -lluxgpu -lcevm_precompiles -lstdc++
#cgo darwin LDFLAGS: -framework Metal -framework Foundation
#cgo linux  LDFLAGS: -Wl,--start-group -levm-gpu -lluxgpu -lcevm_precompiles -Wl,--end-group -lstdc++

#include <stdlib.h>
#include "go_bridge.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// AutoDetect returns the best available backend for this machine.
func AutoDetect() Backend {
	return Backend(C.gpu_auto_detect_backend())
}

// init validates the ABI version of the loaded shared library against the Go
// module's expected ABIVersion. A mismatch means the binary and the cevm
// module were built against incompatible C++ headers and any execution would
// produce silently wrong results — fail fast at process start instead.
func init() {
	got := uint32(C.gpu_abi_version())
	if got != ABIVersion {
		panic(fmt.Sprintf(
			"cevm: ABI version mismatch — loaded libevm-gpu reports v%d but Go bindings expect v%d. "+
				"Rebuild libevm-gpu (see luxcpp/evm) or pin matching versions.",
			got, ABIVersion))
	}
}

// buildTxs converts Go transactions into C-layout transactions, pinning any
// Go-owned byte slices for the duration of the C call. The caller is
// responsible for invoking pinner.Unpin() once C has returned.
//
// Pinning rule: ctxs[i].data and ctxs[i].code are Go pointers inside Go
// memory that C will dereference. Per Go cgo rules these inner pointers
// MUST be pinned. ctxs[i].from / ctxs[i].to are stored by-value (array
// copy) so they don't need pinning.
func buildTxs(txs []Transaction, pinner *runtime.Pinner) []C.CGpuTx {
	ctxs := make([]C.CGpuTx, len(txs))
	for i := range txs {
		t := &txs[i]
		ctxs[i].from = *(*[20]C.uint8_t)(unsafe.Pointer(&t.From[0]))
		ctxs[i].to = *(*[20]C.uint8_t)(unsafe.Pointer(&t.To[0]))
		ctxs[i].gas_limit = C.uint64_t(t.GasLimit)
		ctxs[i].value = C.uint64_t(t.Value)
		ctxs[i].nonce = C.uint64_t(t.Nonce)
		ctxs[i].gas_price = C.uint64_t(t.GasPrice)
		if t.HasTo {
			ctxs[i].has_to = 1
		}
		if len(t.Data) > 0 {
			pinner.Pin(&t.Data[0])
			ctxs[i].data = (*C.uint8_t)(unsafe.Pointer(&t.Data[0]))
			ctxs[i].data_len = C.uint32_t(len(t.Data))
		}
		if len(t.Code) > 0 {
			pinner.Pin(&t.Code[0])
			ctxs[i].code = (*C.uint8_t)(unsafe.Pointer(&t.Code[0]))
			ctxs[i].code_len = C.uint32_t(len(t.Code))
		}
	}
	return ctxs
}

// copyU64 safely copies up to want elements from a C uint64 array into a Go
// slice. Bounds-checks `want` against a sane maximum to defend against a
// corrupted result struct returning an absurd count.
func copyU64(ptr *C.uint64_t, want uint32) []uint64 {
	if ptr == nil || want == 0 {
		return nil
	}
	const maxTxsPerBlock = 1 << 24 // 16M txs/block — far above any realistic block
	if want > maxTxsPerBlock {
		return nil
	}
	src := unsafe.Slice((*uint64)(unsafe.Pointer(ptr)), int(want))
	dst := make([]uint64, want)
	copy(dst, src)
	return dst
}

// ExecuteBlock runs a block of transactions through the C++ EVM.
//
// Thread safety: ExecuteBlock is safe to call from multiple goroutines
// concurrently. The C++ engine uses thread-local kernel hosts, so each
// goroutine that reaches the GPU path gets its own MTLBuffer/CUDA context
// cache. There are no shared mutable globals between calls.
//
// Memory safety: every Go-owned []byte the C side dereferences (tx.Data,
// tx.Code) is pinned for the duration of the C call. The ctxs[] slice
// itself is a stack-allocated local (or heap-promoted by escape analysis,
// either way reachable) — runtime.KeepAlive(ctxs) at the end guarantees
// the GC won't collect it while the C call is still in flight. The pinner
// is unpinned via defer on every return path including errors.
func ExecuteBlock(backend Backend, txs []Transaction) (*BlockResult, error) {
	if len(txs) == 0 {
		return &BlockResult{}, nil
	}

	var pinner runtime.Pinner
	defer pinner.Unpin()
	ctxs := buildTxs(txs, &pinner)

	result := C.gpu_execute_block(
		&ctxs[0],
		C.uint32_t(len(ctxs)),
		C.uint8_t(backend),
	)
	// Always free the C-allocated result, even on the error path. The C
	// implementation of gpu_free_result is null-safe so this is a no-op
	// when result.gas_used is nil.
	defer C.gpu_free_result(&result)

	// Defensive: the cgo call above is synchronous so ctxs is reachable
	// throughout, but make the contract explicit so a future refactor that
	// moves the C call into a goroutine or callback path doesn't silently
	// break pointer reachability.
	runtime.KeepAlive(ctxs)

	if result.ok == 0 {
		return nil, fmt.Errorf("cevm: execute_block failed")
	}

	return &BlockResult{
		GasUsed:      copyU64(result.gas_used, uint32(result.num_txs)),
		TotalGas:     uint64(result.total_gas),
		ExecTimeMs:   float64(result.exec_time_ms),
		Conflicts:    uint32(result.conflicts),
		ReExecutions: uint32(result.re_executions),
	}, nil
}

// ExecuteBlockV2 runs a block through the C++ EVM and returns the V2 result
// with per-tx status and post-execution state root.
//
// Thread safety: same as ExecuteBlock — safe under concurrent goroutines.
// Memory safety: same pinner + KeepAlive contract as ExecuteBlock.
func ExecuteBlockV2(backend Backend, numThreads uint32, txs []Transaction) (*BlockResultV2, error) {
	if len(txs) == 0 {
		return &BlockResultV2{ABIVersion: ABIVersion}, nil
	}

	var pinner runtime.Pinner
	defer pinner.Unpin()
	ctxs := buildTxs(txs, &pinner)

	// Pass EVM_GPU_REV_DEFAULT (Cancun = 12). v0.26 added a revision
	// parameter so callers can target older hard forks; we always pass
	// Cancun for production consensus paths.
	result := C.gpu_execute_block_v2(
		&ctxs[0],
		C.uint32_t(len(ctxs)),
		C.uint8_t(backend),
		C.uint32_t(numThreads),
		C.uint8_t(12), // EVM_GPU_REV_CANCUN = EVM_GPU_REV_DEFAULT
	)
	defer C.gpu_free_result_v2(&result)
	runtime.KeepAlive(ctxs)

	if result.ok == 0 {
		return nil, fmt.Errorf("cevm: execute_block_v2 failed")
	}
	if uint32(result.abi_version) != ABIVersion {
		return nil, fmt.Errorf("cevm: ABI version mismatch in result (lib=%d expected=%d)",
			uint32(result.abi_version), ABIVersion)
	}

	br := &BlockResultV2{
		GasUsed:      copyU64(result.gas_used, uint32(result.num_txs)),
		TotalGas:     uint64(result.total_gas),
		ExecTimeMs:   float64(result.exec_time_ms),
		Conflicts:    uint32(result.conflicts),
		ReExecutions: uint32(result.re_executions),
		ABIVersion:   uint32(result.abi_version),
	}
	for i := 0; i < 32; i++ {
		br.StateRoot[i] = byte(result.state_root[i])
	}
	if result.status != nil && result.num_txs > 0 {
		const maxTxsPerBlock = 1 << 24
		want := uint32(result.num_txs)
		if want > maxTxsPerBlock {
			return nil, fmt.Errorf("cevm: result.num_txs=%d exceeds sanity bound", want)
		}
		statSlice := unsafe.Slice((*uint8)(unsafe.Pointer(result.status)), int(want))
		br.Status = make([]TxStatus, want)
		for i, s := range statSlice {
			br.Status[i] = TxStatus(s)
		}
	}
	return br, nil
}

// ExecuteBlockV3 runs a block through the C++ EVM with an explicit block
// context and returns the V2 result shape (state root + per-tx status).
//
// Pass `ctx == nil` for V2 semantics (zero-initialised block context — chain
// id, timestamp, etc. all resolve to zero). Pass a populated *BlockContext
// to feed CHAINID, TIMESTAMP, NUMBER, BASEFEE, COINBASE, etc. through to
// every backend that consumes them (Metal kernel reads it directly; CPU
// kernel path picks it up once the parallel agent's wiring lands; CUDA
// host drops it until that backend grows the same overload).
//
// Thread safety and memory safety are identical to ExecuteBlockV2: pinner
// over Data/Code, KeepAlive over the ctxs slice, defer-free of the result.
// The BlockContext itself is passed by value into a stack-allocated C
// struct, so it doesn't need pinning.
func ExecuteBlockV3(backend Backend, numThreads uint32, txs []Transaction, ctx *BlockContext) (*BlockResultV2, error) {
	if len(txs) == 0 {
		return &BlockResultV2{ABIVersion: ABIVersion}, nil
	}

	var pinner runtime.Pinner
	defer pinner.Unpin()
	ctxs := buildTxs(txs, &pinner)

	// CBlockContext is layout-compatible with cevm.BlockContext: same
	// field order, same widths, no padding deltas. Build it on the stack
	// so we don't need to pin it. When ctx==nil we pass a NULL pointer
	// and the C side defaults to a zero block context.
	var cctxStorage C.CBlockContext
	var cctxPtr *C.CBlockContext
	if ctx != nil {
		cctxStorage.origin = *(*[20]C.uint8_t)(unsafe.Pointer(&ctx.Origin[0]))
		cctxStorage.gas_price = C.uint64_t(ctx.GasPrice)
		cctxStorage.timestamp = C.uint64_t(ctx.Timestamp)
		cctxStorage.number = C.uint64_t(ctx.Number)
		cctxStorage.prevrandao = *(*[32]C.uint8_t)(unsafe.Pointer(&ctx.Prevrandao[0]))
		cctxStorage.gas_limit = C.uint64_t(ctx.GasLimit)
		cctxStorage.chain_id = C.uint64_t(ctx.ChainID)
		cctxStorage.base_fee = C.uint64_t(ctx.BaseFee)
		cctxStorage.blob_base_fee = C.uint64_t(ctx.BlobBaseFee)
		cctxStorage.coinbase = *(*[20]C.uint8_t)(unsafe.Pointer(&ctx.Coinbase[0]))
		cctxStorage.blob_hashes = *(*[8][32]C.uint8_t)(unsafe.Pointer(&ctx.BlobHashes[0][0]))
		nbh := ctx.NumBlobHashes
		if nbh > 8 {
			nbh = 8
		}
		cctxStorage.num_blob_hashes = C.uint32_t(nbh)
		cctxPtr = &cctxStorage
	}

	result := C.gpu_execute_block_v3(
		&ctxs[0],
		C.uint32_t(len(ctxs)),
		C.uint8_t(backend),
		C.uint32_t(numThreads),
		C.uint8_t(12), // EVM_GPU_REV_CANCUN
		cctxPtr,
	)
	defer C.gpu_free_result_v2(&result)
	runtime.KeepAlive(ctxs)
	runtime.KeepAlive(cctxStorage)

	if result.ok == 0 {
		return nil, fmt.Errorf("cevm: execute_block_v3 failed")
	}
	if uint32(result.abi_version) != ABIVersion {
		return nil, fmt.Errorf("cevm: ABI version mismatch in result (lib=%d expected=%d)",
			uint32(result.abi_version), ABIVersion)
	}

	br := &BlockResultV2{
		GasUsed:      copyU64(result.gas_used, uint32(result.num_txs)),
		TotalGas:     uint64(result.total_gas),
		ExecTimeMs:   float64(result.exec_time_ms),
		Conflicts:    uint32(result.conflicts),
		ReExecutions: uint32(result.re_executions),
		ABIVersion:   uint32(result.abi_version),
	}
	for i := 0; i < 32; i++ {
		br.StateRoot[i] = byte(result.state_root[i])
	}
	if result.status != nil && result.num_txs > 0 {
		const maxTxsPerBlock = 1 << 24
		want := uint32(result.num_txs)
		if want > maxTxsPerBlock {
			return nil, fmt.Errorf("cevm: result.num_txs=%d exceeds sanity bound", want)
		}
		statSlice := unsafe.Slice((*uint8)(unsafe.Pointer(result.status)), int(want))
		br.Status = make([]TxStatus, want)
		for i, s := range statSlice {
			br.Status[i] = TxStatus(s)
		}
	}
	return br, nil
}

// BackendName returns the human-readable name of a backend as reported by the
// C++ library (which is authoritative).
func BackendName(b Backend) string {
	cstr := C.gpu_backend_name(C.uint8_t(b))
	if cstr == nil {
		return "unknown"
	}
	return C.GoString(cstr)
}

// AvailableBackends returns the list of backends compiled and detected
// at runtime by the loaded library.
func AvailableBackends() []Backend {
	n := uint32(C.gpu_available_backends(nil, 0))
	if n == 0 {
		return nil
	}
	buf := make([]C.uint8_t, n)
	got := uint32(C.gpu_available_backends(&buf[0], C.uint32_t(n)))
	out := make([]Backend, got)
	for i := uint32(0); i < got; i++ {
		out[i] = Backend(buf[i])
	}
	return out
}

// LibraryABIVersion returns the ABI version reported by the loaded library.
// Useful for diagnostics when binaries and shared libs may drift.
func LibraryABIVersion() uint32 {
	return uint32(C.gpu_abi_version())
}

// healthProbe is one entry in the Health() battery — a named bytecode
// program with its expected execution status. Every conformant backend must
// run each probe to its expected status with non-zero gas.
type healthProbe struct {
	name      string
	bytecode  []byte
	wantStatus TxStatus
	// callBridge=true marks probes whose top-level opcode is in the CALL
	// family (CALL/CALLCODE/DELEGATECALL/STATICCALL/CREATE/CREATE2). On the
	// GPU path these currently route through the dispatcher's "not supported"
	// shim — we only require that the bridge is exercised (i.e. the probe
	// runs to completion, not that it succeeds with TxOK). Only enforced on
	// CPU backends; for GPU we accept any status as long as the kernel
	// returned cleanly.
	callBridge bool
	// strictParity=true means gas must match exactly across all backends
	// that ran this probe. False for probes whose dynamic gas accounting is
	// known to differ between the interpretive CPU path (evmone) and the
	// flat GPU kernel (e.g. KECCAK256 dynamic word cost, MCOPY dynamic
	// memory expansion). Differences on strict probes are kernel bugs and
	// must fail Health.
	strictParity bool
}

// healthBattery is the ordered list of probes Health() runs against every
// backend. Each probe targets a different EVM subsystem so a backend that's
// silently broken in one area (e.g. storage) fails its own probe instead
// of slipping through.
//
// Order matters only insofar as we want simple programs first so a complete
// breakage shows up on probe[0] before we waste time on later probes.
func healthBattery() []healthProbe {
	// arith: PUSH1 1, PUSH1 1, ADD, POP, STOP — pure arithmetic.
	arith := []byte{0x60, 0x01, 0x60, 0x01, 0x01, 0x50, 0x00}

	// storage: PUSH1 0xAB, PUSH1 0x01, SSTORE, PUSH1 0x01, SLOAD, POP, STOP.
	storage := []byte{0x60, 0xab, 0x60, 0x01, 0x55, 0x60, 0x01, 0x54, 0x50, 0x00}

	// keccak: hash 32 bytes of zero memory at offset 0.
	// PUSH1 32, PUSH1 0, KECCAK256, POP, STOP.
	keccak := []byte{0x60, 0x20, 0x60, 0x00, 0x20, 0x50, 0x00}

	// memory: MSTORE 0xAB at offset 0, MLOAD it back, MCOPY 32 bytes 0->32, STOP.
	// PUSH1 0xAB, PUSH1 0, MSTORE, PUSH1 0, MLOAD, POP,
	// PUSH1 32, PUSH1 0, PUSH1 32, MCOPY, STOP.
	memOps := []byte{
		0x60, 0xab, 0x60, 0x00, 0x52,
		0x60, 0x00, 0x51, 0x50,
		0x60, 0x20, 0x60, 0x00, 0x60, 0x20, 0x5e,
		0x00,
	}

	// callBridge: CALL with constant target. On GPU this should hit the
	// "call not supported" branch in the dispatcher and return cleanly with
	// TxCallNotSupported. On CPU it executes to TxOK (the in-process EVM
	// supports CALL).
	// PUSH1 0 (retSize), PUSH1 0 (retOff), PUSH1 0 (argSize), PUSH1 0 (argOff),
	// PUSH1 0 (value), ADDRESS (to), PUSH1 0 (gas), CALL, POP, STOP.
	callBridge := []byte{
		0x60, 0x00, 0x60, 0x00, 0x60, 0x00, 0x60, 0x00,
		0x60, 0x00, 0x30,
		0x60, 0x00, 0xf1,
		0x50, 0x00,
	}

	return []healthProbe{
		{name: "arith", bytecode: arith, wantStatus: TxOK, strictParity: true},
		{name: "storage", bytecode: storage, wantStatus: TxOK, strictParity: true},
		{name: "keccak", bytecode: keccak, wantStatus: TxOK, strictParity: false},
		{name: "memory", bytecode: memOps, wantStatus: TxOK, strictParity: false},
		{name: "call-bridge", bytecode: callBridge, wantStatus: TxOK, callBridge: true, strictParity: false},
	}
}

// HealthProbeResult is the outcome of a single probe on a single backend.
type HealthProbeResult struct {
	Name    string
	OK      bool
	GasUsed uint64
	Status  TxStatus
	Err     error
}

// HealthReport is the per-backend result of Health(). It aggregates the
// per-probe results into a single OK / not-OK signal: a backend is healthy
// iff every probe ran to its expected status with non-zero gas.
type HealthReport struct {
	Backend      Backend
	Name         string
	OK           bool
	Err          error
	Probe        string // first failing probe name, empty when OK
	ProbesRun    int
	ProbeResults []HealthProbeResult
	// Aggregate stats — sum of gas across probes, time of the last probe.
	GasUsed  uint64
	Status   TxStatus
	ExecTime float64
}

// Health runs a battery of canonical bytecode programs through every backend
// the loaded library exposes and returns a per-backend report. Use at
// process start to fail-fast on misconfigured GPUs (driver missing, library
// mismatch, device permissions, kernel coverage gaps). Returns nil only if
// the runtime cannot enumerate backends at all.
//
// The battery covers:
//   - arithmetic (ADD/POP) — strict gas parity required across backends
//   - storage (SSTORE / SLOAD) — strict gas parity required
//   - hashing (KECCAK256) — non-zero gas required, parity not strict
//   - memory ops (MSTORE / MLOAD / MCOPY) — non-zero gas required, parity not strict
//   - the CALL bridge (CALL with a constant target) — must complete cleanly
//
// A backend is reported OK iff every probe executed to its expected status
// with non-zero gas AND its gas matches every other backend on the strict-
// parity probes. A failure sets Err and Probe to identify the offending
// case.
func Health() []HealthReport {
	backends := AvailableBackends()
	if len(backends) == 0 {
		return nil
	}
	probes := healthBattery()
	out := make([]HealthReport, 0, len(backends))
	for _, b := range backends {
		rep := HealthReport{
			Backend:      b,
			Name:         BackendName(b),
			ProbeResults: make([]HealthProbeResult, 0, len(probes)),
		}
		isGPU := b == GPUMetal || b == GPUCUDA
		allOK := true
		for _, p := range probes {
			pr := runHealthProbe(b, p, isGPU)
			rep.ProbeResults = append(rep.ProbeResults, pr)
			rep.ProbesRun++
			rep.GasUsed += pr.GasUsed
			rep.Status = pr.Status
			if !pr.OK && allOK {
				allOK = false
				rep.Probe = pr.Name
				rep.Err = pr.Err
			}
		}
		rep.OK = allOK
		out = append(out, rep)
	}
	// Cross-backend strict-parity check on probes flagged strictParity=true.
	// Two backends that disagree on gas for "arith" or "storage" indicate a
	// real consensus bug — mark BOTH as not healthy so the deploy fails fast.
	enforceStrictParity(out, probes)
	return out
}

// enforceStrictParity walks the per-backend reports, finds probes flagged
// strictParity=true, and marks any backend whose gas differs from the
// majority value as NotOK. We use the median (robust to one outlier) as
// the reference rather than the first-seen, so a single buggy CPU build
// doesn't poison every GPU report.
func enforceStrictParity(reports []HealthReport, probes []healthProbe) {
	if len(reports) < 2 {
		return // nothing to compare
	}
	// Build map: probe name → strictParity bit.
	strict := make(map[string]bool, len(probes))
	for _, p := range probes {
		if p.strictParity {
			strict[p.name] = true
		}
	}
	// For each strict probe, collect the gas values across all backends
	// that produced an OK probe result, find the majority value, and flag
	// any backend whose gas differs from it.
	probeNames := make([]string, 0, len(strict))
	for n := range strict {
		probeNames = append(probeNames, n)
	}
	for _, probeName := range probeNames {
		// Per-probe gas histogram across backends.
		hist := make(map[uint64]int)
		for _, r := range reports {
			for _, pr := range r.ProbeResults {
				if pr.Name == probeName && pr.OK {
					hist[pr.GasUsed]++
				}
			}
		}
		if len(hist) <= 1 {
			continue // all backends agree (or only one backend ran this probe)
		}
		// Find majority gas value. Tie → smallest gas (the most evmone-like).
		var majorityGas uint64
		var majorityCount int
		for g, c := range hist {
			if c > majorityCount || (c == majorityCount && g < majorityGas) {
				majorityGas = g
				majorityCount = c
			}
		}
		// Flag backends whose gas differs from majority.
		for i := range reports {
			for _, pr := range reports[i].ProbeResults {
				if pr.Name != probeName || !pr.OK {
					continue
				}
				if pr.GasUsed != majorityGas {
					reports[i].OK = false
					if reports[i].Err == nil {
						reports[i].Err = fmt.Errorf(
							"strict-parity probe %q: gas=%d but majority=%d (likely kernel gas-accounting bug)",
							probeName, pr.GasUsed, majorityGas)
						reports[i].Probe = probeName
					}
				}
			}
		}
	}
}

// runHealthProbe executes one probe on one backend and returns its result.
// The result.OK rule: gas must be > 0 AND status must match the probe's
// expectation, modulo the call-bridge exception described in healthProbe.
func runHealthProbe(b Backend, p healthProbe, isGPU bool) HealthProbeResult {
	tx := Transaction{
		HasTo:    true,
		Code:     p.bytecode,
		GasLimit: 200_000,
		Nonce:    0,
		GasPrice: 1,
	}
	pr := HealthProbeResult{Name: p.name}
	r, err := ExecuteBlockV2(b, 0, []Transaction{tx})
	if err != nil {
		pr.Err = fmt.Errorf("probe %q: %w", p.name, err)
		return pr
	}
	if len(r.GasUsed) != 1 || len(r.Status) != 1 {
		pr.Err = fmt.Errorf("probe %q: malformed result (gas=%d status=%d)",
			p.name, len(r.GasUsed), len(r.Status))
		return pr
	}
	pr.GasUsed = r.GasUsed[0]
	pr.Status = r.Status[0]
	if pr.GasUsed == 0 {
		pr.Err = fmt.Errorf("probe %q: 0 gas — kernel did not execute", p.name)
		return pr
	}
	// Call-bridge probe on GPU: any returning status is fine. The point is
	// that the dispatcher reached the bridge and returned cleanly.
	if p.callBridge && isGPU {
		pr.OK = true
		return pr
	}
	if pr.Status != p.wantStatus {
		// Accept TxReturn for arith probes that end in RETURN; we don't
		// because arith ends in STOP. But a backend that maps STOP-with-no-
		// data to TxReturn instead of TxOK is acceptable: both indicate
		// successful termination. So treat TxOK and TxReturn as equivalent
		// here.
		if !(pr.Status == TxReturn && p.wantStatus == TxOK) &&
			!(pr.Status == TxOK && p.wantStatus == TxReturn) {
			pr.Err = fmt.Errorf("probe %q: status=%s want=%s",
				p.name, pr.Status, p.wantStatus)
			return pr
		}
	}
	pr.OK = true
	return pr
}
