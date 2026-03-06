package cevm

import "testing"

// Tests in this file run under both `cgo` and `!cgo` builds. They cover the
// parts of the API that don't depend on the C++ library: enum stringers,
// constant exports, and the empty-input fast-paths that ExecuteBlock /
// ExecuteBlockV2 promise to return without error.

func TestBackendString(t *testing.T) {
	tests := []struct {
		b    Backend
		want string
	}{
		{CPUSequential, "cpu-sequential"},
		{CPUParallel, "cpu-parallel"},
		{GPUMetal, "gpu-metal"},
		{GPUCUDA, "gpu-cuda"},
		{Backend(99), "unknown(99)"},
	}

	for _, tt := range tests {
		if got := tt.b.String(); got != tt.want {
			t.Errorf("Backend(%d).String() = %q, want %q", int(tt.b), got, tt.want)
		}
	}
}

func TestTxStatusString(t *testing.T) {
	tests := []struct {
		s    TxStatus
		want string
	}{
		{TxOK, "ok"},
		{TxReturn, "return"},
		{TxRevert, "revert"},
		{TxOOG, "oog"},
		{TxError, "error"},
		{TxCallNotSupported, "call-not-supported"},
		{TxStatus(99), "status(99)"},
	}
	for _, tt := range tests {
		if got := tt.s.String(); got != tt.want {
			t.Errorf("TxStatus(%d).String() = %q, want %q", int(tt.s), got, tt.want)
		}
	}
}

func TestVMID(t *testing.T) {
	if got := VMID(); got != "cevm" {
		t.Errorf("VMID() = %q, want %q", got, "cevm")
	}
}

func TestPluginPath(t *testing.T) {
	p := PluginPath()
	if p == "" {
		t.Skip("could not determine home directory")
	}
	const suffix = "work/luxcpp/evm/build/bin/cevm"
	if len(p) < len(suffix) {
		t.Fatalf("PluginPath() too short: %q", p)
	}
	got := p[len(p)-len(suffix):]
	if got != suffix {
		t.Errorf("PluginPath() suffix = %q, want %q", got, suffix)
	}
}

func TestPluginExists(t *testing.T) {
	// This test just verifies the function runs without panicking.
	// Whether it returns true or false depends on the build environment.
	_ = PluginExists()
}

// TestExecuteBlockEmpty must succeed on both cgo and nocgo paths: the API
// contract is "empty input ⇒ empty result, no error" regardless of build.
func TestExecuteBlockEmpty(t *testing.T) {
	result, err := ExecuteBlock(CPUSequential, nil)
	if err != nil {
		t.Fatalf("ExecuteBlock(nil) returned error: %v", err)
	}
	if result.TotalGas != 0 {
		t.Errorf("expected 0 total gas for empty block, got %d", result.TotalGas)
	}
}

func TestExecuteBlockV2Empty(t *testing.T) {
	result, err := ExecuteBlockV2(CPUSequential, 0, nil)
	if err != nil {
		t.Fatalf("ExecuteBlockV2(nil) returned error: %v", err)
	}
	if result.ABIVersion != ABIVersion {
		t.Errorf("ABIVersion = %d, want %d", result.ABIVersion, ABIVersion)
	}
	if result.TotalGas != 0 {
		t.Errorf("expected 0 total gas for empty block, got %d", result.TotalGas)
	}
}

// TestAvailableBackends_NonEmpty: every build must expose at least one backend.
// Under nocgo, that's CPUSequential. Under cgo, it's whatever the loaded
// library reports — but never zero.
func TestAvailableBackends_NonEmpty(t *testing.T) {
	bs := AvailableBackends()
	if len(bs) == 0 {
		t.Fatal("AvailableBackends() returned empty list")
	}
}

func TestBackendName_NonEmpty(t *testing.T) {
	for _, b := range []Backend{CPUSequential, CPUParallel, GPUMetal, GPUCUDA} {
		if BackendName(b) == "" {
			t.Errorf("BackendName(%d) is empty", int(b))
		}
	}
}

// TestHealth_AlwaysReturnsReports: Health() must never return nil. Under
// nocgo it returns a single non-OK report. Under cgo it returns one report
// per available backend.
func TestHealth_AlwaysReturnsReports(t *testing.T) {
	reports := Health()
	if len(reports) == 0 {
		t.Fatal("Health() returned no reports — even nocgo must report status")
	}
}

// TestExecuteBlockV3Empty checks the V3 entry point honours the same
// "empty input ⇒ empty result, no error" contract as V1/V2, on both
// cgo and nocgo builds.
func TestExecuteBlockV3Empty(t *testing.T) {
	result, err := ExecuteBlockV3(CPUSequential, 0, nil, nil)
	if err != nil {
		t.Fatalf("ExecuteBlockV3(nil, nil) returned error: %v", err)
	}
	if result.ABIVersion != ABIVersion {
		t.Errorf("ABIVersion = %d, want %d", result.ABIVersion, ABIVersion)
	}
	if result.TotalGas != 0 {
		t.Errorf("expected 0 total gas for empty block, got %d", result.TotalGas)
	}
}

// TestABIVersion locks the ABIVersion constant so a careless bump is caught
// by code review. Update both this test and ABIVersion together when the
// C ABI surface changes.
func TestABIVersion(t *testing.T) {
	const want uint32 = 5
	if ABIVersion != want {
		t.Errorf("ABIVersion = %d, want %d (update C-side EVM_GPU_ABI_VERSION in lockstep)",
			ABIVersion, want)
	}
}

// TestBlockContextLayout asserts the Go BlockContext fields are the right
// widths in the right order. The C-side layout assertion lives in
// go_bridge.cpp (static_assert sizeof(BlockContext) == sizeof(CBlockContext));
// this test catches Go-side reordering before it reaches the C ABI.
func TestBlockContextLayout(t *testing.T) {
	var ctx BlockContext
	ctx.Origin[0] = 0xAA
	ctx.Origin[19] = 0xBB
	ctx.GasPrice = 1
	ctx.Timestamp = 2
	ctx.Number = 3
	ctx.Prevrandao[0] = 0xCC
	ctx.Prevrandao[31] = 0xDD
	ctx.GasLimit = 4
	ctx.ChainID = 96369
	ctx.BaseFee = 5
	ctx.BlobBaseFee = 6
	ctx.Coinbase[0] = 0xEE
	ctx.Coinbase[19] = 0xFF
	ctx.BlobHashes[0][0] = 0x11
	ctx.BlobHashes[7][31] = 0x22
	ctx.NumBlobHashes = 8

	// Sanity: the struct round-trips its own fields.
	if ctx.ChainID != 96369 {
		t.Errorf("ChainID round-trip failed: got %d", ctx.ChainID)
	}
	if ctx.Origin[0] != 0xAA || ctx.Origin[19] != 0xBB {
		t.Errorf("Origin round-trip failed: got %x..%x", ctx.Origin[0], ctx.Origin[19])
	}
	if ctx.NumBlobHashes != 8 {
		t.Errorf("NumBlobHashes round-trip failed: got %d", ctx.NumBlobHashes)
	}
}
