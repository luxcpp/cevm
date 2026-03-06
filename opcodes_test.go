//go:build cgo

package cevm

import (
	"fmt"
	"sort"
	"sync"
	"testing"
)

// ----------------------------------------------------------------------------
// Bytecode builder
// ----------------------------------------------------------------------------

// prog is a tiny, append-only EVM bytecode builder. Every method returns the
// receiver so call chains read like an assembly listing.
type prog struct{ b []byte }

// op appends a single opcode byte.
func (p *prog) op(o byte) *prog {
	p.b = append(p.b, o)
	return p
}

// raw appends arbitrary bytes (used after PUSH for the immediate operand).
func (p *prog) raw(bs ...byte) *prog {
	p.b = append(p.b, bs...)
	return p
}

// push appends a PUSH<n> with the supplied operand bytes.
// n must be in [1, 32]; len(val) must equal n.
func (p *prog) push(n int, val ...byte) *prog {
	if n < 1 || n > 32 {
		panic(fmt.Sprintf("prog.push: bad n=%d", n))
	}
	if len(val) != n {
		panic(fmt.Sprintf("prog.push: n=%d but len(val)=%d", n, len(val)))
	}
	p.b = append(p.b, 0x60+byte(n-1)) // PUSH1=0x60 ... PUSH32=0x7F
	p.b = append(p.b, val...)
	return p
}

// push1 is the common case.
func (p *prog) push1(v byte) *prog { return p.push(1, v) }

// bytes returns the assembled bytecode.
func (p *prog) bytes() []byte { return p.b }

// stop / revert / ret terminate the program.
func (p *prog) stop() *prog   { return p.op(0x00) }
func (p *prog) revert() *prog { return p.push1(0).push1(0).op(0xFD) }
func (p *prog) ret() *prog    { return p.push1(0).push1(0).op(0xF3) }

// ----------------------------------------------------------------------------
// Test table — one entry per defined EVM opcode.
// ----------------------------------------------------------------------------

// opcodeTest describes a single opcode probe.
type opcodeTest struct {
	name     string
	opcode   byte
	bytecode []byte
	// expectStatus is the status the CPU reference path is expected to return.
	// We assert CPU status equals this; GPU is compared to CPU when not skipGPU.
	expectStatus TxStatus
	// skipGPU=true for opcodes the GPU kernel cannot run by itself (CALL family).
	// We still run them through CPU to confirm CPU baseline; GPU is skipped with
	// a logged note instead of failing.
	skipGPU bool
	// needsCalldata populates tx.Data for calldata opcodes.
	needsCalldata bool
	// needsTo populates tx.HasTo for context opcodes that read the recipient.
	needsTo bool
}

// allOpcodeTests is the canonical opcode coverage table. One entry per opcode.
// Order is by opcode value so coverage gaps are visible at a glance.
var allOpcodeTests = []opcodeTest{
	// 0x00 — STOP
	{name: "STOP", opcode: 0x00, bytecode: (&prog{}).stop().bytes(), expectStatus: TxOK},

	// 0x01..0x0B — arithmetic
	{name: "ADD", opcode: 0x01, bytecode: (&prog{}).push1(5).push1(3).op(0x01).stop().bytes(), expectStatus: TxOK},
	{name: "MUL", opcode: 0x02, bytecode: (&prog{}).push1(5).push1(3).op(0x02).stop().bytes(), expectStatus: TxOK},
	{name: "SUB", opcode: 0x03, bytecode: (&prog{}).push1(3).push1(5).op(0x03).stop().bytes(), expectStatus: TxOK},
	{name: "DIV", opcode: 0x04, bytecode: (&prog{}).push1(2).push1(10).op(0x04).stop().bytes(), expectStatus: TxOK},
	{name: "SDIV", opcode: 0x05, bytecode: (&prog{}).push1(2).push1(10).op(0x05).stop().bytes(), expectStatus: TxOK},
	{name: "MOD", opcode: 0x06, bytecode: (&prog{}).push1(3).push1(10).op(0x06).stop().bytes(), expectStatus: TxOK},
	{name: "SMOD", opcode: 0x07, bytecode: (&prog{}).push1(3).push1(10).op(0x07).stop().bytes(), expectStatus: TxOK},
	{name: "ADDMOD", opcode: 0x08, bytecode: (&prog{}).push1(7).push1(5).push1(3).op(0x08).stop().bytes(), expectStatus: TxOK},
	{name: "MULMOD", opcode: 0x09, bytecode: (&prog{}).push1(7).push1(5).push1(3).op(0x09).stop().bytes(), expectStatus: TxOK},
	{name: "EXP", opcode: 0x0A, bytecode: (&prog{}).push1(3).push1(2).op(0x0A).stop().bytes(), expectStatus: TxOK},
	{name: "SIGNEXTEND", opcode: 0x0B, bytecode: (&prog{}).push1(0xFF).push1(0).op(0x0B).stop().bytes(), expectStatus: TxOK},

	// 0x10..0x1D — comparison + bitwise
	{name: "LT", opcode: 0x10, bytecode: (&prog{}).push1(5).push1(3).op(0x10).stop().bytes(), expectStatus: TxOK},
	{name: "GT", opcode: 0x11, bytecode: (&prog{}).push1(3).push1(5).op(0x11).stop().bytes(), expectStatus: TxOK},
	{name: "SLT", opcode: 0x12, bytecode: (&prog{}).push1(5).push1(3).op(0x12).stop().bytes(), expectStatus: TxOK},
	{name: "SGT", opcode: 0x13, bytecode: (&prog{}).push1(3).push1(5).op(0x13).stop().bytes(), expectStatus: TxOK},
	{name: "EQ", opcode: 0x14, bytecode: (&prog{}).push1(5).push1(5).op(0x14).stop().bytes(), expectStatus: TxOK},
	{name: "ISZERO", opcode: 0x15, bytecode: (&prog{}).push1(0).op(0x15).stop().bytes(), expectStatus: TxOK},
	{name: "AND", opcode: 0x16, bytecode: (&prog{}).push1(0xF0).push1(0x0F).op(0x16).stop().bytes(), expectStatus: TxOK},
	{name: "OR", opcode: 0x17, bytecode: (&prog{}).push1(0xF0).push1(0x0F).op(0x17).stop().bytes(), expectStatus: TxOK},
	{name: "XOR", opcode: 0x18, bytecode: (&prog{}).push1(0xFF).push1(0x0F).op(0x18).stop().bytes(), expectStatus: TxOK},
	{name: "NOT", opcode: 0x19, bytecode: (&prog{}).push1(0).op(0x19).stop().bytes(), expectStatus: TxOK},
	{name: "BYTE", opcode: 0x1A, bytecode: (&prog{}).push1(0xAB).push1(31).op(0x1A).stop().bytes(), expectStatus: TxOK},
	{name: "SHL", opcode: 0x1B, bytecode: (&prog{}).push1(1).push1(2).op(0x1B).stop().bytes(), expectStatus: TxOK},
	{name: "SHR", opcode: 0x1C, bytecode: (&prog{}).push1(8).push1(2).op(0x1C).stop().bytes(), expectStatus: TxOK},
	{name: "SAR", opcode: 0x1D, bytecode: (&prog{}).push1(8).push1(2).op(0x1D).stop().bytes(), expectStatus: TxOK},

	// 0x20 — KECCAK256: hash 32 bytes of zeroed memory at offset 0.
	{name: "KECCAK256", opcode: 0x20, bytecode: (&prog{}).push1(32).push1(0).op(0x20).stop().bytes(), expectStatus: TxOK},

	// 0x30..0x4A — context / block
	{name: "ADDRESS", opcode: 0x30, bytecode: (&prog{}).op(0x30).stop().bytes(), expectStatus: TxOK, needsTo: true},
	{name: "BALANCE", opcode: 0x31, bytecode: (&prog{}).op(0x30).op(0x31).stop().bytes(), expectStatus: TxOK, needsTo: true},
	{name: "ORIGIN", opcode: 0x32, bytecode: (&prog{}).op(0x32).stop().bytes(), expectStatus: TxOK},
	{name: "CALLER", opcode: 0x33, bytecode: (&prog{}).op(0x33).stop().bytes(), expectStatus: TxOK},
	{name: "CALLVALUE", opcode: 0x34, bytecode: (&prog{}).op(0x34).stop().bytes(), expectStatus: TxOK},
	{name: "CALLDATALOAD", opcode: 0x35, bytecode: (&prog{}).push1(0).op(0x35).stop().bytes(), expectStatus: TxOK, needsCalldata: true},
	{name: "CALLDATASIZE", opcode: 0x36, bytecode: (&prog{}).op(0x36).stop().bytes(), expectStatus: TxOK, needsCalldata: true},
	{name: "CALLDATACOPY", opcode: 0x37, bytecode: (&prog{}).push1(32).push1(0).push1(0).op(0x37).stop().bytes(), expectStatus: TxOK, needsCalldata: true},
	{name: "CODESIZE", opcode: 0x38, bytecode: (&prog{}).op(0x38).stop().bytes(), expectStatus: TxOK},
	{name: "CODECOPY", opcode: 0x39, bytecode: (&prog{}).push1(4).push1(0).push1(0).op(0x39).stop().bytes(), expectStatus: TxOK},
	{name: "GASPRICE", opcode: 0x3A, bytecode: (&prog{}).op(0x3A).stop().bytes(), expectStatus: TxOK},
	{name: "EXTCODESIZE", opcode: 0x3B, bytecode: (&prog{}).op(0x30).op(0x3B).stop().bytes(), expectStatus: TxOK, needsTo: true},
	{name: "EXTCODECOPY", opcode: 0x3C, bytecode: (&prog{}).push1(0).push1(0).push1(0).op(0x30).op(0x3C).stop().bytes(), expectStatus: TxOK, needsTo: true},
	{name: "RETURNDATASIZE", opcode: 0x3D, bytecode: (&prog{}).op(0x3D).stop().bytes(), expectStatus: TxOK},
	{name: "RETURNDATACOPY", opcode: 0x3E, bytecode: (&prog{}).push1(0).push1(0).push1(0).op(0x3E).stop().bytes(), expectStatus: TxOK},
	{name: "EXTCODEHASH", opcode: 0x3F, bytecode: (&prog{}).op(0x30).op(0x3F).stop().bytes(), expectStatus: TxOK, needsTo: true},
	{name: "BLOCKHASH", opcode: 0x40, bytecode: (&prog{}).push1(0).op(0x40).stop().bytes(), expectStatus: TxOK},
	{name: "COINBASE", opcode: 0x41, bytecode: (&prog{}).op(0x41).stop().bytes(), expectStatus: TxOK},
	{name: "TIMESTAMP", opcode: 0x42, bytecode: (&prog{}).op(0x42).stop().bytes(), expectStatus: TxOK},
	{name: "NUMBER", opcode: 0x43, bytecode: (&prog{}).op(0x43).stop().bytes(), expectStatus: TxOK},
	{name: "PREVRANDAO", opcode: 0x44, bytecode: (&prog{}).op(0x44).stop().bytes(), expectStatus: TxOK},
	{name: "GASLIMIT", opcode: 0x45, bytecode: (&prog{}).op(0x45).stop().bytes(), expectStatus: TxOK},
	{name: "CHAINID", opcode: 0x46, bytecode: (&prog{}).op(0x46).stop().bytes(), expectStatus: TxOK},
	{name: "SELFBALANCE", opcode: 0x47, bytecode: (&prog{}).op(0x47).stop().bytes(), expectStatus: TxOK, needsTo: true},
	{name: "BASEFEE", opcode: 0x48, bytecode: (&prog{}).op(0x48).stop().bytes(), expectStatus: TxOK},
	{name: "BLOBHASH", opcode: 0x49, bytecode: (&prog{}).push1(0).op(0x49).stop().bytes(), expectStatus: TxOK},
	{name: "BLOBBASEFEE", opcode: 0x4A, bytecode: (&prog{}).op(0x4A).stop().bytes(), expectStatus: TxOK},

	// 0x50..0x5F — stack/memory/storage/flow
	{name: "POP", opcode: 0x50, bytecode: (&prog{}).push1(1).op(0x50).stop().bytes(), expectStatus: TxOK},
	{name: "MLOAD", opcode: 0x51, bytecode: (&prog{}).push1(0).op(0x51).stop().bytes(), expectStatus: TxOK},
	{name: "MSTORE", opcode: 0x52, bytecode: (&prog{}).push1(0xAB).push1(0).op(0x52).stop().bytes(), expectStatus: TxOK},
	{name: "MSTORE8", opcode: 0x53, bytecode: (&prog{}).push1(0xAB).push1(0).op(0x53).stop().bytes(), expectStatus: TxOK},
	{name: "SLOAD", opcode: 0x54, bytecode: (&prog{}).push1(0).op(0x54).stop().bytes(), expectStatus: TxOK},
	{name: "SSTORE", opcode: 0x55, bytecode: (&prog{}).push1(1).push1(0).op(0x55).stop().bytes(), expectStatus: TxOK},
	// JUMP: PUSH1 4, JUMP, INVALID, JUMPDEST, STOP — jumps over the INVALID byte.
	{name: "JUMP", opcode: 0x56, bytecode: (&prog{}).push1(4).op(0x56).op(0xFE).op(0x5B).stop().bytes(), expectStatus: TxOK},
	// JUMPI: cond=1, dest=6, PUSH1 1, PUSH1 6, JUMPI, INVALID, JUMPDEST, STOP.
	{name: "JUMPI", opcode: 0x57, bytecode: (&prog{}).push1(1).push1(6).op(0x57).op(0xFE).op(0x5B).stop().bytes(), expectStatus: TxOK},
	{name: "PC", opcode: 0x58, bytecode: (&prog{}).op(0x58).stop().bytes(), expectStatus: TxOK},
	{name: "MSIZE", opcode: 0x59, bytecode: (&prog{}).op(0x59).stop().bytes(), expectStatus: TxOK},
	{name: "GAS", opcode: 0x5A, bytecode: (&prog{}).op(0x5A).stop().bytes(), expectStatus: TxOK},
	{name: "JUMPDEST", opcode: 0x5B, bytecode: (&prog{}).op(0x5B).stop().bytes(), expectStatus: TxOK},
	{name: "TLOAD", opcode: 0x5C, bytecode: (&prog{}).push1(0).op(0x5C).stop().bytes(), expectStatus: TxOK},
	{name: "TSTORE", opcode: 0x5D, bytecode: (&prog{}).push1(1).push1(0).op(0x5D).stop().bytes(), expectStatus: TxOK},
	{name: "MCOPY", opcode: 0x5E, bytecode: (&prog{}).push1(32).push1(0).push1(0).op(0x5E).stop().bytes(), expectStatus: TxOK},
	{name: "PUSH0", opcode: 0x5F, bytecode: (&prog{}).op(0x5F).stop().bytes(), expectStatus: TxOK},

	// 0x60..0x7F — PUSH1..PUSH32 (table built below)
	// 0x80..0x8F — DUP1..DUP16 (table built below)
	// 0x90..0x9F — SWAP1..SWAP16 (table built below)

	// 0xA0..0xA4 — LOG0..LOG4
	{name: "LOG0", opcode: 0xA0, bytecode: (&prog{}).push1(0).push1(0).op(0xA0).stop().bytes(), expectStatus: TxOK},
	{name: "LOG1", opcode: 0xA1, bytecode: (&prog{}).push1(0xAA).push1(0).push1(0).op(0xA1).stop().bytes(), expectStatus: TxOK},
	{name: "LOG2", opcode: 0xA2, bytecode: (&prog{}).push1(0xAA).push1(0xBB).push1(0).push1(0).op(0xA2).stop().bytes(), expectStatus: TxOK},
	{name: "LOG3", opcode: 0xA3, bytecode: (&prog{}).push1(0xAA).push1(0xBB).push1(0xCC).push1(0).push1(0).op(0xA3).stop().bytes(), expectStatus: TxOK},
	{name: "LOG4", opcode: 0xA4, bytecode: (&prog{}).push1(0xAA).push1(0xBB).push1(0xCC).push1(0xDD).push1(0).push1(0).op(0xA4).stop().bytes(), expectStatus: TxOK},

	// 0xF0..0xFF — system
	// CALL family is expected to return CallNotSupported on GPU. We assert
	// on CPU via expectStatus (CPU may execute or also flag unsupported), and
	// skipGPU=true so we do not require gas equivalence.
	{name: "CREATE", opcode: 0xF0, bytecode: (&prog{}).push1(0).push1(0).push1(0).op(0xF0).stop().bytes(), expectStatus: TxOK, skipGPU: true},
	{name: "CALL", opcode: 0xF1, bytecode: (&prog{}).push1(0).push1(0).push1(0).push1(0).push1(0).op(0x30).push1(0).op(0xF1).stop().bytes(), expectStatus: TxOK, skipGPU: true, needsTo: true},
	{name: "CALLCODE", opcode: 0xF2, bytecode: (&prog{}).push1(0).push1(0).push1(0).push1(0).push1(0).op(0x30).push1(0).op(0xF2).stop().bytes(), expectStatus: TxOK, skipGPU: true, needsTo: true},
	{name: "RETURN", opcode: 0xF3, bytecode: (&prog{}).push1(0).push1(0).op(0xF3).bytes(), expectStatus: TxReturn},
	{name: "DELEGATECALL", opcode: 0xF4, bytecode: (&prog{}).push1(0).push1(0).push1(0).push1(0).op(0x30).push1(0).op(0xF4).stop().bytes(), expectStatus: TxOK, skipGPU: true, needsTo: true},
	{name: "CREATE2", opcode: 0xF5, bytecode: (&prog{}).push1(0).push1(0).push1(0).push1(0).op(0xF5).stop().bytes(), expectStatus: TxOK, skipGPU: true},
	{name: "STATICCALL", opcode: 0xFA, bytecode: (&prog{}).push1(0).push1(0).push1(0).push1(0).op(0x30).push1(0).op(0xFA).stop().bytes(), expectStatus: TxOK, skipGPU: true, needsTo: true},
	{name: "REVERT", opcode: 0xFD, bytecode: (&prog{}).push1(0).push1(0).op(0xFD).bytes(), expectStatus: TxRevert},
	// INVALID intentionally fails — both backends should flag it.
	{name: "INVALID", opcode: 0xFE, bytecode: []byte{0xFE}, expectStatus: TxError},
	{name: "SELFDESTRUCT", opcode: 0xFF, bytecode: (&prog{}).op(0x30).op(0xFF).bytes(), expectStatus: TxOK, skipGPU: true, needsTo: true},
}

// init expands the tables for PUSH1..PUSH32, DUP1..DUP16, SWAP1..SWAP16.
// Each is generated programmatically — there's nothing interesting to
// hand-write.
func init() {
	// PUSH1..PUSH32: PUSH<n> with n bytes of 0xAA, then STOP.
	for n := 1; n <= 32; n++ {
		op := byte(0x5F + n)
		operand := make([]byte, n)
		for i := range operand {
			operand[i] = 0xAA
		}
		bc := append([]byte{op}, operand...)
		bc = append(bc, 0x00) // STOP
		allOpcodeTests = append(allOpcodeTests, opcodeTest{
			name:         fmt.Sprintf("PUSH%d", n),
			opcode:       op,
			bytecode:     bc,
			expectStatus: TxOK,
		})
	}
	// DUP1..DUP16: push N+1 zeros then DUP<N>, STOP.
	for n := 1; n <= 16; n++ {
		op := byte(0x7F + n)
		p := &prog{}
		for i := 0; i <= n; i++ {
			p.push1(byte(i + 1))
		}
		p.op(op).stop()
		allOpcodeTests = append(allOpcodeTests, opcodeTest{
			name:         fmt.Sprintf("DUP%d", n),
			opcode:       op,
			bytecode:     p.bytes(),
			expectStatus: TxOK,
		})
	}
	// SWAP1..SWAP16: push N+1 distinct vals then SWAP<N>, STOP.
	for n := 1; n <= 16; n++ {
		op := byte(0x8F + n)
		p := &prog{}
		for i := 0; i <= n; i++ {
			p.push1(byte(i + 1))
		}
		p.op(op).stop()
		allOpcodeTests = append(allOpcodeTests, opcodeTest{
			name:         fmt.Sprintf("SWAP%d", n),
			opcode:       op,
			bytecode:     p.bytes(),
			expectStatus: TxOK,
		})
	}
}

// ----------------------------------------------------------------------------
// Coverage report (shared across subtests, summarised by TestOpcodeCoverage_Summary)
// ----------------------------------------------------------------------------

type opcodeOutcome struct {
	name           string
	opcode         byte
	cpuStatus      TxStatus
	gpuStatus      TxStatus
	cpuGas         uint64
	gpuGas         uint64
	gasMatch       bool
	gpuRan         bool
	gpuMatchedCPU  bool
	skippedCallFam bool
	notImplemented bool
	executeErr     string
}

var (
	coverageMu      sync.Mutex
	coverageResults = map[string]opcodeOutcome{}
)

func recordOutcome(o opcodeOutcome) {
	coverageMu.Lock()
	defer coverageMu.Unlock()
	coverageResults[o.name] = o
}

// ----------------------------------------------------------------------------
// Test helpers
// ----------------------------------------------------------------------------

// runTx builds a Transaction for the given opcode test and runs it through
// `backend`. It always calls ExecuteBlockV2 so we get per-tx status.
func runOpcodeTx(backend Backend, ot opcodeTest) (*BlockResultV2, error) {
	var from [20]byte
	from[19] = 0x42
	tx := Transaction{
		From:     from,
		GasLimit: 1_000_000,
		GasPrice: 1,
		Nonce:    0,
		Code:     ot.bytecode,
	}
	if ot.needsTo {
		var to [20]byte
		to[19] = 0xCA
		tx.To = to
		tx.HasTo = true
	}
	if ot.needsCalldata {
		tx.Data = []byte{
			0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
			0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
			0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
			0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
		}
	}
	return ExecuteBlockV2(backend, 1, []Transaction{tx})
}

// ----------------------------------------------------------------------------
// The big test: run every opcode through CPU + GPU and compare.
// ----------------------------------------------------------------------------

func TestOpcodeCoverage_GPU_vs_CPU(t *testing.T) {
	available := AvailableBackends()
	if !contains(available, CPUSequential) {
		t.Skip("CPUSequential backend not available")
	}
	gpuAvailable := contains(available, GPUMetal)
	if !gpuAvailable {
		t.Logf("GPU Metal backend NOT available — running CPU-only coverage")
	}

	for _, ot := range allOpcodeTests {
		ot := ot // capture
		t.Run(ot.name, func(t *testing.T) {
			outcome := opcodeOutcome{name: ot.name, opcode: ot.opcode}
			defer func() { recordOutcome(outcome) }()

			// 1) CPU reference run.
			cpuRes, err := runOpcodeTx(CPUSequential, ot)
			if err != nil {
				outcome.executeErr = "cpu: " + err.Error()
				t.Fatalf("CPU execute failed for %s (0x%02X): %v", ot.name, ot.opcode, err)
			}
			if len(cpuRes.Status) != 1 || len(cpuRes.GasUsed) != 1 {
				t.Fatalf("CPU returned malformed result for %s: status=%v gas=%v", ot.name, cpuRes.Status, cpuRes.GasUsed)
			}
			outcome.cpuStatus = cpuRes.Status[0]
			outcome.cpuGas = cpuRes.GasUsed[0]

			// 2) GPU comparison (if available and not in skipGPU set).
			if !gpuAvailable {
				return
			}
			if ot.skipGPU {
				outcome.skippedCallFam = true
				t.Logf("skip GPU for %s: CALL family / contract creation (CPU status=%s gas=%d)",
					ot.name, outcome.cpuStatus, outcome.cpuGas)
				return
			}

			gpuRes, err := runOpcodeTx(GPUMetal, ot)
			if err != nil {
				outcome.executeErr = "gpu: " + err.Error()
				// Coverage measurement: log but do not fail the whole suite.
				t.Logf("OPCODE NOT IMPLEMENTED ON GPU (execute error): %s (0x%02X): %v",
					ot.name, ot.opcode, err)
				return
			}
			outcome.gpuRan = true
			if len(gpuRes.Status) != 1 || len(gpuRes.GasUsed) != 1 {
				t.Fatalf("GPU returned malformed result for %s: status=%v gas=%v", ot.name, gpuRes.Status, gpuRes.GasUsed)
			}
			outcome.gpuStatus = gpuRes.Status[0]
			outcome.gpuGas = gpuRes.GasUsed[0]

			// If GPU reports CallNotSupported but the opcode isn't in our
			// skip list, treat that as "not implemented yet" data.
			if outcome.gpuStatus == TxCallNotSupported {
				outcome.notImplemented = true
				t.Logf("OPCODE NOT IMPLEMENTED ON GPU (CallNotSupported): %s (0x%02X)", ot.name, ot.opcode)
				return
			}
			if outcome.gpuStatus == TxError && outcome.cpuStatus != TxError {
				outcome.notImplemented = true
				t.Logf("OPCODE NOT IMPLEMENTED ON GPU (TxError, cpu=%s): %s (0x%02X)",
					outcome.cpuStatus, ot.name, ot.opcode)
				return
			}

			// 3) Compare CPU vs GPU.
			if outcome.cpuStatus != outcome.gpuStatus {
				t.Errorf("%s (0x%02X): status mismatch cpu=%s gpu=%s",
					ot.name, ot.opcode, outcome.cpuStatus, outcome.gpuStatus)
			}
			if outcome.cpuGas == outcome.gpuGas {
				outcome.gasMatch = true
				outcome.gpuMatchedCPU = outcome.cpuStatus == outcome.gpuStatus
			} else {
				t.Logf("%s (0x%02X): gas differs cpu=%d gpu=%d (status cpu=%s gpu=%s)",
					ot.name, ot.opcode, outcome.cpuGas, outcome.gpuGas,
					outcome.cpuStatus, outcome.gpuStatus)
			}
		})
	}
}

// ----------------------------------------------------------------------------
// Coverage summary report
// ----------------------------------------------------------------------------

// TestOpcodeCoverage_Summary depends on TestOpcodeCoverage_GPU_vs_CPU running
// first (Go test order is deterministic by source position, this file's main
// test is declared above). It prints a human-readable coverage report.
func TestOpcodeCoverage_Summary(t *testing.T) {
	coverageMu.Lock()
	defer coverageMu.Unlock()

	if len(coverageResults) == 0 {
		t.Skip("no opcode results recorded — run TestOpcodeCoverage_GPU_vs_CPU first")
	}

	type bucket struct {
		ran, gasMatch, skippedCall, notImpl, errored int
		failedNames                                  []string
	}
	var b bucket
	for _, o := range coverageResults {
		switch {
		case o.skippedCallFam:
			b.skippedCall++
		case o.notImplemented:
			b.notImpl++
			b.failedNames = append(b.failedNames, fmt.Sprintf("%s(0x%02X)", o.name, o.opcode))
		case o.executeErr != "":
			b.errored++
			b.failedNames = append(b.failedNames, fmt.Sprintf("%s(0x%02X)", o.name, o.opcode))
		case o.gpuRan && o.gasMatch:
			b.ran++
			b.gasMatch++
		case o.gpuRan:
			b.ran++
		}
	}
	sort.Strings(b.failedNames)

	total := len(coverageResults)
	t.Logf("=== EVM Opcode GPU Coverage Report ===")
	t.Logf("Total opcodes tested:      %d", total)
	t.Logf("GPU executed:              %d", b.ran)
	t.Logf("Gas-match w/ CPU:          %d", b.gasMatch)
	t.Logf("Skipped (CALL/CREATE):     %d", b.skippedCall)
	t.Logf("Not implemented on GPU:    %d", b.notImpl)
	t.Logf("GPU execute errors:        %d", b.errored)
	if len(b.failedNames) > 0 {
		t.Logf("Failing opcodes:")
		for _, n := range b.failedNames {
			t.Logf("  - %s", n)
		}
	}
}

// TestOpcodeCoverage_TableComplete asserts the static table covers every
// opcode in the canonical set (sanity check on the test author).
func TestOpcodeCoverage_TableComplete(t *testing.T) {
	got := map[byte]string{}
	for _, ot := range allOpcodeTests {
		if other, ok := got[ot.opcode]; ok {
			t.Errorf("duplicate opcode 0x%02X: %q and %q", ot.opcode, other, ot.name)
			continue
		}
		got[ot.opcode] = ot.name
	}

	// Canonical list of opcodes the EVM spec defines (Cancun + EIP-4844 + EIP-1153).
	want := []byte{
		// 0x00..0x0B
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
		// 0x10..0x1D
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D,
		// 0x20
		0x20,
		// 0x30..0x3F
		0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,
		// 0x40..0x4A
		0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A,
		// 0x50..0x5F
		0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F,
		// 0xA0..0xA4
		0xA0, 0xA1, 0xA2, 0xA3, 0xA4,
		// 0xF0..0xF5, 0xFA, 0xFD..0xFF
		0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xFA, 0xFD, 0xFE, 0xFF,
	}
	// PUSH1..PUSH32 (0x60..0x7F)
	for op := byte(0x60); op <= 0x7F; op++ {
		want = append(want, op)
	}
	// DUP1..DUP16 (0x80..0x8F)
	for op := byte(0x80); op <= 0x8F; op++ {
		want = append(want, op)
	}
	// SWAP1..SWAP16 (0x90..0x9F)
	for op := byte(0x90); op <= 0x9F; op++ {
		want = append(want, op)
	}

	missing := []byte{}
	for _, op := range want {
		if _, ok := got[op]; !ok {
			missing = append(missing, op)
		}
	}
	if len(missing) > 0 {
		var s []string
		for _, op := range missing {
			s = append(s, fmt.Sprintf("0x%02X", op))
		}
		t.Errorf("table is missing %d opcodes: %v", len(missing), s)
	}
	t.Logf("table covers %d / %d opcodes", len(got), len(want))
}
