package ggml

// Round-trip correctness for the indexed-addressing path
// (EncodeKAt → DequantKAt with scattered physical-slot locs).
//
// The contiguous fast path is exercised by every other TQ test; this file
// specifically pins down that:
//   - Encode writes to physical slot locs[i] (not firstCell+i)
//   - Dequant reads from physical slot locs[i] back into dense row i
//   - Both pieces produce values that match the contiguous baseline applied
//     to the same dense token data (the data is independent of slot layout).
//
// The kernels are validated against themselves rather than a CPU reference
// because indexed-mode addressing only changes WHERE data lives in the cache
// buffer, not HOW it is encoded — so the encoded values must equal the
// contiguous-mode encoded values placed at the same physical slots.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

func TestTQEncodeKAtFragmentedRoundTrip(t *testing.T) {
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 128
		nKVHeads     = 4
		batchSize    = 6
		capacity     = 32 // larger than batchSize so we can scatter
		bits         = 3
		outlierBits  = 4
		outlierCount = 32
	)

	rotationSeed := turboquant.PresetTQ3K.RotationSeed

	mgrAny := b.NewTQCompressedKManager(
		headDim, nKVHeads, bits,
		rotationSeed,
		0, // vBits = 0 (K-only)
		outlierBits, outlierCount,
		false, // symmetric for a clean CPU-side comparison
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil (GPU not available)")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, capacity)

	rng := rand.New(rand.NewSource(0xfeed))
	kData := make([]float32, headDim*nKVHeads*batchSize)
	for i := range kData {
		kData[i] = float32(rng.NormFloat64()) * 0.7
	}
	kTensor := ctx.FromFloats(kData, headDim, nKVHeads, batchSize)

	// A scattered loc set: tokens i=0..5 → slots [5, 17, 1, 23, 8, 30].
	locs := []int32{5, 17, 1, 23, 8, 30}
	locsTensor := ctx.FromInts(locs, batchSize)

	encAt := mgr.EncodeKAt(ctx, 0, kTensor, locsTensor)
	if encAt == nil {
		t.Fatalf("EncodeKAt returned nil")
	}
	// nCells in the dequant view: read back at exactly the same slots.
	deqAt := mgr.DequantKAt(ctx, 0, encAt, locsTensor)
	if deqAt == nil {
		t.Fatalf("DequantKAt returned nil")
	}
	// DequantKAt returns f16; Floats() on an f16 tensor returns raw f16 bytes
	// reinterpreted as float32 (i.e. garbage). Cast to f32 first so Floats()
	// reads actual float values.
	deqAtF32 := deqAt.(*Tensor).Cast(ctx, ml.DTypeF32)
	ctx.Forward(encAt, deqAtF32).Compute(encAt, deqAtF32)

	gpuOut := deqAtF32.Floats()

	// Round-trip check: GPU's DequantK applies WHT-undo on the way out, so
	// gpuOut is in UNROTATED space — same coordinate system as kData. Compare
	// directly. Also build a CPU encode+decode round-trip (rotated→quantized→
	// rotated→unrotated) as a sanity reference; both should agree within
	// scalar-quantization slop.
	//
	// Per-layer rotation seed: the manager derives it as
	// `rotationSeed XOR (layer+1)` (turboquant_compressed.go EnsureLayer)
	// so layer 0 uses `rotationSeed XOR 1`. The CPU reference must use the
	// same per-layer seed.
	layerSeed := rotationSeed ^ uint64(1)
	cpuPreset := turboquant.Preset{
		RotationSeed:   layerSeed,
		KeyPrimaryBits: bits,
		OutlierBits:    outlierBits,
		OutlierCount:   outlierCount,
	}
	rotation := turboquant.BuildRotation(headDim, layerSeed)

	maxDiffGpuVsOriginal := float32(0)
	maxDiffCpuVsOriginal := float32(0)
	maxDiffGpuVsCpu := float32(0)
	for tok := range batchSize {
		for h := range nKVHeads {
			origPerHead := kData[(tok*nKVHeads+h)*headDim : (tok*nKVHeads+h+1)*headDim]
			cpuEnc, err := turboquant.EncodeKeyPerHeadOutlier(origPerHead, cpuPreset)
			if err != nil {
				t.Fatalf("CPU encode tok=%d h=%d: %v", tok, h, err)
			}
			// CPU dequant lands in rotated space; un-rotate (WHT is its own
			// inverse) to bring it into the same coordinate system as gpuOut
			// and the original kData.
			cpuRotated := turboquant.DequantKeyPerHeadOutlier(cpuEnc, cpuPreset, headDim)
			cpuUnrotated := turboquant.ApplyRotation(cpuRotated, rotation)

			gpuBase := (tok*nKVHeads + h) * headDim
			for d := range headDim {
				dGpuO := float32(math.Abs(float64(gpuOut[gpuBase+d] - origPerHead[d])))
				dCpuO := float32(math.Abs(float64(cpuUnrotated[d] - origPerHead[d])))
				dGpuCpu := float32(math.Abs(float64(gpuOut[gpuBase+d] - cpuUnrotated[d])))
				if dGpuO > maxDiffGpuVsOriginal {
					maxDiffGpuVsOriginal = dGpuO
				}
				if dCpuO > maxDiffCpuVsOriginal {
					maxDiffCpuVsOriginal = dCpuO
				}
				if dGpuCpu > maxDiffGpuVsCpu {
					maxDiffGpuVsCpu = dGpuCpu
				}
			}
		}
	}
	// Round-trip noise floor: 3-bit primary + 4-bit outliers + WHT + f16
	// rounding. CPU side measures ~0.3, GPU adds f16 + shfl-reorder noise
	// pushing peak per-element diff into the 0.6–0.9 range. Tol=1.0 catches
	// real bugs (wrong slot, missing rotation, missing outlier remap give
	// diffs in the 5–10 range — see git history) without being so tight
	// that hardware-level f16 noise causes flake.
	const tol float32 = 1.0
	t.Logf("scattered locs %v: GPU vs original=%f  CPU vs original=%f  GPU vs CPU=%f (tol=%f)",
		locs, maxDiffGpuVsOriginal, maxDiffCpuVsOriginal, maxDiffGpuVsCpu, tol)
	if maxDiffGpuVsOriginal > tol {
		t.Errorf("GPU round-trip vs original kData: maxDiff=%f (want < %f) — indexed encode/dequant likely reading or writing wrong physical slot",
			maxDiffGpuVsOriginal, tol)
	}
	if maxDiffCpuVsOriginal > tol {
		t.Errorf("CPU round-trip vs original kData: maxDiff=%f (want < %f) — CPU reference encoder broken",
			maxDiffCpuVsOriginal, tol)
	}
	if maxDiffGpuVsCpu > tol {
		t.Errorf("GPU vs CPU round-trip: maxDiff=%f (want < %f) — encoder paths diverge", maxDiffGpuVsCpu, tol)
	}
}

func TestTQEncodeKAtMatchesContiguousAtSameSlots(t *testing.T) {
	// Pins down the equivalence: encoding tokens [0..N-1] with
	// firstCell=F should leave the cache buffer in the same state as
	// encoding tokens [0..N-1] with locs = [F, F+1, ..., F+N-1].
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 128
		nKVHeads     = 2
		batchSize    = 4
		capacity     = 16
		bits         = 3
		outlierBits  = 4
		outlierCount = 32
		firstCell    = 3
	)

	rotationSeed := turboquant.PresetTQ3K.RotationSeed
	mgrAny := b.NewTQCompressedKManager(headDim, nKVHeads, bits, rotationSeed, 0, outlierBits, outlierCount, false)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, capacity)

	rng := rand.New(rand.NewSource(0xcafe))
	kData := make([]float32, headDim*nKVHeads*batchSize)
	for i := range kData {
		kData[i] = float32(rng.NormFloat64()) * 0.5
	}
	kTensor := ctx.FromFloats(kData, headDim, nKVHeads, batchSize)

	// Path 1: contiguous EncodeK starting at firstCell, then DequantK from the
	// same range.
	enc1 := mgr.EncodeK(ctx, 0, kTensor, firstCell)
	if enc1 == nil {
		t.Fatalf("EncodeK returned nil")
	}
	deq1 := mgr.DequantK(ctx, 0, enc1, firstCell, batchSize)
	if deq1 == nil {
		t.Fatalf("DequantK returned nil")
	}
	ctx.Forward(enc1, deq1).Compute(enc1, deq1)
	contiguous := append([]float32(nil), deq1.Floats()...)

	// Reset the cache state for a clean comparison.
	mgr.EnsureLayer(0, capacity)

	// Path 2: indexed EncodeKAt with locs = [F, F+1, F+2, F+3], then
	// DequantKAt with the same locs.
	locs := []int32{firstCell, firstCell + 1, firstCell + 2, firstCell + 3}
	locsTensor := ctx.FromInts(locs, batchSize)
	enc2 := mgr.EncodeKAt(ctx, 0, kTensor, locsTensor)
	if enc2 == nil {
		t.Fatalf("EncodeKAt returned nil")
	}
	deq2 := mgr.DequantKAt(ctx, 0, enc2, locsTensor)
	if deq2 == nil {
		t.Fatalf("DequantKAt returned nil")
	}
	ctx.Forward(enc2, deq2).Compute(enc2, deq2)
	indexed := deq2.Floats()

	if len(contiguous) != len(indexed) {
		t.Fatalf("dequant output length: contiguous=%d indexed=%d", len(contiguous), len(indexed))
	}
	maxDiff := float32(0)
	for i := range contiguous {
		diff := float32(math.Abs(float64(contiguous[i] - indexed[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	// The two paths write to the same physical slots, but the encode kernel
	// has separate `firstCell + c` and `locs[c]` code paths (different
	// pointer-arithmetic ordering, different shfl-reduction sequence in the
	// EDEN refinement loops). Floating-point reductions reorder slightly
	// between the two, producing sub-quantization noise on the order of the
	// f16 round-trip floor. Tolerance is loose enough to admit that noise but
	// tight enough to catch a real path divergence (e.g. wrong physical slot,
	// missing rotation, missing zero correction — those produce diffs > 0.1).
	const tol float32 = 0.1
	if maxDiff > tol {
		t.Fatalf("contiguous (firstCell=%d) and indexed (locs=%v) diverge beyond tolerance: maxDiff=%f (tol=%f)",
			firstCell, locs, maxDiff, tol)
	}
	t.Logf("contiguous-equivalent locs %v: maxDiff=%f (tol=%f)", locs, maxDiff, tol)
}
