package ggml

// TestTQFusedFA_MaskAtLargeNCells: verify that the FA kernel correctly applies
// the SWA mask for cells beyond the kernel's per-iteration block (nthreads=128).
//
// HYPOTHESIS: the kernel reads the mask once per cell. When nCells > nthreads
// the kernel processes cells in chunks; if the chunk-2+ mask read aliases the
// chunk-1 entries, the SWA-hidden cells silently contribute. On Metal this
// failure mode existed because the kernel did not advance the maskh pointer
// across iterations and read at thread-local index — fixed by switching to
// absolute cell_rel indexing. On CUDA/HIP the per-iter pointer advancement
// already covered this, so this test passes on those backends as a regression
// check.
//
// Test plan:
//   - Build K with nCells=256 (= 2 chunks of 128).
//   - K cells 128..255 carry HUGE values (scale=100) so their Q·K dominates
//     softmax IF the mask is misapplied.
//   - Mask cells 0..127 = 0 (visible), cells 128..255 = -INF (hidden).
//   - V cells 0..127 = 1, V cells 128..255 = 100.
//   - With CORRECT mask suppression, only cells 0..127 (V=1) contribute, so
//     every output element ≈ 1.0. With buggy mask, cells 128..255 (V=100)
//     dominate and output ≈ 100.

import (
	"math"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

func TestTQFusedFA_MaskAtLargeNCells(t *testing.T) {
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 128
		nKVHeads     = 4
		nQHeads      = 8 // gqa=2, but doesn't matter here
		nTokensQ     = 1
		nCells       = 256 // > nthreads=128 to expose any mask-aliasing bug
		bits         = 4
		outlierBits  = 5
		outlierCount = 32
	)

	rotationSeed := turboquant.PresetTQ4K.RotationSeed ^ uint64(0xd128a5)
	mgrAny := b.NewTQCompressedKManager(headDim, nKVHeads, bits, rotationSeed,
		0, outlierBits, outlierCount, true)
	if mgrAny == nil {
		t.Skip("manager nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, nCells)
	if !mgr.fusedKernelSupports() {
		t.Skip("not supported")
	}

	var rngState uint64 = 0xd128a5_cafe
	layerSeed := rotationSeed ^ uint64(1)
	rotation := turboquant.BuildRotation(headDim, layerSeed)

	// Q: random per head (rotated to match the K WHT domain).
	qRotated := make([]float32, headDim*nQHeads*nTokensQ)
	for h := range nQHeads {
		v := make([]float32, headDim)
		for d := range headDim {
			v[d] = float32(tqGaussian(&rngState)) * 0.5
		}
		copy(qRotated[h*headDim:], turboquant.ApplyRotation(v, rotation))
	}

	// K is wrapped in a tensor with shape [headDim, nKVHeads, nCells]. ggml
	// stores ne[0] (headDim) innermost, then ne[1] (nKVHeads), then ne[2]
	// (nCells). So index into the float buffer as
	// kData[c*nKVHeads*headDim + h*headDim + d] for element (d, h, c).
	kData := make([]float32, headDim*nKVHeads*nCells)
	for c := range nCells {
		for h := range nKVHeads {
			scale := float32(1.0)
			if c >= 128 {
				scale = 100.0 // make hidden K HUGE so they would dominate without the mask
			}
			for d := range headDim {
				kData[c*nKVHeads*headDim+h*headDim+d] = float32(tqGaussian(&rngState)) * scale
			}
		}
	}

	// V is wrapped in a tensor with shape [headDim, nCells, nKVHeads]. Note
	// the dimension order differs from K — V's nCells comes BEFORE nKVHeads.
	// Index correctly as vF32[h*nCells*headDim + c*headDim + d] for element
	// (d, c, h). Cells 0..127 hold V=1.0 (visible). Cells 128..255 hold V=100
	// (hidden by the mask). If suppression works, output is ~1.0; if not, the
	// V=100 entries leak through and output is ~100.
	vF32 := make([]float32, headDim*nCells*nKVHeads)
	for h := range nKVHeads {
		for c := range nCells {
			val := float32(1.0)
			if c >= 128 {
				val = 100.0
			}
			for d := range headDim {
				vF32[h*nCells*headDim+c*headDim+d] = val
			}
		}
	}
	vF16Bytes := tqF32SliceToF16Bytes(vF32)

	// Mask: shape [nCells, nTokensQ]. Cells 0..127 visible (0), 128..255 hidden (-INF).
	// Build directly as f16 bytes so the kernel sees the value verbatim
	// (avoids a Cast graph op).
	maskF32 := make([]float32, nCells*nTokensQ)
	for tok := range nTokensQ {
		for c := range nCells {
			if c >= 128 {
				maskF32[tok*nCells+c] = float32(math.Inf(-1))
			}
		}
	}
	maskF16Bytes := tqF32SliceToF16Bytes(maskF32)

	qT := ctx.FromFloats(qRotated, headDim, nTokensQ, nQHeads)
	kT := ctx.FromFloats(kData, headDim, nKVHeads, nCells)
	vT := ctx.FromBytes(ml.DTypeF16, vF16Bytes, headDim, nCells, nKVHeads)
	maskT := ctx.FromBytes(ml.DTypeF16, maskF16Bytes, nCells, nTokensQ)

	enc := mgr.EncodeK(ctx, 0, kT, 0)
	tqkRaw, ok := mgr.GetAsTQTensor(ctx, 0, enc, 0, nCells)
	if !ok {
		t.Fatalf("GetAsTQTensor failed")
	}
	tqk := tqkRaw.(*tqTensor)

	attnScale := 1.0 / math.Sqrt(float64(headDim))
	attnOut := b.tqFlashAttention(ctx, qT.(*Tensor), tqk, vT.(*Tensor), maskT, attnScale, 0)
	ctx.Forward(enc, attnOut).Compute(enc, attnOut)

	gpuOut := attnOut.Floats()

	// Expected: output should be ~1.0 everywhere (only V cells with value 1.0 visible).
	// If buggy: V=100 cells contribute, output deviates dramatically from 1.0.
	var maxDev float32
	var nGT5 int
	for _, v := range gpuOut {
		dev := float32(math.Abs(float64(v) - 1.0))
		if dev > maxDev {
			maxDev = dev
		}
		if dev > 5.0 {
			nGT5++
		}
	}
	t.Logf("nCells=%d (=2 kernel chunks): expected output=1.0, max_dev=%.4f, %d/%d elements >5",
		nCells, maxDev, nGT5, len(gpuOut))

	if maxDev > 5.0 {
		t.Fatalf("max_dev=%.4f > 5.0 — kernel ignores SWA mask for cells beyond nthreads", maxDev)
	}
}
