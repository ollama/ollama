package ggml

// Pure GPU encode→dequant V round-trip with bits=0 (V-only manager
// configuration). Mirrors TestTQGPUOnlyRoundTripContiguous but exercises:
//   - manager construction with KeyPrimaryBits=0 (kSkipped path)
//   - EnsureVLayer + EncodeV + DequantV chain in isolation
//
// Catches V-side regressions independently of any K plumbing. Outliers are
// disabled because the current V encode kernel does not implement an outlier
// split (it's a K-only feature in the existing pipeline). The V dequant
// kernel does fuse WHT undo, so the output is in the original (unrotated)
// coordinate system that stock flash-attn expects.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

func TestTQGPUOnlyVRoundTripContiguous(t *testing.T) {
	const (
		headDim   = 128
		nKVHeads  = 4
		batchSize = 6
		capacity  = 32
	)

	for _, vBits := range []int{2, 3, 4} {
		t.Run(presetName(vBits), func(t *testing.T) {
			ctx, be := setupTQAny(t)
			b := be.(*Backend)
			rotationSeed := turboquant.PresetTQ3V.RotationSeed
			// V-only construction: bits=0 skips all K state, vBits>0 sets V.
			// asymmetric=false: V encoder doesn't currently consume the
			// asymmetric primary path (no zeros tensor for V).
			mgrAny := b.NewTQCompressedKManager(
				headDim, nKVHeads, 0,
				rotationSeed,
				vBits,
				0, 0,
				false,
			)
			if mgrAny == nil {
				t.Skip("manager nil")
			}
			mgr := mgrAny.(*ggmlTQCompressedK)
			mgr.EnsureVLayer(0, capacity)

			rng := rand.New(rand.NewSource(0xfeed1234))
			vData := make([]float32, headDim*nKVHeads*batchSize)
			for i := range vData {
				vData[i] = float32(rng.NormFloat64()) * 0.7
			}
			vTensor := ctx.FromFloats(vData, headDim, nKVHeads, batchSize)

			enc := mgr.EncodeV(ctx, 0, vTensor, 0)
			if enc == nil {
				t.Fatalf("EncodeV returned nil")
			}
			deq := mgr.DequantV(ctx, 0, enc, 0, batchSize)
			if deq == nil {
				t.Fatalf("DequantV returned nil")
			}
			// DequantV returns rotated-space V; un-rotate via WHTUndo (which is
			// self-inverse) before comparing against the original input. The
			// fused dequant+WHT path used by the K dequant kernel doesn't
			// exist for V yet — see the V-only fused FA work for the
			// equivalent. The Cast(F32) is required because Floats() on an
			// f16 tensor returns raw bytes reinterpreted as f32 (garbage).
			if mgr.HasRotation() {
				deq = mgr.WHTUndo(ctx, 0, deq)
			}
			deqF32 := deq.(*Tensor).Cast(ctx, ml.DTypeF32)
			ctx.Forward(enc, deqF32).Compute(enc, deqF32)

			out := deqF32.Floats()
			if len(out) != len(vData) {
				t.Fatalf("output len %d != input len %d", len(out), len(vData))
			}

			var sumDiff, sumDiffSq, sumIn, sumOut float64
			var maxDiff float32
			for i := range vData {
				d := float64(out[i] - vData[i])
				sumDiff += math.Abs(d)
				sumDiffSq += d * d
				sumIn += float64(vData[i]) * float64(vData[i])
				sumOut += float64(out[i]) * float64(out[i])
				if ad := float32(math.Abs(d)); ad > maxDiff {
					maxDiff = ad
				}
			}
			n := float64(len(vData))
			rmsIn := math.Sqrt(sumIn / n)
			rmsOut := math.Sqrt(sumOut / n)
			rmsDiff := math.Sqrt(sumDiffSq / n)

			t.Logf("vBits=%d RMS_in=%.4f RMS_out=%.4f RMS_diff=%.4f max_diff=%.4f mean_abs_diff=%.4f",
				vBits, rmsIn, rmsOut, rmsDiff, maxDiff, sumDiff/n)

			if rmsOut < rmsIn*0.2 || rmsOut > rmsIn*5.0 {
				t.Fatalf("GPU V dequant magnitude off: RMS_in=%.4f RMS_out=%.4f", rmsIn, rmsOut)
			}
			// 2-bit gets 60% slop; 3-bit 40%; 4-bit 20%. Empirical bands
			// loose enough to absorb f16 + WHT-undo noise but tight enough
			// to catch a wrong-rotation or wrong-codebook regression.
			tol := map[int]float64{2: 0.6, 3: 0.4, 4: 0.25}[vBits]
			if rmsDiff > rmsIn*tol {
				t.Fatalf("V encode+dequant RMS_diff=%.4f exceeds %.0f%% of RMS_in=%.4f",
					rmsDiff, tol*100, rmsIn)
			}
		})
	}
}

func presetName(vBits int) string {
	switch vBits {
	case 2:
		return "tq2v"
	case 3:
		return "tq3v"
	case 4:
		return "tq4v"
	}
	return "unknown"
}
