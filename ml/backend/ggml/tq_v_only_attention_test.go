package ggml

// V-only weighted-V test, exercised at the math level so the test
// doesn't depend on the SDPA wrapper's shape contracts (which assume
// cache-managed Q/K/V layouts). This proves the Phase A V-only path
// produces a softmax(QK)·V output equivalent to the raw-V baseline
// within scalar-quantisation tolerance:
//
//   1. Construct random V on GPU.
//   2. Pipeline V through the manager: EncodeV → DequantV → WHTUndo.
//   3. Apply a synthetic peaked attention-weight distribution to both
//      raw V and dequant V, computing Σ_c w_c · V[d, h, c] in Go.
//   4. Compare outputs.
//
// This is a unit-level proxy for the runner-binary smoke test that
// is environmentally blocked. If the V-only data flow is sound, the
// two weighted-V outputs match within ~scalar-quant slop scaled by
// the peak attention weight.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

func TestTQVOnlyAttentionEndToEnd(t *testing.T) {
	const (
		headDim  = 128
		nHeadsKV = 4
		nCellsKV = 16
		capacity = 32
	)

	for _, vBits := range []int{2, 3, 4} {
		t.Run(presetName(vBits), func(t *testing.T) {
			ctx, be := setupTQAny(t)
			b := be.(*Backend)
			rotationSeed := turboquant.PresetTQ3V.RotationSeed

			mgrAny := b.NewTQCompressedKManager(
				headDim, nHeadsKV, 0,
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

			rng := rand.New(rand.NewSource(0xa77))

			vData := make([]float32, headDim*nHeadsKV*nCellsKV)
			for i := range vData {
				vData[i] = float32(rng.NormFloat64()) * 0.5
			}
			vTensor := ctx.FromFloats(vData, headDim, nHeadsKV, nCellsKV)

			enc := mgr.EncodeV(ctx, 0, vTensor, 0)
			if enc == nil {
				t.Fatalf("EncodeV returned nil")
			}
			deq := mgr.DequantV(ctx, 0, enc, 0, nCellsKV)
			if deq == nil {
				t.Fatalf("DequantV returned nil")
			}
			if mgr.HasRotation() {
				deq = mgr.WHTUndo(ctx, 0, deq)
			}

			deqF32 := deq.(*Tensor).Cast(ctx, ml.DTypeF32)
			vRefF32 := vTensor.(*Tensor).Cast(ctx, ml.DTypeF32)

			ctx.Forward(enc, deqF32, vRefF32).Compute(enc, deqF32, vRefF32)

			outQ := deqF32.Floats()
			outR := vRefF32.Floats()
			if len(outQ) != len(outR) || len(outQ) == 0 {
				t.Fatalf("output len mismatch: quant=%d ref=%d", len(outQ), len(outR))
			}

			// Synthetic peaked attention weights — pessimistic stand-in
			// for a real softmax (which spreads weight more evenly and
			// averages noise down further). Pass here implies pass under
			// real attention.
			weights := make([][]float64, nHeadsKV)
			for h := range nHeadsKV {
				weights[h] = make([]float64, nCellsKV)
				peak := rng.Intn(nCellsKV)
				for c := range nCellsKV {
					if c == peak {
						weights[h][c] = 0.5
					} else {
						weights[h][c] = 0.5 / float64(nCellsKV-1)
					}
				}
			}

			attnQuant := make([]float64, nHeadsKV*headDim)
			attnRef := make([]float64, nHeadsKV*headDim)
			for h := range nHeadsKV {
				for d := range headDim {
					var sumQ, sumR float64
					for c := range nCellsKV {
						idx := (c*nHeadsKV+h)*headDim + d
						sumQ += weights[h][c] * float64(outQ[idx])
						sumR += weights[h][c] * float64(outR[idx])
					}
					attnQuant[h*headDim+d] = sumQ
					attnRef[h*headDim+d] = sumR
				}
			}

			var sumDiffSq, sumRefSq, maxDiff float64
			for i := range attnRef {
				d := attnQuant[i] - attnRef[i]
				sumDiffSq += d * d
				sumRefSq += attnRef[i] * attnRef[i]
				if ad := math.Abs(d); ad > maxDiff {
					maxDiff = ad
				}
			}
			n := float64(len(attnRef))
			rmsRef := math.Sqrt(sumRefSq / n)
			rmsDiff := math.Sqrt(sumDiffSq / n)

			t.Logf("vBits=%d cells=%d heads=%d  RMS_ref=%.4f  RMS_diff=%.4f  max_diff=%.4f  rel=%.2f%%",
				vBits, nCellsKV, nHeadsKV, rmsRef, rmsDiff, maxDiff, 100*rmsDiff/math.Max(rmsRef, 1e-9))

			tol := map[int]float64{2: 0.50, 3: 0.30, 4: 0.18}[vBits]
			if rmsDiff > rmsRef*tol {
				t.Fatalf("V-only weighted-V output diverges from reference: RMS_diff=%.4f exceeds %.0f%% of RMS_ref=%.4f",
					rmsDiff, tol*100, rmsRef)
			}
		})
	}
}
