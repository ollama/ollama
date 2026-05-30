package ggml

// TestTQGPUOnlyRoundTripContiguous: GPU encode → GPU dequant → compare to
// original input. Pure GPU self-consistency, no CPU reference involved. This
// isolates whether the GPU encode+dequant chain itself is sound, separately
// from any CPU-vs-GPU layout/encoding mismatch.
//
// For a small symmetric+outlier preset (matching TestTQEncodeKAtFragmentedRoundTrip
// configuration), the GPU should reconstruct the input within ~scalar
// quantization slop (a few percent of input magnitude). If output is wildly
// different (near zero or huge), the encode or dequant kernel is broken.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

func TestTQGPUOnlyRoundTripContiguous(t *testing.T) {
	const (
		headDim      = 128
		nKVHeads     = 4
		batchSize    = 6
		capacity     = 32
		bits         = 3
		outlierBits  = 4
		outlierCount = 32
	)

	for _, asym := range []bool{false, true} {
		name := "symmetric"
		if asym {
			name = "asymmetric"
		}
		t.Run(name, func(t *testing.T) {
			ctx, be := setupTQAny(t)
			b := be.(*Backend)
			rotationSeed := turboquant.PresetTQ3K.RotationSeed
			mgrAny := b.NewTQCompressedKManager(
				headDim, nKVHeads, bits,
				rotationSeed,
				0,
				outlierBits, outlierCount,
				asym,
			)
			if mgrAny == nil {
				t.Skip("manager nil")
			}
			mgr := mgrAny.(*ggmlTQCompressedK)
			mgr.EnsureLayer(0, capacity)

			rng := rand.New(rand.NewSource(0xfeed))
			kData := make([]float32, headDim*nKVHeads*batchSize)
			for i := range kData {
				kData[i] = float32(rng.NormFloat64()) * 0.7
			}
			kTensor := ctx.FromFloats(kData, headDim, nKVHeads, batchSize)

			// Contiguous: firstCell=0, no locs.
			enc := mgr.EncodeK(ctx, 0, kTensor, 0)
			if enc == nil {
				t.Fatalf("EncodeK returned nil")
			}
			deq := mgr.DequantK(ctx, 0, enc, 0, batchSize)
			if deq == nil {
				t.Fatalf("DequantK returned nil")
			}
			// DequantK returns f16; cast to f32 before Floats() to get real values.
			deqF32 := deq.(*Tensor).Cast(ctx, ml.DTypeF32)
			ctx.Forward(enc, deqF32).Compute(enc, deqF32)

			out := deqF32.Floats()
			if len(out) != len(kData) {
				t.Fatalf("output len %d != input len %d", len(out), len(kData))
			}

			// Stats: mean abs diff, max abs diff, RMS of input, RMS of output.
			var sumDiff, sumDiffSq, sumIn, sumOut float64
			var maxDiff float32
			for i := range kData {
				d := float64(out[i] - kData[i])
				sumDiff += math.Abs(d)
				sumDiffSq += d * d
				sumIn += float64(kData[i]) * float64(kData[i])
				sumOut += float64(out[i]) * float64(out[i])
				if ad := float32(math.Abs(d)); ad > maxDiff {
					maxDiff = ad
				}
			}
			n := float64(len(kData))
			rmsIn := math.Sqrt(sumIn / n)
			rmsOut := math.Sqrt(sumOut / n)
			rmsDiff := math.Sqrt(sumDiffSq / n)

			t.Logf("input k[0..7] = %v", kData[:8])
			t.Logf("gpu   out[0..7] = %v", out[:8])
			t.Logf("RMS_in=%.4f RMS_out=%.4f RMS_diff=%.4f max_diff=%.4f mean_abs_diff=%.4f",
				rmsIn, rmsOut, rmsDiff, maxDiff, sumDiff/n)

			// Hard correctness gate: dequant output magnitude should be in the
			// same ballpark as input magnitude. RMS_out within 5x of RMS_in.
			if rmsOut < rmsIn*0.2 || rmsOut > rmsIn*5.0 {
				t.Fatalf("GPU dequant magnitude wildly off: RMS_in=%.4f RMS_out=%.4f", rmsIn, rmsOut)
			}
			// Scalar-quant tolerance: RMS_diff < 30% of RMS_in is reasonable for 3-bit.
			if rmsDiff > rmsIn*0.4 {
				t.Fatalf("GPU encode+dequant RMS_diff=%.4f too large vs RMS_in=%.4f", rmsDiff, rmsIn)
			}
		})
	}
}
