package ggml

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

// TestAsymmetricQJLGPURoundtripFidelity runs the GPU-native asymmetric +
// QJL encode/dequant path on a real CUDA GPU and asserts that the output
// is (a) quantized — NOT bit-identical to the input — and (b) within the
// expected reconstruction-error envelope for each preset.
//
// This catches the original "silent f16 fallback" regression: if the GPU
// path didn't actually activate and the cache was storing unmodified K,
// the output would match the input rotated-space representation to fp16
// noise (≈1e-3 relative). Requiring a non-trivial reconstruction error
// floor (but below the upper tolerance) asserts that quantization is
// real and not a passthrough.
//
// The test runs only when a TQ-capable GPU is present (skips otherwise).
func TestAsymmetricQJLGPURoundtripFidelity(t *testing.T) {
	// Inline-construct asym+outliers+QJL presets at 3-bit and 2-bit. These
	// are not shipping presets but exercise the GPU encode/decode path in
	// the same configuration the test was originally written against.
	tq3qa := turboquant.Preset{
		ID: 103, Name: "tq3qa", RotationSeed: 0x35c0ffee,
		KeyPrimaryBits: 3, ValueBits: 3, QJLRowsDivisor: 1,
		OutlierBits: 4, OutlierCount: 32, AsymmetricPrimary: true,
	}
	tq2qa := turboquant.Preset{
		ID: 113, Name: "tq2qa", RotationSeed: 0x25c0ffee,
		KeyPrimaryBits: 2, ValueBits: 2, QJLRowsDivisor: 1,
		OutlierBits: 3, OutlierCount: 32, AsymmetricPrimary: true,
	}
	cases := []struct {
		name    string
		preset  turboquant.Preset
		vBits   int
		headDim int
	}{
		{"tq3qa_d128_h8", tq3qa, 3, 128},
		{"tq2qa_d128_h8", tq2qa, 2, 128},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, be := setupGPUTB(t)
			ggmlBackend := be.(*Backend)

			const (
				numKVHeads = 8
				nCells     = 4
			)
			qjlRows := tc.preset.KeyQJLRows(tc.headDim)
			mgrAny := ggmlBackend.NewTQCompressedKManager(
				tc.headDim, numKVHeads, tc.preset.KeyPrimaryBits,
				tc.preset.RotationSeed,
				tc.vBits,
				tc.preset.OutlierBits, tc.preset.OutlierCount,
				tc.preset.AsymmetricPrimary, qjlRows,
			)
			if mgrAny == nil {
				t.Skip("asymmetric+QJL manager returned nil (not a CUDA backend?)")
			}
			mgr, ok := mgrAny.(*ggmlTQCompressedK)
			if !ok {
				t.Fatalf("unexpected manager type %T", mgrAny)
			}
			mgr.EnsureLayer(0, nCells)

			// Deterministic synthetic K with non-zero mean per channel and a
			// few heavy-tail outliers — the Qwen-2-style distribution these
			// presets are designed for.
			rng := rand.New(rand.NewSource(0xfeed_cafe | int64(tc.headDim)))
			kData := make([]float32, tc.headDim*numKVHeads*nCells)
			bias := make([]float32, tc.headDim)
			for i := range bias {
				bias[i] = 0.25 * float32(rng.NormFloat64())
			}
			for i := 0; i < len(kData); i++ {
				headChan := i % tc.headDim
				kData[i] = float32(rng.NormFloat64()) + bias[headChan]
			}
			// Inject per-cell-per-head outliers.
			for c := range nCells {
				for h := range numKVHeads {
					for range 8 {
						d := rng.Intn(tc.headDim)
						k := (c*numKVHeads+h)*tc.headDim + d
						kData[k] += float32(math.Copysign(2.0+rng.Float64(), rng.NormFloat64()))
					}
				}
			}

			// GPU path: encode then dequant. DequantK returns [headDim,
			// numKVHeads, nCells] f16 in rotated space.
			kT := ctx.FromFloats(kData, tc.headDim, numKVHeads, nCells)
			enc := mgr.EncodeK(ctx, 0, kT, 0)
			if enc == nil {
				t.Fatalf("EncodeK returned nil")
			}
			ctx.Forward(enc)
			out := mgr.DequantK(ctx, 0, enc, 0, nCells)
			if out == nil {
				t.Fatalf("DequantK returned nil")
			}
			ctx.Forward(out).Compute(out)
			gpuOut := out.Floats()
			if len(gpuOut) != tc.headDim*numKVHeads*nCells {
				t.Fatalf("gpu output len = %d, want %d",
					len(gpuOut), tc.headDim*numKVHeads*nCells)
			}

			// Reference: apply the same rotation as the GPU kernel so we
			// compare in rotated space (what the cache stores).
			rotation := turboquant.BuildRotation(tc.headDim, tc.preset.RotationSeed)
			var (
				sumSq, sumSqInput float64
				maxAbsDiff        float32
				nonZeroDiffs      int
			)
			for c := range nCells {
				for h := range numKVHeads {
					slab := make([]float32, tc.headDim)
					for d := range tc.headDim {
						slab[d] = kData[(c*numKVHeads+h)*tc.headDim+d]
					}
					rotated := turboquant.ApplyRotation(slab, rotation)
					for d := range tc.headDim {
						gpuVal := gpuOut[(c*numKVHeads+h)*tc.headDim+d]
						expectedVal := rotated[d]
						diff := gpuVal - expectedVal
						absDiff := diff
						if absDiff < 0 {
							absDiff = -absDiff
						}
						if absDiff > maxAbsDiff {
							maxAbsDiff = absDiff
						}
						// Count genuinely distinct values beyond f16 rounding noise
						// (the f16 round-trip of the stored buffer alone produces
						// ~1e-3 differences; real quantization produces ≫ 1e-3).
						if absDiff > 1e-3 {
							nonZeroDiffs++
						}
						sumSq += float64(diff) * float64(diff)
						sumSqInput += float64(expectedVal) * float64(expectedVal)
					}
				}
			}
			nElems := tc.headDim * numKVHeads * nCells
			rmse := math.Sqrt(sumSq / float64(nElems))
			inputRms := math.Sqrt(sumSqInput / float64(nElems))
			relErr := rmse / inputRms

			t.Logf("%s: rmse=%.4f input_rms=%.4f rel_err=%.4f max_abs=%.4f non_zero_diffs=%d/%d",
				tc.name, rmse, inputRms, relErr, maxAbsDiff, nonZeroDiffs, nElems)

			// Assert A (anti-passthrough): at least 5% of elements must
			// differ from the input by more than fp16-rounding noise. A true
			// passthrough (silent fallback) would give zero non-trivial
			// diffs here.
			minNonZero := nElems / 20
			if nonZeroDiffs < minNonZero {
				t.Fatalf("%s looks like a passthrough: only %d/%d elems differ by >1e-3 from the rotated input — GPU kernel probably not actually running",
					tc.name, nonZeroDiffs, nElems)
			}

			// Assert B (sanity ceiling): relative RMSE should be below 1.0.
			// Real quantization at 2-3 bits is typically 0.1-0.5 relative
			// RMSE on this distribution; anything above 1.0 means the kernel
			// is producing garbage (e.g. wrong pack layout, missing zero
			// offset, broken QJL residual correction).
			if relErr > 1.0 {
				t.Fatalf("%s relative RMSE %.3f > 1.0 — GPU kernel output is not tracking the input",
					tc.name, relErr)
			}
		})
	}
}

// Compile-time check that ml.Tensor is what we expect.
var _ ml.Tensor = (*Tensor)(nil)
