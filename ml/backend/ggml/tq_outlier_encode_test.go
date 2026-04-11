package ggml

import (
	"math"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

// TestOutlierEncodeDequantGPUCPUEquivalence runs tq_encode_kernel_outlier
// + tq_dequant_multihead_kernel_outlier on the GPU for a synthetic K
// batch and compares the decoded output against the CPU reference
// (EncodeKeyPerHeadOutlier + DequantKeyPerHeadOutlier in
// turboquant/encode.go). Catches kernel algorithmic drift and tests the
// exact multi-KV-head layout that broke llama3.2:3b / qwen2.5:7b under
// the __shfl_sync divergence bug.
//
// CURRENT LIMITATION: this test skips in CI / on plain `go test` runs
// because setup()'s synthetic GGUF has no tensors, so no GPU buffer
// types end up in b.schedBufts, so scanTQDevices() finds no TQ-capable
// GPU even on a machine with a P40. The test as written is correct and
// runnable under a test harness that loads a real model-backed backend
// (e.g. the tq_outlier_encode_test.go:setup function could be replaced
// with a helper that loads a tiny real .gguf with GPU-assigned layers).
// Until that harness lands, the CPU reference tests in
// turboquant/encode_test.go (TestOutlierSplitVsUniformHeavyTailed,
// TestOutlierPerHeadRoundTrip) cover algorithmic correctness and the
// full tqbench matrix covers real-model runtime verification.
//
// The test is kept in place because:
//   1. It's the correct scaffolding for a future GPU unit-test harness.
//   2. It documents the exact CPU↔GPU equivalence contract the kernels
//      must satisfy.
//   3. Once schedBufts is populated (either by a better harness or by
//      a future ggml change), the test becomes a regression gate
//      automatically — no further wiring needed.
func TestOutlierEncodeDequantGPUCPUEquivalence(t *testing.T) {
	cases := []struct {
		name         string
		headDim      int
		numKVHeads   int
		bits         int
		outlierBits  int
		outlierCount int
		preset       turboquant.Preset
	}{
		{"d128_h8_tq3k", 128, 8, 3, 4, 32, turboquant.PresetTQ3K},
		{"d128_h4_tq3k", 128, 4, 3, 4, 32, turboquant.PresetTQ3K},
		{"d128_h1_tq3k", 128, 1, 3, 4, 32, turboquant.PresetTQ3K},
		{"d256_h1_tq3k", 256, 1, 3, 4, 32, turboquant.PresetTQ3K},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := setup(t)
			b := ctx.(*Context).b

			mgrAny := b.NewTQCompressedKManager(
				tc.headDim, tc.numKVHeads, tc.bits,
				tc.preset.RotationSeed,
				0, // vBits (K-only)
				tc.outlierBits, tc.outlierCount,
			)
			if mgrAny == nil {
				t.Skip("no TQ-capable GPU available (need compute capability >= 6.0)")
			}
			mgr, ok := mgrAny.(*ggmlTQCompressedK)
			if !ok {
				t.Fatalf("unexpected TQ manager type %T", mgrAny)
			}

			const nCells = 4
			capacity := nCells
			mgr.EnsureLayer(0, capacity)

			// Build a deterministic synthetic K batch: Gaussian noise
			// via the same splitmix64 as the CPU tests.
			kData := make([]float32, tc.headDim*tc.numKVHeads*nCells)
			var rngState uint64 = 0xface_feed_cafe_0000 | uint64(tc.headDim)
			rng := &rngState
			for i := range kData {
				kData[i] = float32(testGaussian(rng))
			}

			// GPU path: create a K tensor, run EncodeK + DequantK, read
			// back the dequanted output.
			kTensor := ctx.FromFloats(kData, tc.headDim, tc.numKVHeads, nCells)
			encodeResult := mgr.EncodeK(ctx, 0, kTensor, 0)
			if encodeResult == nil {
				t.Fatalf("EncodeK returned nil")
			}
			ctx.Forward(encodeResult)

			dequant := mgr.DequantK(ctx, 0, encodeResult, 0, nCells)
			if dequant == nil {
				t.Fatalf("DequantK returned nil")
			}
			ctx.Forward(dequant).Compute(dequant)

			gpuOut := dequant.Floats()
			if len(gpuOut) != tc.headDim*tc.numKVHeads*nCells {
				t.Fatalf("gpu output len = %d, want %d",
					len(gpuOut), tc.headDim*tc.numKVHeads*nCells)
			}

			// CPU reference: encode + dequant each (cell, head) slab
			// with the same preset, compute the expected rotated-space
			// output, and compare elementwise.
			//
			// Note: the CPU reference uses float64 blockScale while the
			// GPU kernel uses float32 reductions, so scales differ by
			// round-off at the 1e-6 level. The tolerance accommodates
			// that plus quantizer boundary ambiguity (when a rotated
			// value sits exactly on a Lloyd-Max boundary, float32 vs
			// float64 accumulation can pick adjacent codebook slots).
			const tol float32 = 5e-2 // loose; tight enough to catch algo bugs
			var maxErr float32
			var mismatches int
			for c := 0; c < nCells; c++ {
				for h := 0; h < tc.numKVHeads; h++ {
					slab := make([]float32, tc.headDim)
					for d := 0; d < tc.headDim; d++ {
						slab[d] = kData[(c*tc.numKVHeads+h)*tc.headDim+d]
					}

					cpuEnc, err := turboquant.EncodeKeyPerHeadOutlier(slab, tc.preset)
					if err != nil {
						t.Fatalf("cpu encode: %v", err)
					}
					cpuOut := turboquant.DequantKeyPerHeadOutlier(cpuEnc, tc.preset, tc.headDim)

					for d := 0; d < tc.headDim; d++ {
						gpuVal := gpuOut[(c*tc.numKVHeads+h)*tc.headDim+d]
						cpuVal := cpuOut[d]
						diff := gpuVal - cpuVal
						if diff < 0 {
							diff = -diff
						}
						if diff > maxErr {
							maxErr = diff
						}
						if diff > tol {
							mismatches++
							if mismatches <= 5 {
								t.Logf("mismatch c=%d h=%d d=%d gpu=%f cpu=%f diff=%f",
									c, h, d, gpuVal, cpuVal, diff)
							}
						}
					}
				}
			}
			t.Logf("%s: max elementwise err = %f (%d mismatches > %.3f)", tc.name, maxErr, mismatches, tol)
			if mismatches > 0 {
				t.Fatalf("%s: %d elements differ beyond tolerance %.3f", tc.name, mismatches, tol)
			}
		})
	}
}

// testGaussian is a local Box-Muller generator that doesn't depend on
// the unexported helpers in the turboquant package. Uses a tiny
// splitmix64 inlined to avoid adding a test dep.
func testGaussian(state *uint64) float64 {
	u1 := testUniform(state)
	u2 := testUniform(state)
	// Guard against log(0).
	if u1 < 1e-12 {
		u1 = 1e-12
	}
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

func testUniform(state *uint64) float64 {
	*state += 0x9e3779b97f4a7c15
	z := *state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z = z ^ (z >> 31)
	return float64(z>>11) / float64(1<<53)
}

// Compile-time check that ml.Tensor is what we expect.
var _ ml.Tensor = (*Tensor)(nil)
