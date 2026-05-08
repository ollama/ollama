package ggml

// Correctness tests for the D=256 outlier fused FA kernel on Metal
// (kernel_tq_fattn_vec_f16_outlier_d256 / kernel_tq_fattn_vec_packed_outlier_d256)
// and the D=128 prefill+GQA path.
//
// All tests use CPU-side encode+decode as the K reference to avoid ggml
// scheduler buffer aliasing: when dequantKF32 and attnOut are in the same
// compute graph, the scheduler may reuse dequantKF32's buffer for attnOut's
// output (both reference enc), making kRef == gpuOut before comparison.
// The CPU path has max_diff ≤ 0.0003 vs GPU dequant, well within tolerances.
//
// D=256 covers Gemma3 (attn head_dim=256). D=128 covers llama3.1/qwen2.5.
// Tests skip automatically on CUDA/ROCm (fusedKernelSupports gates D=256 on
// preferFusedAttention; D=128 outlier fused path is CUDA-supported).

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

// tqCPUKRef encodes and decodes K via the CPU reference path, returning a
// slice indexed by [c*nKVHeads+h] → []float32 of length headDim.
func tqCPUKRef(t *testing.T, kData []float32, nCells, nKVHeads, headDim int, preset turboquant.Preset) [][]float32 {
	t.Helper()
	kDecoded := make([][]float32, nCells*nKVHeads)
	for c := range nCells {
		for h := range nKVHeads {
			slab := kData[(c*nKVHeads+h)*headDim : (c*nKVHeads+h+1)*headDim]
			enc, err := turboquant.EncodeKeyPerHeadOutlier(slab, preset)
			if err != nil {
				t.Fatalf("CPU encode c=%d h=%d: %v", c, h, err)
			}
			kDecoded[c*nKVHeads+h] = turboquant.DequantKeyPerHeadOutlier(enc, preset, headDim)
		}
	}
	return kDecoded
}

// tqCPUKRefNoOutlier is like tqCPUKRef but for presets with no outlier split.
func tqCPUKRefNoOutlier(t *testing.T, kData []float32, nCells, nKVHeads, headDim int, preset turboquant.Preset) [][]float32 {
	t.Helper()
	kDecoded := make([][]float32, nCells*nKVHeads)
	for c := range nCells {
		for h := range nKVHeads {
			slab := kData[(c*nKVHeads+h)*headDim : (c*nKVHeads+h+1)*headDim]
			packed, scale, err := turboquant.EncodeKeyPerHead(slab, preset)
			if err != nil {
				t.Fatalf("CPU encode c=%d h=%d: %v", c, h, err)
			}
			kDecoded[c*nKVHeads+h] = turboquant.DequantKeyPerHead(packed, scale, headDim, preset.KeyPrimaryBits)
		}
	}
	return kDecoded
}

// tqCheckAttn compares GPU FA output against a CPU reference built from
// kDecoded (indexed [c*nKVHeads+kvHead]) and vF32. Returns (maxDiff, nMismatches).
func tqCheckAttn(
	t *testing.T,
	gpuOut []float32,
	qRotated []float32,
	kDecoded [][]float32,
	vF32 []float32,
	nTokensQ, nQHeads, nKVHeads, nCells, headDim int,
	attnScale float64,
	tol float32,
	label string,
) (maxDiff float32, nMismatches int) {
	t.Helper()
	gqaRatio := nQHeads / nKVHeads
	for tok := range nTokensQ {
		for h := range nQHeads {
			kvHead := h / gqaRatio
			qH := qRotated[(h*nTokensQ+tok)*headDim : (h*nTokensQ+tok+1)*headDim]

			scores := make([]float64, nCells)
			for c := range nCells {
				kH := kDecoded[c*nKVHeads+kvHead]
				var dot float64
				for d := range headDim {
					dot += float64(qH[d]) * float64(kH[d])
				}
				scores[c] = dot * attnScale
			}
			maxS := scores[0]
			for _, s := range scores[1:] {
				if s > maxS {
					maxS = s
				}
			}
			sumExp := 0.0
			for c := range nCells {
				scores[c] = math.Exp(scores[c] - maxS)
				sumExp += scores[c]
			}
			for c := range nCells {
				scores[c] /= sumExp
			}

			cpuOut := make([]float64, headDim)
			for c := range nCells {
				vBase := c*headDim + kvHead*headDim*nCells
				for d := range headDim {
					cpuOut[d] += scores[c] * float64(vF32[vBase+d])
				}
			}

			gpuBase := (tok*nQHeads + h) * headDim
			headMismatches := 0
			var headMaxDiff float32
			for d := range headDim {
				diff := float32(math.Abs(float64(gpuOut[gpuBase+d]) - cpuOut[d]))
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff > headMaxDiff {
					headMaxDiff = diff
				}
				if diff > tol {
					if headMismatches < 4 {
						t.Logf("MISMATCH %s tok=%d h=%d(kv=%d) d=%d gpu=%.6f cpu=%.6f diff=%.4f",
							label, tok, h, kvHead, d, gpuOut[gpuBase+d], float32(cpuOut[d]), diff)
					}
					nMismatches++
					headMismatches++
				}
			}
		}
	}
	return maxDiff, nMismatches
}

// bitConfig describes a production bit-width configuration for D=256 outlier tests.
type d256BitConfig struct {
	bits        int
	outlierBits int
	tol         float32
}

var d256Configs = []d256BitConfig{
	{bits: 2, outlierBits: 3, tol: 0.15},
	{bits: 3, outlierBits: 4, tol: 0.12},
	{bits: 4, outlierBits: 5, tol: 0.10},
}

// TestTQFusedFlashAttentionOutlierD256 verifies kernel_tq_fattn_vec_f16_outlier_d256
// (symmetric path) for all three production bit widths using decode mode (nTokensQ=1).
func TestTQFusedFlashAttentionOutlierD256(t *testing.T) {
	for _, cfg := range d256Configs {
		cfg := cfg
		t.Run(fmt.Sprintf("bits%d_outlierBits%d", cfg.bits, cfg.outlierBits), func(t *testing.T) {
			ctx, be := setupTQAny(t)
			b := be.(*Backend)

			const (
				headDim      = 256
				nKVHeads     = 4
				nCells       = 8
				outlierCount = 32
			)

			rotationSeed := turboquant.PresetTQ4K.RotationSeed ^ uint64(cfg.bits)

			mgrAny := b.NewTQCompressedKManager(
				headDim, nKVHeads, cfg.bits,
				rotationSeed,
				0,
				cfg.outlierBits, outlierCount,
				false, // symmetric
				0,
			)
			if mgrAny == nil {
				t.Skip("NewTQCompressedKManager returned nil (GPU unavailable or unsupported)")
			}
			mgr := mgrAny.(*ggmlTQCompressedK)
			mgr.EnsureLayer(0, nCells)

			if !mgr.hasOutliers() {
				t.Fatalf("hasOutliers()=false for outlierCount=%d", outlierCount)
			}
			if !mgr.fusedKernelSupports() {
				t.Skipf("fused kernel not supported for headDim=%d bits=%d (Metal only)", headDim, cfg.bits)
			}

			var rngState uint64 = 0xd256_dead_0000_0000 | uint64(cfg.bits)

			kData := make([]float32, headDim*nKVHeads*nCells)
			for i := range kData {
				kData[i] = float32(tqGaussian(&rngState))
			}
			for c := range nCells {
				for h := range nKVHeads {
					for range 8 {
						d := int(tqSplitmix64(&rngState) % uint64(headDim))
						kData[(c*nKVHeads+h)*headDim+d] *= 4.0
					}
				}
			}

			layerSeed := rotationSeed ^ uint64(1)
			rotation := turboquant.BuildRotation(headDim, layerSeed)

			qRaw := make([]float32, headDim*nKVHeads)
			for i := range qRaw {
				qRaw[i] = float32(tqGaussian(&rngState)) * 0.5
			}
			qRotated := make([]float32, headDim*nKVHeads)
			for h := range nKVHeads {
				rotQ := turboquant.ApplyRotation(qRaw[h*headDim:(h+1)*headDim], rotation)
				copy(qRotated[h*headDim:], rotQ)
			}

			vF32 := make([]float32, headDim*nCells*nKVHeads)
			for i := range vF32 {
				vF32[i] = float32(tqGaussian(&rngState)) * 0.25
			}
			vF16Bytes := tqF32SliceToF16Bytes(vF32)

			kTensor := ctx.FromFloats(kData, headDim, nKVHeads, nCells)
			qTensor := ctx.FromFloats(qRotated, headDim, 1, nKVHeads)
			vTensor := ctx.FromBytes(ml.DTypeF16, vF16Bytes, headDim, nCells, nKVHeads)

			enc := mgr.EncodeK(ctx, 0, kTensor, 0)
			if enc == nil {
				t.Fatalf("EncodeK returned nil")
			}

			tqkRaw, ok := mgr.GetAsTQTensor(ctx, 0, enc, 0, nCells)
			if !ok || tqkRaw == nil {
				t.Fatalf("GetAsTQTensor returned (nil, %v)", ok)
			}
			tqk := tqkRaw.(*tqTensor)

			if tqk.outlierPacked == nil {
				t.Fatalf("tqTensor.outlierPacked is nil — outlier encode not wired for D=256")
			}

			attnScale := 1.0 / math.Sqrt(float64(headDim))
			attnOut := b.tqFlashAttention(ctx, qTensor.(*Tensor), tqk, vTensor.(*Tensor), nil, attnScale, 0)
			if attnOut == nil {
				t.Fatalf("tqFlashAttention returned nil")
			}

			ctx.Forward(enc, attnOut).Compute(enc, attnOut)

			gpuOut := attnOut.Floats()
			if len(gpuOut) != headDim*nKVHeads {
				t.Fatalf("gpu output len=%d want %d", len(gpuOut), headDim*nKVHeads)
			}
			for i, v := range gpuOut {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("gpuOut[%d]=%v (NaN/Inf)", i, v)
				}
			}

			cpuPreset := turboquant.Preset{
				RotationSeed:  layerSeed,
				KeyPrimaryBits: cfg.bits,
				OutlierBits:   cfg.outlierBits,
				OutlierCount:  outlierCount,
			}
			kDecoded := tqCPUKRef(t, kData, nCells, nKVHeads, headDim, cpuPreset)

			// For decode (nTokensQ=1), qRotated layout is [headDim*nKVHeads]; adapt to tqCheckAttn's
			// [h*nTokensQ+tok] convention by using nTokensQ=1 so (h*1+0)*headDim = h*headDim.
			maxDiff, nMismatches := tqCheckAttn(t, gpuOut, qRotated, kDecoded, vF32,
				1, nKVHeads, nKVHeads, nCells, headDim, attnScale, cfg.tol,
				fmt.Sprintf("D=256 decode bits=%d", cfg.bits))

			t.Logf("kernel_tq_fattn_vec_f16_outlier_d256 %d-bit (sym): max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
				cfg.bits, maxDiff, nMismatches, headDim*nKVHeads, cfg.tol)

			if nMismatches > 0 {
				t.Fatalf("%d elements differ beyond %.2f for bits=%d D=256",
					nMismatches, cfg.tol, cfg.bits)
			}
		})
	}
}

// TestTQFusedFlashAttentionOutlierD256Prefill verifies the ncols=2 (prefill) code path
// with GQA 2:1 (8 Q heads, 4 KV heads), mirroring gemma3-4b production inference.
func TestTQFusedFlashAttentionOutlierD256Prefill(t *testing.T) {
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 256
		nKVHeads     = 4
		nQHeads      = 8
		nTokensQ     = 4
		nCells       = 4
		bits         = 4
		outlierBits  = 5
		outlierCount = 32
	)

	rotationSeed := turboquant.PresetTQ4K.RotationSeed ^ 0xd256_face

	mgrAny := b.NewTQCompressedKManager(
		headDim, nKVHeads, bits,
		rotationSeed,
		0,
		outlierBits, outlierCount,
		true, // asymmetricPrimary — production tq4k
		0,
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, nCells)

	if !mgr.fusedKernelSupports() {
		t.Skipf("fused kernel not supported for headDim=%d bits=%d (Metal only)", headDim, bits)
	}

	var rngState uint64 = 0xd256_face_cafe_0000

	layerSeed := rotationSeed ^ uint64(1)
	rotation := turboquant.BuildRotation(headDim, layerSeed)

	kData := make([]float32, headDim*nKVHeads*nCells)
	for c := range nCells {
		for h := range nKVHeads {
			base := (c*nKVHeads + h) * headDim
			for d := range headDim {
				kData[base+d] = float32(tqGaussian(&rngState)) * 0.8
			}
			for range 8 {
				d := int(tqSplitmix64(&rngState) % uint64(headDim))
				kData[base+d] *= 4.0
			}
		}
	}

	qRaw := make([][]float32, nQHeads*nTokensQ)
	for h := range nQHeads {
		for tok := range nTokensQ {
			v := make([]float32, headDim)
			for d := range headDim {
				v[d] = float32(tqGaussian(&rngState)) * 0.5
			}
			qRaw[h*nTokensQ+tok] = v
		}
	}
	qRotated := make([]float32, headDim*nQHeads*nTokensQ)
	for h := range nQHeads {
		for tok := range nTokensQ {
			rotQ := turboquant.ApplyRotation(qRaw[h*nTokensQ+tok], rotation)
			copy(qRotated[(h*nTokensQ+tok)*headDim:], rotQ)
		}
	}

	vF32 := make([]float32, headDim*nCells*nKVHeads)
	for i := range vF32 {
		vF32[i] = float32(tqGaussian(&rngState)) * 0.25
	}
	vF16Bytes := tqF32SliceToF16Bytes(vF32)

	qTensor := ctx.FromFloats(qRotated, headDim, nTokensQ, nQHeads)
	kTensor := ctx.FromFloats(kData, headDim, nKVHeads, nCells)
	vTensor := ctx.FromBytes(ml.DTypeF16, vF16Bytes, headDim, nCells, nKVHeads)

	enc := mgr.EncodeK(ctx, 0, kTensor, 0)
	if enc == nil {
		t.Fatalf("EncodeK returned nil")
	}

	tqkRaw, ok := mgr.GetAsTQTensor(ctx, 0, enc, 0, nCells)
	if !ok || tqkRaw == nil {
		t.Fatalf("GetAsTQTensor returned (nil, %v)", ok)
	}
	tqk := tqkRaw.(*tqTensor)

	if tqk.outlierPacked == nil {
		t.Fatalf("tqTensor.outlierPacked is nil")
	}

	attnScale := 1.0 / math.Sqrt(float64(headDim))
	attnOut := b.tqFlashAttention(ctx, qTensor.(*Tensor), tqk, vTensor.(*Tensor), nil, attnScale, 0)
	if attnOut == nil {
		t.Fatalf("tqFlashAttention returned nil")
	}

	ctx.Forward(enc, attnOut).Compute(enc, attnOut)

	gpuOut := attnOut.Floats()
	for i, v := range gpuOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("gpuOut[%d]=%v (NaN/Inf)", i, v)
		}
	}

	cpuPreset := turboquant.Preset{
		RotationSeed:      layerSeed,
		KeyPrimaryBits:    bits,
		OutlierBits:       outlierBits,
		OutlierCount:      outlierCount,
		AsymmetricPrimary: true,
	}
	kDecoded := tqCPUKRef(t, kData, nCells, nKVHeads, headDim, cpuPreset)

	const tol float32 = 0.20
	maxDiff, nMismatches := tqCheckAttn(t, gpuOut, qRotated, kDecoded, vF32,
		nTokensQ, nQHeads, nKVHeads, nCells, headDim, attnScale, tol,
		"D=256 prefill GQA-2:1")

	total := headDim * nQHeads * nTokensQ
	t.Logf("kernel_tq_fattn_vec_f16_outlier_d256 prefill (ncols=2, GQA 8:4): max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
		maxDiff, nMismatches, total, tol)

	if nMismatches > 0 {
		t.Fatalf("%d elements differ beyond %.2f for D=256 prefill (ncols=2)", nMismatches, tol)
	}
}

// TestTQFusedFlashAttentionOutlierD256GQA2x1Decode verifies the GQA 2:1 decode path
// (nTokensQ=1, ncols=1) with 8 Q heads and 4 KV heads.
func TestTQFusedFlashAttentionOutlierD256GQA2x1Decode(t *testing.T) {
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 256
		nKVHeads     = 4
		nQHeads      = 8
		nTokensQ     = 1
		nCells       = 4
		bits         = 4
		outlierBits  = 5
		outlierCount = 32
	)

	rotationSeed := turboquant.PresetTQ4K.RotationSeed ^ 0xd256_b001

	mgrAny := b.NewTQCompressedKManager(
		headDim, nKVHeads, bits,
		rotationSeed,
		0,
		outlierBits, outlierCount,
		true, // asymmetric
		0,
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, nCells)

	if !mgr.fusedKernelSupports() {
		t.Skipf("fused kernel not supported for headDim=%d bits=%d (Metal only)", headDim, bits)
	}

	var rngState uint64 = 0xd256_b001_cafe_0000

	layerSeed := rotationSeed ^ uint64(1)
	rotation := turboquant.BuildRotation(headDim, layerSeed)

	kData := make([]float32, headDim*nKVHeads*nCells)
	for c := range nCells {
		for h := range nKVHeads {
			base := (c*nKVHeads + h) * headDim
			for d := range headDim {
				kData[base+d] = float32(tqGaussian(&rngState)) * 0.8
			}
			for range 8 {
				d := int(tqSplitmix64(&rngState) % uint64(headDim))
				kData[base+d] *= 4.0
			}
		}
	}

	qRaw := make([][]float32, nQHeads)
	for h := range nQHeads {
		v := make([]float32, headDim)
		for d := range headDim {
			v[d] = float32(tqGaussian(&rngState)) * 0.5
		}
		qRaw[h] = v
	}
	qRotated := make([]float32, headDim*nQHeads)
	for h := range nQHeads {
		rotQ := turboquant.ApplyRotation(qRaw[h], rotation)
		copy(qRotated[h*headDim:], rotQ)
	}

	vF32 := make([]float32, headDim*nCells*nKVHeads)
	for i := range vF32 {
		vF32[i] = float32(tqGaussian(&rngState)) * 0.25
	}
	vF16Bytes := tqF32SliceToF16Bytes(vF32)

	qTensor := ctx.FromFloats(qRotated, headDim, nTokensQ, nQHeads)
	kTensor := ctx.FromFloats(kData, headDim, nKVHeads, nCells)
	vTensor := ctx.FromBytes(ml.DTypeF16, vF16Bytes, headDim, nCells, nKVHeads)

	enc := mgr.EncodeK(ctx, 0, kTensor, 0)
	if enc == nil {
		t.Fatalf("EncodeK returned nil")
	}

	tqkRaw, ok := mgr.GetAsTQTensor(ctx, 0, enc, 0, nCells)
	if !ok || tqkRaw == nil {
		t.Fatalf("GetAsTQTensor returned (nil, %v)", ok)
	}
	tqk := tqkRaw.(*tqTensor)

	attnScale := 1.0 / math.Sqrt(float64(headDim))
	attnOut := b.tqFlashAttention(ctx, qTensor.(*Tensor), tqk, vTensor.(*Tensor), nil, attnScale, 0)
	if attnOut == nil {
		t.Fatalf("tqFlashAttention returned nil")
	}

	ctx.Forward(enc, attnOut).Compute(enc, attnOut)

	gpuOut := attnOut.Floats()
	for i, v := range gpuOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("gpuOut[%d]=%v (NaN/Inf)", i, v)
		}
	}

	cpuPreset := turboquant.Preset{
		RotationSeed:      layerSeed,
		KeyPrimaryBits:    bits,
		OutlierBits:       outlierBits,
		OutlierCount:      outlierCount,
		AsymmetricPrimary: true,
	}
	kDecoded := tqCPUKRef(t, kData, nCells, nKVHeads, headDim, cpuPreset)

	const tol float32 = 0.20
	maxDiff, nMismatches := tqCheckAttn(t, gpuOut, qRotated, kDecoded, vF32,
		nTokensQ, nQHeads, nKVHeads, nCells, headDim, attnScale, tol,
		"D=256 decode GQA-2:1")

	t.Logf("kernel_tq_fattn_vec_f16_outlier_d256 decode (ncols=1, GQA 2:1): max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
		maxDiff, nMismatches, headDim*nQHeads*nTokensQ, tol)

	if nMismatches > 0 {
		t.Fatalf("%d elements differ beyond %.2f for D=256 decode GQA 2:1", nMismatches, tol)
	}
}

// TestTQFusedFlashAttentionOutlierD256Asymmetric verifies the asymmetric primary path
// (asymmetricPrimary=true, tq4k production preset) in decode mode.
func TestTQFusedFlashAttentionOutlierD256Asymmetric(t *testing.T) {
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 256
		nKVHeads     = 4
		nCells       = 8
		bits         = 4
		outlierBits  = 5
		outlierCount = 32
	)

	rotationSeed := turboquant.PresetTQ4K.RotationSeed ^ 0xd256

	mgrAny := b.NewTQCompressedKManager(
		headDim, nKVHeads, bits,
		rotationSeed,
		0,
		outlierBits, outlierCount,
		true, // asymmetricPrimary — production tq4k
		0,
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, nCells)

	if !mgr.fusedKernelSupports() {
		t.Skipf("fused kernel not supported for headDim=%d bits=%d (Metal only)", headDim, bits)
	}

	var rngState uint64 = 0xd256_a5b1_0000_0000

	bias := make([]float32, headDim)
	for d := range headDim {
		bias[d] = float32(tqGaussian(&rngState)) * 0.5
	}

	kData := make([]float32, headDim*nKVHeads*nCells)
	for c := range nCells {
		for h := range nKVHeads {
			base := (c*nKVHeads + h) * headDim
			for d := range headDim {
				kData[base+d] = float32(tqGaussian(&rngState)) + bias[d]
			}
		}
	}
	for c := range nCells {
		for h := range nKVHeads {
			for range 8 {
				d := int(tqSplitmix64(&rngState) % uint64(headDim))
				kData[(c*nKVHeads+h)*headDim+d] *= 4.0
			}
		}
	}

	layerSeed := rotationSeed ^ uint64(1)
	rotation := turboquant.BuildRotation(headDim, layerSeed)

	qRaw := make([]float32, headDim*nKVHeads)
	for i := range qRaw {
		qRaw[i] = float32(tqGaussian(&rngState)) * 0.5
	}
	qRotated := make([]float32, headDim*nKVHeads)
	for h := range nKVHeads {
		rotQ := turboquant.ApplyRotation(qRaw[h*headDim:(h+1)*headDim], rotation)
		copy(qRotated[h*headDim:], rotQ)
	}

	vF32 := make([]float32, headDim*nCells*nKVHeads)
	for i := range vF32 {
		vF32[i] = float32(tqGaussian(&rngState)) * 0.25
	}
	vF16Bytes := tqF32SliceToF16Bytes(vF32)

	kTensor := ctx.FromFloats(kData, headDim, nKVHeads, nCells)
	qTensor := ctx.FromFloats(qRotated, headDim, 1, nKVHeads)
	vTensor := ctx.FromBytes(ml.DTypeF16, vF16Bytes, headDim, nCells, nKVHeads)

	enc := mgr.EncodeK(ctx, 0, kTensor, 0)
	if enc == nil {
		t.Fatalf("EncodeK returned nil")
	}

	tqkRaw, ok := mgr.GetAsTQTensor(ctx, 0, enc, 0, nCells)
	if !ok || tqkRaw == nil {
		t.Fatalf("GetAsTQTensor returned (nil, %v)", ok)
	}
	tqk := tqkRaw.(*tqTensor)

	if tqk.outlierPacked == nil {
		t.Fatalf("tqTensor.outlierPacked is nil")
	}
	if !tqk.asymmetric {
		t.Fatalf("tqTensor.asymmetric=false with asymmetricPrimary=true manager")
	}
	if tqk.zeros == nil {
		t.Fatalf("tqTensor.zeros is nil for asymmetric manager")
	}

	attnScale := 1.0 / math.Sqrt(float64(headDim))
	attnOut := b.tqFlashAttention(ctx, qTensor.(*Tensor), tqk, vTensor.(*Tensor), nil, attnScale, 0)
	if attnOut == nil {
		t.Fatalf("tqFlashAttention returned nil")
	}

	ctx.Forward(enc, attnOut).Compute(enc, attnOut)

	gpuOut := attnOut.Floats()
	for i, v := range gpuOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("gpuOut[%d]=%v (NaN/Inf)", i, v)
		}
	}

	cpuPreset := turboquant.Preset{
		RotationSeed:      layerSeed,
		KeyPrimaryBits:    bits,
		OutlierBits:       outlierBits,
		OutlierCount:      outlierCount,
		AsymmetricPrimary: true,
	}
	kDecoded := tqCPUKRef(t, kData, nCells, nKVHeads, headDim, cpuPreset)

	const tol float32 = 0.20
	maxDiff, nMismatches := tqCheckAttn(t, gpuOut, qRotated, kDecoded, vF32,
		1, nKVHeads, nKVHeads, nCells, headDim, attnScale, tol,
		"D=256 decode asymmetric")

	t.Logf("kernel_tq_fattn_vec_f16_outlier_d256 asymmetric decode: max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
		maxDiff, nMismatches, headDim*nKVHeads, tol)

	if nMismatches > 0 {
		t.Fatalf("%d elements differ beyond %.2f for D=256 asymmetric decode", nMismatches, tol)
	}
}

// TestTQFusedFlashAttentionOutlierD256PrefillNoOutliers verifies the non-outlier
// kernel (kernel_tq_fattn_vec_f16_d256) in prefill mode (ncols=2, GQA 2:1).
func TestTQFusedFlashAttentionOutlierD256PrefillNoOutliers(t *testing.T) {
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 256
		nKVHeads     = 4
		nQHeads      = 8
		nTokensQ     = 4
		nCells       = 4
		bits         = 4
		outlierBits  = 0
		outlierCount = 0
	)

	rotationSeed := turboquant.PresetTQ4K.RotationSeed ^ 0xd256_f001

	mgrAny := b.NewTQCompressedKManager(
		headDim, nKVHeads, bits,
		rotationSeed,
		0,
		outlierBits, outlierCount,
		true,
		0,
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, nCells)

	if !mgr.fusedKernelSupports() {
		t.Skipf("fused kernel not supported for headDim=%d bits=%d (Metal only)", headDim, bits)
	}

	var rngState uint64 = 0xd256_f001_cafe_0000

	layerSeed := rotationSeed ^ uint64(1)
	rotation := turboquant.BuildRotation(headDim, layerSeed)

	kData := make([]float32, headDim*nKVHeads*nCells)
	for c := range nCells {
		for h := range nKVHeads {
			base := (c*nKVHeads + h) * headDim
			for d := range headDim {
				kData[base+d] = float32(tqGaussian(&rngState)) * 0.8
			}
		}
	}

	qRaw := make([][]float32, nQHeads*nTokensQ)
	for h := range nQHeads {
		for tok := range nTokensQ {
			v := make([]float32, headDim)
			for d := range headDim {
				v[d] = float32(tqGaussian(&rngState)) * 0.5
			}
			qRaw[h*nTokensQ+tok] = v
		}
	}
	qRotated := make([]float32, headDim*nQHeads*nTokensQ)
	for h := range nQHeads {
		for tok := range nTokensQ {
			rotQ := turboquant.ApplyRotation(qRaw[h*nTokensQ+tok], rotation)
			copy(qRotated[(h*nTokensQ+tok)*headDim:], rotQ)
		}
	}

	vF32 := make([]float32, headDim*nCells*nKVHeads)
	for i := range vF32 {
		vF32[i] = float32(tqGaussian(&rngState)) * 0.25
	}
	vF16Bytes := tqF32SliceToF16Bytes(vF32)

	qTensor := ctx.FromFloats(qRotated, headDim, nTokensQ, nQHeads)
	kTensor := ctx.FromFloats(kData, headDim, nKVHeads, nCells)
	vTensor := ctx.FromBytes(ml.DTypeF16, vF16Bytes, headDim, nCells, nKVHeads)

	enc := mgr.EncodeK(ctx, 0, kTensor, 0)
	if enc == nil {
		t.Fatalf("EncodeK returned nil")
	}

	tqkRaw, ok := mgr.GetAsTQTensor(ctx, 0, enc, 0, nCells)
	if !ok || tqkRaw == nil {
		t.Fatalf("GetAsTQTensor returned (nil, %v)", ok)
	}
	tqk := tqkRaw.(*tqTensor)

	if tqk.outlierPacked != nil {
		t.Fatalf("outlierPacked non-nil with outlierCount=0 — routing error")
	}

	attnScale := 1.0 / math.Sqrt(float64(headDim))
	attnOut := b.tqFlashAttention(ctx, qTensor.(*Tensor), tqk, vTensor.(*Tensor), nil, attnScale, 0)
	if attnOut == nil {
		t.Fatalf("tqFlashAttention returned nil")
	}

	ctx.Forward(enc, attnOut).Compute(enc, attnOut)

	gpuOut := attnOut.Floats()
	for i, v := range gpuOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("gpuOut[%d]=%v (NaN/Inf)", i, v)
		}
	}

	cpuPreset := turboquant.Preset{
		RotationSeed:      layerSeed,
		KeyPrimaryBits:    bits,
		AsymmetricPrimary: true,
	}
	kDecoded := tqCPUKRefNoOutlier(t, kData, nCells, nKVHeads, headDim, cpuPreset)

	const tol float32 = 0.10
	maxDiff, nMismatches := tqCheckAttn(t, gpuOut, qRotated, kDecoded, vF32,
		nTokensQ, nQHeads, nKVHeads, nCells, headDim, attnScale, tol,
		"D=256 prefill no-outliers GQA-2:1")

	total := headDim * nQHeads * nTokensQ
	t.Logf("D=256 prefill no-outliers (ncols=2, GQA 2:1): max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
		maxDiff, nMismatches, total, tol)

	if nMismatches > 0 {
		t.Fatalf("%d elements differ beyond %.2f for D=256 prefill no-outliers", nMismatches, tol)
	}
}

// TestTQFusedFlashAttentionOutlierD256PrefillSymmetric runs the prefill config
// (ncols=2, GQA 2:1, nCells=4, outlierCount=32) with symmetric=false.
func TestTQFusedFlashAttentionOutlierD256PrefillSymmetric(t *testing.T) {
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 256
		nKVHeads     = 4
		nQHeads      = 8
		nTokensQ     = 4
		nCells       = 4
		bits         = 4
		outlierBits  = 5
		outlierCount = 32
	)

	rotationSeed := turboquant.PresetTQ4K.RotationSeed ^ 0xd256_f002

	mgrAny := b.NewTQCompressedKManager(
		headDim, nKVHeads, bits,
		rotationSeed,
		0,
		outlierBits, outlierCount,
		false, // symmetric
		0,
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, nCells)

	if !mgr.hasOutliers() {
		t.Fatalf("hasOutliers()=false for outlierCount=%d", outlierCount)
	}
	if !mgr.fusedKernelSupports() {
		t.Skipf("fused kernel not supported for headDim=%d bits=%d (Metal only)", headDim, bits)
	}

	var rngState uint64 = 0xd256_f002_cafe_0000

	layerSeed := rotationSeed ^ uint64(1)
	rotation := turboquant.BuildRotation(headDim, layerSeed)

	kData := make([]float32, headDim*nKVHeads*nCells)
	for c := range nCells {
		for h := range nKVHeads {
			base := (c*nKVHeads + h) * headDim
			for d := range headDim {
				kData[base+d] = float32(tqGaussian(&rngState)) * 0.8
			}
			for range 8 {
				d := int(tqSplitmix64(&rngState) % uint64(headDim))
				kData[base+d] *= 4.0
			}
		}
	}

	qRaw := make([][]float32, nQHeads*nTokensQ)
	for h := range nQHeads {
		for tok := range nTokensQ {
			v := make([]float32, headDim)
			for d := range headDim {
				v[d] = float32(tqGaussian(&rngState)) * 0.5
			}
			qRaw[h*nTokensQ+tok] = v
		}
	}
	qRotated := make([]float32, headDim*nQHeads*nTokensQ)
	for h := range nQHeads {
		for tok := range nTokensQ {
			rotQ := turboquant.ApplyRotation(qRaw[h*nTokensQ+tok], rotation)
			copy(qRotated[(h*nTokensQ+tok)*headDim:], rotQ)
		}
	}

	vF32 := make([]float32, headDim*nCells*nKVHeads)
	for i := range vF32 {
		vF32[i] = float32(tqGaussian(&rngState)) * 0.25
	}
	vF16Bytes := tqF32SliceToF16Bytes(vF32)

	qTensor := ctx.FromFloats(qRotated, headDim, nTokensQ, nQHeads)
	kTensor := ctx.FromFloats(kData, headDim, nKVHeads, nCells)
	vTensor := ctx.FromBytes(ml.DTypeF16, vF16Bytes, headDim, nCells, nKVHeads)

	enc := mgr.EncodeK(ctx, 0, kTensor, 0)
	if enc == nil {
		t.Fatalf("EncodeK returned nil")
	}

	tqkRaw, ok := mgr.GetAsTQTensor(ctx, 0, enc, 0, nCells)
	if !ok || tqkRaw == nil {
		t.Fatalf("GetAsTQTensor returned (nil, %v)", ok)
	}
	tqk := tqkRaw.(*tqTensor)

	if tqk.outlierPacked == nil {
		t.Fatalf("tqTensor.outlierPacked is nil for symmetric outlier manager")
	}
	if tqk.asymmetric {
		t.Fatalf("tqTensor.asymmetric=true with symmetric manager")
	}

	attnScale := 1.0 / math.Sqrt(float64(headDim))
	attnOut := b.tqFlashAttention(ctx, qTensor.(*Tensor), tqk, vTensor.(*Tensor), nil, attnScale, 0)
	if attnOut == nil {
		t.Fatalf("tqFlashAttention returned nil")
	}

	ctx.Forward(enc, attnOut).Compute(enc, attnOut)

	gpuOut := attnOut.Floats()
	for i, v := range gpuOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("gpuOut[%d]=%v (NaN/Inf)", i, v)
		}
	}

	cpuPreset := turboquant.Preset{
		RotationSeed:   layerSeed,
		KeyPrimaryBits: bits,
		OutlierBits:    outlierBits,
		OutlierCount:   outlierCount,
	}
	kDecoded := tqCPUKRef(t, kData, nCells, nKVHeads, headDim, cpuPreset)

	const tol float32 = 0.12
	maxDiff, nMismatches := tqCheckAttn(t, gpuOut, qRotated, kDecoded, vF32,
		nTokensQ, nQHeads, nKVHeads, nCells, headDim, attnScale, tol,
		"D=256 prefill symmetric GQA-2:1")

	total := headDim * nQHeads * nTokensQ
	t.Logf("D=256 prefill symmetric (ncols=2, GQA 2:1): max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
		maxDiff, nMismatches, total, tol)

	if nMismatches > 0 {
		t.Fatalf("%d elements differ beyond %.2f for D=256 prefill symmetric", nMismatches, tol)
	}
}

// TestTQFusedFlashAttentionD128PrefillGQA verifies the D=128 outlier fused kernel
// in prefill mode (ncols=2, GQA 2:1), covering llama3.1/qwen2.5 production geometry.
func TestTQFusedFlashAttentionD128PrefillGQA(t *testing.T) {
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 128
		nKVHeads     = 4
		nQHeads      = 8
		nTokensQ     = 4
		nCells       = 4
		bits         = 4
		outlierBits  = 5
		outlierCount = 32
	)

	rotationSeed := turboquant.PresetTQ4K.RotationSeed ^ 0xd128_f001

	mgrAny := b.NewTQCompressedKManager(
		headDim, nKVHeads, bits,
		rotationSeed,
		0,
		outlierBits, outlierCount,
		true,
		0,
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, nCells)

	if !mgr.hasOutliers() {
		t.Fatalf("hasOutliers()=false for outlierCount=%d", outlierCount)
	}
	if !mgr.fusedKernelSupports() {
		t.Skipf("fused kernel not supported for headDim=%d bits=%d (Metal only)", headDim, bits)
	}

	var rngState uint64 = 0xd128_f001_cafe_0000

	layerSeed := rotationSeed ^ uint64(1)
	rotation := turboquant.BuildRotation(headDim, layerSeed)

	kData := make([]float32, headDim*nKVHeads*nCells)
	for c := range nCells {
		for h := range nKVHeads {
			base := (c*nKVHeads + h) * headDim
			for d := range headDim {
				kData[base+d] = float32(tqGaussian(&rngState)) * 0.8
			}
			for range 8 {
				d := int(tqSplitmix64(&rngState) % uint64(headDim))
				kData[base+d] *= 4.0
			}
		}
	}

	qRaw := make([][]float32, nQHeads*nTokensQ)
	for h := range nQHeads {
		for tok := range nTokensQ {
			v := make([]float32, headDim)
			for d := range headDim {
				v[d] = float32(tqGaussian(&rngState)) * 0.5
			}
			qRaw[h*nTokensQ+tok] = v
		}
	}
	qRotated := make([]float32, headDim*nQHeads*nTokensQ)
	for h := range nQHeads {
		for tok := range nTokensQ {
			rotQ := turboquant.ApplyRotation(qRaw[h*nTokensQ+tok], rotation)
			copy(qRotated[(h*nTokensQ+tok)*headDim:], rotQ)
		}
	}

	vF32 := make([]float32, headDim*nCells*nKVHeads)
	for i := range vF32 {
		vF32[i] = float32(tqGaussian(&rngState)) * 0.25
	}
	vF16Bytes := tqF32SliceToF16Bytes(vF32)

	qTensor := ctx.FromFloats(qRotated, headDim, nTokensQ, nQHeads)
	kTensor := ctx.FromFloats(kData, headDim, nKVHeads, nCells)
	vTensor := ctx.FromBytes(ml.DTypeF16, vF16Bytes, headDim, nCells, nKVHeads)

	enc := mgr.EncodeK(ctx, 0, kTensor, 0)
	if enc == nil {
		t.Fatalf("EncodeK returned nil")
	}

	tqkRaw, ok := mgr.GetAsTQTensor(ctx, 0, enc, 0, nCells)
	if !ok || tqkRaw == nil {
		t.Fatalf("GetAsTQTensor returned (nil, %v)", ok)
	}
	tqk := tqkRaw.(*tqTensor)

	if tqk.outlierPacked == nil {
		t.Fatalf("tqTensor.outlierPacked is nil for outlier manager")
	}

	attnScale := 1.0 / math.Sqrt(float64(headDim))
	attnOut := b.tqFlashAttention(ctx, qTensor.(*Tensor), tqk, vTensor.(*Tensor), nil, attnScale, 0)
	if attnOut == nil {
		t.Fatalf("tqFlashAttention returned nil")
	}

	ctx.Forward(enc, attnOut).Compute(enc, attnOut)

	gpuOut := attnOut.Floats()
	for i, v := range gpuOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("gpuOut[%d]=%v (NaN/Inf)", i, v)
		}
	}

	cpuPreset := turboquant.Preset{
		RotationSeed:      layerSeed,
		KeyPrimaryBits:    bits,
		OutlierBits:       outlierBits,
		OutlierCount:      outlierCount,
		AsymmetricPrimary: true,
	}
	kDecoded := tqCPUKRef(t, kData, nCells, nKVHeads, headDim, cpuPreset)

	const tol float32 = 0.10
	maxDiff, nMismatches := tqCheckAttn(t, gpuOut, qRotated, kDecoded, vF32,
		nTokensQ, nQHeads, nKVHeads, nCells, headDim, attnScale, tol,
		"D=128 prefill GQA-2:1")

	total := headDim * nQHeads * nTokensQ
	t.Logf("D=128 prefill GQA (ncols=2, GQA 2:1, outlier kernel): max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
		maxDiff, nMismatches, total, tol)

	if nMismatches > 0 {
		t.Fatalf("%d elements differ beyond %.2f for D=128 prefill GQA", nMismatches, tol)
	}
}
