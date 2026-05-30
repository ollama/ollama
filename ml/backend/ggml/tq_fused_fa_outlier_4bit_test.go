package ggml

// TestTQFusedFlashAttentionOutlier4bit and TestTQFusedFlashAttentionOutlier4bitAsymmetric
// exercise the bits=4 / outlierBits=5 fused flash-attention path used by the
// production tq4 / tq4k presets (32-entry outlier codebook). Compared
// against the CPU decode reference.
//
// Coverage:
//   - Symmetric:  EncodeKeyPerHeadOutlier → fused FA
//   - Asymmetric: bits=4, outlierBits=5 — production tq4k configuration.

import (
	"math"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

func TestTQFusedFlashAttentionOutlier4bit(t *testing.T) {
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 128
		nKVHeads     = 8
		nCells       = 8 // small N exercises the per-cell stride path
		bits         = 4
		outlierBits  = 5
		outlierCount = 32
	)

	rotationSeed := turboquant.PresetTQ4K.RotationSeed

	mgrAny := b.NewTQCompressedKManager(
		headDim, nKVHeads, bits,
		rotationSeed,
		0, // vBits=0: K-only, routes to f16_outlier kernel
		outlierBits, outlierCount,
		false, // asymmetricPrimary=false for clean CPU reference
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
		t.Skipf("fused kernel not supported for headDim=%d bits=%d", headDim, bits)
	}

	var rngState uint64 = 0xfeed_babe_0000_0000 | uint64(bits)

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
		t.Fatalf("tqTensor.outlierPacked is nil — outlier encode not wired")
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
		ID: 201, Name: "test_4bit_outlier_sym",
		RotationSeed:   layerSeed,
		KeyPrimaryBits: bits,
		OutlierBits:    outlierBits,
		OutlierCount:   outlierCount,
	}

	kDecoded := make([][]float32, nCells*nKVHeads)
	for c := range nCells {
		for h := range nKVHeads {
			slab := kData[(c*nKVHeads+h)*headDim : (c*nKVHeads+h+1)*headDim]
			cpuEnc, err := turboquant.EncodeKeyPerHeadOutlier(slab, cpuPreset)
			if err != nil {
				t.Fatalf("CPU encode c=%d h=%d: %v", c, h, err)
			}
			kDecoded[c*nKVHeads+h] = turboquant.DequantKeyPerHeadOutlier(cpuEnc, cpuPreset, headDim)
		}
	}

	const tol float32 = 0.10
	var maxDiff float32
	var nMismatches int

	for h := range nKVHeads {
		qH := qRotated[h*headDim : (h+1)*headDim]

		scores := make([]float64, nCells)
		for c := range nCells {
			kH := kDecoded[c*nKVHeads+h]
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
			vBase := c*headDim + h*headDim*nCells
			for d := range headDim {
				cpuOut[d] += scores[c] * float64(vF32[vBase+d])
			}
		}

		gpuBase := h * headDim
		for d := range headDim {
			diff := float32(math.Abs(float64(gpuOut[gpuBase+d]) - cpuOut[d]))
			if diff > maxDiff {
				maxDiff = diff
			}
			if diff > tol {
				nMismatches++
				if nMismatches <= 5 {
					t.Logf("mismatch h=%d d=%d gpu=%.4f cpu=%.4f diff=%.4f",
						h, d, gpuOut[gpuBase+d], cpuOut[d], diff)
				}
			}
		}
	}

	t.Logf("kernel_tq_fattn_vec_f16_outlier 4-bit (sym): max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
		maxDiff, nMismatches, headDim*nKVHeads, tol)

	if nMismatches > 0 {
		t.Fatalf("%d elements differ beyond %.2f — 5-bit outlier decode likely broken (sign flips indicate fallback to 2-bit)", nMismatches, tol)
	}
}

// TestTQFusedFlashAttentionOutlier4bitAsymmetric verifies the fused FA kernel
// for the production tq4k configuration (bits=4, outlierBits=5, asymmetric=true).
// The CPU reference comes from the GPU's own DequantK output so the comparison
// isolates only the fused kernel's inline K decode, not encode-side EDEN gaps.
func TestTQFusedFlashAttentionOutlier4bitAsymmetric(t *testing.T) {
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 128
		nKVHeads     = 8
		nCells       = 8
		bits         = 4
		outlierBits  = 5
		outlierCount = 32
	)

	rotationSeed := turboquant.PresetTQ4K.RotationSeed

	mgrAny := b.NewTQCompressedKManager(
		headDim, nKVHeads, bits,
		rotationSeed,
		0,
		outlierBits, outlierCount,
		true, // asymmetricPrimary=true — production tq4k
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, nCells)

	if !mgr.fusedKernelSupports() {
		t.Skipf("fused kernel not supported for headDim=%d bits=%d", headDim, bits)
	}

	var rngState uint64 = 0xdead_4b17_0000_0000 | uint64(bits)

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
	if len(gpuOut) != headDim*nKVHeads {
		t.Fatalf("attnOut len=%d want %d", len(gpuOut), headDim*nKVHeads)
	}

	for i, v := range gpuOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("gpuOut[%d]=%v (NaN/Inf)", i, v)
		}
	}

	// CPU asymmetric+EDEN reference: independent encode+decode, no GPU dequant
	// in the same compute graph (avoids kRef aliasing). See feedback memory
	// `feedback_ggml_test_kref_aliasing`.
	cpuPreset := turboquant.Preset{
		ID: 202, Name: "test_outlier_asym_4bit",
		RotationSeed:      layerSeed,
		KeyPrimaryBits:    bits,
		OutlierBits:       outlierBits,
		OutlierCount:      outlierCount,
		AsymmetricPrimary: true,
	}
	kDecoded := make([][]float32, nCells*nKVHeads)
	for c := range nCells {
		for h := range nKVHeads {
			slab := kData[(c*nKVHeads+h)*headDim : (c*nKVHeads+h+1)*headDim]
			cpuEnc, err := turboquant.EncodeKeyPerHeadOutlier(slab, cpuPreset)
			if err != nil {
				t.Fatalf("CPU encode c=%d h=%d: %v", c, h, err)
			}
			kDecoded[c*nKVHeads+h] = turboquant.DequantKeyPerHeadOutlier(cpuEnc, cpuPreset, headDim)
		}
	}

	// Tolerance 0.20: same as 3-bit asymmetric. If the 5-bit outlier decode is
	// wrong, errors are sign flips and 5× magnitude divergence — far above 0.20.
	const tol float32 = 0.20
	var maxDiff float32
	var nMismatches int

	for h := range nKVHeads {
		qH := qRotated[h*headDim : (h+1)*headDim]

		scores := make([]float64, nCells)
		for c := range nCells {
			kH := kDecoded[c*nKVHeads+h]
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
			vBase := c*headDim + h*headDim*nCells
			for d := range headDim {
				cpuOut[d] += scores[c] * float64(vF32[vBase+d])
			}
		}

		gpuBase := h * headDim
		for d := range headDim {
			diff := float32(math.Abs(float64(gpuOut[gpuBase+d]) - cpuOut[d]))
			if diff > maxDiff {
				maxDiff = diff
			}
			if diff > tol {
				nMismatches++
				if nMismatches <= 5 {
					t.Logf("mismatch h=%d d=%d gpu=%.4f cpu=%.4f diff=%.4f",
						h, d, gpuOut[gpuBase+d], cpuOut[d], diff)
				}
			}
		}
	}

	t.Logf("kernel_tq_fattn_vec_f16_outlier 4-bit (asymmetric): max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
		maxDiff, nMismatches, headDim*nKVHeads, tol)

	if nMismatches > 0 {
		t.Fatalf("%d elements differ beyond %.2f — sign flips indicate 5-bit outlier fallback to 2-bit decode", nMismatches, tol)
	}
}
