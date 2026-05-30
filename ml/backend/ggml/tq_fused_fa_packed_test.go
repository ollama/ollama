package ggml

// Tests for the kernel_tq_fattn_vec_packed_outlier_* family:
// K+V fused inline-decode flash attention (outlier split, various head dims).
// This is the Metal-default path for tq2/tq3/tq4 K+V presets on Apple Silicon.
//
// Coverage:
//   packed_outlier_d64  — llama3.2:3b head_dim=64
//   packed_outlier      — D=128 (qwen2.5, llama3.1)
//   packed_outlier_d256 — D=256 (gemma3)
//   packed_outlier_d512 — D=512 (gemma4 global attention layers)

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

type tqPackedFAConfig struct {
	headDim, nKVHeads, nCells int
	bits, vBits               int
	outlierBits, outlierCount int
	asymmetric                bool
	tol                       float32
	label                     string
	nWarmup                   int
}

func tqRunPackedFA(t *testing.T, cfg tqPackedFAConfig) {
	t.Helper()
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	rotationSeed := turboquant.PresetTQ3.RotationSeed ^ uint64(cfg.headDim)

	mgrAny := b.NewTQCompressedKManager(
		cfg.headDim, cfg.nKVHeads, cfg.bits,
		rotationSeed,
		cfg.vBits, // K+V mode
		cfg.outlierBits, cfg.outlierCount,
		cfg.asymmetric,
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil (GPU unavailable or unsupported)")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, cfg.nCells)
	mgr.EnsureVLayer(0, cfg.nCells)

	if !mgr.fusedKernelSupports() {
		t.Skipf("fused kernel not supported for headDim=%d bits=%d", cfg.headDim, cfg.bits)
	}

	var rngState uint64 = 0xfab1_c400_0000_0000 | uint64(cfg.headDim)

	kData := make([]float32, cfg.headDim*cfg.nKVHeads*cfg.nCells)
	for i := range kData {
		kData[i] = float32(tqGaussian(&rngState))
	}
	for c := range cfg.nCells {
		for h := range cfg.nKVHeads {
			nSpikes := cfg.outlierCount / 4
			for range nSpikes {
				d := int(tqSplitmix64(&rngState) % uint64(cfg.headDim))
				kData[(c*cfg.nKVHeads+h)*cfg.headDim+d] *= 5.0
			}
		}
	}

	layerSeed := rotationSeed ^ uint64(1)
	rotation := turboquant.BuildRotation(cfg.headDim, layerSeed)

	qRaw := make([]float32, cfg.headDim*cfg.nKVHeads)
	for i := range qRaw {
		qRaw[i] = float32(tqGaussian(&rngState)) * 0.5
	}
	qRotated := make([]float32, cfg.headDim*cfg.nKVHeads)
	for h := range cfg.nKVHeads {
		rotQ := turboquant.ApplyRotation(qRaw[h*cfg.headDim:(h+1)*cfg.headDim], rotation)
		copy(qRotated[h*cfg.headDim:], rotQ)
	}

	vF32 := make([]float32, cfg.headDim*cfg.nCells*cfg.nKVHeads)
	for i := range vF32 {
		vF32[i] = float32(tqGaussian(&rngState)) * 0.25
	}

	// EncodeV expects [headDim, nKVHeads, nCells] so it stores slot c*nKVHeads+h,
	// matching what the FA kernel reads. vF32 is in [headDim, nCells, nKVHeads]
	// (same layout the CPU reference uses), so transpose (nCells, nKVHeads) → (nKVHeads, nCells).
	vF32ForEncode := make([]float32, len(vF32))
	for h := range cfg.nKVHeads {
		for c := range cfg.nCells {
			for d := range cfg.headDim {
				src := d + c*cfg.headDim + h*cfg.headDim*cfg.nCells
				dst := d + h*cfg.headDim + c*cfg.headDim*cfg.nKVHeads
				vF32ForEncode[dst] = vF32[src]
			}
		}
	}

	kTensor := ctx.FromFloats(kData, cfg.headDim, cfg.nKVHeads, cfg.nCells)
	qTensor := ctx.FromFloats(qRotated, cfg.headDim, 1, cfg.nKVHeads)
	vTensorF32 := ctx.FromFloats(vF32ForEncode, cfg.headDim, cfg.nKVHeads, cfg.nCells)

	encK := mgr.EncodeK(ctx, 0, kTensor, 0)
	if encK == nil {
		t.Fatal("EncodeK returned nil")
	}
	encV := mgr.EncodeV(ctx, 0, vTensorF32, 0)
	if encV == nil {
		t.Fatal("EncodeV returned nil — vBits may be 0 or EnsureVLayer not called")
	}

	// GetAsTQTensorKV wraps K+V packed tensors; tqFlashAttention dispatches to
	// packed_outlier_* kernel (v_packed=true path in ggml-metal-ops.cpp).
	tqkvRaw, ok := mgr.GetAsTQTensorKV(ctx, 0, encK, encV, 0, cfg.nCells)
	if !ok || tqkvRaw == nil {
		t.Skip("GetAsTQTensorKV returned (nil, false) — K+V fused FA not supported on this device")
	}
	tqkv := tqkvRaw.(*tqTensor)
	if tqkv.vPacked == nil {
		t.Fatal("tqTensor.vPacked is nil after GetAsTQTensorKV")
	}

	// The K+V packed dispatch asserts mask != nullptr to derive nCells from
	// mask->ne[0]. Provide an all-zero (no masking) f16 mask [nCells, 1].
	maskF32 := make([]float32, cfg.nCells)
	maskF16Bytes := tqF32SliceToF16Bytes(maskF32)
	maskTensor := ctx.FromBytes(ml.DTypeF16, maskF16Bytes, cfg.nCells, 1)

	attnScale := 1.0 / math.Sqrt(float64(cfg.headDim))
	// Pass tqkv.vPacked as the value tensor — matches the production K+V fused path.
	attnOut := b.tqFlashAttention(ctx, qTensor.(*Tensor), tqkv, tqkv.vPacked, maskTensor, attnScale, 0)
	if attnOut == nil {
		t.Fatal("tqFlashAttention returned nil")
	}

	// Undo WHT rotation — matches production path (ggml.go TQApplyWHT after FA).
	// FA kernel outputs in WHT-rotated V space; undo gives back original V space.
	var attnFinal ml.Tensor = attnOut
	if tqkv.signs != nil {
		attnFinal = attnOut.(*Tensor).TQApplyWHT(ctx, tqkv.signs)
	}

	// BGO warmup: D64 packed-outlier kernels hit a Metal fast-tier JIT LICM bug
	// when nKVHeads>1. Dispatching ~25+ times triggers BGO (the correct optimizer).
	// Each test run creates a fresh pipeline state, so warmup is needed every run.
	for range cfg.nWarmup {
		ctx.Forward(attnFinal).Compute(attnFinal)
	}
	if cfg.nWarmup > 0 {
		// BGO compiles asynchronously after sufficient dispatches. 5s gives ample margin.
		time.Sleep(5 * time.Second)
	}

	// Include attnOut in Compute so its sync callback is set (needed for Floats() on intermediate node).
	ctx.Forward(encK, encV, attnFinal).Compute(encK, encV, attnOut, attnFinal)
	_ = maskTensor // used as input to attnOut graph node

	// Diagnostic: log pre-WHT values for all heads.
	var preWHT []float32
	if tqkv.signs != nil {
		preWHT = attnOut.Floats()
		for h := range cfg.nKVHeads {
			off := h * cfg.headDim
			t.Logf("%s diag h=%d d=0..2 pre=%.4f,%.4f,%.4f", cfg.label, h, preWHT[off], preWHT[off+1], preWHT[off+2])
		}
	}

	// Diagnostic: V packed bytes for h=1 cells 0 and 1.
	if cfg.nKVHeads >= 2 {
		vPackedBytesPerSlot := cfg.headDim * cfg.vBits / 8
		if vpb := mgr.vPackedTensors[0].BackendGetBytes(); len(vpb) >= 6*vPackedBytesPerSlot {
			// slot = c*nKVHeads+h; for h=1,c=0 slot=1; for h=1,c=1 slot=5
			s1 := vpb[1*vPackedBytesPerSlot : 1*vPackedBytesPerSlot+8]
			s5 := vpb[5*vPackedBytesPerSlot : 5*vPackedBytesPerSlot+8]
			t.Logf("%s vpacked h1c0[0..7]=%v h1c1[0..7]=%v", cfg.label, s1, s5)
		}
	}

	gpuOut := attnFinal.Floats()
	for i, v := range gpuOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s: gpuOut[%d]=%v (NaN/Inf)", cfg.label, i, v)
		}
	}

	// Diagnostic: V scales — GPU done after above Floats() call.
	// vScalesTensors[0] shape = [nKVHeads, capacity] f32; element [h,c] at h+c*nKVHeads.
	if vScaleBytes := mgr.vScalesTensors[0].BackendGetBytes(); len(vScaleBytes) >= cfg.nKVHeads*cfg.nCells*4 {
		vScaleF32 := make([]float32, len(vScaleBytes)/4)
		for i := range vScaleF32 {
			bits := uint32(vScaleBytes[4*i]) | uint32(vScaleBytes[4*i+1])<<8 | uint32(vScaleBytes[4*i+2])<<16 | uint32(vScaleBytes[4*i+3])<<24
			vScaleF32[i] = math.Float32frombits(bits)
		}
		h0c0 := vScaleF32[0+0*cfg.nKVHeads]
		h0c1 := vScaleF32[0+1*cfg.nKVHeads]
		h1c0 := vScaleF32[1+0*cfg.nKVHeads]
		h1c1 := vScaleF32[1+1*cfg.nKVHeads]
		t.Logf("%s vscales h0c0=%.6f h0c1=%.6f h1c0=%.6f h1c1=%.6f",
			cfg.label, h0c0, h0c1, h1c0, h1c1)
	}

	// Diagnostic: K main scales — check if they vary between PASS/FAIL.
	// scalesTensors[0] shape = [nKVHeads, capacity] f32; element [h,c] at h+c*nKVHeads.
	if cfg.nKVHeads >= 2 {
		if ksb := mgr.scalesTensors[0].BackendGetBytes(); len(ksb) >= cfg.nKVHeads*cfg.nCells*4 {
			kScaleF32 := make([]float32, len(ksb)/4)
			for i := range kScaleF32 {
				b := uint32(ksb[4*i]) | uint32(ksb[4*i+1])<<8 | uint32(ksb[4*i+2])<<16 | uint32(ksb[4*i+3])<<24
				kScaleF32[i] = math.Float32frombits(b)
			}
			kh1 := make([]float32, cfg.nCells)
			for c := range cfg.nCells {
				kh1[c] = kScaleF32[1+c*cfg.nKVHeads]
			}
			t.Logf("%s kscales h1[c0..c7]=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
				cfg.label, kh1[0], kh1[1], kh1[2], kh1[3], kh1[4], kh1[5], kh1[6], kh1[7])
		}
	}

	// Diagnostic: K outlier packed codes — check if they vary between PASS/FAIL.
	if mgr.outlierPackedTensors[0] != nil {
		if ob := mgr.outlierPackedTensors[0].BackendGetBytes(); len(ob) >= 16 {
			t.Logf("%s outl_packed[0..15]=%v", cfg.label, ob[0:16])
		}
	}
	// Diagnostic: K outlier scales — check if they vary between PASS/FAIL.
	// outlierScalesTensors[0] shape = [nKVHeads, capacity] f32.
	if mgr.outlierScalesTensors[0] != nil {
		if osb := mgr.outlierScalesTensors[0].BackendGetBytes(); len(osb) >= cfg.nKVHeads*cfg.nCells*4 {
			oScaleF32 := make([]float32, len(osb)/4)
			for i := range oScaleF32 {
				bits := uint32(osb[4*i]) | uint32(osb[4*i+1])<<8 | uint32(osb[4*i+2])<<16 | uint32(osb[4*i+3])<<24
				oScaleF32[i] = math.Float32frombits(bits)
			}
			t.Logf("%s outl_scales h1c0=%.6f h1c1=%.6f", cfg.label, oScaleF32[1+0*cfg.nKVHeads], oScaleF32[1+1*cfg.nKVHeads])
		}
	}

	cpuPreset := turboquant.Preset{
		RotationSeed:      layerSeed,
		KeyPrimaryBits:    cfg.bits,
		OutlierBits:       cfg.outlierBits,
		OutlierCount:      cfg.outlierCount,
		AsymmetricPrimary: cfg.asymmetric,
	}
	kDecoded := tqCPUKRef(t, kData, cfg.nCells, cfg.nKVHeads, cfg.headDim, cpuPreset)

	// Diagnostic: compare GPU pre-WHT vs expected pre-WHT (CPU attn output rotated back).
	// If they match, the FA accumulation is correct and the issue is in WHT undo.
	// If they differ, the issue is in the FA kernel (KQ or V decode).
	if preWHT != nil && len(preWHT) >= cfg.headDim*cfg.nKVHeads {
		for h := range cfg.nKVHeads {
			kvHead := h
			qH := qRotated[h*cfg.headDim : (h+1)*cfg.headDim]
			scores := make([]float64, cfg.nCells)
			for c := range cfg.nCells {
				kH := kDecoded[c*cfg.nKVHeads+kvHead]
				var dot float64
				for d := range cfg.headDim {
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
			for c := range cfg.nCells {
				scores[c] = math.Exp(scores[c] - maxS)
				sumExp += scores[c]
			}
			for c := range cfg.nCells {
				scores[c] /= sumExp
			}
			cpuRef := make([]float32, cfg.headDim)
			for c := range cfg.nCells {
				vBase := c*cfg.headDim + kvHead*cfg.headDim*cfg.nCells
				for d := range cfg.headDim {
					cpuRef[d] += float32(scores[c]) * vF32[vBase+d]
				}
			}
			expectedPre := turboquant.ApplyRotation(cpuRef, rotation)
			gpuPre := preWHT[h*cfg.headDim : (h+1)*cfg.headDim]
			maxPreDiff := float32(0)
			for d := range cfg.headDim {
				diff := gpuPre[d] - expectedPre[d]
				if diff < 0 {
					diff = -diff
				}
				if diff > maxPreDiff {
					maxPreDiff = diff
				}
			}
			scoreStr := ""
			for c := range cfg.nCells {
				if c > 0 {
					scoreStr += ","
				}
				scoreStr += fmt.Sprintf("%.3f", scores[c])
			}
			t.Logf("%s pre-WHT h=%d: gpu[0..2]=%.4f,%.4f,%.4f exp[0..2]=%.4f,%.4f,%.4f maxDiff=%.4f softmax=[%s]",
				cfg.label, h,
				gpuPre[0], gpuPre[1], gpuPre[2],
				expectedPre[0], expectedPre[1], expectedPre[2],
				maxPreDiff, scoreStr)

			// Hypothesis: GPU uses V[c=0] for all cells. Compute expected pre-WHT
			// as if only cell 0 contributed (unquantized), then compare to GPU.
			c0Ref := make([]float32, cfg.headDim)
			vBase0 := 0*cfg.headDim + kvHead*cfg.headDim*cfg.nCells
			for d := range cfg.headDim {
				c0Ref[d] = vF32[vBase0+d]
			}
			c0Pre := turboquant.ApplyRotation(c0Ref, rotation)
			maxC0Diff := float32(0)
			for d := range cfg.headDim {
				diff := gpuPre[d] - c0Pre[d]
				if diff < 0 {
					diff = -diff
				}
				if diff > maxC0Diff {
					maxC0Diff = diff
				}
			}
			t.Logf("%s hyp-c0 h=%d: c0pre[0..2]=%.4f,%.4f,%.4f maxDiffVsGPU=%.4f",
				cfg.label, h, c0Pre[0], c0Pre[1], c0Pre[2], maxC0Diff)
		}
	}

	// Compare against raw (unrotated) vF32 — WHT undo above brings GPU output back
	// to original V space, so the only difference is quantization noise.
	maxDiff, nMismatches := tqCheckAttn(t, gpuOut, qRotated, kDecoded, vF32,
		1, cfg.nKVHeads, cfg.nKVHeads, cfg.nCells, cfg.headDim, attnScale, cfg.tol,
		cfg.label)
	t.Logf("%s: max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
		cfg.label, maxDiff, nMismatches, cfg.headDim*cfg.nKVHeads, cfg.tol)
	if nMismatches > 0 {
		t.Fatalf("%s: %d mismatches beyond tol=%.2f", cfg.label, nMismatches, cfg.tol)
	}
}

// TestTQFusedFlashAttentionPackedOutlierD64 — kernel_tq_fattn_vec_packed_outlier_d64
// (llama3.2:3b head_dim=64, K+V preset on Metal).
func TestTQFusedFlashAttentionPackedOutlierD64(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 64, nKVHeads: 4, nCells: 8,
		bits: 3, vBits: 3, outlierBits: 4, outlierCount: 16,
		tol: 0.55, label: "packed_outlier_d64",
		// 64 warmup dispatches exceed the Metal BGO threshold (~25 needed).
		// Each test run creates a fresh pipeline state, so warmup runs every time.
		nWarmup: 64,
	})
}

// TestTQFusedFlashAttentionPackedOutlierD128 — kernel_tq_fattn_vec_packed_outlier
// (qwen2.5/llama3.1 head_dim=128, K+V preset on Metal).
func TestTQFusedFlashAttentionPackedOutlierD128(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 128, nKVHeads: 8, nCells: 8,
		bits: 3, vBits: 3, outlierBits: 4, outlierCount: 32,
		tol: 0.65, label: "packed_outlier_d128",
	})
}

// TestTQFusedFlashAttentionPackedOutlierD256 — kernel_tq_fattn_vec_packed_outlier_d256
// (gemma3 head_dim=256, K+V preset on Metal).
func TestTQFusedFlashAttentionPackedOutlierD256(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 256, nKVHeads: 4, nCells: 8,
		bits: 3, vBits: 3, outlierBits: 4, outlierCount: 32,
		tol: 0.85, label: "packed_outlier_d256",
	})
}

// TestTQFusedFlashAttentionPackedOutlierD512 — kernel_tq_fattn_vec_packed_outlier_d512
// (gemma4 global attention head_dim=512, K+V preset on Metal).
func TestTQFusedFlashAttentionPackedOutlierD512(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 512, nKVHeads: 4, nCells: 8,
		bits: 3, vBits: 3, outlierBits: 4, outlierCount: 32,
		tol: 0.65, label: "packed_outlier_d512",
	})
}

func TestTQFusedFlashAttentionPackedOutlierD64_1Head(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 64, nKVHeads: 1, nCells: 8,
		bits: 3, vBits: 3, outlierBits: 4, outlierCount: 16,
		tol: 0.5, label: "packed_outlier_d64_1h",
	})
}

func TestTQFusedFlashAttentionPackedOutlierD64_2Head(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 64, nKVHeads: 2, nCells: 8,
		bits: 3, vBits: 3, outlierBits: 4, outlierCount: 16,
		tol: 0.5, label: "packed_outlier_d64_2h",
		nWarmup: 64,
	})
}

// Symmetric 4-bit test — bits=4, outlierBits=5 (same as tq4 but symmetric).
func TestTQFusedFlashAttentionPackedOutlierD128Symmetric4bit(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 128, nKVHeads: 8, nCells: 8,
		bits: 4, vBits: 4, outlierBits: 5, outlierCount: 32,
		asymmetric: false,
		tol: 0.5, label: "packed_outlier_d128_sym_4bit",
	})
}

// Asymmetric K+V fused tests — tq2/tq3/tq4 production presets (asymmetricPrimary=true).

func TestTQFusedFlashAttentionPackedOutlierD128Asymmetric2bit(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 128, nKVHeads: 8, nCells: 8,
		bits: 2, vBits: 2, outlierBits: 3, outlierCount: 32,
		asymmetric: true,
		tol: 0.8, label: "packed_outlier_d128_asym_tq2",
	})
}

func TestTQFusedFlashAttentionPackedOutlierD128Asymmetric3bit(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 128, nKVHeads: 8, nCells: 8,
		bits: 3, vBits: 3, outlierBits: 4, outlierCount: 32,
		asymmetric: true,
		tol: 0.65, label: "packed_outlier_d128_asym_tq3",
	})
}

func TestTQFusedFlashAttentionPackedOutlierD128Asymmetric4bit(t *testing.T) {
	tqRunPackedFA(t, tqPackedFAConfig{
		headDim: 128, nKVHeads: 8, nCells: 8,
		bits: 4, vBits: 4, outlierBits: 5, outlierCount: 32,
		asymmetric: true,
		tol: 0.5, label: "packed_outlier_d128_asym_tq4",
	})
}
