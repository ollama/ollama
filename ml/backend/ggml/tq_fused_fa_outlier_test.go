package ggml

// TestTQFusedFlashAttentionOutlier verifies kernel_tq_fattn_vec_f16_outlier
// on the first available TQ-capable GPU (Metal on Mac, CUDA/ROCm on Linux).
//
// Kernel routing: the outlier variant is selected when outlier_count > 0 and
// v_packed == false (K-only preset). This test uses asymmetricPrimary=false
// so the CPU reference encode (EncodeKeyPerHeadOutlier) matches the GPU
// encode up to float32 vs float64 accumulation noise.
//
// Single-Compute contract: the whole graph (EncodeK → fused FA) is built
// first, then Compute() is called exactly once. Multiple Compute() calls each
// invoke ggml_backend_sched_reset() internally, invalidating buffer assignments
// for every tensor touched in earlier passes.

import (
	"bytes"
	"math"
	"math/rand"
	"os"
	"testing"

	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

// setupTQAny finds the first TQ-capable GPU (Metal, CUDA Pascal+, or ROCm
// RDNA1+) and returns a context backed by a one-layer synthetic GGUF assigned
// to that device. Skips the test if no suitable device is found.
func setupTQAny(tb testing.TB) (ml.Context, ml.Backend) {
	tb.Helper()

	initDevices()
	gpus := GPUDevices()
	if len(gpus) == 0 {
		tb.Skip("no GPU devices visible; skipping TQ fused-FA outlier test")
	}

	var target GPUDeviceInfo
	for _, g := range gpus {
		if accepted, _ := tqDeviceAccepted(g.Library, g.CCMajor); accepted {
			target = g
			break
		}
	}
	if target.Name == "" {
		tb.Skip("no TQ-capable GPU (need Metal, NVIDIA Pascal cc6+, or AMD RDNA1+)")
	}
	tb.Logf("using GPU: %s (%s cc %d.%d)", target.Name, target.Library, target.CCMajor, target.CCMinor)

	kv := fsggml.KV{
		"general.architecture": "test",
		"block_count":          uint32(1),
	}
	seed := make([]byte, 64)
	rand.New(rand.NewSource(42)).Read(seed)
	tensors := []*fsggml.Tensor{
		{
			Name:     "blk.0.attn_q.weight",
			Kind:     uint32(fsggml.TensorTypeF32),
			Shape:    []uint64{4, 4},
			WriterTo: bytes.NewReader(seed),
		},
	}

	f, err := os.CreateTemp(tb.TempDir(), "*.bin")
	if err != nil {
		tb.Fatalf("CreateTemp: %v", err)
	}
	defer f.Close()
	if err := fsggml.WriteGGUF(f, kv, tensors); err != nil {
		tb.Fatalf("WriteGGUF: %v", err)
	}

	backend, err := ml.NewBackend(f.Name(), ml.BackendParams{
		AllocMemory: true,
		GPULayers: ml.GPULayersList{
			{
				DeviceID: ml.DeviceID{ID: target.ID, Library: target.Library},
				Layers:   []int{0},
			},
		},
	})
	if err != nil {
		tb.Fatalf("NewBackend: %v", err)
	}

	ctx := backend.NewContext().Input()
	tb.Cleanup(func() {
		ctx.Close()
		backend.Close()
	})
	return ctx, backend
}

func TestTQFusedFlashAttentionOutlier(t *testing.T) {
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 128
		nKVHeads     = 8
		nCells       = 16
		bits         = 3
		outlierBits  = 4
		outlierCount = 32
	)

	// Use the production rotation seed from PresetTQ3K (K-only, 3-bit).
	// asymmetricPrimary=false so EncodeKeyPerHeadOutlier is bit-comparable.
	rotationSeed := turboquant.PresetTQ3K.RotationSeed

	mgrAny := b.NewTQCompressedKManager(
		headDim, nKVHeads, bits,
		rotationSeed,
		0, // vBits = 0 (K-only — avoids v_packed, routes to outlier FA)
		outlierBits, outlierCount,
		false, // asymmetricPrimary — symmetric for clean CPU reference
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil (GPU not available or preset unsupported)")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, nCells)

	if !mgr.hasOutliers() {
		t.Fatalf("hasOutliers()=false for outlierCount=%d — check manager wiring", outlierCount)
	}
	if !mgr.fusedKernelSupports() {
		t.Skipf("fused kernel not supported for headDim=%d bits=%d", headDim, bits)
	}

	// ── Synthetic data ──────────────────────────────────────────────────────

	var rngState uint64 = 0xcafe_babe_0000_0000 | uint64(headDim)

	// K: [headDim, nKVHeads, nCells] f32, Gaussian + heavy-tail outliers to
	// exercise the outlier-split quantizer path in the encode kernel.
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

	// Q: [headDim, nKVHeads] f32, WHT-rotated before upload.
	// The kernel expects Q already in WHT-rotated space (in production this is
	// done by TQApplyWHT). Layer 0 uses rotationSeed ^ 1.
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

	// V: [headDim, nCells, nKVHeads] f16.
	// Metal kernel reads: nCells=V->ne[1], nKVHeads=V->ne[2].
	// We generate f32 and convert to f16 for upload.
	vF32 := make([]float32, headDim*nCells*nKVHeads)
	for i := range vF32 {
		vF32[i] = float32(tqGaussian(&rngState)) * 0.25
	}
	vF16Bytes := tqF32SliceToF16Bytes(vF32)

	// ── GPU graph (single Compute) ───────────────────────────────────────────

	// kTensor: ne[0]=headDim, ne[1]=nKVHeads, ne[2]=nCells
	kTensor := ctx.FromFloats(kData, headDim, nKVHeads, nCells)
	// qTensor: ne[0]=headDim, ne[1]=1 (one query token), ne[2]=nKVHeads
	qTensor := ctx.FromFloats(qRotated, headDim, 1, nKVHeads)
	// vTensor: ne[0]=headDim, ne[1]=nCells, ne[2]=nKVHeads (f16)
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

	// Confirm that the outlier buffers are wired: outlier_count > 0 in the
	// kernel args triggers the outlier decode path in ggml_metal_op_tq_fattn.
	if tqk.outlierPacked == nil {
		t.Fatalf("tqTensor.outlierPacked is nil — outlier encode path not wired")
	}
	if tqk.outlierCount != outlierCount {
		t.Fatalf("tqTensor.outlierCount = %d, want %d", tqk.outlierCount, outlierCount)
	}

	attnScale := 1.0 / math.Sqrt(float64(headDim))
	attnOut := b.tqFlashAttention(
		ctx,
		qTensor.(*Tensor),
		tqk,
		vTensor.(*Tensor),
		nil, // no causal mask for single-token query over full cache
		attnScale, 0,
	)
	if attnOut == nil {
		t.Fatalf("tqFlashAttention returned nil")
	}

	// Single Compute() — enc and attnOut share the encode subgraph so both
	// compute correctly in one pass without buffer reassignment between passes.
	ctx.Forward(enc, attnOut).Compute(enc, attnOut)

	gpuOut := attnOut.Floats()
	if len(gpuOut) != headDim*nKVHeads {
		t.Fatalf("gpu output len = %d, want %d", len(gpuOut), headDim*nKVHeads)
	}

	// ── Smoke: no NaN / Inf ─────────────────────────────────────────────────
	for i, v := range gpuOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("gpuOut[%d] = %v — outlier kernel reading uninitialised memory or has wrong buffer layout", i, v)
		}
	}

	// ── Fidelity: CPU softmax(scale·Q·K^T)·V ────────────────────────────────
	//
	// CPU K comes from EncodeKeyPerHeadOutlier + DequantKeyPerHeadOutlier,
	// using the same layerSeed so the rotated-space K matches what the GPU
	// encoder stores. The preset is symmetric (no asymmetric centering) to
	// match the asymmetricPrimary=false manager above.
	//
	// NOTE: the GPU EDEN scale-refinement step (two-pass MSE-optimal scale)
	// runs on GPU but NOT in EncodeKeyPerHeadOutlier, producing a systematic
	// scale gap. tolerance=0.50 accommodates this plus f16 rounding.
	cpuPreset := turboquant.Preset{
		ID: 200, Name: "test_outlier_sym",
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

	const tol float32 = 0.50
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

		// V layout: ne[0]=headDim, ne[1]=nCells, ne[2]=nKVHeads
		// flat index v[d,c,h] = d + c*headDim + h*headDim*nCells
		cpuOut := make([]float64, headDim)
		for c := range nCells {
			vBase := c*headDim + h*headDim*nCells
			for d := range headDim {
				cpuOut[d] += scores[c] * float64(vF32[vBase+d])
			}
		}

		// GPU output layout: ne[0]=headDim, ne[1]=1, ne[2]=nKVHeads
		// flat index out[d,0,h] = d + h*headDim
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

	t.Logf("kernel_tq_fattn_vec_f16_outlier: max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
		maxDiff, nMismatches, headDim*nKVHeads, tol)

	if nMismatches > 0 {
		t.Fatalf("%d elements differ beyond tolerance %.2f — outlier inline-decode path likely broken", nMismatches, tol)
	}
}

// TestTQFusedFlashAttentionOutlierAsymmetric verifies kernel_tq_fattn_vec_f16_outlier
// with asymmetric=true (the production PresetTQ3 configuration).
//
// The CPU reference K comes from the GPU's own DequantK output (cast f16→f32
// in the same Compute() call), so the comparison is insensitive to the
// asymmetric mean-centering mismatch that would occur if EncodeKeyPerHeadOutlier
// were used as the CPU-side encoder. This isolates exactly one question:
//
//	Does the fused FA kernel's inline K decode produce the same K that the
//	standalone DequantK kernel produces?
//
// If the asymmetric zero correction is missing or wrong in the fused kernel,
// every attention score will be shifted by Q·mean(K)/√D, causing the softmax
// to distribute weight incorrectly. The resulting output error will be much
// larger than the f16 accumulation noise that is the only other error source.
//
// Graph layout (single Compute()):
//
//	kTensor ──► EncodeK ──┬──► DequantK(f16) ──► Cast(f32) ─► Floats() [CPU ref]
//	                      └──► GetAsTQTensor ──► FusedFA ────► Floats() [GPU out]
func TestTQFusedFlashAttentionOutlierAsymmetric(t *testing.T) {
	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	const (
		headDim      = 128
		nKVHeads     = 8
		nCells       = 16
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
		true, // asymmetricPrimary = true — this is what production tq3 uses
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, nCells)

	if !mgr.fusedKernelSupports() {
		t.Skipf("fused kernel not supported for headDim=%d bits=%d", headDim, bits)
	}

	// ── Synthetic data ──────────────────────────────────────────────────────
	// Use per-channel biases to create a non-zero mean per channel in rotated
	// space — the exact condition that asymmetric mean centering is designed to
	// handle, and that produces the largest possible error if the zero
	// correction is missing in the fused kernel.

	var rngState uint64 = 0xdead_c0de_0000_0000 | uint64(headDim)

	// Per-channel bias (constant across heads and cells).
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
	// Outlier injection.
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

	// ── GPU graph (single Compute) ───────────────────────────────────────────

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
		t.Fatalf("tqTensor.asymmetric=false even though manager has asymmetricPrimary=true")
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
		t.Fatalf("attnOut len = %d, want %d", len(gpuOut), headDim*nKVHeads)
	}

	// ── Smoke: no NaN / Inf ─────────────────────────────────────────────────
	for i, v := range gpuOut {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("gpuOut[%d] = %v (NaN/Inf) — asymmetric outlier kernel reading uninitialised memory", i, v)
		}
	}

	// ── CPU softmax(scale·Q·K^T)·V using independent CPU encode+decode ──────
	//
	// Build kRef via the asymmetric-aware CPU reference encoder rather than
	// re-using a GPU dequant in the same compute graph — kRef and attnOut
	// would otherwise alias under ggml's scheduler. See feedback memory
	// `feedback_ggml_test_kref_aliasing`.
	cpuPreset := turboquant.Preset{
		ID: 201, Name: "test_outlier_asym",
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

	// Tolerance 0.20: asymmetric centring + EDEN refinement on CPU narrows
	// the gap to the GPU kernel, but f16-quantized means and shfl-reduction
	// reorder still account for noticeable noise above the symmetric path's
	// ~0.05 floor. If the zero correction is MISSING in the kernel, the
	// score shift Q·zeros/√D ≈ 2–5 produces output errors 10–50× this
	// threshold — that's the failure mode this test is here to catch.
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

	t.Logf("kernel_tq_fattn_vec_f16_outlier (asymmetric): max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
		maxDiff, nMismatches, headDim*nKVHeads, tol)

	if nMismatches > 0 {
		t.Fatalf("%d elements differ beyond tolerance %.2f — if errors are 2–5× the expected score shift, zero correction is missing; if ~0.20, tolerance may need widening further for this hardware's f16 noise floor", nMismatches, tol)
	}
}

// tqGaussian generates a standard-normal sample via Box-Muller. Uses tqSplitmix64
// for uniform samples. Distinct from testGaussian (same algorithm, separate state).
func tqGaussian(state *uint64) float64 {
	u1 := tqUniform(state)
	u2 := tqUniform(state)
	if u1 < 1e-12 {
		u1 = 1e-12
	}
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

func tqUniform(state *uint64) float64 {
	return float64(tqSplitmix64(state)>>11) / float64(1<<53)
}

// tqSplitmix64 advances the splitmix64 state and returns the mixed output.
func tqSplitmix64(state *uint64) uint64 {
	*state += 0x9e3779b97f4a7c15
	z := *state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// tqF32SliceToF16Bytes converts []float32 to IEEE 754 fp16 bytes (little-endian),
// suitable for ctx.FromBytes(ml.DTypeF16, ...).
func tqF32SliceToF16Bytes(fs []float32) []byte {
	out := make([]byte, len(fs)*2)
	for i, f := range fs {
		h := tqFloat32ToF16(f)
		out[i*2] = byte(h)
		out[i*2+1] = byte(h >> 8)
	}
	return out
}

// tqFloat32ToF16 converts a float32 to its IEEE 754 fp16 bit pattern.
func tqFloat32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 31) & 1)
	exp32 := int32((b>>23)&0xff) - 127
	frac32 := b & 0x7fffff
	if (b>>23)&0xff == 0xff { // NaN or Inf
		return sign<<15 | 0x7c00 | uint16(frac32>>13)
	}
	exp16 := exp32 + 15
	if exp16 <= 0 {
		return sign << 15
	}
	if exp16 >= 31 {
		return sign<<15 | 0x7c00
	}
	return sign<<15 | uint16(exp16)<<10 | uint16(frac32>>13)
}
