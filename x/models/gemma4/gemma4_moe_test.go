package gemma4

import (
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func useMLXTestThread(t *testing.T) {
	t.Helper()

	runtime.LockOSThread()
	initialized := false
	t.Cleanup(func() {
		if initialized {
			mlx.Sweep()
			mlx.ClearCache()
			if mlx.GPUIsAvailable() {
				mlx.SetDefaultDeviceGPU()
			}
		}
		runtime.UnlockOSThread()
	})

	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	initialized = true
	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
	}
}

// onesLike creates a tensor of the given shape filled with a small constant.
func onesLike(shape ...int) *mlx.Array {
	return mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, shape...), 0.01)
}

func tinyMoEConfig() *TextConfig {
	return &TextConfig{
		HiddenSize:             16, // tiny for testing
		NumAttentionHeads:      2,
		NumKeyValueHeads:       1,
		NumGlobalKeyValueHeads: 1,
		HeadDim:                8,
		GlobalHeadDim:          8,
		NumExperts:             4,
		TopKExperts:            2,
		ExpertIntermediateSize: 8,
		EnableMoeBlock:         true,
		AttentionKEqV:          false,
		RMSNormEps:             1e-6,
		SlidingScale:           1.0,
		FullScale:              1.0,
	}
}

func newRouter(cfg *TextConfig) *Router {
	return &Router{
		Proj:  linearFromWeight(onesLike(int(cfg.NumExperts), int(cfg.HiddenSize))),
		Scale: onesLike(int(cfg.HiddenSize)),
	}
}

func newMoEBlock(cfg *TextConfig) *MoEBlock {
	return &MoEBlock{
		GateWeight:     onesLike(int(cfg.NumExperts), int(cfg.HiddenSize), int(cfg.ExpertIntermediateSize)),
		UpWeight:       onesLike(int(cfg.NumExperts), int(cfg.HiddenSize), int(cfg.ExpertIntermediateSize)),
		DownWeight:     onesLike(int(cfg.NumExperts), int(cfg.ExpertIntermediateSize), int(cfg.HiddenSize)),
		PerExpertScale: onesLike(int(cfg.NumExperts)),
	}
}

func TestMoERouterForward(t *testing.T) {
	useMLXTestThread(t)

	cfg := tinyMoEConfig()
	B, L := int32(1), int32(3)
	x := onesLike(int(B), int(L), int(cfg.HiddenSize))
	router := newRouter(cfg)

	scores, inds := router.Forward(x, cfg)
	mlx.Eval(scores, inds)

	sDims := scores.Dims()
	iDims := inds.Dims()
	t.Logf("scores shape: %v, inds shape: %v", sDims, iDims)

	if len(sDims) != 2 || sDims[0] != int(B*L) || sDims[1] != int(cfg.TopKExperts) {
		t.Errorf("scores shape = %v, want [%d, %d]", sDims, B*L, cfg.TopKExperts)
	}
	if len(iDims) != 2 || iDims[0] != int(B*L) || iDims[1] != int(cfg.TopKExperts) {
		t.Errorf("inds shape = %v, want [%d, %d]", iDims, B*L, cfg.TopKExperts)
	}
}

func TestMoEBlockForward(t *testing.T) {
	useMLXTestThread(t)

	cfg := tinyMoEConfig()
	B, L := int32(1), int32(3)
	x := onesLike(int(B), int(L), int(cfg.HiddenSize))
	router := newRouter(cfg)
	moe := newMoEBlock(cfg)

	scores, inds := router.Forward(x, cfg)
	mlx.Eval(scores, inds)

	out := moe.Forward(x, scores, inds, cfg)
	mlx.Eval(out)

	outDims := out.Dims()
	t.Logf("MoE output shape: %v", outDims)

	if len(outDims) != 3 || outDims[0] != int(B) || outDims[1] != int(L) || outDims[2] != int(cfg.HiddenSize) {
		t.Errorf("output shape = %v, want [%d, %d, %d]", outDims, B, L, cfg.HiddenSize)
	}
}

func TestMoEBlockSortedForward(t *testing.T) {
	useMLXTestThread(t)

	cfg := tinyMoEConfig()
	B, L := int32(1), int32(128)
	x := onesLike(int(B), int(L), int(cfg.HiddenSize))
	router := newRouter(cfg)
	moe := newMoEBlock(cfg)

	scores, inds := router.Forward(x, cfg)
	mlx.Eval(scores, inds)

	out := moe.Forward(x, scores, inds, cfg)
	mlx.Eval(out)

	outDims := out.Dims()
	t.Logf("MoE sorted output shape: %v", outDims)

	if len(outDims) != 3 || outDims[0] != int(B) || outDims[1] != int(L) || outDims[2] != int(cfg.HiddenSize) {
		t.Errorf("output shape = %v, want [%d, %d, %d]", outDims, B, L, cfg.HiddenSize)
	}
}

// TestLoadFusedExpertsQuantized verifies that a quantized, fused gate_up
// projection is loaded onto the GatherQMM path under every name the experts
// ship as — including gemma's bare ".experts." name, which previously fell
// through to the dense branch and was loaded unquantized (the memory bloat
// bug this fix addresses).
func TestLoadFusedExpertsQuantized(t *testing.T) {
	skipIfNoMLX(t)

	const E, I, H = 4, 8, 16
	m := &Model{TextConfig: &TextConfig{QuantGroupSize: 16, QuantBits: 4, QuantMode: "nvfp4"}}

	for _, prefix := range []string{
		"model.language_model.layers.0.experts",        // gemma HF (bare .experts.)
		"model.language_model.layers.0.moe.switch_mlp", // create pipeline
	} {
		t.Run(prefix, func(t *testing.T) {
			gateUpKey := prefix + ".gate_up_proj"
			downKey := prefix + ".down_proj"
			tensors := map[string]*mlx.Array{
				gateUpKey:            onesLike(E, 2*I, H),
				gateUpKey + "_scale": onesLike(E, 2*I, H/16),
				gateUpKey + "_qbias": onesLike(E, 2*I, H/16),
				downKey:              onesLike(E, H, I),
				downKey + "_scale":   onesLike(E, H, I/16),
				downKey + "_qbias":   onesLike(E, H, I/16),
			}

			moe := &MoEBlock{}
			m.loadFusedExperts(moe, tensors, gateUpKey, tensors[gateUpKey], downKey, tensors[downKey])

			if !moe.UseQuantized {
				t.Error("UseQuantized = false, want true")
			}
			if !moe.UseFusedGateUp {
				t.Error("UseFusedGateUp = false, want true")
			}
			if moe.GateUpWeightQ == nil || moe.GateUpScales == nil || moe.GateUpBiases == nil {
				t.Error("quantized gate_up weight/scale/bias not all set")
			}
			if moe.DownWeightQ == nil || moe.DownScales == nil || moe.DownBiases == nil {
				t.Error("quantized down weight/scale/bias not all set")
			}
			// Dense fields must stay nil so Forward takes the GatherQMM path.
			if moe.GateUpWeight != nil || moe.DownWeight != nil {
				t.Error("dense weights set on a quantized block")
			}
		})
	}
}

// TestLoadFusedExpertsDense verifies that a fused gate_up projection with no
// scale companions is loaded onto the dense GatherMM path, kept fused.
func TestLoadFusedExpertsDense(t *testing.T) {
	skipIfNoMLX(t)

	const E, I, H = 4, 8, 16
	m := &Model{TextConfig: &TextConfig{}}

	gateUpKey := "model.language_model.layers.0.experts.gate_up_proj"
	downKey := "model.language_model.layers.0.experts.down_proj"
	tensors := map[string]*mlx.Array{
		gateUpKey: onesLike(E, 2*I, H),
		downKey:   onesLike(E, I, H),
	}

	moe := &MoEBlock{}
	m.loadFusedExperts(moe, tensors, gateUpKey, tensors[gateUpKey], downKey, tensors[downKey])

	if moe.UseQuantized {
		t.Error("UseQuantized = true, want false (no scales present)")
	}
	if !moe.UseFusedGateUp {
		t.Error("UseFusedGateUp = false, want true")
	}
	if moe.GateUpWeight == nil || moe.DownWeight == nil {
		t.Error("dense fused weights not set")
	}
	if moe.GateUpWeightQ != nil || moe.DownWeightQ != nil {
		t.Error("quantized weights set on a dense block")
	}
}

// TestRouterForwardMatchesLegacy verifies the optimized Router.Forward —
// which takes the top-k of the raw logits and softmaxes only the selected
// values — produces the same indices and (within tolerance) the same
// normalized scores as the legacy path that softmaxes over every expert
// first, gathers the top-k probabilities, then renormalizes.
func TestRouterForwardMatchesLegacy(t *testing.T) {
	useMLXTestThread(t)

	cfg := &TextConfig{
		HiddenSize:  8,
		NumExperts:  4,
		TopKExperts: 2,
		RMSNormEps:  1e-6,
		RouterScale: 0.5,
	}

	// Distinct per-expert weight rows so top-k has a well-defined ordering
	// (tied scores would let argpartition pick either tied expert and make
	// the index comparison below flaky).
	projWeight := mlx.FromValues([]float32{
		0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, // expert 0
		0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, // expert 1
		-0.05, -0.06, -0.07, -0.08, -0.09, -0.10, -0.11, -0.12, // expert 2
		0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, // expert 3
	}, int(cfg.NumExperts), int(cfg.HiddenSize))

	scale := mlx.FromValues([]float32{
		1.0, 0.9, 1.1, 1.0, 1.2, 0.8, 1.0, 1.05,
	}, int(cfg.HiddenSize))

	r := &Router{
		Proj:  linearFromWeight(projWeight),
		Scale: scale,
	}

	// Varied x so different positions potentially hit different top-k.
	x := mlx.FromValues([]float32{
		0.2, -0.1, 0.3, 0.0, 0.4, -0.2, 0.1, 0.05,
		-0.3, 0.2, -0.1, 0.4, -0.05, 0.3, 0.0, 0.2,
		0.5, 0.4, -0.2, 0.1, -0.3, 0.0, 0.3, -0.1,
	}, 1, 3, int(cfg.HiddenSize))

	gotScores, gotInds := r.Forward(x, cfg)
	wantScores, wantInds := legacyRouterForward(r, x, cfg)
	gotInds = gotInds.AsType(mlx.DTypeInt32)
	wantInds = wantInds.AsType(mlx.DTypeInt32)
	mlx.Eval(gotScores, gotInds, wantScores, wantInds)

	if got, want := gotInds.Ints(), wantInds.Ints(); !intSlicesEqual(got, want) {
		t.Fatalf("indices mismatch:\n  got  %v\n  want %v", got, want)
	}
	if got, want := gotScores.Floats(), wantScores.Floats(); !floatSlicesClose(got, want, 1e-5) {
		t.Fatalf("scores mismatch:\n  got  %v\n  want %v", got, want)
	}
}

// legacyRouterForward implements the pre-optimization router: full softmax
// over every expert, gather the top-k probabilities, then renormalize them
// to sum to 1. Algebraically identical to the fused form in Router.Forward.
func legacyRouterForward(r *Router, x *mlx.Array, cfg *TextConfig) (*mlx.Array, *mlx.Array) {
	dims := x.Dims()
	BL := int32(dims[0]) * int32(dims[1])

	xFlat := mlx.Reshape(x, BL, cfg.HiddenSize)
	normed := mlx.RMSNormFn(xFlat, nil, cfg.RMSNormEps)
	normed = mlx.MulScalar(normed, cfg.RouterScale)
	normed = mlx.Mul(normed, r.Scale)

	expertScores := r.Proj.Forward(normed)
	probs := mlx.SoftmaxAxis(expertScores, -1, true)

	neg := mlx.Neg(expertScores)
	inds := mlx.Argpartition(neg, int(cfg.TopKExperts)-1, -1)
	inds = mlx.SliceStartStop(inds,
		[]int32{0, 0},
		[]int32{BL, cfg.TopKExperts},
	)

	scores := mlx.TakeAlongAxis(probs, inds, -1)
	sumScores := mlx.Sum(scores, -1, true)
	scores = mlx.Div(scores, sumScores)
	return scores, inds
}

func intSlicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func floatSlicesClose(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > tol {
			return false
		}
	}
	return true
}

// linearFromWeight creates a simple nn.LinearLayer from a weight tensor (no bias).
func linearFromWeight(w *mlx.Array) *simpleLinear {
	return &simpleLinear{weight: w}
}

type simpleLinear struct {
	weight *mlx.Array
}

func (l *simpleLinear) Forward(x *mlx.Array) *mlx.Array {
	return x.Matmul(mlx.Transpose(l.weight, 1, 0))
}

func (l *simpleLinear) OutputDim() int32 {
	return int32(l.weight.Dims()[0])
}
