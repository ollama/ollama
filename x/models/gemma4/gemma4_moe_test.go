package gemma4

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// onesLike creates a tensor of the given shape filled with a small constant.
func onesLike(shape ...int) *mlx.Array {
	return mlx.AddScalar(mlx.Zeros(mlx.DTypeBFloat16, shape...), 0.01)
}

func TestMoEForward(t *testing.T) {
	skipIfNoMLX(t)

	// Small config matching 26b architecture pattern.
	cfg := &TextConfig{
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

	B, L := int32(1), int32(3)
	x := onesLike(int(B), int(L), int(cfg.HiddenSize))

	// Test Router.Forward.
	router := &Router{
		Proj:  linearFromWeight(onesLike(int(cfg.NumExperts), int(cfg.HiddenSize))),
		Scale: onesLike(int(cfg.HiddenSize)),
	}

	t.Run("Router", func(t *testing.T) {
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
	})

	// Test MoEBlock.Forward.
	moe := &MoEBlock{
		GateWeight:     onesLike(int(cfg.NumExperts), int(cfg.HiddenSize), int(cfg.ExpertIntermediateSize)),
		UpWeight:       onesLike(int(cfg.NumExperts), int(cfg.HiddenSize), int(cfg.ExpertIntermediateSize)),
		DownWeight:     onesLike(int(cfg.NumExperts), int(cfg.ExpertIntermediateSize), int(cfg.HiddenSize)),
		PerExpertScale: onesLike(int(cfg.NumExperts)),
	}

	t.Run("MoEBlock", func(t *testing.T) {
		scores, inds := router.Forward(x, cfg)
		mlx.Eval(scores, inds)

		out := moe.Forward(x, scores, inds, cfg)
		mlx.Eval(out)

		outDims := out.Dims()
		t.Logf("MoE output shape: %v", outDims)

		if len(outDims) != 3 || outDims[0] != int(B) || outDims[1] != int(L) || outDims[2] != int(cfg.HiddenSize) {
			t.Errorf("output shape = %v, want [%d, %d, %d]", outDims, B, L, cfg.HiddenSize)
		}
	})

	// Test with larger batch to exercise the sorted GatherMM path (B*L >= 64).
	t.Run("MoEBlock_sorted", func(t *testing.T) {
		bigB, bigL := int32(1), int32(128)
		bigX := onesLike(int(bigB), int(bigL), int(cfg.HiddenSize))

		scores, inds := router.Forward(bigX, cfg)
		mlx.Eval(scores, inds)

		out := moe.Forward(bigX, scores, inds, cfg)
		mlx.Eval(out)

		outDims := out.Dims()
		t.Logf("MoE sorted output shape: %v", outDims)

		if len(outDims) != 3 || outDims[0] != int(bigB) || outDims[1] != int(bigL) || outDims[2] != int(cfg.HiddenSize) {
			t.Errorf("output shape = %v, want [%d, %d, %d]", outDims, bigB, bigL, cfg.HiddenSize)
		}
	})
}

// TestRouterForwardMatchesLegacy verifies the optimized Router.Forward —
// which takes the top-k of the raw logits and softmaxes only the selected
// values — produces the same indices and (within tolerance) the same
// normalized scores as the legacy path that softmaxes over every expert
// first, gathers the top-k probabilities, then renormalizes.
func TestRouterForwardMatchesLegacy(t *testing.T) {
	skipIfNoMLX(t)

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
