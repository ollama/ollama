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
