package dflash

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestLiLiCorrWeightsAreCollectable(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		var want []*mlx.Array
		weight := func() *mlx.Array {
			a := mlx.FromValue(float32(1))
			want = append(want, a)
			return a
		}
		linear := func() *nn.Linear { return &nn.Linear{Weight: weight(), Bias: weight()} }
		norm := func() *nn.RMSNorm { return nn.NewRMSNorm(weight(), 1e-5) }
		conv := func() *convolution { return &convolution{Base: weight(), Projection: linear()} }
		var m *lilicorrModel
		got := mlx.ScopedArrays(func() []*mlx.Array {
			m = &lilicorrModel{
				Model: &Model{Layers: []*Layer{{AttentionConv: conv(), MLPConv: conv()}}},
				Head: &lilicorrHead{
					Token: linear(), Pass: linear(), Context: linear(),
					FeatureNorm: &nn.LayerNorm{Weight: weight(), Bias: weight()},
					FeatureUp:   linear(), FeatureDown: linear(),
					Slot: weight(), Rank: weight(), AttentionBias: weight(),
					Layers: []lilicorrLayer{{AttnNorm: norm(), MLPNorm: norm(), QKV: linear(), Output: linear(), Up: linear(), Down: linear()}},
					Norm:   norm(), AnchorNorm: norm(), Factor: []*nn.Linear{linear(), linear(), linear()},
					Outgoing: linear(), Incoming: linear(), AnchorOutgoing: linear(),
				},
			}
			return mlx.Collect(m)
		})
		if len(got) != len(want) {
			t.Fatalf("collected %d weights, want %d", len(got), len(want))
		}
		for i, a := range want {
			if !slices.Contains(got, a) {
				t.Fatalf("weight %d was not retained", i)
			}
			mlx.Eval(a)
			if a.Floats()[0] != 1 {
				t.Fatalf("weight %d did not survive load scope", i)
			}
		}
	})
}
