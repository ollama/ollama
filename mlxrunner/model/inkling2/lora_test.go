package inkling2

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model/lora"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestLoRAExpertRoutes(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			// Two tokens select opposite experts. Gate/up share A, down shares B.
			input := mlx.FromValues([]float32{1, 2, 4, 1}, 2, 1, 1, 2)
			idx := mlx.FromValues([]int32{1, 0, 0, 1}, 2, 2)
			up := &lora.Factors{AT: mlx.FromValues([]float32{1, 2}, 2, 1), BT: mlx.FromValues([]float32{3, 5}, 2, 1, 1), Scale: 2}
			out := addExpert(mlx.FromValue(float32(0)), input, idx, up)
			if got := out.Floats(); !slices.Equal(got, []float32{50, 30, 36, 60}) {
				t.Fatalf("gate/up routes: %v", got)
			}
			down := &lora.Factors{AT: mlx.FromValues([]float32{7, 11}, 2, 1, 1), BT: mlx.FromValues([]float32{1, 2}, 1, 2), Scale: 2}
			if got := addExpert(mlx.FromValue(float32(0)), out, idx, down).Floats(); !slices.Equal(got, []float32{1100, 2200, 420, 840, 504, 1008, 1320, 2640}) {
				t.Fatalf("down routes: %v", got)
			}
		})
	})
}

func TestLoRAAttentionSlices(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			linear := func(rows ...float32) *nn.Linear {
				return nn.NewLinear(mlx.FromValues(rows, len(rows), 1), nil)
			}
			delta := func(b float32) *lora.Factors {
				return &lora.Factors{AT: mlx.FromValues([]float32{1}, 1, 1), BT: mlx.FromValues([]float32{b}, 1, 1), Scale: 2}
			}
			// Q, relative, gate, sink, K, K-convolution, V, V-convolution.
			a := selfAttention{heads: 1, headDim: 1, kvHeads: 1, PositionProjection: mlx.FromValues([]float32{1}, 1, 1),
				Projection: linear(1, 2, 3, 4, 5, 6, 7, 8), QueryRows: linear(1, 2, 3, 4), KeyRows: linear(5, 6), ValueRows: linear(7, 8), Output: linear(9),
				lora: &attentionLoRA{query: delta(10), relative: delta(20), gate: delta(30), key: delta(40), value: delta(50)}, outputLoRA: delta(60)}
			input := func(v float32) *mlx.Array { return mlx.FromValues([]float32{v}, 1, 1, 1) }
			check := func(x *mlx.Array, want ...float32) {
				if got := x.Floats(); !slices.Equal(got, want) {
					t.Fatalf("got %v want %v", got, want)
				}
			}
			x := input(2)
			q, k, v := a.project(x, x, x)
			check(q, 42, 84, 126, 8)
			check(k, 170, 12)
			check(v, 214, 16)
			q, k, v = a.project(x, input(3), input(4))
			check(q, 42, 84, 126, 8)
			check(k, 255, 18)
			check(v, 428, 32)
			check(a.projectOutput(x), 258)
		})
	})
}

func TestLoRAHeadBeforeSoftcap(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			head := &lora.Factors{AT: mlx.FromValues([]float32{1}, 1, 1), BT: mlx.FromValues([]float32{2, 3}, 1, 2), Scale: 2}
			m := targetModel{Output: nn.NewLinear(mlx.FromValues([]float32{2, 4}, 2, 1), nil), headLoRA: head, config: textConfig{LogitCap: 10}}
			got := m.Unembed(mlx.FromValues([]float32{1}, 1, 1, 1)).Floats()
			if len(got) != 2 {
				t.Fatalf("unexpected vocabulary dimension: %v", got)
			}
			for i, raw := range []float64{6, 10} {
				if math.Abs(float64(got[i])-10*math.Tanh(raw/10)) > 1e-5 {
					t.Fatalf("incorrect softcapped logit %d: %f", i, got[i])
				}
			}
		})
	})
}
