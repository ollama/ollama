package laya

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestBidirectionalAttention(t *testing.T) {
	for _, tc := range []struct {
		name  string
		local bool
		want  []float32
	}{
		{"global", false, []float32{3, 13, 3, 13, 3, 13, 3, 13}},
		{"local", true, []float32{1, 11, 2, 12, 4, 14, 5, 15}},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			mlx.Scoped(func() {
				// Zero queries/keys give uniform attention. The two heads average
				// values [0,2,4,6] and [10,12,14,16], either over all positions
				// or over the current position and its immediate neighbors.
				qkv := mlx.FromValues([]float32{
					0, 0, 0, 0, 0, 10,
					0, 0, 0, 0, 2, 12,
					0, 0, 0, 0, 4, 14,
					0, 0, 0, 0, 6, 16,
				}, 1, 4, 6)
				b := &batch.Batch{
					InputIDs:     mlx.FromValues([]int32{0, 1, 2, 3}, 1, 4),
					SeqOffsets:   []int32{0},
					SeqQueryLens: []int32{4},
				}
				var mask nn.AttentionMask
				if tc.local {
					blocked := float32(math.Inf(-1))
					mask = nn.ArrayMask(mlx.FromValues([]float32{
						0, 0, blocked, blocked,
						0, 0, 0, blocked,
						blocked, 0, 0, 0,
						blocked, blocked, 0, 0,
					}, 4, 4))
				}
				out := attention(b, qkv, 2, 0, mask).Reshape(-1)
				mlx.Eval(out)
				for i, got := range out.Floats() {
					if !(math.Abs(float64(got-tc.want[i])) <= 1e-5) {
						t.Fatalf("output %d = %g, want %g", i, got, tc.want[i])
					}
				}
			})
		})
	}
}

func TestBatchMatchesIndividualDecisions(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			// A small encoder with global and local layers followed by the
			// decision head. Batch rows and their question types stay independent.
			weight := func(rows, cols int) *mlx.Array {
				return mlx.Sin(mlx.Arange(0, float64(rows*cols), 1, mlx.DTypeFloat32)).Reshape(rows, cols)
			}
			linear := func(out, in int) *nn.Linear { return &nn.Linear{Weight: weight(out, in)} }
			norm := &nn.LayerNorm{Weight: mlx.FromValues([]float32{1, 1, 1, 1}, 4), Eps: 1e-5}
			m := Model{
				encoder:    encoderConfig{HiddenSize: 4, Heads: 2, GlobalEvery: 2, LocalAttention: 2, GlobalTheta: 160000, LocalTheta: 10000},
				Embeddings: weight(8, 4), TypeEmbedding: weight(3, 4), EmbeddingNorm: norm, FinalNorm: norm,
				Layers: []encoderLayer{
					{MLPNorm: norm, QKV: linear(12, 4), Out: linear(4, 4), In: linear(16, 4), Down: linear(4, 8)},
					{AttentionNorm: norm, MLPNorm: norm, QKV: linear(12, 4), Out: linear(4, 4), In: linear(16, 4), Down: linear(4, 8)},
				},
				Head:      []headLayer{{Norm1: norm, Norm2: norm, QKV: linear(12, 4), Out: linear(4, 4), In: linear(16, 4), Down: linear(4, 16)}},
				ScoreNorm: norm, ScoreIn: linear(4, 4), ScoreOut: linear(1, 4),
			}
			rows := [][]int32{{1, 2, 3, 4}, {5, 6, 7, 5}}
			types := []int32{0, 2}
			b := &batch.Batch{
				InputIDs:     mlx.FromValues([]int32{1, 2, 3, 4, 5, 6, 7, 5}, 2, 4),
				SeqOffsets:   []int32{0, 0},
				SeqQueryLens: []int32{4, 4},
				Layout:       []any{types[0], types[1]},
			}
			hidden, _ := m.Forward(b, nil)
			out := m.Unembed(hidden).Reshape(-1)
			mlx.Eval(out)
			got := out.Floats()
			for i, ids := range rows {
				single := &batch.Batch{
					InputIDs:     mlx.FromValues(ids, 1, len(ids)),
					SeqOffsets:   []int32{0},
					SeqQueryLens: []int32{int32(len(ids))},
					Layout:       []any{types[i]},
				}
				one, _ := m.Forward(single, nil)
				scores := m.Unembed(one).Reshape(-1)
				mlx.Eval(scores)
				for j, want := range scores.Floats() {
					if !(math.Abs(float64(got[i*4+j]-want)) <= 1e-5) {
						t.Fatalf("row %d token %d: batched score %g, individual score %g", i, j, got[i*4+j], want)
					}
				}
			}
		})
	})
}
