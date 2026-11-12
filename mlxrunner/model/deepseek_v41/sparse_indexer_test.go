package deepseek_v41

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestSparseIndexerSelectsVisiblePositions(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		indexer := &sparseIndexer{
			Query:   nn.NewLinear(mlx.FromValues([]float32{1, 0, 0, 1}, 2, 2), nil),
			Weights: nn.NewLinear(mlx.Zeros(mlx.DTypeFloat32, 1, 2), mlx.FromValues([]float32{1}, 1)),
			Heads:   1,
			Width:   2,
			TopK:    1,
		}
		x := mlx.Zeros(mlx.DTypeFloat32, 1, 3, 2)
		queries := mlx.FromValues([]float32{1, 0, 1, 0, 0, 1}, 1, 3, 2)
		keys := mlx.FromValues([]float32{10, 0, 0, 1, 0, 10}, 1, 3, 2)
		selected, _ := indexer.selectKeys(x, queries, keys, nil, 0, 1, []float32{0})
		mlx.Eval(selected)
		if got, want := selected.Ints(), []int32{0, 0, 2}; !slices.Equal(got, want) {
			t.Fatalf("selected %v, want %v", got, want)
		}

		paired, _ := indexer.selectKeys(x, queries, keys, nil, 0, 2, []float32{0})
		mlx.Eval(paired)
		if got, want := paired.Ints(), []int32{-1, 0, 0}; !slices.Equal(got, want) {
			t.Fatalf("paired selection %v, want %v", got, want)
		}
	})
}

func TestSparseIndexerSharesCandidateBlocks(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		source := &sparseIndexer{
			Query:              nn.NewLinear(mlx.FromValues([]float32{1, 0, 0, 1}, 2, 2), nil),
			Weights:            nn.NewLinear(mlx.Zeros(mlx.DTypeFloat32, 1, 2), mlx.FromValues([]float32{1}, 1)),
			Heads:              1,
			Width:              2,
			TopK:               1,
			CandidateSource:    true,
			CandidateBlocks:    1,
			CandidateBlockSize: 1,
		}
		x := mlx.Zeros(mlx.DTypeFloat32, 1, 3, 2)
		queries := mlx.FromValues([]float32{1, 0, 1, 0, 0, 1}, 1, 3, 2)
		keys := mlx.FromValues([]float32{10, 0, 0, 1, 0, 10}, 1, 3, 2)
		_, candidates := source.selectKeys(x, queries, keys, nil, 0, 1, []float32{0})
		later := *source
		later.CandidateSource = false
		later.UsesCandidates = true
		selected, _ := later.selectKeys(x, queries, keys, candidates, 0, 1, []float32{0})
		mlx.Eval(selected)
		if got, want := selected.Ints(), []int32{0, 1, 2}; !slices.Equal(got, want) {
			t.Fatalf("candidate-filtered selection %v, want %v", got, want)
		}
	})
}

func TestCandidateBlocksPinNewestVisibleBlock(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		values := make([]float32, 36)
		for i := range values {
			query, key := i/12, i%12
			if key >= []int{0, 5, 12}[query] {
				values[i] = -math.MaxFloat32
			} else {
				values[i] = float32(key + 1)
			}
		}
		scores := mlx.FromValues(values, 1, 3, 12)
		selected := selectCandidateBlocks(scores, []int32{0, 5, 12}, 2, 2)
		mlx.Eval(selected)
		want := make([]int32, 36)
		for _, i := range []int{14, 15, 16, 17, 32, 33, 34, 35} {
			want[i] = 1
		}
		if got := selected.AsType(mlx.DTypeInt32).Ints(); !slices.Equal(got, want) {
			t.Fatalf("candidate positions %v, want %v", got, want)
		}
	})
}

func TestCandidateBlocksPadPartialLastBlock(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		scores := mlx.FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 1, 1, 11)
		selected := selectCandidateBlocks(scores, []int32{11}, 2, 3)
		mlx.Eval(selected)
		if got, want := selected.AsType(mlx.DTypeInt32).Ints(), []int32{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1}; !slices.Equal(got, want) {
			t.Fatalf("partial final block %v, want %v", got, want)
		}
	})
}
