package deepseek_v41

import (
	"math"
	"math/rand/v2"
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

// Scoring queries in row blocks bounds the scores' memory without changing any
// selection: each query's scores, causal limit, and candidate blocks are its own.
func TestSparseIndexerRowBlocksMatchOneBlock(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const heads, width, low, hidden, length, count, start = 32, 128, 64, 64, 2047, 3000, 2000
		rng := rand.New(rand.NewPCG(1, 2))
		values := func(n int, scale float64) []float32 {
			v := make([]float32, n)
			for i := range v {
				v[i] = float32(rng.NormFloat64() * scale)
			}
			return v
		}
		source := &sparseIndexer{
			Query:              nn.NewLinear(mlx.FromValues(values(heads*width*low, 0.1), heads*width, low), nil),
			Weights:            nn.NewLinear(mlx.FromValues(values(heads*hidden, 0.1), heads, hidden), nil),
			Heads:              heads,
			Width:              width,
			TopK:               512,
			CandidateSource:    true,
			CandidateBlocks:    8,
			CandidateBlockSize: 128,
		}
		x := mlx.FromValues(values(length*hidden, 1), 1, length, hidden)
		queryLow := mlx.FromValues(values(length*low, 1), 1, length, low)
		keys := mlx.FromValues(values(count*width, 1), 1, count, width)
		frequencies := []float32{1, 0.1, 0.01, 0.001}

		later := *source
		later.CandidateSource = false
		later.UsesCandidates = true
		for _, rows := range []int{256, 100} {
			wholeSel, wholeMask := source.selectKeysInRows(x, queryLow, keys, nil, start, 1, frequencies, length)
			blockSel, blockMask := source.selectKeysInRows(x, queryLow, keys, nil, start, 1, frequencies, rows)
			wholeLater, _ := later.selectKeysInRows(x, queryLow, keys, wholeMask, start, 1, frequencies, length)
			blockLater, _ := later.selectKeysInRows(x, queryLow, keys, blockMask, start, 1, frequencies, rows)
			wm, bm := wholeMask.AsType(mlx.DTypeInt32), blockMask.AsType(mlx.DTypeInt32)
			mlx.Eval(wholeSel, blockSel, wm, bm, wholeLater, blockLater)
			if !slices.Equal(wholeSel.Ints(), blockSel.Ints()) {
				t.Fatalf("%d-row blocks select differently from one block", rows)
			}
			if !slices.Equal(wm.Ints(), bm.Ints()) {
				t.Fatalf("%d-row blocks keep different candidate blocks", rows)
			}
			if !slices.Equal(wholeLater.Ints(), blockLater.Ints()) {
				t.Fatalf("%d-row blocks select differently under candidates", rows)
			}
		}
	})
}
