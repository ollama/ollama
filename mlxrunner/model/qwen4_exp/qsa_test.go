package qwen4_exp

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestQSASelectsCompressedBlocksAndCausalTail(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := &Config{IndexerBudget: 8, IndexerCompressRatio: 4}
		scores := mlx.FromValues([]float32{0.1, 4, 2, 3}, 1, 1, 4)
		b := &batch.Batch{SeqOffsets: []int32{16}}

		blocks, queryEnds := qsaLogicalBlocks(scores, b, 16, cfg)
		if blocks.DType() != mlx.DTypeInt32 {
			t.Fatalf("blocks dtype = %v, want int32", blocks.DType())
		}
		mlx.Eval(blocks, queryEnds)

		got := blocks.Ints()
		slices.Sort(got)
		if want := []int32{1, 3}; !slices.Equal(got, want) {
			t.Fatalf("selected blocks = %v, want %v", got, want)
		}
		if got, want := queryEnds.Ints(), []int32{17}; !slices.Equal(got, want) {
			t.Fatalf("query ends = %v, want %v", got, want)
		}
	})
}

func TestQSASelectionMasksFutureBlocks(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := &Config{IndexerBudget: 8, IndexerCompressRatio: 4}
		// Only block 0 and token 4 are visible. Give every future block a much
		// larger score so the test fails if selection sees cached-but-causal-junk.
		scores := mlx.FromValues([]float32{0.1, 100, 90, 80, 70}, 1, 1, 5)
		b := &batch.Batch{SeqOffsets: []int32{4}}

		blocks, queryEnds := qsaLogicalBlocks(scores, b, 20, cfg)
		mlx.Eval(blocks, queryEnds)

		got := blocks.Ints()
		if want := []int32{0, -1}; !slices.Equal(got, want) {
			t.Fatalf("selected blocks = %v, want %v", got, want)
		}
		if got, want := queryEnds.Ints(), []int32{5}; !slices.Equal(got, want) {
			t.Fatalf("query ends = %v, want %v", got, want)
		}
	})
}

func TestQSASparseAttentionMatchesReference(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := &Config{NumKeyValueHeads: 1, IndexerCompressRatio: 1, Scale: 1}
		q := mlx.FromValues([]float32{1, 0}, 1, 1, 1, 2)
		k := mlx.FromValues([]float32{1, 0, 0, 1, 2, 0}, 1, 1, 3, 2)
		v := mlx.FromValues([]float32{10, 1, 20, 2, 30, 3}, 1, 1, 3, 2)
		blocks := mlx.FromValues([]int32{2, 0}, 1, 1, 2)
		queryEnds := mlx.FromValues([]int32{3}, 1, 1)

		out := qsaSparseAttention(q, nn.NewKVHistory(k, v, nil), blocks, queryEnds, cfg)
		out = out.AsType(mlx.DTypeFloat32)
		mlx.Eval(out)
		got := out.Floats()
		p2 := float32(math.Exp(2) / (math.Exp(2) + math.Exp(1)))
		want := []float32{p2*30 + (1-p2)*10, p2*3 + (1-p2)*1}
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-4 {
				t.Fatalf("sparse attention[%d] = %v, want %v", i, got[i], want[i])
			}
		}
	})
}

func TestQSASparseAttentionIgnoresInvalidRows(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := &Config{NumKeyValueHeads: 1, IndexerCompressRatio: 1, Scale: 1}
		q := mlx.FromValues([]float32{1, 0}, 1, 1, 1, 2)
		// Row 0 is intentionally dominant junk. Only row 1 is logically valid.
		k := mlx.FromValues([]float32{100, 0, 1, 0}, 1, 1, 2, 2)
		v := mlx.FromValues([]float32{999, 999, 7, 3}, 1, 1, 2, 2)
		blocks := mlx.FromValues([]int32{-1, 1}, 1, 1, 2)
		queryEnds := mlx.FromValues([]int32{2}, 1, 1)

		out := qsaSparseAttention(q, nn.NewKVHistory(k, v, nil), blocks, queryEnds, cfg)
		out = out.AsType(mlx.DTypeFloat32)
		mlx.Eval(out)
		if got, want := out.Floats(), []float32{7, 3}; !slices.Equal(got, want) {
			t.Fatalf("sparse attention = %v, want %v", got, want)
		}
	})
}

func TestQSASparseAttentionKeepsBatchRowsIndependent(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := &Config{NumKeyValueHeads: 1, IndexerCompressRatio: 1, Scale: 1}
		q := mlx.FromValues([]float32{1, 0, 1, 0}, 2, 1, 1, 2)
		k := mlx.FromValues([]float32{
			1, 0, 0, 1,
			1, 0, 0, 1,
		}, 2, 1, 2, 2)
		v := mlx.FromValues([]float32{
			10, 1, 20, 2,
			30, 3, 40, 4,
		}, 2, 1, 2, 2)
		blocks := mlx.FromValues([]int32{0, 1}, 2, 1, 1)
		queryEnds := mlx.FromValues([]int32{2, 2}, 2, 1)

		out := qsaSparseAttention(q, nn.NewKVHistory(k, v, nil), blocks, queryEnds, cfg)
		out = out.AsType(mlx.DTypeFloat32)
		mlx.Eval(out)
		if got, want := out.Floats(), []float32{10, 1, 40, 4}; !slices.Equal(got, want) {
			t.Fatalf("sparse attention = %v, want %v", got, want)
		}
	})
}

func TestQSAIndependentHistories(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := &Config{NumKeyValueHeads: 1, Scale: .7, IndexerBudget: 8, IndexerCompressRatio: 4, IndexerHeadDim: 2, RopeDim: 2, RMSNormEps: 1e-6, RopeParameters: ropeParameters{RopeTheta: 10000}}
		a := &fullAttention{Indexer: &attentionIndexer{KNorm: nn.NewRMSNorm(mlx.FromValues([]float32{1, 1}, 2), 1e-6)}}
		var main, index []*nn.KVHistory
		offsets := []int32{4, 18, 12}
		for row, offset := range offsets {
			n := int(offset + 1)
			k, v := make([]float32, n*2), make([]float32, n*2)
			positions := make([]int32, n*3)
			for i := range n {
				k[2*i], k[2*i+1] = float32((i+row*3)%7)*.2, float32((i*3+row)%11)*.1
				v[2*i], v[2*i+1] = float32(row*100+i), float32(i%3)
				positions[3*i], positions[3*i+1], positions[3*i+2] = int32(i), int32(i), int32(i)
			}
			keys := mlx.FromValues(k, 1, 1, n, 2)
			main = append(main, nn.NewKVHistory(keys, mlx.FromValues(v, 1, 1, n, 2), nil))
			index = append(index, nn.NewKVHistory(keys, mlx.FromValues(positions, 1, 1, n, 3), nil))
		}
		// Mix dense and sparse rows, then reorder and retire rows. Comparing to
		// independent forwards catches accidental use of the longest row's cache.
		for _, rows := range [][]int{{0, 1, 2}, {2, 0}, {1, 2}} {
			b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, len(rows), 1)}
			var histories, indices []*nn.KVHistory
			for _, row := range rows {
				b.SeqOffsets = append(b.SeqOffsets, offsets[row])
				b.SeqQueryLens = append(b.SeqQueryLens, 1)
				histories = append(histories, main[row])
				indices = append(indices, index[row])
			}
			q := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, len(rows), 1, 1, 2), .5)
			got := a.attend(b, q, q, nn.NewRowKVHistory(histories), nn.NewRowKVHistory(indices), cfg)
			var separate []*mlx.Array
			for i := range rows {
				qi := q.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(), mlx.Slice())
				separate = append(separate, a.attend(b.Row(i), qi, qi, histories[i], indices[i], cfg))
			}
			want := mlx.Concatenate(separate, 0)
			mlx.Eval(got, want)
			actual, expected := got.Floats(), want.Floats()
			if len(actual) == 0 || len(actual) != len(expected) {
				t.Fatalf("rows %v value count = %d, want %d", rows, len(actual), len(expected))
			}
			for i, v := range expected {
				if delta := math.Abs(float64(actual[i] - v)); math.IsNaN(delta) || delta > 1e-4 {
					t.Fatalf("rows %v element %d got %g want %g", rows, i, actual[i], v)
				}
			}
		}
	})
}
