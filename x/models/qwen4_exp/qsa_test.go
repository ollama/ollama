package qwen4_exp

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func TestQSASelectsCompressedBlocksAndCausalTail(t *testing.T) {
	mlxtest.Setup(t)
	cfg := &Config{IndexerBudget: 8, IndexerCompressRatio: 4}
	scores := mlx.FromValues([]float32{0.1, 4, 2, 3}, 1, 1, 4)
	b := &batch.Batch{SeqOffsets: []int32{16}}

	indices, valid := qsaLogicalIndices(scores, b, 16, cfg)
	indices = indices.AsType(mlx.DTypeInt32)
	valid = valid.AsType(mlx.DTypeInt32)
	mlx.Eval(indices, valid)

	values, mask := indices.Ints(), valid.Ints()
	selected := make([]int32, 0, len(values))
	for i, value := range values {
		if mask[i] != 0 {
			selected = append(selected, value)
		}
	}
	slices.Sort(selected)
	want := []int32{4, 5, 6, 7, 12, 13, 14, 15, 16}
	if !slices.Equal(selected, want) {
		t.Fatalf("selected indices = %v, want %v", selected, want)
	}
}

func TestQSASelectionMasksFutureBlocks(t *testing.T) {
	mlxtest.Setup(t)
	cfg := &Config{IndexerBudget: 8, IndexerCompressRatio: 4}
	// Only block 0 and token 4 are visible. Give every future block a much
	// larger score so the test fails if selection sees cached-but-causal-junk.
	scores := mlx.FromValues([]float32{0.1, 100, 90, 80, 70}, 1, 1, 5)
	b := &batch.Batch{SeqOffsets: []int32{4}}

	indices, valid := qsaLogicalIndices(scores, b, 20, cfg)
	indices = indices.AsType(mlx.DTypeInt32)
	valid = valid.AsType(mlx.DTypeInt32)
	mlx.Eval(indices, valid)

	var selected []int32
	for i, value := range indices.Ints() {
		if valid.Ints()[i] != 0 {
			selected = append(selected, value)
		}
	}
	slices.Sort(selected)
	if want := []int32{0, 1, 2, 3, 4}; !slices.Equal(selected, want) {
		t.Fatalf("selected indices = %v, want %v", selected, want)
	}
}

func TestQSASparseAttentionMatchesReference(t *testing.T) {
	mlxtest.Setup(t)
	cfg := &Config{NumKeyValueHeads: 1, Scale: 1}
	q := mlx.FromValues([]float32{1, 0}, 1, 1, 1, 2)
	k := mlx.FromValues([]float32{1, 0, 0, 1, 2, 0}, 1, 1, 3, 2)
	v := mlx.FromValues([]float32{10, 1, 20, 2, 30, 3}, 1, 1, 3, 2)
	indices := mlx.FromValues([]int32{2, 0}, 1, 1, 2)
	valid := mlx.FromValues([]bool{true, true}, 1, 1, 2)

	out := qsaSparseAttention(q, nn.NewKVHistory(k, v, nil), indices, valid, cfg)
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
}

func TestQSASparseAttentionIgnoresInvalidRows(t *testing.T) {
	mlxtest.Setup(t)
	cfg := &Config{NumKeyValueHeads: 1, Scale: 1}
	q := mlx.FromValues([]float32{1, 0}, 1, 1, 1, 2)
	// Row 0 is intentionally dominant junk. Only row 1 is logically valid.
	k := mlx.FromValues([]float32{100, 0, 1, 0}, 1, 1, 2, 2)
	v := mlx.FromValues([]float32{999, 999, 7, 3}, 1, 1, 2, 2)
	indices := mlx.FromValues([]int32{0, 1}, 1, 1, 2)
	valid := mlx.FromValues([]bool{false, true}, 1, 1, 2)

	out := qsaSparseAttention(q, nn.NewKVHistory(k, v, nil), indices, valid, cfg)
	out = out.AsType(mlx.DTypeFloat32)
	mlx.Eval(out)
	if got, want := out.Floats(), []float32{7, 3}; !slices.Equal(got, want) {
		t.Fatalf("sparse attention = %v, want %v", got, want)
	}
}

func TestQSASparseAttentionKeepsBatchRowsIndependent(t *testing.T) {
	mlxtest.Setup(t)
	cfg := &Config{NumKeyValueHeads: 1, Scale: 1}
	q := mlx.FromValues([]float32{1, 0, 1, 0}, 2, 1, 1, 2)
	k := mlx.FromValues([]float32{
		1, 0, 0, 1,
		1, 0, 0, 1,
	}, 2, 1, 2, 2)
	v := mlx.FromValues([]float32{
		10, 1, 20, 2,
		30, 3, 40, 4,
	}, 2, 1, 2, 2)
	indices := mlx.FromValues([]int32{0, 1}, 2, 1, 1)
	valid := mlx.FromValues([]bool{true, true}, 2, 1, 1)

	out := qsaSparseAttention(q, nn.NewKVHistory(k, v, nil), indices, valid, cfg)
	out = out.AsType(mlx.DTypeFloat32)
	mlx.Eval(out)
	if got, want := out.Floats(), []float32{10, 1, 40, 4}; !slices.Equal(got, want) {
		t.Fatalf("sparse attention = %v, want %v", got, want)
	}
}
