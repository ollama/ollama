package qwen4_exp

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestEngramHashes(t *testing.T) {
	mlxtest.Setup(t)

	p := &PLE{
		LayerMultipliers: mlx.FromValues([]int64{3, 5, 7}, 3),
		HeadVocabSizes:   mlx.FromValues([]int64{11, 13, 17, 19}, 4),
		HeadOffsets:      mlx.FromValues([]int64{0, 11, 24, 41}, 4),
	}
	cfg := &Config{NGramSize: 3, HeadsPerNGram: 2, EOSTokenID: 9}
	b := &batch.Batch{InputIDs: mlx.FromValues([]int32{1, 9, 2, 3}, 1, 4)}
	history := mlx.FromValues([]int64{7, 8}, 1, 2)

	got, _ := p.hashes(b, history, cfg)
	got = got.AsType(mlx.DTypeInt32)
	mlx.Eval(got)
	values := got.Ints()
	want := []int32{10, 15, 33, 48, 8, 15, 28, 41, 10, 15, 27, 42, 3, 14, 33, 44}
	if !slices.Equal(values, want) {
		t.Fatalf("hashes = %v, want %v", values, want)
	}
}

func TestEngramCacheCarriesChunkHistory(t *testing.T) {
	mlxtest.Setup(t)
	c := newEngramCache(2, 3, 1, 9)
	t.Cleanup(c.Free)
	b := &batch.Batch{
		InputIDs:     mlx.FromValues([]int32{1, 2}, 1, 2),
		SeqQueryLens: []int32{2},
	}
	c.put(b, mlx.FromValues([]int64{1, 2}, 1, 2), mlx.FromValues([]float32{10, 20}, 1, 2, 1))
	b.InputIDs = mlx.FromValues([]int32{3, 4}, 1, 2)
	c.put(b, mlx.FromValues([]int64{3, 4}, 1, 2), mlx.FromValues([]float32{30, 40}, 1, 2, 1))

	history := c.history.AsType(mlx.DTypeInt32)
	mlx.Eval(history, c.convHistory)
	if got, want := history.Ints(), []int32{3, 4}; !slices.Equal(got, want) {
		t.Fatalf("token history = %v, want %v", got, want)
	}
	if got, want := c.convHistory.Floats(), []float32{20, 30, 40}; !slices.Equal(got, want) {
		t.Fatalf("conv history = %v, want %v", got, want)
	}
}

func TestEngramCacheRestoresScheduledSnapshot(t *testing.T) {
	mlxtest.Setup(t)
	c := newEngramCache(2, 3, 1, 9)
	t.Cleanup(c.Free)
	b := &batch.Batch{
		InputIDs:     mlx.FromValues([]int32{1, 2}, 1, 2),
		SeqQueryLens: []int32{2},
	}
	c.put(b, mlx.FromValues([]int64{1, 2}, 1, 2), mlx.FromValues([]float32{10, 20}, 1, 2, 1))

	c.PrepareSnapshots([]int{3, 4})
	b.InputIDs = mlx.FromValues([]int32{3, 4}, 1, 2)
	inputIDs := mlx.FromValues([]int64{3, 4}, 1, 2)
	convInput := mlx.FromValues([]float32{30, 40}, 1, 2, 1)
	c.put(b, inputIDs, convInput)
	snapshots := c.TakeSnapshots()
	if len(snapshots) != 2 || snapshots[0] == nil || snapshots[1] == nil {
		t.Fatalf("TakeSnapshots() = %v, want two captured snapshots", snapshots)
	}
	for _, snapshot := range snapshots {
		t.Cleanup(snapshot.Close)
	}
	first := snapshots[0].(*engramSnapshot)
	mlx.Eval(first.history, first.convHistory)
	if got, want := first.convHistory.Floats(), []float32{10, 20, 30}; !slices.Equal(got, want) {
		t.Fatalf("captured convolution history = %v, want %v", got, want)
	}

	b.InputIDs = mlx.FromValues([]int32{5, 6}, 1, 2)
	c.put(b, mlx.FromValues([]int64{5, 6}, 1, 2), mlx.FromValues([]float32{50, 60}, 1, 2, 1))
	if !c.Restore(snapshots[0], 3) {
		t.Fatal("Restore(snapshot, 3) failed")
	}
	history := c.history.AsType(mlx.DTypeInt32)
	mlx.Eval(history, c.convHistory)
	if got, want := history.Ints(), []int32{2, 3}; !slices.Equal(got, want) {
		t.Fatalf("restored token history = %v, want %v", got, want)
	}
	if got, want := c.convHistory.Floats(), []float32{10, 20, 30}; !slices.Equal(got, want) {
		t.Fatalf("restored convolution history = %v, want %v", got, want)
	}

	parent, child := c.Split(snapshots[1], 3)
	if parent != nil || child != snapshots[1] {
		t.Fatalf("Split(snapshot) = (%v, %v), want (nil, snapshot)", parent, child)
	}
	if merged := c.Merge(nil, child); merged != child {
		t.Fatalf("Merge(nil, child) = %v, want child", merged)
	}
}
