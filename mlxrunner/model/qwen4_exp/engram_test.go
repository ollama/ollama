package qwen4_exp

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
)

func TestEngramHashes(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
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
	})
}

func TestEngramCacheCarriesChunkHistory(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
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
	})
}

func TestEngramCacheRestoresScheduledSnapshot(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
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
	})
}

func TestEngramIndependentOwners(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		a, b := newEngramCache(2, 3, 1, 9), newEngramCache(2, 3, 1, 9)
		defer a.Free()
		defer b.Free()
		for i, c := range []*engramCache{a, b} {
			ids := mlx.FromValues([]int32{int32(i + 1), int32(i + 2)}, 1, 2)
			batch := &batch.Batch{InputIDs: ids, SeqOffsets: []int32{0}, SeqQueryLens: []int32{2}}
			c.put(batch, ids.AsType(mlx.DTypeInt64), mlx.MulScalar(ids.AsType(mlx.DTypeFloat32).ExpandDims(2), 10))
			c.PrepareSnapshots([]int{3})
		}
		layers, err := cache.Batch([][]cache.Cache{{b}, {a}})
		if err != nil {
			t.Fatal(err)
		}
		view := layers[0].(engramState)
		rb := &batch.Batch{InputIDs: mlx.FromValues([]int32{4, 7}, 2, 1), SeqOffsets: []int32{2, 2}, SeqQueryLens: []int32{1, 1}}
		history := view.get(rb).AsType(mlx.DTypeInt32)
		mlx.Eval(history)
		if got := history.Ints(); !slices.Equal(got, []int32{2, 3, 1, 2}) {
			t.Fatalf("gathered histories = %v", got)
		}
		view.put(rb, rb.InputIDs.AsType(mlx.DTypeInt64), mlx.FromValues([]float32{40, 70}, 2, 1, 1))
		for i, c := range []*engramCache{b, a} {
			snapshot := c.TakeSnapshots()[0]
			if snapshot == nil {
				t.Fatal("missing row snapshot")
			}
			defer snapshot.Close()
			if !c.Restore(snapshot, 3) {
				t.Fatal("failed to restore batched state")
			}
			h := c.history.AsType(mlx.DTypeInt32)
			mlx.Eval(h, c.convHistory)
			wantTokens := [][]int32{{3, 4}, {2, 7}}[i]
			wantConv := [][]float32{{20, 30, 40}, {10, 20, 70}}[i]
			if !slices.Equal(h.Ints(), wantTokens) || !slices.Equal(c.convHistory.Floats(), wantConv) {
				t.Fatalf("row %d history=%v conv=%v", i, h.Ints(), c.convHistory.Floats())
			}
		}
		incompatible := newEngramCache(2, 4, 1, 9)
		defer incompatible.Free()
		if _, err := cache.Batch([][]cache.Cache{{a}, {incompatible}}); err == nil {
			t.Fatal("accepted incompatible Engram owners")
		}
	})
}
