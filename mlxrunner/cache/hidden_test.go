package cache

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestHiddenCacheSnapshots(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := NewHiddenCache()
		defer c.Free()
		c.PrepareSnapshots([]int{2, 3})
		c.Append(mlx.FromValues([]float32{1, 2, 3, 4}, 1, 4, 1))
		captured := c.TakeSnapshots()
		whole := c.Snapshot(0)
		defer whole.Close()
		if !c.Restore(nil, 2) {
			t.Fatal("could not rewind")
		}
		c.Append(mlx.FromValues([]float32{30, 40}, 1, 2, 1))
		mlx.Eval(c.State()...)
		if got := c.State()[0].Floats(); !slices.Equal(got, []float32{1, 2, 30, 40}) {
			t.Fatalf("branch = %v", got)
		}
		c.Free()
		if !c.Restore(captured[0], 2) || !c.Restore(captured[1], 3) || !c.Restore(whole, 4) {
			t.Fatal("could not restore captured prefix")
		}
		for _, s := range captured {
			s.Close()
		}
		mlx.Eval(c.State()...)
		if got := c.State()[0].Floats(); !slices.Equal(got, []float32{1, 2, 3, 4}) {
			t.Fatalf("restored original = %v", got)
		}
		left, right := c.Split(c.Snapshot(0), 1)
		if left.Size() < 16 || right.Size() < 16 {
			t.Fatal("snapshot view undercounts backing storage")
		}
		joined := c.Merge(left, right)
		defer joined.Close()
		c.Free()
		if !c.Restore(joined, 4) || c.Restore(nil, 5) || c.Restore(nil, -1) {
			t.Fatal("incorrect restore bounds")
		}
		mlx.Eval(c.State()...)
		if got := c.State()[0].Floats(); !slices.Equal(got, []float32{1, 2, 3, 4}) {
			t.Fatalf("merged original = %v", got)
		}
	})
}
