package cache

import (
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestEstimateMemory(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		b := newKVBatch(0, 1)
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		full := NewKVCache()
		window := NewRotatingKVCache(4)
		full.Update(b, k, k)
		window.Update(b, k, k)
		recurrent := NewRecurrentCache(3, 8, 2, 4, 8)
		recurrent.Get(b, mlx.DTypeFloat16)
		recurrent.Put(b, []*mlx.Array{recurrent.convState}, []*mlx.Array{recurrent.deltaState})
		defer full.Free()
		defer window.Free()
		defer recurrent.Free()
		for _, tc := range []struct {
			name                string
			cache               Cache
			tokens, checkpoints int
			want                int64
		}{
			{"full", full, 1024, 2, 2 * 1024 * 128},
			{"capacity_rounding", full, 257, 0, 2 * 512 * 128},
			{"window", window, 1024, 2, 4 * 4 * 128},
			{"short_window", window, 2, 0, 2 * 2 * 128},
			{"recurrent", recurrent, 1024, 2, 4 * (3*8*2 + 2*4*8*4)},
			{"shared_layer", nil, 1024, 2, 0},
		} {
			got, ok := EstimateMemory(tc.cache, tc.tokens, tc.checkpoints)
			if !ok || got != tc.want {
				t.Errorf("%s: bytes=%d known=%v, want %d", tc.name, got, ok, tc.want)
			}
		}
		if _, ok := EstimateMemory(NewKVCache(), 1024, 2); ok {
			t.Fatal("uninitialized attention cache was treated as known")
		}
		if _, ok := EstimateMemory(full, -1, 2); ok {
			t.Fatal("invalid context was treated as known")
		}
	})
}
