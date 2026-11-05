package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
)

func TestRunnerCloseReleasesGenerationCache(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		activeMemory := func() int {
			mlx.DefaultStream().Synchronize()
			mlx.ClearCache()
			return mlx.ActiveMemory()
		}
		baseline := activeMemory()
		pc := newPrefixCache([]cache.Cache{cache.NewKVCache(), nil, cache.NewRotatingKVCache(4)})
		defer pc.close()
		r := &Runner{cache: pc}

		for _, inputs := range [][]int32{{1, 2, 3, 4}, {5, 6, 7, 8}} {
			mlx.Scoped(func() {
				session := pc.begin(inputs, nil)
				defer session.close()
				n := len(session.remaining)
				b := &batch.Batch{
					InputIDs:     mlx.FromValues(session.remaining, 1, n),
					SeqOffsets:   []int32{int32(pc.minCacheOffset())},
					SeqQueryLens: []int32{int32(n)},
				}
				keys := mlx.Zeros(mlx.DTypeFloat32, 1, 1, n, 8)
				values := mlx.Zeros(mlx.DTypeFloat32, 1, 1, n, 8)
				for _, kv := range pc.caches {
					if kv != nil {
						kv.(cache.Attention).Update(b, keys, values)
						mlx.Eval(kv.State()...)
					}
				}
			})
		}
		if pc.minCacheOffset() == 0 || pc.pagedOutBytes == 0 || activeMemory() <= baseline {
			t.Fatal("requests did not populate live and paged-out cache state")
		}

		r.Close()
		if r.cache != nil {
			t.Error("closed runner retains generation cache")
		}
		if pc.root != nil || len(pc.activePath) != 0 || pc.pagedOutBytes != 0 {
			t.Error("closed runner retains prefix snapshots")
		}
		for i, kv := range pc.caches {
			if kv != nil && kv.Offset() != 0 {
				t.Errorf("cache %d retains offset %d after close", i, kv.Offset())
			}
		}
		if got := activeMemory(); got != baseline {
			t.Errorf("close retained %d active bytes above baseline %d", got-baseline, baseline)
		}
		r.Close()
		if got := activeMemory(); got != baseline {
			t.Errorf("repeated close retained %d active bytes above baseline %d", got-baseline, baseline)
		}
	})
}

func TestRunnerCloseEmpty(t *testing.T) {
	for _, r := range []*Runner{{}, {cache: newPrefixCache(nil)}} {
		r.Close()
		r.Close()
		if r.cache != nil {
			t.Error("closed runner retains empty generation cache")
		}
	}
}
