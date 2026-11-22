package inkling2

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

const gateTestHeads, gateTestKVHeads, gateTestWidth = 24, 8, 128

// gatedTestHistory writes a prefix and then a block of queries rows (live of
// them real) into a fresh cache per request and returns the block's batch and
// history. The caller frees the returned caches.
func gatedTestHistory(prefixes []int, queries int, live int32, window int, values func(shape ...int) *mlx.Array) (*batch.Batch, *nn.KVHistory, []cache.Cache) {
	rows := len(prefixes)
	caches := make([]cache.Cache, rows)
	histories := make([]*nn.KVHistory, rows)
	offsets, lens := make([]int32, rows), make([]int32, rows)
	for r, prefix := range prefixes {
		if window > 0 {
			caches[r] = cache.NewOrderedWindowKVCache(window)
		} else {
			caches[r] = cache.NewKVCache()
		}
		attention := caches[r].(cache.Attention)
		first := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, prefix), SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(prefix)}}
		attention.Update(first, values(1, gateTestKVHeads, prefix, gateTestWidth), values(1, gateTestKVHeads, prefix, gateTestWidth))
		block := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, queries), SeqOffsets: []int32{int32(prefix)}, SeqQueryLens: []int32{live}}
		histories[r] = attention.Update(block, values(1, gateTestKVHeads, queries, gateTestWidth), values(1, gateTestKVHeads, queries, gateTestWidth))
		offsets[r], lens[r] = int32(prefix), live
	}
	history := histories[0]
	if rows > 1 {
		history = nn.NewRowKVHistory(histories)
	}
	return &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, rows, queries), SeqOffsets: offsets, SeqQueryLens: lens}, history, caches
}

func closeCaches(caches []cache.Cache) {
	for _, c := range caches {
		c.Free()
	}
}

// For two requests, the gate folded into the attention merge must give the
// same BF16 bits as relativeSDPA followed by the graph gate (tolerance zero):
// local and global caches and every short block width, with the gate a column
// view of the stacked projection as in forward.
func TestGatedShortAttentionMatchesStockGate(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		const heads, width, distances, rows = gateTestHeads, gateTestWidth, 1024, 2
		rng := rand.New(rand.NewPCG(7, 23))
		normal := func(scale float64, dtype mlx.DType, shape ...int) *mlx.Array {
			n := 1
			for _, d := range shape {
				n *= d
			}
			data := make([]float32, n)
			for i := range data {
				data[i] = float32(rng.NormFloat64() * scale)
			}
			return mlx.FromValues(data, shape...).AsType(dtype)
		}
		bf16 := mlx.DTypeBFloat16
		kv := func(shape ...int) *mlx.Array { return normal(1, bf16, shape...) }
		scale := float32(1 / math.Sqrt(width))
		for _, window := range []int{1024, 0} {
			for _, queries := range []int{1, 3, 4, 8} {
				mlx.Scoped(func() {
					b, history, caches := gatedTestHistory([]int{1100, 1137}, queries, int32(queries), window, kv)
					defer closeCaches(caches)
					q := normal(0.3, mlx.DTypeFloat32, rows, heads, queries, width)
					table := normal(1, bf16, rows, heads, queries, distances)
					sink := normal(1, bf16, rows, heads, queries, 1)
					// Gates reach the saturated ends of the sigmoid.
					projection := normal(4, bf16, rows, queries, 3*heads*width)
					gate := columns(projection, heads*width, 2*heads*width)

					got := gatedShortAttention(b, q, table, sink, gate, history, window, scale)
					if got == nil {
						t.Fatalf("window %d, %d queries: fused gate declined", window, queries)
					}
					attended := relativeSDPA(b, q, nil, nil, table, sink, history, window, scale)
					want := gateAttention(attended.Transpose(0, 2, 1, 3).Reshape(rows, queries, heads*width), gate, bf16)
					gotBits := mlx.Contiguous(got.View(mlx.DTypeUint16).AsType(mlx.DTypeInt32), false).Ints()
					wantBits := mlx.Contiguous(want.View(mlx.DTypeUint16).AsType(mlx.DTypeInt32), false).Ints()
					if len(gotBits) != len(wantBits) {
						t.Fatalf("fused gate produced %d values, want %d", len(gotBits), len(wantBits))
					}
					for i := range wantBits {
						if gotBits[i] != wantBits[i] {
							t.Fatalf("window %d, %d queries: value %d is %#04x, want %#04x", window, queries, i, gotBits[i], wantBits[i])
						}
					}
				})
			}
		}
	})
}

// A single request, a padded row, a block past shortTargetRows or a history
// the kernel does not serve keeps the separate attention route.
func TestGatedShortAttentionDeclines(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		const heads, width = gateTestHeads, gateTestWidth
		bf16 := mlx.DTypeBFloat16
		zeros := func(shape ...int) *mlx.Array { return mlx.Zeros(bf16, shape...) }
		for _, tc := range []struct {
			name           string
			prefixes       []int
			queries        int
			live           int32
			window         int
			withoutHistory bool
			launch         bool
		}{
			{"two requests", []int{1100, 1137}, 4, 4, 1024, false, true},
			{"single request", []int{1100}, 4, 4, 1024, false, false},
			{"padded rows", []int{1100, 1137}, 4, 3, 1024, false, false},
			{"long block", []int{1100, 1137}, shortTargetRows + 1, shortTargetRows + 1, 1024, false, false},
			{"short history", []int{500, 537}, 4, 4, 0, false, false},
			{"no cache", []int{1100, 1137}, 4, 4, 1024, true, false},
		} {
			mlx.Scoped(func() {
				b, history, caches := gatedTestHistory(tc.prefixes, tc.queries, tc.live, tc.window, zeros)
				defer closeCaches(caches)
				if tc.withoutHistory {
					history = nil
				}
				rows := len(tc.prefixes)
				q := mlx.Zeros(mlx.DTypeFloat32, rows, heads, tc.queries, width)
				table := zeros(rows, heads, tc.queries, 1024)
				sink := zeros(rows, heads, tc.queries, 1)
				gate := zeros(rows, tc.queries, heads*width)
				if got := gatedShortAttention(b, q, table, sink, gate, history, tc.window, 0.125); (got != nil) != tc.launch {
					t.Fatalf("%s: fused gate launched %t, want %t", tc.name, got != nil, tc.launch)
				}
			})
		}
	})
}
