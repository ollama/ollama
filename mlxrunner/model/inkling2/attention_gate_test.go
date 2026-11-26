package inkling2

import (
	"math"
	"math/rand/v2"
	"slices"
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

// relativeAttentionQualified reports whether this device runs the split
// attention kernels; others, such as paravirtual GPUs, take the stock graph.
func relativeAttentionQualified() bool {
	if !mlx.MetalIsAvailable() {
		return false
	}
	var launched bool
	mlx.Scoped(func() {
		bf16 := mlx.DTypeBFloat16
		q := mlx.Zeros(mlx.DTypeFloat32, 1, gateTestHeads, 1, gateTestWidth)
		k := mlx.Zeros(bf16, 1, gateTestKVHeads, 1100, gateTestWidth)
		table := mlx.Zeros(bf16, 1, gateTestHeads, 1, 1024)
		sink := mlx.Zeros(bf16, 1, gateTestHeads, 1, 1)
		launched = mlx.RelativePositionAttention(q, k, k, table, sink, 0, 0.125) != nil
	})
	return launched
}

// For two requests, the gate folded into the attention merge must give the
// same BF16 bits as relativeSDPA followed by the graph gate (tolerance zero):
// local and global caches and every short block width, with the gate a column
// view of the stacked projection as in forward.
func TestGatedShortAttentionMatchesStockGate(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !relativeAttentionQualified() {
			t.Skip("requires a qualified relative-attention Metal device")
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

// A padded row, a block past shortTargetRows or a history the kernel does not
// serve keeps the separate attention route.
func TestGatedShortAttentionDeclines(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		// Devices without the split kernels must decline every case.
		qualified := relativeAttentionQualified()
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
			{"single request", []int{1100}, 4, 4, 1024, false, true},
			{"single padded request", []int{1100}, 4, 3, 1024, false, false},
			{"padded rows", []int{1100, 1137}, 4, 3, 1024, false, false},
			{"long block", []int{1100, 1137}, shortTargetRows + 1, shortTargetRows + 1, 1024, false, false},
			{"short history", []int{500, 537}, 4, 4, 0, false, true},
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
				want := tc.launch && qualified
				if got := gatedShortAttention(b, q, table, sink, gate, history, tc.window, 0.125); (got != nil) != want {
					t.Fatalf("%s: fused gate launched %t, want %t", tc.name, got != nil, want)
				}
			})
		}
	})
}

// A verification block whose first query sees fewer keys than the split
// kernels serve and whose last query sees more must still score every query
// bit for bit as single-token decode does, for global and windowed caches.
func TestRelativeSDPABlockMatchesDecodeAcrossThreshold(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !relativeAttentionQualified() {
			t.Skip("requires a qualified relative-attention Metal device")
		}
		const queries, prefix = 4, 1021
		rng := rand.New(rand.NewPCG(3, 5))
		normal := func(dtype mlx.DType, shape ...int) *mlx.Array {
			n := 1
			for _, d := range shape {
				n *= d
			}
			data := make([]float32, n)
			for i := range data {
				data[i] = float32(rng.NormFloat64())
			}
			return mlx.FromValues(data, shape...).AsType(dtype)
		}
		bf16 := mlx.DTypeBFloat16
		scale := float32(1 / math.Sqrt(gateTestWidth))
		for _, window := range []int{0, 1024} {
			mlx.Scoped(func() {
				k := normal(bf16, 1, gateTestKVHeads, prefix+queries, gateTestWidth)
				v := normal(bf16, 1, gateTestKVHeads, prefix+queries, gateTestWidth)
				q := normal(mlx.DTypeFloat32, 1, gateTestHeads, queries, gateTestWidth)
				table := normal(bf16, 1, gateTestHeads, queries, 1024)
				sink := normal(bf16, 1, gateTestHeads, queries, 1)
				span := func(x *mlx.Array, from, to int) *mlx.Array {
					return x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(from, to), mlx.Slice())
				}
				rows := func(from, to int) *batch.Batch {
					return &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, to-from), SeqOffsets: []int32{int32(from)}, SeqQueryLens: []int32{int32(to - from)}}
				}
				// decode writes the first n positions and returns the history of the last.
				decode := func(n int) (*nn.KVHistory, cache.Cache) {
					var c cache.Cache = cache.NewKVCache()
					if window > 0 {
						c = cache.NewOrderedWindowKVCache(window)
					}
					attention := c.(cache.Attention)
					attention.Update(rows(0, n-1), span(k, 0, n-1), span(v, 0, n-1))
					return attention.Update(rows(n-1, n), span(k, n-1, n), span(v, n-1, n)), c
				}
				var c cache.Cache = cache.NewKVCache()
				if window > 0 {
					c = cache.NewOrderedWindowKVCache(window)
				}
				defer c.Free()
				attention := c.(cache.Attention)
				attention.Update(rows(0, prefix), span(k, 0, prefix), span(v, 0, prefix))
				history := attention.Update(rows(prefix, prefix+queries), span(k, prefix, prefix+queries), span(v, prefix, prefix+queries))
				block := relativeSDPA(rows(prefix, prefix+queries), q, nil, nil, table, sink, history, window, scale)
				for i := range queries {
					h, single := decode(prefix + i + 1)
					want := relativeSDPA(rows(prefix+i, prefix+i+1), span(q, i, i+1), nil, nil, span(table, i, i+1), span(sink, i, i+1), h, window, scale)
					got := mlx.Contiguous(span(block, i, i+1), false).Floats()
					wantValues := mlx.Contiguous(want, false).Floats()
					for j := range wantValues {
						if math.Float32bits(got[j]) != math.Float32bits(wantValues[j]) {
							t.Fatalf("window %d: query %d value %d is %v, single-token decode gives %v", window, i, j, got[j], wantValues[j])
						}
					}
					single.Free()
				}
			})
		}
	})
}

// A rectangular single-request history uses the fused merge and gate too:
// its BF16 output matches relativeSDPA followed by the graph gate exactly.
func TestGatedShortAttentionSingleRequestMatchesStockGate(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !relativeAttentionQualified() {
			t.Skip("requires a qualified relative-attention Metal device")
		}
		const heads, width, distances = gateTestHeads, gateTestWidth, 1024
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
					b, history, caches := gatedTestHistory([]int{1100}, queries, int32(queries), window, kv)
					defer closeCaches(caches)
					q := normal(0.3, mlx.DTypeFloat32, 1, heads, queries, width)
					table := normal(1, bf16, 1, heads, queries, distances)
					sink := normal(1, bf16, 1, heads, queries, 1)
					projection := normal(4, bf16, 1, queries, 3*heads*width)
					gate := columns(projection, heads*width, 2*heads*width)
					got := gatedShortAttention(b, q, table, sink, gate, history, window, scale)
					if got == nil {
						t.Fatalf("window %d, %d queries: fused gate declined", window, queries)
					}
					attended := relativeSDPA(b, q, nil, nil, table, sink, history, window, scale)
					want := gateAttention(attended.Transpose(0, 2, 1, 3).Reshape(1, queries, heads*width), gate, bf16)
					gotBits := mlx.Contiguous(got.View(mlx.DTypeUint16).AsType(mlx.DTypeInt32), false).Ints()
					wantBits := mlx.Contiguous(want.View(mlx.DTypeUint16).AsType(mlx.DTypeInt32), false).Ints()
					if !slices.Equal(gotBits, wantBits) {
						t.Fatalf("window %d, %d queries: fused gate differs from graph", window, queries)
					}
				})
			}
		}
	})
}
