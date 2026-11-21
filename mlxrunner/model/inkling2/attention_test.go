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

func TestAttentionRoundsAfterGate(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			bf16 := func(n float32) *mlx.Array {
				return mlx.FromValues([]float32{n}, 1, 1, 1, 1).AsType(mlx.DTypeBFloat16)
			}
			const width = 128
			q := mlx.Zeros(mlx.DTypeBFloat16, 1, 1, 1, width)
			for _, keys := range []int{1, 1024} {
				b := &batch.Batch{
					InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 1), SeqOffsets: []int32{int32(keys - 1)}, SeqQueryLens: []int32{1},
				}
				k := mlx.Zeros(mlx.DTypeBFloat16, 1, 1, keys, width)
				values := make([]float32, keys*width)
				for i := (keys - 1) * width; i < len(values); i++ {
					values[i] = 8
				}
				v := mlx.FromValues(values, 1, 1, keys, width).AsType(mlx.DTypeBFloat16)
				store := cache.NewKVCache()
				histories := []*nn.KVHistory{store.Update(b, k, v)}
				if keys == 1 {
					histories = append(histories, nil)
				}
				for _, tc := range []struct {
					name             string
					bias, gate, want float32
				}{
					{"attention output", 0, 2, 6.21875},
					{"relative bias and sink", 1.0 / 128, 0, 3.53125},
				} {
					// Only the last key contributes at BF16 output precision, giving
					// 8 * sigmoid(bias + 2) * sigmoid(gate), rounded after the gate.
					bias := make([]float32, keys)
					bias[0] = tc.bias
					for i := 1; i < keys; i++ {
						bias[i] = -96
					}
					table := mlx.FromValues(bias, 1, 1, 1, keys).AsType(mlx.DTypeBFloat16)
					for _, history := range histories {
						attended := relativeSDPA(b, q, k, v, table, bf16(-2), history, 0, 1)
						out := gateAttention(attended, bf16(tc.gate), mlx.DTypeBFloat16)
						if got := out.AsType(mlx.DTypeFloat32).Floats()[0]; got != tc.want {
							t.Errorf("%s (%d keys, cached %t): gated output = %v, want %v", tc.name, keys, history != nil, got, tc.want)
						}
					}
				}
				store.Free()
			}
		})
	})
}

func TestRelativeAttentionWithTokenDependentSinks(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			b := &batch.Batch{
				InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, 2),
				SeqOffsets:   []int32{0},
				SeqQueryLens: []int32{2},
			}
			qk := mlx.Zeros(mlx.DTypeFloat32, 1, 1, 2, 1)
			values := mlx.FromValues([]float32{4, 10}, 1, 1, 2, 1)
			relative := mlx.FromValues([]float32{math.Ln2, math.Ln2}, 1, 1, 2, 1)
			sink := mlx.FromValues([]float32{0, float32(math.Log(3))}, 1, 1, 2, 1)
			// The dot product scale does not affect the relative bias or sinks.
			got := relativeSDPA(b, qk, qk, values, relative, sink, nil, 0, 0.5).Floats()
			want := []float64{8.0 / 3, 4}
			if len(got) != len(want) {
				t.Fatalf("attention produced %d values, want %d", len(got), len(want))
			}
			for i := range want {
				if !(math.Abs(float64(got[i])-want[i]) <= 1e-5) {
					t.Errorf("query %d: attention = %g, want %g", i, got[i], want[i])
				}
			}
		})
	})
}

func TestRelativeAttentionMatchesSerialAfterCacheRestore(t *testing.T) {
	for _, tc := range []struct {
		name   string
		window int
	}{{"full", 0}, {"sliding", 1024}} {
		t.Run(tc.name, func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.Scoped(func() {
					const (
						prefix   = 1037
						block    = 8
						accepted = 3
					)
					rng := rand.New(rand.NewPCG(5, 19))
					values := func(heads, tokens, width int) *mlx.Array {
						data := make([]float32, heads*tokens*width)
						for i := range data {
							data[i] = float32(rng.NormFloat64())
						}
						return mlx.FromValues(data, 1, heads, tokens, width).AsType(mlx.DTypeBFloat16)
					}
					q := values(8, block, 64)
					k, v := values(2, prefix+block, 64), values(2, prefix+block, 64)
					relative, sink := values(8, block, 16), values(8, block, 1)
					querySlice := func(x *mlx.Array, start, end int) *mlx.Array {
						return x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
					}
					batchOf := func(start, length int) *batch.Batch {
						return &batch.Batch{
							InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, length), SeqOffsets: []int32{int32(start)}, SeqQueryLens: []int32{int32(length)},
						}
					}
					prefill := func() cache.Cache {
						var c cache.Cache
						if tc.window > 0 {
							c = cache.NewOrderedWindowKVCache(tc.window)
						} else {
							c = cache.NewKVCache()
						}
						c.(cache.Attention).Update(batchOf(0, prefix), querySlice(k, 0, prefix), querySlice(v, 0, prefix))
						return c
					}
					decode := func(c cache.Cache, start, end int) *mlx.Array {
						b := batchOf(prefix+start, end-start)
						history := c.(cache.Attention).Update(b, querySlice(k, prefix+start, prefix+end), querySlice(v, prefix+start, prefix+end))
						return relativeSDPA(b, querySlice(q, start, end), nil, nil, querySlice(relative, start, end), querySlice(sink, start, end), history, tc.window, 0.125)
					}
					serial := prefill()
					defer serial.Free()
					want := make([]*mlx.Array, block)
					for i := range block {
						want[i] = decode(serial, i, i+1)
					}
					blockCache := prefill()
					defer blockCache.Free()
					blockCache.PrepareSnapshots([]int{prefix + accepted})
					logicalValues := func(x *mlx.Array) []float32 {
						return mlx.Contiguous(x.AsType(mlx.DTypeFloat32), false).Floats()
					}
					got := logicalValues(decode(blockCache, 0, block))
					if !slices.Equal(got, logicalValues(mlx.Concatenate(want, 2))) {
						t.Error("block attention differs from serial decode")
					}
					snapshots := blockCache.TakeSnapshots()
					if len(snapshots) != 1 || snapshots[0] == nil {
						t.Fatal("speculative block did not capture its rollback boundary")
					}
					defer snapshots[0].Close()
					if !blockCache.Restore(snapshots[0], prefix+accepted) {
						t.Fatal("cache could not restore the accepted prefix")
					}
					got = logicalValues(decode(blockCache, accepted, block))
					if !slices.Equal(got, logicalValues(mlx.Concatenate(want[accepted:], 2))) {
						t.Error("attention after rollback differs from serial decode")
					}
				})
			})
		})
	}
}

func TestRelativeAttentionKeepsDistancesAfterWindowRestore(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			const window = 4
			run := func(c *cache.RotatingKVCache, start, length int) []float32 {
				b := &batch.Batch{
					InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, length),
					SeqOffsets:   []int32{int32(start)},
					SeqQueryLens: []int32{int32(length)},
				}
				q := mlx.Zeros(mlx.DTypeFloat32, 1, 1, length, 1)
				values := mlx.Arange(float64(start+1), float64(start+length+1), 1, mlx.DTypeFloat32).
					Reshape(1, 1, length, 1)
				row := mlx.FromValues([]float32{0, -0.3, -0.8, -1.7}, 1, 1, 1, window)
				table := mlx.BroadcastTo(row, 1, 1, int32(length), window)
				sinks := mlx.Zeros(mlx.DTypeFloat32, 1, 1, length, 1)
				history := c.Update(b, q, values)
				return relativeSDPA(b, q, nil, nil, table, sinks, history, window, 1).Floats()
			}

			cold := cache.NewRotatingKVCache(window)
			defer cold.Free()
			cold.PrepareSnapshots([]int{6})
			prefill := run(cold, 0, 8)
			snapshots := cold.TakeSnapshots()
			if len(snapshots) != 1 || snapshots[0] == nil {
				t.Fatal("expected a snapshot in the oversize prefill")
			}
			defer snapshots[0].Close()
			want := append(prefill[6:], run(cold, 8, 1)...)

			warm := cache.NewRotatingKVCache(window)
			defer warm.Free()
			if !warm.Restore(snapshots[0], 6) {
				t.Fatal("failed to restore the prefill snapshot")
			}
			got := append(run(warm, 6, 2), run(warm, 8, 1)...)
			if len(want) != 3 || len(got) != len(want) {
				t.Fatalf("expected three attention values, got %d restored and %d cold", len(got), len(want))
			}
			for i := range want {
				if !(math.Abs(float64(got[i]-want[i])) <= 1e-5) {
					t.Errorf("position %d: restored attention = %g, cold attention = %g", 6+i, got[i], want[i])
				}
			}
		})
	})
}

func TestRelativePositionDecodeBias(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			for _, extent := range []int{16, 128, 1024} {
				// Select one query from a query-major table, as in speculative verification.
				table := mlx.Arange(0, float64(3*24*extent), 1, mlx.DTypeFloat32).AsType(mlx.DTypeBFloat16).Reshape(1, 3, 24, extent).Transpose(0, 2, 1, 3).Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(1, 2), mlx.Slice())
				sink := mlx.Arange(0, 3*24, 1, mlx.DTypeFloat32).AsType(mlx.DTypeBFloat16).Reshape(1, 3, 24, 1).Transpose(0, 2, 1, 3).Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(1, 2), mlx.Slice())
				for _, keys := range []int{1, 17, 127, 1023} {
					got := mlx.RelativePositionDecodeBias(table, sink, keys)
					if got == nil {
						t.Skip("requires a qualified relative-attention Metal device")
					}
					want := relativeBiasGraph(table, sink, keys)
					if !slices.Equal(got.Floats(), want.Floats()) {
						t.Fatalf("extent=%d keys=%d: decode shader differs from graph", extent, keys)
					}
				}
			}
		})
	})
}

func TestRelativeAttentionInputsMatchGraph(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		const heads, kvHeads, width, rel, distances = 24, 8, 128, 16, 1024
		pattern := func(dtype mlx.DType, scale float32, seed int, shape ...int) *mlx.Array {
			n := 1
			for _, d := range shape {
				n *= d
			}
			values := make([]float32, n)
			for i := range values {
				values[i] = float32((i*seed)%997-498) * scale
			}
			return mlx.FromValues(values, shape...).AsType(dtype)
		}
		close := func(name string, got, want *mlx.Array) {
			g := mlx.Contiguous(got.AsType(mlx.DTypeFloat32), false).Floats()
			w := mlx.Contiguous(want.AsType(mlx.DTypeFloat32), false).Floats()
			var peak float64
			for _, v := range w {
				peak = max(peak, math.Abs(float64(v)))
			}
			for i := range w {
				// Sums of squares and the table product round in another order.
				if math.Abs(float64(g[i]-w[i])) > peak/64 {
					t.Fatalf("%s element %d = %g, want %g", name, i, g[i], w[i])
				}
			}
		}
		for _, tokens := range []int{1, 9} {
			for _, temperature := range []bool{false, true} {
				mlx.Scoped(func() {
					stride := heads*(2*width+rel+1) + 32
					source := pattern(mlx.DTypeBFloat16, 0.01, 31, 1, tokens, stride)
					k := pattern(mlx.DTypeBFloat16, 0.02, 17, 1, tokens, kvHeads*width)
					v := pattern(mlx.DTypeBFloat16, 0.03, 23, 1, tokens, kvHeads*width)
					queryScale := pattern(mlx.DTypeBFloat16, 0.002, 7, heads, width)
					relativeScale := pattern(mlx.DTypeBFloat16, 0.003, 13, heads, rel)
					projection := pattern(mlx.DTypeBFloat16, 0.001, 19, rel, distances)
					var tau, tokenTau *mlx.Array
					if temperature {
						tokenTau = mlx.AddScalar(mlx.MulScalar(mlx.Arange(0, float64(tokens), 1, mlx.DTypeFloat32), 0.01), 1).Reshape(1, tokens)
						tau = tokenTau.Reshape(1, tokens, 1, 1)
					}
					q, keys, values, table, sink := mlx.RelativeAttentionInputs(source, k, v, queryScale, relativeScale, projection,
						tokenTau, heads, kvHeads, 22.6, 8, 1e-6)
					if q == nil {
						t.Fatal("kernel did not run")
					}
					cols := func(a, b int) *mlx.Array { return source.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(a, b)) }
					wantQ := clipL2(cols(0, heads*width).Reshape(1, tokens, heads, width), queryScale, tau, 22.6).Transpose(0, 2, 1, 3)
					r := clipL2(cols(heads*width, heads*(width+rel)).Reshape(1, tokens, heads, rel), relativeScale, tau, 8)
					close("q", q, wantQ)
					close("k", keys, clipL2(k.Reshape(1, tokens, kvHeads, width), nil, nil, 22.6).Transpose(0, 2, 1, 3))
					close("v", values, mlx.RMSNormFn(v.Reshape(1, tokens, kvHeads, width), nil, 1e-6).Transpose(0, 2, 1, 3))
					close("table", table, r.Matmul(projection).Transpose(0, 2, 1, 3))
					start := heads * (2*width + rel)
					close("sink", sink, cols(start, start+heads).Transpose(0, 2, 1).ExpandDims(-1))
				})
			}
		}
	})
}
