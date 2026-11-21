package inkling2

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestRelativePrefillMixedHistories(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		const width = 320
		for _, tc := range []struct{ window, prefix, paddedLive int }{
			{0, 155, 197},
			{1024, 155, 0},
			{1024, 155, 1},
			{1024, 1211, 8},
			{1024, 155, 9},
			{1024, 155, 197},
			{1024, 1211, 197},
		} {
			window := tc.window
			mlx.Scoped(func() {
				b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 2, width), SeqOffsets: []int32{1211, int32(tc.prefix)}, SeqQueryLens: []int32{width, int32(tc.paddedLive)}}
				values := func(rows, heads, steps, channels int, factor float32) *mlx.Array {
					data := mlx.Arange(0, float64(rows*heads*steps*channels), 1, mlx.DTypeFloat32)
					return mlx.Sin(mlx.MulScalar(data, factor)).AsType(mlx.DTypeBFloat16).Reshape(rows, steps, heads, channels).Transpose(0, 2, 1, 3)
				}
				q, table, sink := values(2, 24, width, 128, .017), values(2, 24, width, 1024, -.009), values(2, 24, width, 1, .005)
				histories := make([]*nn.KVHistory, 2)
				expected := make([]*mlx.Array, 2)
				scale := float32(1 / math.Sqrt(128))
				for i := range histories {
					var store interface {
						cache.Attention
						Free()
					} = cache.NewKVCache()
					if window != 0 {
						store = cache.NewOrderedWindowKVCache(window)
					}
					defer store.Free()
					prefix := int(b.SeqOffsets[i])
					row := b.Row(i)
					first := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, prefix), SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(prefix)}}
					prefixK, prefixV := values(1, 8, prefix, 128, .014), values(1, 8, prefix, 128, .023)
					newK, newV := values(1, 8, width, 128, -.008), values(1, 8, width, 128, .031)
					store.Update(first, prefixK, prefixV)
					h := store.Update(row, newK, newV)
					histories[i] = h
					pick := func(x *mlx.Array) *mlx.Array {
						return x.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(), mlx.Slice())
					}
					if window == 0 {
						expected[i] = relativeWindowSDPA(row, pick(q).AsType(mlx.DTypeFloat32), pick(table), pick(sink), h.K().Dim(2), 0, scale, nn.WithKVHistory(h.AsType(mlx.DTypeFloat32)))
						continue
					}
					// Build a separate history from only the live tokens. This checks both
					// the absolute key positions after eviction and zeroed padding.
					live := int(row.SeqQueryLens[0])
					if live == 0 {
						expected[i] = mlx.Zeros(mlx.DTypeFloat32, 1, 24, width, 128)
						continue
					}
					trim := func(x *mlx.Array) *mlx.Array {
						return x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, live), mlx.Slice())
					}
					single := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, live), SeqOffsets: []int32{int32(prefix)}, SeqQueryLens: []int32{int32(live)}}
					serial := cache.NewOrderedWindowKVCache(window)
					defer serial.Free()
					serial.Update(first, prefixK, prefixV)
					h = serial.Update(single, trim(newK), trim(newV))
					out := relativeSDPA(single, trim(pick(q)), nil, nil, trim(pick(table)), trim(pick(sink)), h, window, scale)
					expected[i] = mlx.PadConstant(out, []int{2}, []int{0}, []int{width - live})
				}
				got := relativeSDPA(b, q, nil, nil, table, sink, nn.NewRowKVHistory(histories), window, scale)
				want := mlx.Concatenate(expected, 0)
				if got.DType() != mlx.DTypeFloat32 || want.DType() != mlx.DTypeFloat32 {
					t.Fatal("relative attention must preserve FP32 until the output gate")
				}
				same := got.View(mlx.DTypeUint32).Equal(want.View(mlx.DTypeUint32)).AsType(mlx.DTypeInt32).Reshape(-1).SumAxis(0, false).Int()
				if int(same) != want.Size() {
					t.Fatalf("window %d with %d live tokens: %d of %d mixed-row FP32 entries differ", window, tc.paddedLive, want.Size()-int(same), want.Size())
				}
			})
		}
	})
}
