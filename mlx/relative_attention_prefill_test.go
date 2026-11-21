package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestRelativeAttentionPrefill(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		if !supportsRelativeAttentionPrefill(kernelDevice()) {
			t.Skip("requires a Metal device with qualified tiled prefill")
		}
		check := func(queries, keys, heads, kvHeads, width, distances, window int) {
			scale := float32(1 / math.Sqrt(float64(width)))
			Scoped(func() {
				q, k, v, table, sink := relativeAttentionInputs(queries+3, keys, heads, kvHeads, width, distances)
				// Crop a tail from strided views to test offsets in all five operands.
				crop := func(x *Array) *Array { return x.Slice(Slice(), Slice(), Slice(3, queries+3), Slice()) }
				q, table, sink = crop(q), crop(table), crop(sink)
				pos := Arange(float64(keys-queries), float64(keys), 1, DTypeInt32).Reshape(1, 1, queries, 1)
				distance := pos.Subtract(Arange(0, float64(keys), 1, DTypeInt32))
				index := Clip(distance, FromValue(0), FromValue(distances-1))
				relative := Where(distance.Equal(index), table.TakeAlongAxis(index, -1).AsType(DTypeFloat32), FromValue(float32(0)))
				bias := relative.Subtract(sink.AsType(DTypeFloat32))
				inf := FromValue(float32(math.Inf(-1)))
				bias = Where(distance.Less(FromValue(0)), inf, bias)
				if window > 0 {
					bias = Where(FromValue(window-1).Less(distance), inf, bias)
				}
				for _, dtype := range []DType{DTypeBFloat16, DTypeFloat32} {
					query := q.AsType(dtype)
					got := RelativePositionAttention(query, k, v, table, sink, window, scale)
					if got == nil || got.DType() != dtype {
						t.Fatalf("prefill did not return dtype %v", dtype)
					}
					want := FastScaledDotProductAttentionWithSinks(query, k.AsType(dtype), v.AsType(dtype), scale, FastSDPANoWindow, "", bias.AsType(dtype), Zeros(dtype, heads))
					if err := requireExact("tiled prefill", Contiguous(got, false), Contiguous(want, false)); err != nil {
						t.Fatalf("prefill %d/%d window %d dtype %v: %v", queries, keys, window, dtype, err)
					}
				}
			})
		}
		for _, tc := range []struct{ queries, keys, window int }{
			{9, 9, 0},
			{63, 84, 1024},
			{64, 1031, 1024},
			{65, 68, 0},
			{257, 257, 1024},
			{319, 1542, 0},
			{320, 1543, 0},
			{400, 1400, 1024},
			{400, 1423, 1024},
			{1023, 1120, 1024},
			{1024, 1120, 1024},
			{1089, 2110, 1024},
		} {
			check(tc.queries, tc.keys, 24, 8, 128, 1024, tc.window)
		}
		// Vary the table and the window independently, including a single
		// learned distance, and exercise MHA and GQA with different head widths.
		check(75, 135, 4, 4, 64, 1, 31)
		check(319, 517, 8, 2, 64, 29, 133)
		check(327, 934, 6, 2, 128, 77, 0)
		check(21, 27, 1, 1, 128, 51, 1)
		const scale = 0.125
		q, k, v, table, sink := relativeAttentionInputs(9, 32, 24, 8, 128, 1024)
		if RelativePositionAttention(q, k, v, table, sink, -1, scale) != nil ||
			RelativePositionAttention(q.AsType(DTypeFloat16), k, v, table, sink, 0, scale) != nil ||
			RelativePositionAttention(q, k.Slice(Slice(), Slice(), Slice(0, 8), Slice()), v, table, sink, 0, scale) != nil ||
			RelativePositionAttention(q, nil, v, table, sink, 0, scale) != nil {
			t.Fatal("unsupported input selected prefill")
		}
		q, k, v, table, sink = relativeAttentionInputs(8, 32, 24, 8, 128, 1024)
		if RelativePositionAttention(q, k, v, table, sink, 0, scale) != nil {
			t.Fatal("verification width selected prefill")
		}
	})
}
