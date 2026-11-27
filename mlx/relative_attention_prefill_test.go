package mlx

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestRelativePositionDecodeBiasCUDA(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !CUDAIsAvailable() {
			t.Skip("requires CUDA")
		}
		for _, tc := range []struct{ heads, keys, distances int }{
			{1, 8, 32},
			{1, 64, 32},
			{1, 256, 1024},
			{2, 8, 1024},
			{3, 64, 32},
			{4, 256, 1024},
		} {
			Scoped(func() {
				table, sink := decodeRelativeAttentionInputs(tc.heads, tc.distances)
				got := RelativePositionDecodeBias(table, sink, tc.keys)
				if got == nil {
					t.Fatalf("CUDA decode bias declined heads=%d keys=%d distances=%d", tc.heads, tc.keys, tc.distances)
				}
				want := relativePositionDecodeBiasReference(table, sink, tc.keys)
				if err := requireExact(fmt.Sprintf("decode heads=%d keys=%d distances=%d", tc.heads, tc.keys, tc.distances), got, want); err != nil {
					t.Fatal(err)
				}
			})
		}
	})
}

// decodeRelativeAttentionInputs builds BF16 decode operands: a query-major
// table [1, heads, 1, distances] and one sink per head [1, heads, 1, 1].
func decodeRelativeAttentionInputs(heads, distances int) (table, sink *Array) {
	table = patternArray(DTypeBFloat16, []int{1, heads, 1, distances}, 0, 0.0078125, 9, 127)
	sink = patternArray(DTypeBFloat16, []int{1, heads, 1, 1}, 0, 0.1, 13, 67)
	return table, sink
}

// relativePositionDecodeBiasReference mirrors the model graph fallback
// (relativeBiasGraph) for a single final query, including its FP32 widening.
func relativePositionDecodeBiasReference(table, sink *Array, keys int) *Array {
	queries := table.Dim(2)
	shiftedQueries := Arange(float64(keys-queries), float64(keys), 1, DTypeInt32).Reshape(1, 1, queries, 1)
	keyPositions := Arange(0, float64(keys), 1, DTypeInt32)
	distance := shiftedQueries.Subtract(keyPositions)
	index := Clip(distance, FromValue(0), FromValue(table.Dim(-1)-1))
	bias := table.TakeAlongAxis(index, -1).AsType(DTypeFloat32)
	bias = Where(index.Equal(distance), bias, FromValue(float32(0)))
	return bias.Subtract(sink.AsType(DTypeFloat32))
}

func TestRelativeAttentionPrefill(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		useCUDA := CUDAIsAvailable()
		if !useCUDA && !MetalIsAvailable() {
			t.Skip("requires Metal or CUDA")
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
					gotC, wantBias := Contiguous(got, false), bias.AsType(dtype)
					if useCUDA {
						// The tiled path keeps the stored BF16 keys and values; compare
						// against the same dtypes so the check isolates tiling/mask
						// correctness rather than BF16-versus-FP32 SDPA rounding.
						want := FastScaledDotProductAttentionWithSinks(query, k, v, scale, FastSDPANoWindow, "", wantBias, Zeros(dtype, heads))
						if err := requireClose("tiled prefill", gotC, Contiguous(want, false), 2e-4); err != nil {
							t.Fatalf("prefill %d/%d window %d dtype %v: %v", queries, keys, window, dtype, err)
						}
					} else {
						want := FastScaledDotProductAttentionWithSinks(query, k.AsType(dtype), v.AsType(dtype), scale, FastSDPANoWindow, "", wantBias, Zeros(dtype, heads))
						if err := requireExact("tiled prefill", gotC, Contiguous(want, false)); err != nil {
							t.Fatalf("prefill %d/%d window %d dtype %v: %v", queries, keys, window, dtype, err)
						}
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
		// A verification-width block takes the split pass, not prefill. GPUs
		// without the split kernels, such as Intel or virtual Macs, decline it
		// so the caller keeps the stock graph.
		q, k, v, table, sink = relativeAttentionInputs(8, 32, 24, 8, 128, 1024)
		kernel, parts := relativeSplit(8, 32)
		got := RelativePositionAttention(q, k, v, table, sink, 0, scale)
		if kernel == nil {
			if got != nil {
				t.Fatal("verification width ran without the split kernels")
			}
			return
		}
		want := relativeAttentionSplitMerge(kernel, parts, q, k, v, table, sink, 0, scale)
		if got == nil || want == nil || requireClose("verification width", got, want, 0) != nil {
			t.Fatal("verification width selected prefill")
		}
	})
}
