package mlx

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestRelativeAttentionMetalDevices(t *testing.T) {
	for _, tc := range []struct {
		name, arch string
		parts      int
		prefill    bool
	}{
		{"Apple M2 Pro", "applegpu_g14s", 0, true},
		{"Apple M2 Ultra", "applegpu_g14d", 128, true},
		{"Apple M3 Ultra", "applegpu_g15d", 128, true},
		{"Apple M4 Pro", "applegpu_g16s", 64, true},
		{"Apple M5 Pro", "applegpu_g17s", 64, true},
		{"Apple M5 Pro", "applegpu_g17d", 0, false},
		{"Apple M5 Max", "applegpu_g17s", 64, true},
		{"Apple M5 Ultra", "applegpu_g17d", 128, true},
		{"Apple M5 Ultra", "applegpu_g15d", 0, false},
		{"Apple M1 Ultra", "applegpu_g13d", 0, false},
		{"Apple M3 Max", "applegpu_g15s", 0, false},
		{"Apple M5 Max", "unqualified_arch_s", 0, false},
	} {
		if got := relativeAttentionSplitParts(tc.name, tc.arch); got != tc.parts {
			t.Errorf("%s/%s: split parts %d, want %d", tc.name, tc.arch, got, tc.parts)
		}
		if got := supportsRelativeAttentionPrefill(tc.name, tc.arch); got != tc.prefill {
			t.Errorf("%s/%s: tiled prefill %t, want %t", tc.name, tc.arch, got, tc.prefill)
		}
	}
}

func TestRelativeAttentionCUDAFallback(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !CUDAIsAvailable() {
			t.Skip("requires CUDA")
		}
		for _, width := range []int{1, 4, 8, 9, 64} {
			q, k, v, table, sink := relativeAttentionInputs(width, 1041, 8, 2, 128, 1024)
			for _, dtype := range []DType{DTypeFloat32, DTypeBFloat16} {
				if out := RelativePositionAttention(q.AsType(dtype), k, v, table, sink, 1024, 0.125); out != nil {
					t.Fatalf("CUDA attention with %d queries of dtype %v must use native SDPA", width, dtype)
				}
			}
		}
	})
}

func TestRelativeAttentionOrderedViews(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if relativeAttentionParts() == 0 {
			t.Skip("requires a qualified relative-attention device")
		}
		scale := float32(1 / math.Sqrt(128))
		for _, width := range []int{1, 8} {
			q, k, v, table, sink := relativeAttentionInputs(width, 1041, 24, 8, 128, 1024)
			for _, window := range []int{0, 1024} {
				got := RelativePositionAttention(q, k, v, table, sink, window, scale)
				if got == nil {
					t.Fatal("eligible route declined")
				}
				want := stockRelativeAttention(q, k, v, table, sink, window, scale)
				if err := requireClose(fmt.Sprintf("width %d window %d", width, window), got, want, 0); err != nil {
					t.Fatal(err)
				}
				query := q.AsType(DTypeFloat32)
				got = RelativePositionAttention(query, k, v, table, sink, window, scale)
				if got == nil || got.DType() != DTypeFloat32 {
					t.Fatal("FP32 query requires an FP32 result")
				}
				want = stockRelativeAttention(query, k.AsType(DTypeFloat32), v.AsType(DTypeFloat32), table, sink, window, scale)
				if err := requireExact("FP32 per-query attention", Contiguous(got, false), Contiguous(want, false)); err != nil {
					t.Fatal(err)
				}
			}
			if RelativePositionAttention(q, k, v, table.AsType(DTypeFloat32), sink, 0, scale) != nil {
				t.Fatal("unexpected dispatch for FP32 bias")
			}
			if RelativePositionAttention(q, k, v, table, sink, 32, scale) != nil {
				t.Fatal("unexpected dispatch with short local window")
			}
			short := func(x *Array) *Array { return x.Slice(Slice(), Slice(), Slice(0, 1023), Slice()) }
			if RelativePositionAttention(q, short(k), short(v), table, sink, 0, scale) != nil {
				t.Fatal("unexpected dispatch under split threshold")
			}
			if width == 1 {
				// Verify that a rejected launch returns control to the model's
				// stock attention path, including after the first pass succeeded.
				for _, stage := range []**gpuKernel{&relativeAttentionPart, &relativeAttentionMerge} {
					original := *stage
					*stage = &gpuKernel{metalDisabled: true, cudaDisabled: true}
					got := RelativePositionAttention(q, k, v, table, sink, 0, scale)
					*stage = original
					if got != nil {
						t.Fatal("disabled GPU kernel produced output")
					}
				}
			}
		}
	})
}

func TestRelativeAttentionHeadGroups(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if relativeAttentionParts() == 0 {
			t.Skip("requires a qualified GPU")
		}
		for _, tc := range []struct{ queries, heads, kvHeads, distances, window int }{
			{1, 4, 4, 1, 0},
			{3, 8, 2, 73, 1061},
			{8, 6, 2, 1600, 1200},
		} {
			Scoped(func() {
				q, k, v, table, sink := relativeAttentionInputs(tc.queries, 1243, tc.heads, tc.kvHeads, 128, tc.distances)
				scale := float32(1 / math.Sqrt(128))
				got := RelativePositionAttention(q, k, v, table, sink, tc.window, scale)
				if got == nil {
					t.Fatal("eligible head group declined")
				}
				want := stockRelativeAttention(q, k, v, table, sink, tc.window, scale)
				same := got.View(DTypeUint16).Equal(want.View(DTypeUint16)).AsType(DTypeInt32).Reshape(-1).SumAxis(0, false).Int()
				if int(same) != got.Size() {
					t.Fatalf("heads %d/%d, queries %d, distances %d, window %d: %d differing BF16 elements", tc.heads, tc.kvHeads, tc.queries, tc.distances, tc.window, got.Size()-int(same))
				}
			})
		}
	})
}

func TestRelativeAttentionRejectsIncompatibleShapes(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		q, k, v, table, sink := relativeAttentionInputs(3, 1027, 6, 2, 128, 31)
		shortHeads := func(x *Array, heads int) *Array { return x.Slice(Slice(), Slice(0, heads), Slice(), Slice()) }
		emptyDistances := table.Slice(Slice(), Slice(), Slice(), Slice(0, 0))
		for _, tc := range []struct {
			name                 string
			q, k, v, table, sink *Array
			window               int
		}{
			{"ungrouped heads", shortHeads(q, 5), k, v, shortHeads(table, 5), shortHeads(sink, 5), 0},
			{"wrong value heads", q, k, shortHeads(v, 1), table, sink, 0},
			{"wrong table heads", q, k, v, shortHeads(table, 3), sink, 0},
			{"missing distances", q, k, v, emptyDistances, sink, 0},
			{"negative window", q, k, v, table, sink, -1},
		} {
			if RelativePositionAttention(tc.q, tc.k, tc.v, tc.table, tc.sink, tc.window, 0.125) != nil {
				t.Fatalf("%s unexpectedly launched", tc.name)
			}
		}
	})
}

func stockRelativeAttention(q, k, v, table, sink *Array, window int, scale float32) *Array {
	keys, rows := k.Dim(2), q.Dim(2)
	rowSlice := func(x *Array, row int) *Array {
		return x.Slice(Slice(), Slice(), Slice(row, row+1), Slice())
	}
	var out []*Array
	for row := range rows {
		end := keys - rows + row + 1
		begin := 0
		if window > 0 {
			begin = max(0, end-window)
		}
		distance := Arange(float64(end-begin-1), -1, -1, DTypeInt32)
		positions := Clip(distance, FromValue(0), FromValue(table.Dim(3)-1))
		bias := rowSlice(table, row).TakeAlongAxis(positions.Reshape(1, 1, 1, end-begin), -1).AsType(DTypeFloat32)
		bias = Where(positions.Equal(distance), bias, FromValue(float32(0)))
		mask := bias.Subtract(rowSlice(sink, row).AsType(DTypeFloat32)).AsType(q.DType())
		rowK := k.Slice(Slice(), Slice(), Slice(begin, end), Slice())
		rowV := v.Slice(Slice(), Slice(), Slice(begin, end), Slice())
		out = append(out, FastScaledDotProductAttentionWithSinks(rowSlice(q, row), rowK, rowV, scale,
			FastSDPANoWindow, "", mask, Zeros(q.DType(), q.Dim(1))))
	}
	return Concatenate(out, 2)
}

func relativeAttentionInputs(queries, keys, heads, kvHeads, width, distances int) (q, k, v, table, sink *Array) {
	// Leave unused capacity on both sides of each head's visible cache. The
	// logical sequence length differs from the physical head stride.
	history := func(stride, modulus int) *Array {
		raw := patternArray(DTypeBFloat16, []int{1, kvHeads, keys + 19, width}, 0, 0.015625, stride, modulus)
		return raw.Slice(Slice(), Slice(), Slice(5, 5+keys), Slice())
	}
	q = patternArray(DTypeBFloat16, []int{1, heads, queries, width}, 0, 0.03125, 7, 71)
	table = patternArray(DTypeBFloat16, []int{1, heads, queries, distances}, 0, 0.0078125, 9, 127)
	sink = patternArray(DTypeBFloat16, []int{1, heads, queries, 1}, 0, 0.1, 13, 67)
	return q, history(17, 79), history(31, 113), table, sink
}

func BenchmarkRelativePositionAttentionEager(b *testing.B) {
	benchmarkRelativeAttention(b, stockRelativeAttention)
}

func BenchmarkRelativePositionAttentionKernel(b *testing.B) {
	benchmarkRelativeAttention(b, RelativePositionAttention)
}

func benchmarkRelativeAttention(b *testing.B, attend func(q, k, v, table, sink *Array, window int, scale float32) *Array) {
	thread := mlxTestThread(b)
	var supported bool
	if err := thread.Do(b.Context(), func() error {
		supported = relativeAttentionParts() != 0
		return nil
	}); err != nil {
		b.Fatal(err)
	}
	if !supported {
		b.Skip("requires a qualified relative-attention device")
	}
	// The eager arm evaluates native vector SDPA separately for each query.
	// The checkpoint's full serving route is measured by the model benchmark.
	for _, width := range []int{1, 8} {
		for _, tc := range []struct {
			name         string
			keys, window int
		}{
			{"local", 1024 + width - 1, 1024}, {"global", 3484, 0},
		} {
			b.Run(fmt.Sprintf("%s/width%d", tc.name, width), func(b *testing.B) {
				err := thread.Do(b.Context(), func() error {
					var launchErr error
					Scoped(func() {
						q, k, v, table, sink := relativeAttentionInputs(width, tc.keys, 24, 8, 128, 1024)
						Eval(q, k, v, table, sink)
						run := func() *Array { return attend(q, k, v, table, sink, tc.window, float32(1/math.Sqrt(128))) }
						warm := run()
						if warm == nil {
							launchErr = fmt.Errorf("attention declined the %s inputs", tc.name)
							return
						}
						Eval(warm)
						b.ResetTimer()
						for range b.N {
							Scoped(func() { Eval(run()) })
						}
						b.StopTimer()
					})
					return launchErr
				})
				if err != nil {
					b.Fatal(err)
				}
			})
		}
	}
}
