package mlx

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestRelativeAttentionSplitParts(t *testing.T) {
	for arch, want := range map[string]int{
		"applegpu_g14d":      128,
		"applegpu_g17d":      128,
		"applegpu_g14s":      64,
		"applegpu_g17s":      64,
		"applegpu_g18g":      32,
		"unqualified_arch_s": 0,
		"":                   0,
	} {
		if got := relativeAttentionSplitParts(arch); got != want {
			t.Errorf("%s: split parts %d, want %d", arch, got, want)
		}
	}
}

// The tuning ladder sizes partitions by history alone and stages keys only
// for blocks of queries over long histories.
func TestRelativeSplitConfig(t *testing.T) {
	for _, tc := range []struct {
		arch             string
		queries, visible int
		kernel           *gpuKernel
		parts            int
	}{
		{"applegpu_g17s", 1, 1024, relativeAttentionBlockPart, 32},
		{"applegpu_g17s", 8, 4096, relativeAttentionBlockPart, 32},
		{"applegpu_g17s", 8, 4097, relativeAttentionSharedPart, 64},
		{"applegpu_g17s", 1, 16384, relativeAttentionBlockPart, 64},
		{"applegpu_g17s", 4, 32768, relativeAttentionSharedPart, 128},
		{"applegpu_g17d", 4, 32768, relativeAttentionPart, 128},
		{"applegpu_g16s", 4, 32768, relativeAttentionSharedPart, 128},
		{"applegpu_g15d", 4, 32768, relativeAttentionPart, 128},
		{"applegpu_g14d", 1, 1024, relativeAttentionPart, 128},
		{"applegpu_g18g", 1, 1024, relativeAttentionPart, 32},
	} {
		base := relativeAttentionSplitParts(tc.arch)
		kernel, parts := relativeSplitConfig(base, tc.queries, tc.visible, tc.arch)
		if kernel != tc.kernel || parts != tc.parts {
			t.Errorf("%s, %d queries over %d keys: %s with %d parts, want %s with %d", tc.arch, tc.queries, tc.visible,
				kernel.name, parts, tc.kernel.name, tc.parts)
		}
	}
}

// The blocked kernels, and partition counts other than the device's native
// one, sum keys in a different order from native vector SDPA, so the two agree
// to rounding: within two BF16 ULPs of values below one, and FP32 accumulation
// error for FP32 queries.
const (
	relativeAttentionBF16Tol = 1.0 / 128
	relativeAttentionFP32Tol = 2e-5
)

// requireStockAttention compares a split result with stockRelativeAttention,
// which runs MLX's single-query vector SDPA per query. Ultra and Pro/Max GPUs
// split that SDPA into relativeAttentionParts() partitions, so the per-key
// kernel at that count must match bit for bit; anything else within tol.
func requireStockAttention(label string, got, want *Array, kernel *gpuKernel, parts int, tol float64) error {
	_, size := appleGPU(kernelArchitecture())
	if kernel == relativeAttentionPart && parts == relativeAttentionParts() && (size == 'd' || size == 's') {
		tol = 0
	}
	return requireClose(label, Contiguous(got, false), Contiguous(want, false), tol)
}

// Draft verification must score each query exactly as single-token decode does
// at its position, so which drafts are accepted cannot depend on the block
// width. Cover blocks that straddle a history length where the split changes,
// long blocks past it, and a local window.
func TestRelativeAttentionBlockMatchesDecode(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if relativeAttentionParts() == 0 {
			t.Skip("requires an Apple GPU")
		}
		scale := float32(1 / math.Sqrt(128))
		for _, tc := range []struct {
			queries, keys, window int
			// rolling views start at the newest window, as an ordered window
			// cache hands them out: a block's view starts at its first query's
			// window, single-token decode's at its own.
			rolling bool
		}{
			{8, 4100, 0, false},
			{4, 4098, 0, false},
			{8, 5000, 0, false},
			{8, 16388, 0, false},
			{3, 2050, 0, false},
			{8, 1100, 1024, false},
			{4, 3000, 1024, true},
			{8, 3000, 1024, true},
		} {
			Scoped(func() {
				q, k, v, table, sink := relativeAttentionInputs(tc.queries, tc.keys, 24, 8, 128, 1024)
				q = q.AsType(DTypeFloat32)
				first := tc.keys - tc.queries
				start := func(end int) int {
					if tc.rolling {
						return max(0, end-tc.window)
					}
					return 0
				}
				view := func(x *Array, from, to int) *Array { return x.Slice(Slice(), Slice(), Slice(from, to), Slice()) }
				from := start(first + 1)
				block := RelativePositionAttention(q, view(k, from, tc.keys), view(v, from, tc.keys), table, sink, tc.window, scale)
				if block == nil {
					t.Fatalf("%d queries over %d keys declined", tc.queries, tc.keys)
				}
				for j := range tc.queries {
					query := func(x *Array) *Array { return x.Slice(Slice(), Slice(), Slice(j, j+1), Slice()) }
					end := first + j + 1
					single := RelativePositionAttention(query(q), view(k, start(end), end), view(v, start(end), end), query(table), query(sink), tc.window, scale)
					got, want := Contiguous(query(block), false).Floats(), Contiguous(single, false).Floats()
					for i := range want {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Errorf("%d queries over %d keys, window %d: query %d value %d is %v, single-token decode gives %v",
								tc.queries, tc.keys, tc.window, j, i, got[i], want[i])
							return
						}
					}
				}
			})
		}
	})
}

// Long global histories take larger partition counts than the 1-8K the
// split was first tuned at. Every count must still agree with native SDPA.
func TestRelativeAttentionLongHistories(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if relativeAttentionParts() == 0 {
			t.Skip("requires an Apple GPU")
		}
		scale := float32(1 / math.Sqrt(128))
		for _, keys := range []int{9000, 20000, 40000} {
			for _, queries := range []int{1, 3, 8} {
				Scoped(func() {
					q, k, v, table, sink := relativeAttentionInputs(queries, keys, 24, 8, 128, 1024)
					q = q.AsType(DTypeFloat32)
					got := RelativePositionAttention(q, k, v, table, sink, 0, scale)
					if got == nil {
						t.Fatalf("%d queries over %d keys declined", queries, keys)
					}
					want := stockRelativeAttention(q, k.AsType(DTypeFloat32), v.AsType(DTypeFloat32), table, sink, 0, scale)
					kernel, parts := relativeSplit(queries, keys)
					label := fmt.Sprintf("%d queries over %d keys (%s, %d parts)", queries, keys, kernel.name, parts)
					if err := requireStockAttention(label, got, want, kernel, parts, relativeAttentionFP32Tol); err != nil {
						t.Fatal(err)
					}
				})
			}
		}
	})
}

// Every split kernel must agree with native SDPA for each block width,
// partition count and window, whichever one the device selection picks.
func TestRelativeAttentionSplitKernels(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if relativeAttentionParts() == 0 {
			t.Skip("requires an Apple GPU")
		}
		scale := float32(1 / math.Sqrt(128))
		kernels := map[string]*gpuKernel{
			"per-key": relativeAttentionPart, "block": relativeAttentionBlockPart, "shared": relativeAttentionSharedPart,
		}
		for _, tc := range []struct{ keys, window int }{{1100, 1024}, {1031, 0}, {5003, 0}} {
			for _, queries := range []int{1, 2, 3, 4, 8} {
				Scoped(func() {
					q, k, v, table, sink := relativeAttentionInputs(queries, tc.keys, 24, 8, 128, 1024)
					q = q.AsType(DTypeFloat32)
					want := stockRelativeAttention(q, k.AsType(DTypeFloat32), v.AsType(DTypeFloat32), table, sink, tc.window, scale)
					for name, kernel := range kernels {
						for _, parts := range []int{relativeAttentionParts() / 2, relativeAttentionParts(), 2 * relativeAttentionParts()} {
							label := fmt.Sprintf("%s kernel, %d parts, %d queries over %d keys, window %d", name, parts, queries, tc.keys, tc.window)
							got := relativeAttentionSplitMerge(kernel, parts, q, k, v, table, sink, tc.window, scale)
							if got == nil {
								t.Fatalf("%s: declined", label)
							}
							if err := requireStockAttention(label, got, want, kernel, parts, relativeAttentionFP32Tol); err != nil {
								t.Fatal(err)
							}
						}
					}
				})
			}
		}
	})
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
			t.Skip("requires an Apple GPU")
		}
		scale := float32(1 / math.Sqrt(128))
		for _, width := range []int{1, 8} {
			q, k, v, table, sink := relativeAttentionInputs(width, 1041, 24, 8, 128, 1024)
			for _, window := range []int{0, 1024} {
				kernel, parts := relativeSplit(width, relativeAttentionVisible(k, window))
				got := RelativePositionAttention(q, k, v, table, sink, window, scale)
				if got == nil {
					t.Fatal("eligible route declined")
				}
				want := stockRelativeAttention(q, k, v, table, sink, window, scale)
				if err := requireStockAttention(fmt.Sprintf("width %d window %d", width, window), got, want, kernel, parts, relativeAttentionBF16Tol); err != nil {
					t.Fatal(err)
				}
				query := q.AsType(DTypeFloat32)
				got = RelativePositionAttention(query, k, v, table, sink, window, scale)
				if got == nil || got.DType() != DTypeFloat32 {
					t.Fatal("FP32 query requires an FP32 result")
				}
				want = stockRelativeAttention(query, k.AsType(DTypeFloat32), v.AsType(DTypeFloat32), table, sink, window, scale)
				if err := requireStockAttention("FP32 per-query attention", got, want, kernel, parts, relativeAttentionFP32Tol); err != nil {
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
				passes := []**gpuKernel{&relativeAttentionPart, &relativeAttentionBlockPart, &relativeAttentionSharedPart}
				for _, stages := range [][]**gpuKernel{passes, {&relativeAttentionMerge}} {
					var originals []*gpuKernel
					for _, stage := range stages {
						originals = append(originals, *stage)
						*stage = &gpuKernel{metalDisabled: true, cudaDisabled: true}
					}
					got := RelativePositionAttention(q, k, v, table, sink, 0, scale)
					for i, stage := range stages {
						*stage = originals[i]
					}
					if got != nil {
						t.Fatal("disabled GPU kernel produced output")
					}
				}
			}
		}
	})
}

// Cache windows and query-major projections reach the kernel without a copy and
// give the same bits as row-contiguous inputs.
func TestRelativeAttentionReadsStridedInputs(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if relativeAttentionParts() == 0 {
			t.Skip("requires an Apple GPU")
		}
		scale := float32(1 / math.Sqrt(128))
		for _, tc := range []struct{ queries, keys, window int }{
			{1, 1024, 1024}, {4, 1027, 1024}, {8, 1031, 1024}, {3, 2050, 0}, {8, 8199, 0},
		} {
			Scoped(func() {
				q, k, v, table, sink := relativeAttentionInputs(tc.queries, tc.keys, 24, 8, 128, 1024)
				queryMajor := func(x *Array) *Array {
					return Contiguous(x.Transpose(0, 2, 1, 3), false).Transpose(0, 2, 1, 3)
				}
				q, table, sink = queryMajor(q.AsType(DTypeFloat32)), queryMajor(table), queryMajor(sink)
				dense := func(x *Array) *Array { return Contiguous(x, false) }
				want := RelativePositionAttention(dense(q), dense(k), dense(v), dense(table), dense(sink), tc.window, scale)
				if want == nil {
					t.Fatal("eligible contiguous inputs declined")
				}
				Eval(q, k, v, table, sink, want)

				// Measure from an empty allocator cache so earlier tests' buffers do not count.
				ClearCache()
				before := ActiveMemory()
				var got *Array
				peak := MeasurePeakMemory(func() {
					if got = RelativePositionAttention(q, k, v, table, sink, tc.window, scale); got != nil {
						Eval(got)
					}
				})
				if got == nil {
					t.Fatalf("%d queries over %d keys: strided inputs declined", tc.queries, tc.keys)
				}
				if err := requireExact(fmt.Sprintf("%d queries over %d keys", tc.queries, tc.keys), got, want); err != nil {
					t.Fatal(err)
				}
				// Partial sums, their statistics and the output; copying K and V
				// would add twice the window.
				rows := 24 * tc.queries
				_, parts := relativeSplit(tc.queries, relativeAttentionVisible(k, tc.window))
				outputs := rows*parts*(128*4+8) + rows*128*4
				if extra := peak - before - outputs; extra >= k.NumBytes() {
					t.Fatalf("%d queries over %d keys allocated %d bytes beyond the outputs; window is %d bytes",
						tc.queries, tc.keys, extra, k.NumBytes())
				}
			})
		}
		// Token-major keys and values do not have contiguous rows. Every split
		// kernel reads them through their strides and agrees with the stock graph.
		Scoped(func() {
			q, k, v, table, sink := relativeAttentionInputs(4, 1031, 24, 8, 128, 1024)
			q = q.AsType(DTypeFloat32)
			tokenMajor := func(x *Array) *Array { return Contiguous(x.Transpose(0, 2, 1, 3), false).Transpose(0, 2, 1, 3) }
			want := stockRelativeAttention(q, k.AsType(DTypeFloat32), v.AsType(DTypeFloat32), table, sink, 1024, scale)
			parts := relativeAttentionParts()
			for name, kernel := range map[string]*gpuKernel{
				"per-key": relativeAttentionPart, "block": relativeAttentionBlockPart, "shared": relativeAttentionSharedPart,
			} {
				got := relativeAttentionSplitMerge(kernel, parts, q, tokenMajor(k), tokenMajor(v), table, sink, 1024, scale)
				if got == nil {
					t.Fatalf("%s kernel declined token-major keys", name)
				}
				if err := requireStockAttention(name+" kernel, token-major keys", got, want, kernel, parts, relativeAttentionFP32Tol); err != nil {
					t.Fatal(err)
				}
			}
			got := RelativePositionAttention(q, tokenMajor(k), v, table, sink, 1024, scale)
			if got == nil {
				t.Fatal("eligible shape declined")
			}
			if err := requireStockAttention("token-major keys", got, want, nil, 0, relativeAttentionFP32Tol); err != nil {
				t.Fatal(err)
			}
		})
	})
}

// Folding the output gate into the merge must give the same BF16 bits as the
// separate merge and a graph gate (tolerance zero), for one and two requests of
// every short width, with the gate read through a column view. The gates run
// through every BF16 bit pattern across the cases.
func TestRelativeAttentionGatedMatchesSeparateGate(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if relativeAttentionParts() == 0 {
			t.Skip("requires an Apple GPU")
		}
		const heads, kvHeads, width = 24, 8, 128
		scale := float32(1 / math.Sqrt(width))
		pattern, covered := uint16(0), 0
		// The second pair of histories straddles a partition-count change on
		// devices whose split follows the history length; each request must
		// still match its own separate merge.
		for _, lengths := range [][2]int{{1031, 1098}, {3000, 5000}} {
			for _, window := range []int{1024, 0} {
				for _, requests := range []int{1, 2} {
					for _, queries := range []int{1, 3, 4, 8} {
						Scoped(func() {
							label := fmt.Sprintf("histories %v, window %d, %d requests, %d queries", lengths, window, requests, queries)
							var q, k, v, table, sink, separate []*Array
							for r := range requests {
								rq, rk, rv, rt, rs := relativeAttentionInputs(queries, lengths[r], heads, kvHeads, width, 1024)
								rq = rq.AsType(DTypeFloat32)
								q, k, v, table, sink = append(q, rq), append(k, rk), append(v, rv), append(table, rt), append(sink, rs)
								attended := RelativePositionAttention(rq, rk, rv, rt, rs, window, scale)
								if attended == nil {
									t.Fatalf("%s: separate merge declined", label)
								}
								separate = append(separate, attended.Transpose(0, 2, 1, 3).Reshape(1, queries, heads*width))
							}
							bits := make([]uint16, requests*queries*heads*width)
							for i := range bits {
								bits[i] = pattern
								pattern++
							}
							gate := FromValues(bits, requests, queries, heads*width).View(DTypeBFloat16)
							// Read the gate as a column view of a wider projection.
							pad := func(n int) *Array { return Zeros(DTypeBFloat16, requests, queries, n) }
							gate = Concatenate([]*Array{pad(72), gate, pad(128)}, 2).Slice(Slice(), Slice(), Slice(72, 72+heads*width))
							want := Mul(Concatenate(separate, 0), gate.AsType(DTypeFloat32).Sigmoid()).AsType(DTypeBFloat16)
							got := RelativePositionAttentionGated(q, k, v, table, sink, gate, window, scale)
							if got == nil {
								t.Fatalf("%s: gated merge declined", label)
							}
							gotBits := Contiguous(got.View(DTypeUint16).AsType(DTypeInt32), false).Ints()
							wantBits := Contiguous(want.View(DTypeUint16).AsType(DTypeInt32), false).Ints()
							if len(gotBits) != len(wantBits) {
								t.Fatalf("%s: %d values, want %d", label, len(gotBits), len(wantBits))
							}
							for i := range wantBits {
								if gotBits[i] != wantBits[i] {
									t.Fatalf("%s: value %d is %#04x, want %#04x", label, i, gotBits[i], wantBits[i])
								}
							}
							covered += len(wantBits)
						})
					}
				}
			}
		}
		if covered < 1<<16 {
			t.Fatalf("covered only %d gate values", covered)
		}
		q, k, v, table, sink := relativeAttentionInputs(4, 1031, heads, kvHeads, width, 1024)
		q = q.AsType(DTypeFloat32)
		one := func(x *Array) []*Array { return []*Array{x} }
		gate := Zeros(DTypeBFloat16, 1, 4, heads*width)
		for _, tc := range []struct {
			name                 string
			q, k, v, table, sink []*Array
			gate                 *Array
		}{
			{"FP32 gate", one(q), one(k), one(v), one(table), one(sink), gate.AsType(DTypeFloat32)},
			{"narrow gate", one(q), one(k), one(v), one(table), one(sink), gate.Slice(Slice(), Slice(), Slice(0, heads*width-128))},
			{"three requests", []*Array{q, q, q}, []*Array{k, k, k}, []*Array{v, v, v}, []*Array{table, table, table}, []*Array{sink, sink, sink}, Zeros(DTypeBFloat16, 3, 4, heads*width)},
			{"short history", one(q), one(k.Slice(Slice(), Slice(), Slice(0, 1026), Slice())), one(v.Slice(Slice(), Slice(), Slice(0, 1026), Slice())), one(table), one(sink), gate},
		} {
			if RelativePositionAttentionGated(tc.q, tc.k, tc.v, tc.table, tc.sink, tc.gate, 0, scale) != nil {
				t.Fatalf("%s unexpectedly launched", tc.name)
			}
		}
	})
}

func TestRelativeAttentionHeadGroups(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if relativeAttentionParts() == 0 {
			t.Skip("requires an Apple GPU")
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
				kernel, parts := relativeSplit(tc.queries, relativeAttentionVisible(k, tc.window))
				label := fmt.Sprintf("heads %d/%d, queries %d, distances %d, window %d", tc.heads, tc.kvHeads, tc.queries, tc.distances, tc.window)
				if err := requireStockAttention(label, got, want, kernel, parts, relativeAttentionBF16Tol); err != nil {
					t.Fatal(err)
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
		b.Skip("requires an Apple GPU")
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
