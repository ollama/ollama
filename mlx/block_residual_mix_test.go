package mlx

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

// Each case runs the mixer alone and with its normalization. Forwards of at
// most 8 rows give each query its own threadgroup; longer ones share one
// threadgroup among a row's queries.
func TestBlockResidualMixMatchesGraph(t *testing.T) {
	const width = 2304
	for _, tc := range []struct {
		name                            string
		batch, tokens, sources, queries int
		vector, strided                 bool
		queryType                       DType
	}{
		{"decode", 1, 1, 2, 1, true, false, DTypeFloat32},
		{"decode_query_matrix", 1, 1, 3, 3, false, false, DTypeBFloat16},
		{"batched_single_query_matrix", 4, 1, 3, 1, false, false, DTypeBFloat16},
		{"strided_verification", 2, 8, 4, 4, false, true, DTypeBFloat16},
		{"verification_query_matrix", 2, 8, 3, 3, false, false, DTypeBFloat16},
		{"short_prompt_tail", 1, 15, 5, 3, false, false, DTypeFloat32},
		{"prefill_boundary", 1, 16, 5, 3, false, false, DTypeFloat32},
		{"prefill", 1, 65, 5, 3, false, false, DTypeFloat32},
	} {
		for _, norm := range []bool{false, true} {
			name := tc.name
			if norm {
				name += "_norm"
			}
			t.Run(name, func(t *testing.T) {
				withMLXThread(t, func(t *mlxthreadtest.T) {
					shape := []int{tc.batch, tc.tokens, width}
					sources := make([]*Array, tc.sources)
					for i := range sources {
						dtype := DTypeBFloat16
						if i%2 == 1 {
							dtype = DTypeFloat32
						}
						if tc.strided {
							sources[i] = patternArray(dtype, []int{tc.batch, width, tc.tokens}, 0.05, 0.004, 13+2*i, 263).Transpose(0, 2, 1)
						} else {
							sources[i] = patternArray(dtype, shape, 0.05, 0.004, 13+2*i, 263)
						}
					}
					query := patternArray(tc.queryType, []int{width, tc.queries}, -0.07, 0.013, 37, 271).Transpose(1, 0)
					weight := patternArray(DTypeFloat32, []int{tc.queries, width}, 1, 0.01, 17, 61)
					if tc.vector {
						query, weight = query.Squeeze(0), weight.Squeeze(0)
					}
					var got *Array
					if norm {
						got = BlockResidualMixNorm(sources, query, weight, 1e-6, 1e-6)
					} else {
						got = BlockResidualMix(sources, query, 1e-6)
					}
					if arch := kernelArchitecture(); !isAppleGPU() || !blockResidualMixRuns(tc.tokens, norm, arch) {
						if got != nil {
							t.Fatalf("selected the Metal residual mixer on %s for %d tokens", arch, tc.tokens)
						}
						return
					}
					if got == nil {
						t.Fatal("eligible Metal inputs did not select the residual mixer")
					}

					// Reproduce the release graph, including scaling the query in its original
					// dtype before FP32 multiplication and both native reduction orders.
					keys := RMSNormFn(Stack(sources, -2).AsType(DTypeFloat32), nil, 1e-6)
					scaled := MulScalar(query.Reshape(tc.queries, width), float32(1/math.Sqrt(width)))
					scores := Sum(Mul(keys.ExpandDims(-3), scaled.ExpandDims(-2)), -1, false)
					weights := SoftmaxAxis(scores, -1, true).ExpandDims(-1)
					want := Sum(Mul(keys.ExpandDims(-3), weights), -2, false)
					wantType := DTypeFloat32
					if norm {
						// The fused normalization reproduces RMSNormRoundBF16 of the
						// graph's mixture with each query's weights, bit for bit.
						weight = weight.Reshape(tc.queries, width)
						parts := make([]*Array, tc.queries)
						for q := range tc.queries {
							mixture := want.Slice(Slice(), Slice(), Slice(q, q+1), Slice()).Squeeze(2)
							if parts[q] = RMSNormRoundBF16(mixture, weight.Slice(Slice(q, q+1), Slice()).Squeeze(0), 1e-6); parts[q] == nil {
								t.Fatal("reference normalization did not run")
							}
						}
						want, wantType = Stack(parts, 2), DTypeBFloat16
					}
					if tc.vector {
						want = want.Squeeze(-2)
					}
					if got.DType() != wantType || !slices.Equal(got.Dims(), want.Dims()) {
						t.Fatalf("got %v %v, want %v %v", got.DType(), got.Dims(), wantType, want.Dims())
					}
					// Query matrices return a view of query-major storage.
					if err := requireExact(name, Contiguous(got, false), want); err != nil {
						t.Fatal(err)
					}
				})
			})
		}
	}
}

func TestBlockResidualMixFallback(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		const width, tokens = 2304, 16
		x := Zeros(DTypeBFloat16, 1, tokens, width)
		q := Zeros(DTypeFloat32, width)
		valid := []*Array{x, x}
		for _, tc := range []struct {
			name    string
			sources []*Array
			query   *Array
			epsilon float32
		}{
			{"one_source", []*Array{x}, q, 1e-6},
			{"too_many_sources", []*Array{x, x, x, x, x, x}, q, 1e-6},
			{"missing_source", []*Array{x, nil}, q, 1e-6},
			{"source_dimensions", []*Array{x, x.Reshape(tokens, width)}, q, 1e-6},
			{"different_rows", []*Array{x, x.Reshape(tokens, 1, width)}, q, 1e-6},
			{"source_dtype", []*Array{x, x.AsType(DTypeFloat16)}, q, 1e-6},
			{"source_width", []*Array{x.Slice(Slice(), Slice(), Slice(0, 64)), x.Slice(Slice(), Slice(), Slice(0, 64))}, Zeros(DTypeFloat32, 64), 1e-6},
			{"missing_query", valid, nil, 1e-6},
			{"query_rank", valid, q.Reshape(1, 1, width), 1e-6},
			{"query_width", valid, q.Reshape(2, width/2), 1e-6},
			{"too_many_queries", valid, Zeros(DTypeFloat32, 5, width), 1e-6},
			{"query_dtype", valid, q.AsType(DTypeFloat16), 1e-6},
			{"zero_epsilon", valid, q, 0},
			{"other_epsilon", valid, q, 1e-3},
			{"nonfinite_epsilon", valid, q, float32(math.NaN())},
		} {
			if got := BlockResidualMix(tc.sources, tc.query, tc.epsilon); got != nil {
				t.Errorf("%s selected the Metal residual mixer", tc.name)
			}
		}
	})
}
