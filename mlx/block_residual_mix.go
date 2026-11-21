package mlx

import (
	"math"
	"slices"
)

// BlockResidualMix normalizes each [batch, tokens, width] residual source and
// mixes the sources independently for each query. Fusing RMS and query scoring
// avoids the [batch, tokens, queries, sources, width] product. At width 2304
// this improved Inkling's cold prefill by 4–8% on M4 Pro and M5 Max.
// Other shapes, dtypes, epsilons and devices return nil for the caller's graph.
func BlockResidualMix(sources []*Array, query *Array, epsilon float32) *Array {
	const width, maxSources = 2304, 5
	if len(sources) < 2 || len(sources) > maxSources || epsilon != 1e-6 || !MetalIsAvailable() || !isAppleGPU() {
		return nil
	}
	supported := func(x *Array) bool {
		if x == nil {
			return false
		}
		dtype := x.DType()
		return dtype == DTypeBFloat16 || dtype == DTypeFloat32
	}
	if !supported(query) || !supported(sources[0]) {
		return nil
	}
	shape := sources[0].Dims()
	if len(shape) != 3 || shape[0] < 1 || shape[1] < 1 || shape[2] != width {
		return nil
	}
	// M2 Ultra gained 3.8% on cold prefill, but fixed-work decode was 1.6% slower.
	if name, _ := kernelDevice(); name == "Apple M2 Ultra" && shape[1] < 16 {
		return nil
	}
	for _, source := range sources[1:] {
		if !supported(source) || !slices.Equal(shape, source.Dims()) {
			return nil
		}
	}
	qshape := query.Dims()
	queries := 1
	switch len(qshape) {
	case 1:
	case 2:
		queries = qshape[0]
	default:
		return nil
	}
	if queries < 1 || queries > 4 || qshape[len(qshape)-1] != width {
		return nil
	}
	batch, tokens, count := shape[0], shape[1], len(sources)
	if batch > math.MaxUint32/width/max(queries, count)/tokens {
		return nil // Shader element offsets are unsigned 32-bit indices.
	}
	// The score kernel has a fixed signature for mixed BF16/FP32 sources.
	// Extra bindings alias a live source and are never read.
	var inputs [maxSources + 1]*Array
	for i := range maxSources {
		inputs[i] = sources[min(i, count-1)]
	}
	inputs[maxSources] = MulScalar(query, float32(1/math.Sqrt(width)))
	args := []gpuIntArg{{"Width", width}, {"Sources", count}, {"Queries", queries}}
	b, t, s, q := int32(batch), int32(tokens), int32(count), int32(queries)
	out, ok := blockResidualScores.applyMetal(gpuLaunch{
		inputs: inputs[:],
		ints:   args,
		outputs: []gpuOutputSpec{
			{"values", []int32{b, t, s, width}, DTypeFloat32},
			{"scores", []int32{b, t, q, s}, DTypeFloat32},
		},
		grid: [3]int{width / 4, count, batch * tokens}, threadGroup: [3]int{width / 4, 1, 1},
	})
	if !ok {
		return nil
	}
	outputShape := []int32{b, t, width}
	if len(qshape) == 2 {
		outputShape = []int32{b, t, q, width}
	}
	mixed, ok := blockResidualSum.applyMetal(gpuLaunch{
		inputs:  []*Array{out[0], SoftmaxAxis(out[1], -1, true)},
		ints:    args,
		outputs: []gpuOutputSpec{{"mixed", outputShape, DTypeFloat32}},
		grid:    [3]int{width, queries, batch * tokens}, threadGroup: [3]int{128, 1, 1},
	})
	if !ok {
		return nil
	}
	return mixed[0]
}

var blockResidualScores = &gpuKernel{
	name:    "block_residual_scores",
	inputs:  []string{"x0", "x1", "x2", "x3", "x4", "query"},
	outputs: []string{"values", "scores"},
	metal: gpuSource{header: `
// The pinned MLX width-2304 RMS and row sum both reduce four adjacent
// values per thread, then the 18 SIMD-group totals in a second SIMD sum.
inline float block_residual_sum(float x, threadgroup float* partials, uint lane, uint group) {
    float subtotal = simd_sum(x);
    if (lane == 0) partials[group] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0) return simd_sum(lane < 18 ? partials[lane] : 0.0f);
    return 0.0f;
}
`, source: `
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup, group = simdgroup_index_in_threadgroup;
uint source = threadgroup_position_in_grid.y, row = threadgroup_position_in_grid.z;
uint column = tid * 4, offset = row * Width + column;
threadgroup float partials[18];
threadgroup float invRMS;
float v[4], squares = 0.0f;
for (uint j = 0; j < 4; j++) {
    switch (source) {
        case 0: v[j] = float(x0[offset + j]); break;
        case 1: v[j] = float(x1[offset + j]); break;
        case 2: v[j] = float(x2[offset + j]); break;
        case 3: v[j] = float(x3[offset + j]); break;
        default: v[j] = float(x4[offset + j]); break;
    }
    squares += v[j] * v[j];
}
float total = block_residual_sum(squares, partials, lane, group);
if (tid == 0) invRMS = metal::precise::rsqrt(total / float(Width) + 1e-6f);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint j = 0; j < 4; j++) {
    v[j] *= invRMS;
    values[(row * Sources + source) * Width + column + j] = v[j];
}
for (uint q = 0; q < Queries; q++) {
    float dot = 0.0f;
    for (uint j = 0; j < 4; j++) {
        // Match a materialized FP32 product followed by MLX Sum.
        volatile float product = v[j] * float(query[q * Width + column + j]);
        dot += product;
    }
    float score = block_residual_sum(dot, partials, lane, group);
    if (tid == 0) scores[(row * Queries + q) * Sources + source] = score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
`},
}

// Each query uses the same normalized sources. Accumulating the individually
// rounded products avoids a query-expanded tensor and preserves the stock sum.
var blockResidualSum = &gpuKernel{
	name:    "block_residual_weighted_sum",
	inputs:  []string{"values", "weights"},
	outputs: []string{"mixed"},
	metal: gpuSource{source: `
uint h = thread_position_in_grid.x, q = thread_position_in_grid.y, row = thread_position_in_grid.z;
float sum = 0.0f;
for (uint s = 0; s < Sources; s++) {
    volatile float product = values[(row * Sources + s) * Width + h] * weights[(row * Queries + q) * Sources + s];
    sum += product;
}
mixed[(row * Queries + q) * Width + h] = sum;
`},
}
