package mlx

import (
	"math"
	"slices"
)

// BlockResidualMix normalizes each [batch, tokens, width] source and mixes them
// independently for each query. Keeping the FP32 values and softmax weights in
// one Metal threadgroup avoids storing and rereading them between kernels.
// Unsupported inputs return nil for the caller's graph.
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
	// The kernel has a fixed signature for mixed BF16/FP32 sources.
	// Extra bindings alias a live source and are never read.
	var inputs [maxSources + 1]*Array
	for i := range maxSources {
		inputs[i] = sources[min(i, count-1)]
	}
	inputs[maxSources] = query
	outputShape := []int32{int32(batch), int32(tokens), width}
	if len(qshape) == 2 {
		outputShape = []int32{int32(batch), int32(tokens), int32(queries), width}
	}
	out, ok := blockResidualMix.applyMetal(gpuLaunch{
		inputs:  inputs[:],
		dtypes:  []gpuDTypeArg{{"QueryT", query.DType()}},
		ints:    []gpuIntArg{{"Width", width}, {"Sources", count}, {"Queries", queries}},
		outputs: []gpuOutputSpec{{"mixed", outputShape, DTypeFloat32}},
		grid:    [3]int{width / 4, queries, batch * tokens}, threadGroup: [3]int{width / 4, 1, 1},
	})
	if !ok {
		return nil
	}
	return out[0]
}

// One threadgroup owns a (token, query) pair. The four adjacent values per
// lane and the second SIMD sum preserve the native width-2304 reduction order.
var blockResidualMix = &gpuKernel{
	name:    "block_residual_mix",
	inputs:  []string{"x0", "x1", "x2", "x3", "x4", "query"},
	outputs: []string{"mixed"},
	metal: gpuSource{source: `
uint row = threadgroup_position_in_grid.z, q = threadgroup_position_in_grid.y;
uint first = thread_index_in_threadgroup * 4, offset = row * Width + first;
uint lane = thread_index_in_simdgroup, group = simdgroup_index_in_threadgroup;
constexpr uint Groups = Width / (4 * 32);
threadgroup float squares[Sources][Groups], dots[Sources][Groups];
float values[Sources][4], key[4], weights[Sources];
for (uint j = 0; j < 4; ++j) {
    // The graph scales in the query's dtype before multiplying FP32 keys.
    key[j] = float(QueryT(query[q * Width + first + j] * QueryT(1.0f / metal::precise::sqrt(float(Width)))));
}
for (uint s = 0; s < Sources; ++s) {
    float total = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        float v;
        switch (s) {
            case 0: v = float(x0[offset + j]); break;
            case 1: v = float(x1[offset + j]); break;
            case 2: v = float(x2[offset + j]); break;
            case 3: v = float(x3[offset + j]); break;
            default: v = float(x4[offset + j]); break;
        }
        values[s][j] = v;
        total += v * v;
    }
    float partial = simd_sum(total);
    if (lane == 0) squares[s][group] = partial;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint s = 0; s < Sources; ++s) {
    float total = simd_sum(lane < Groups ? squares[s][lane] : 0.0f);
    float invRMS = metal::precise::rsqrt(total / float(Width) + 1e-6f);
    float score = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        values[s][j] *= invRMS;
        volatile float product = values[s][j] * key[j];
        score += product;
    }
    float partial = simd_sum(score);
    if (lane == 0) dots[s][group] = partial;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float peak = -INFINITY;
for (uint s = 0; s < Sources; ++s) {
    weights[s] = simd_sum(lane < Groups ? dots[s][lane] : 0.0f);
    peak = max(peak, weights[s]);
}
float sum = 0.0f;
for (uint s = 0; s < Sources; ++s) {
    weights[s] = metal::fast::exp(weights[s] - peak);
    sum += weights[s];
}
float inverse = 1.0f / sum;
for (uint s = 0; s < Sources; ++s) weights[s] *= inverse;
for (uint j = 0; j < 4; ++j) {
    float value = 0.0f;
    for (uint s = 0; s < Sources; ++s) {
        // Match the graph's rounded products before summing across sources.
        volatile float product = values[s][j] * weights[s];
        value += product;
    }
    mixed[(row * Queries + q) * Width + first + j] = value;
}
`},
}
