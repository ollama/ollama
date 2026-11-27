package mlx

import (
	"math"
	"slices"
)

// BlockResidualMix normalizes each [batch, tokens, width] source and mixes them
// independently for each query. Keeping the FP32 values and softmax weights in
// one Metal threadgroup avoids storing and rereading them between kernels.
// A vector query [width] returns [batch, tokens, width]. A matrix query
// [queries, width] returns [batch, tokens, queries, width] as a transposed
// view of query-major storage, so each query's mixture stays contiguous.
// Unsupported inputs return nil for the caller's graph.
func BlockResidualMix(sources []*Array, query *Array, epsilon float32) *Array {
	return blockResidualMix(sources, query, nil, epsilon, 0)
}

// BlockResidualMixNorm is BlockResidualMix followed by an RMS normalization of
// each mixture with FP32 weight [queries, width] (or [width] for a vector
// query) and normEpsilon, rounded once to BF16. It reproduces
// RMSNormRoundBF16 of the FP32 mixture, including its reduction order. The
// result has BlockResidualMix's shape and layout.
func BlockResidualMixNorm(sources []*Array, query, weight *Array, epsilon, normEpsilon float32) *Array {
	if weight == nil || weight.DType() != DTypeFloat32 || query == nil || weight.Size() != query.Size() {
		return nil
	}
	return blockResidualMix(sources, query, weight, epsilon, normEpsilon)
}

func blockResidualMix(sources []*Array, query, norm *Array, epsilon, normEpsilon float32) *Array {
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
	if !blockResidualMixRuns(shape[1], norm != nil, kernelArchitecture()) {
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
	var inputs [maxSources + 2]*Array
	for i := range maxSources {
		inputs[i] = sources[min(i, count-1)]
	}
	inputs[maxSources] = query
	// Without a norm, the query fills the norm slot and is not read there.
	hasNorm, dtype := 0, DTypeFloat32
	inputs[maxSources+1] = query
	if norm != nil {
		hasNorm, dtype = 1, DTypeBFloat16
		inputs[maxSources+1] = norm
	}
	// Short forwards spread their queries over threadgroups to keep the GPU
	// busy; longer ones normalize each row once for all of its queries.
	perGroup := queries
	if batch*tokens <= 8 {
		perGroup = 1
	}
	// Each query's mixture is stored contiguously so selecting one is a view.
	out, ok := blockResidualMixKernel.applyMetal(gpuLaunch{
		inputs: inputs[:],
		dtypes: []gpuDTypeArg{{"QueryT", query.DType()}, {"OutT", dtype}},
		ints: []gpuIntArg{
			{"Width", width},
			{"Sources", count},
			{"PerGroup", perGroup},
			{"HasNorm", hasNorm},
			{"NormEpsBits", int(math.Float32bits(normEpsilon))},
		},
		outputs: []gpuOutputSpec{{"mixed", []int32{int32(queries), int32(batch), int32(tokens), width}, dtype}},
		grid:    [3]int{width / 4, queries / perGroup, batch * tokens}, threadGroup: [3]int{width / 4, 1, 1},
	})
	if !ok {
		return nil
	}
	if len(qshape) == 2 {
		return out[0].Transpose(1, 2, 0, 3)
	}
	return out[0].Reshape(batch, tokens, width)
}

// blockResidualMixRuns reports whether the fused mixer beats the stock graph
// for a forward of tokens rows on a device architecture, normalizing its
// output when norm is set.
func blockResidualMixRuns(tokens int, norm bool, architecture string) bool {
	generation, size := appleGPU(architecture)
	if generation < 15 && size == 'd' && tokens < 16 {
		// M2 Ultra gained 3.8% on cold prefill, but fixed-work decode was 1.6% slower.
		return false
	}
	if generation < 15 && norm {
		// M2 can reject the normalizing variant's 576-thread groups, which hold
		// more registers, when they are evaluated; no fallback catches that.
		return false
	}
	return true
}

// One threadgroup owns a token and PerGroup consecutive queries, normalizing
// the sources once for all of them. The four adjacent values per lane and the
// second SIMD sum preserve the native width-2304 reduction order. With
// HasNorm, each mixture is RMS normalized in RMSNormRoundBF16's order.
var blockResidualMixKernel = &gpuKernel{
	name:    "block_residual_mix",
	inputs:  []string{"x0", "x1", "x2", "x3", "x4", "query", "norm"},
	outputs: []string{"mixed"},
	metal: gpuSource{source: `
uint row = threadgroup_position_in_grid.z, rows = threadgroups_per_grid.z;
uint firstQuery = threadgroup_position_in_grid.y * PerGroup;
uint first = thread_index_in_threadgroup * 4, offset = row * Width + first;
uint lane = thread_index_in_simdgroup, group = simdgroup_index_in_threadgroup;
constexpr uint Groups = Width / (4 * 32);
// RMSNormRoundBF16 sums squares over eight consecutive values per lane, then
// SIMD sums per 32 lanes, then those sums in order.
constexpr uint NormLanes = Width / 8, NormGroups = (NormLanes + 31) / 32;
threadgroup float squares[Sources][Groups], dots[PerGroup][Sources][Groups];
threadgroup float staged[HasNorm ? Width : 1], normPartial[NormGroups];
float values[Sources][4], weights[Sources];
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
    for (uint j = 0; j < 4; ++j) values[s][j] *= invRMS;
}
for (uint q = 0; q < PerGroup; ++q) {
    float key[4];
    for (uint j = 0; j < 4; ++j) {
        // The graph scales in the query's dtype before multiplying FP32 keys.
        key[j] = float(QueryT(query[(firstQuery + q) * Width + first + j] * QueryT(1.0f / metal::precise::sqrt(float(Width)))));
    }
    for (uint s = 0; s < Sources; ++s) {
        float score = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            volatile float product = values[s][j] * key[j];
            score += product;
        }
        float partial = simd_sum(score);
        if (lane == 0) dots[q][s][group] = partial;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint q = 0; q < PerGroup; ++q) {
    float peak = -INFINITY;
    for (uint s = 0; s < Sources; ++s) {
        weights[s] = simd_sum(lane < Groups ? dots[q][s][lane] : 0.0f);
        peak = max(peak, weights[s]);
    }
    float sum = 0.0f;
    for (uint s = 0; s < Sources; ++s) {
        weights[s] = metal::fast::exp(weights[s] - peak);
        sum += weights[s];
    }
    float inverse = 1.0f / sum;
    for (uint s = 0; s < Sources; ++s) weights[s] *= inverse;
    float mix[4];
    for (uint j = 0; j < 4; ++j) {
        float value = 0.0f;
        for (uint s = 0; s < Sources; ++s) {
            // Match the graph's rounded products before summing across sources.
            volatile float product = values[s][j] * weights[s];
            value += product;
        }
        mix[j] = value;
    }
    uint at = ((firstQuery + q) * rows + row) * Width + first;
    if (!HasNorm) {
        for (uint j = 0; j < 4; ++j) mixed[at + j] = OutT(mix[j]);
        continue;
    }
    // Stage the mixture so one lane holds each run of eight values.
    for (uint j = 0; j < 4; ++j) staged[first + j] = mix[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint t = thread_index_in_threadgroup;
    if (t < NormGroups * 32) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            float v = t < NormLanes ? staged[t * 8 + i] : 0.0f;
            sum += v * v;
        }
        sum = simd_sum(sum);
        if (lane == 0) normPartial[group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint g = 0; g < NormGroups; ++g) total += normPartial[g];
    float inv = metal::precise::rsqrt(total / float(Width) + as_type<float>(NormEpsBits));
    for (uint j = 0; j < 4; ++j) {
        float y = norm[(firstQuery + q) * Width + first + j] * (mix[j] * inv);
        mixed[at + j] = OutT(y);
    }
}
`},
}
