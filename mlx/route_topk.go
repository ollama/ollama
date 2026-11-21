package mlx

import "math"

// RouteSigmoidTopK selects experts for each row of FP32 logits [rows, experts].
// Experts rank by sigmoid(logit) plus an optional FP32 bias [experts], highest
// first with ties to the lower index. Each chosen expert's weight is its
// sigmoid divided by the sum over the chosen experts, times scale. It returns
// uint32 indices and FP32 weights, both [rows, topK], or nils when the Metal
// kernel cannot run.
func RouteSigmoidTopK(logits, bias *Array, topK int, scale float32) (indices, weights *Array) {
	if !MetalIsAvailable() || logits == nil || logits.NumDims() != 2 || logits.DType() != DTypeFloat32 {
		return nil, nil
	}
	rows, experts := logits.Dim(0), logits.Dim(1)
	if rows < 1 || experts < 1 || experts > 1024 || experts%32 != 0 || topK < 1 || topK > 32 || topK > experts ||
		uint64(logits.Size()) > math.MaxInt32 {
		return nil, nil
	}
	hasBias := 0
	if bias != nil {
		if bias.DType() != DTypeFloat32 || bias.Size() != experts {
			return nil, nil
		}
		hasBias = 1
	} else {
		bias = logits
	}
	out, ok := routeSigmoidTopK.applyMetal(gpuLaunch{
		inputs: []*Array{logits, bias},
		ints: []gpuIntArg{{"Experts", experts}, {"TopK", topK}, {"HasBias", hasBias},
			{"ScaleBits", int(math.Float32bits(scale))}},
		outputs: []gpuOutputSpec{
			{"route_indices", []int32{int32(rows), int32(topK)}, DTypeUint32},
			{"route_weights", []int32{int32(rows), int32(topK)}, DTypeFloat32},
		},
		grid:        [3]int{experts, rows, 1},
		threadGroup: [3]int{experts, 1, 1},
	})
	if !ok {
		return nil, nil
	}
	return out[0], out[1]
}

var routeSigmoidTopK = &gpuKernel{
	name:    "route_sigmoid_topk",
	inputs:  []string{"logits", "bias"},
	outputs: []string{"indices", "weights"},
	metal: gpuSource{source: `
constexpr uint Groups = Experts / 32;
uint row = threadgroup_position_in_grid.y, e = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup, group = simdgroup_index_in_threadgroup;
float logit = logits[row * Experts + e];
float gate = 1.0f / (1.0f + metal::precise::exp(-logit));
float score = HasBias ? gate + bias[e] : gate;
threadgroup float bestScore[Groups];
threadgroup uint bestIndex[Groups];
threadgroup uint chosen[TopK];
for (uint k = 0; k < TopK; ++k) {
    // Highest score, ties to the lower expert index.
    float s = score;
    uint i = e;
    for (uint offset = 16; offset > 0; offset /= 2) {
        float os = simd_shuffle_down(s, offset);
        uint oi = simd_shuffle_down(i, offset);
        if (lane + offset < 32 && (os > s || (os == s && oi < i))) {
            s = os;
            i = oi;
        }
    }
    if (lane == 0) {
        bestScore[group] = s;
        bestIndex[group] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
        float bs = bestScore[0];
        uint bi = bestIndex[0];
        for (uint g = 1; g < Groups; ++g) {
            if (bestScore[g] > bs || (bestScore[g] == bs && bestIndex[g] < bi)) {
                bs = bestScore[g];
                bi = bestIndex[g];
            }
        }
        chosen[k] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == chosen[k]) score = -INFINITY;
}
if (e < TopK) {
    uint index = chosen[e];
    float g = 1.0f / (1.0f + metal::precise::exp(-logits[row * Experts + index]));
    float total = 0.0f;
    for (uint k = 0; k < TopK; ++k) {
        total += 1.0f / (1.0f + metal::precise::exp(-logits[row * Experts + chosen[k]]));
    }
    indices[row * TopK + e] = index;
    weights[row * TopK + e] = g / total * as_type<float>(ScaleBits);
}
`},
}
