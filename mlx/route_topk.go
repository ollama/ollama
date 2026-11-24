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
	if rows < 1 || experts < 1 || experts > 1024 || topK < 1 || topK > 32 || topK > experts ||
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
		ints: []gpuIntArg{
			{"Experts", experts},
			{"TopK", topK},
			{"HasBias", hasBias},
			{"ScaleBits", int(math.Float32bits(scale))},
		},
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
uint row = threadgroup_position_in_grid.y, e = thread_index_in_threadgroup;
float logit = logits[row * Experts + e];
float gate = 1.0f / (1.0f + metal::precise::exp(-logit));
float score = HasBias ? gate + bias[e] : gate;
threadgroup float ranked[Experts];
threadgroup float chosenGate[TopK];
threadgroup float chosenLogit[TopK];
threadgroup uint chosen[TopK];
// NaN ranks last, so the ranks stay a permutation and every slot is chosen.
score = isnan(score) ? -INFINITY : score;
ranked[e] = score;
threadgroup_barrier(mem_flags::mem_threadgroup);
// An expert's rank counts the experts ahead of it: higher scores, and equal
// scores at lower indices.
uint rank = 0;
for (uint other = 0; other < Experts; ++other) {
    float s = ranked[other];
    rank += (s > score || (s == score && other < e)) ? 1u : 0u;
}
if (rank < TopK) {
    chosen[rank] = e;
    chosenGate[rank] = gate;
    chosenLogit[rank] = logit;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (e < TopK) {
    float total = 0.0f;
    bool underflow = false;
    for (uint k = 0; k < TopK; ++k) {
        total += chosenGate[k];
        underflow = underflow || chosenGate[k] < FLT_MIN;
    }
    float weight = chosenGate[e] / total;
    if (underflow) {
        // A chosen gate underflowed (logits below about -87) and lost its
        // share, so normalize the log-sigmoids relative to the largest instead.
        float top = -INFINITY;
        for (uint k = 0; k < TopK; ++k) top = max(top, log_sigmoid(chosenLogit[k]));
        float sum = 0.0f;
        for (uint k = 0; k < TopK; ++k) sum += metal::precise::exp(log_sigmoid(chosenLogit[k]) - top);
        weight = metal::precise::exp(log_sigmoid(chosenLogit[e]) - top) / sum;
    }
    indices[row * TopK + e] = chosen[e];
    weights[row * TopK + e] = weight * as_type<float>(ScaleBits);
}
`, header: `
inline float log_sigmoid(float x) {
    return -(metal::max(-x, 0.0f) + metal::precise::log(1.0f + metal::precise::exp(-metal::fabs(x))));
}
`},
}
