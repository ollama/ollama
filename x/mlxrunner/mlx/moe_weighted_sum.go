package mlx

const moeWeightedSumMetalKernelSource = `
auto elem = thread_position_in_grid.x;
auto total = B * L * H;
if (elem >= total) {
  return;
}

auto h = elem % H;
auto token = elem / H;
auto expert_base = token * TopK * H + h;
auto score_base = token * TopK;

float acc = 0.0f;
for (int k = 0; k < TopK; ++k) {
  acc += static_cast<float>(expert[expert_base + k * H]) * static_cast<float>(scores[score_base + k]);
}
y[elem] = static_cast<OutT>(acc);
`

var moeWeightedSum = &gpuKernel{
	name:    "moe_weighted_sum",
	inputs:  []string{"expert", "scores"},
	outputs: []string{"y"},
	metal:   gpuSource{source: moeWeightedSumMetalKernelSource},
}

func moeWeightedSumValidate(expert, scores *Array, outDType DType) (B, L, topK, H int, ok bool) {
	if expert == nil || scores == nil {
		return 0, 0, 0, 0, false
	}
	if !moeWeightedSumSupportedDType(expert.DType()) || !moeWeightedSumSupportedDType(scores.DType()) || !moeWeightedSumSupportedDType(outDType) {
		return 0, 0, 0, 0, false
	}
	ed := expert.Dims()
	sd := scores.Dims()
	if len(ed) != 4 || len(sd) != 3 {
		return 0, 0, 0, 0, false
	}
	B, L, topK, H = ed[0], ed[1], ed[2], ed[3]
	if B <= 0 || L <= 0 || topK <= 0 || H <= 0 {
		return 0, 0, 0, 0, false
	}
	if sd[0] != B || sd[1] != L || sd[2] != topK {
		return 0, 0, 0, 0, false
	}
	return B, L, topK, H, true
}

func moeWeightedSumSupportedDType(dtype DType) bool {
	return dtype == DTypeFloat32 || dtype == DTypeFloat16 || dtype == DTypeBFloat16
}

func moeWeightedSumTemplateArgs(B, L, topK, H int) []gpuIntArg {
	return []gpuIntArg{
		{"B", B},
		{"L", L},
		{"TopK", topK},
		{"H", H},
	}
}

// FastMoEWeightedSum computes sum_k expert[B,L,k,H] * scores[B,L,k] and
// returns [B,L,H]. It returns ok=false for unsupported backends or shapes so
// callers can use a backend-neutral fallback.
func FastMoEWeightedSum(expert, scores *Array, outDType DType) (y *Array, ok bool) {
	B, L, topK, H, ok := moeWeightedSumValidate(expert, scores, outDType)
	if !ok {
		return nil, false
	}

	total := B * L * H
	gridX := (total + 255) / 256 * 256
	outs, ok := moeWeightedSum.applyMetal(gpuLaunch{
		dtypes: []gpuDTypeArg{{"OutT", outDType}},
		ints:   moeWeightedSumTemplateArgs(B, L, topK, H),
		outputs: []gpuOutputSpec{
			{"MOE_WEIGHTED_SUM", []int32{int32(B), int32(L), int32(H)}, outDType},
		},
		grid:        [3]int{gridX, 1, 1},
		threadGroup: [3]int{256, 1, 1},
		inputs:      []*Array{expert, scores},
	})
	if !ok {
		return nil, false
	}
	return outs[0], true
}
