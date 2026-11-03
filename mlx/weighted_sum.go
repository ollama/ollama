package mlx

const (
	weightedSumValues = iota
	weightedSumScores
	weightedSumIndices
)

const (
	weightedSumWidth = iota
	weightedSumExperts
	weightedSumGather
)

var gatherWeightedSum = &gpuKernel{
	name:    "gather_weighted_sum",
	inputs:  []string{weightedSumValues: "values", weightedSumScores: "scores", weightedSumIndices: "indices"},
	outputs: []string{"out"},
	metal: gpuSource{source: `
auto col = thread_position_in_grid.x;
auto row = thread_position_in_grid.y;
if (col >= D) return;
float sum = 0;
for (int k = 0; k < K; ++k) {
  auto slot = row * K + k;
  auto source = GATHER ? indices[slot] : slot;
  sum += float(values[source * D + col]) * scores[slot];
}
out[row * D + col] = static_cast<T>(sum);
`},
	fallback: func(launch gpuLaunch) []*Array {
		in := launch.inputs
		indices := in[weightedSumIndices]
		if launch.ints[weightedSumGather].value == 0 {
			indices = nil
		}
		return []*Array{gatherWeightedSumGraph(in[weightedSumValues], in[weightedSumScores], indices)}
	},
}

func gatherWeightedSumGraph(values, scores, indices *Array) *Array {
	if indices != nil {
		values = Take(values, Flatten(indices), 0)
	}
	values = Reshape(values, int32(scores.Dim(0)), int32(scores.Dim(1)), int32(values.Dim(1)))
	return Sum(Mul(values.AsType(DTypeFloat32), ExpandDims(scores, -1)), 1, false).AsType(values.DType())
}

// GatherWeightedSum reduces K expert rows per token in FP32, then rounds to
// values' dtype. values is [N*K, D], scores is [N, K], and optional indices
// maps each token/expert slot to a row in values. Fusing the gather and reduction
// avoids materializing reordered expert outputs and their FP32 weighted product.
func GatherWeightedSum(values, scores, indices *Array) *Array {
	if values.NumDims() != 2 || scores.NumDims() != 2 || values.Dim(0) != scores.Size() {
		panic("mlx: GatherWeightedSum requires values [N*K, D] and scores [N, K]")
	}
	if indices != nil && indices.Size() != scores.Size() {
		panic("mlx: GatherWeightedSum indices must have N*K entries")
	}
	scores = scores.AsType(DTypeFloat32)
	if values.DType() != DTypeFloat32 && values.DType() != DTypeBFloat16 && values.DType() != DTypeFloat16 {
		return gatherWeightedSumGraph(values, scores, indices)
	}
	N, K, D := scores.Dim(0), scores.Dim(1), values.Dim(1)
	if N == 0 || K == 0 || D == 0 {
		return gatherWeightedSumGraph(values, scores, indices)
	}
	gather := 0
	if indices == nil {
		indices = FromValues([]int32{0}, 1)
	} else {
		indices = indices.AsType(DTypeInt32)
		gather = 1
	}
	return gatherWeightedSum.run(gpuLaunch{
		dtypes:      []gpuDTypeArg{{"T", values.DType()}},
		ints:        []gpuIntArg{weightedSumWidth: {"D", D}, weightedSumExperts: {"K", K}, weightedSumGather: {"GATHER", gather}},
		outputs:     []gpuOutputSpec{{"WEIGHTED_SUM", []int32{int32(N), int32(D)}, values.DType()}},
		grid:        [3]int{D, N, 1},
		threadGroup: [3]int{min(D, 256), 1, 1},
		inputs:      []*Array{weightedSumValues: values, weightedSumScores: scores, weightedSumIndices: indices},
	})[0]
}
