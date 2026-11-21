package mlx

import "math"

// LoopControllerMix applies a looped layer's message gate. For each row,
// gain and forget are the FP32 products of compressed [rows, rank] with
// weight [2*width, rank], and the result is
// input * (1 - tanh(forget)) + message * tanh(gain) in FP32.
// input and message are [..., width]. It returns nil when the Metal kernel
// cannot run.
func LoopControllerMix(input, message, compressed, weight *Array) *Array {
	if !MetalIsAvailable() || input == nil || message == nil || compressed == nil || weight == nil ||
		input.NumDims() == 0 || weight.NumDims() != 2 || weight.DType() != DTypeFloat32 ||
		compressed.DType() != DTypeFloat32 {
		return nil
	}
	width, rank := input.Dim(-1), weight.Dim(1)
	rows := input.Size() / max(width, 1)
	if width == 0 || rank == 0 || rank > 256 || weight.Dim(0) != 2*width || message.Size() != input.Size() ||
		message.Dim(-1) != width || compressed.Size() != rows*rank || uint64(input.Size()) > math.MaxInt32 {
		return nil
	}
	const rowsPerGroup, threads = 8, 256
	shape := make([]int32, input.NumDims())
	for i := range shape {
		shape[i] = int32(input.Dim(i))
	}
	out, ok := loopControllerMix.applyMetal(gpuLaunch{
		inputs:      []*Array{input, message, compressed, weight},
		ints:        []gpuIntArg{{"Width", width}, {"Rank", rank}, {"RowsPerGroup", rowsPerGroup}},
		outputs:     []gpuOutputSpec{{"loop_controller_mix", shape, DTypeFloat32}},
		grid:        [3]int{(width + threads - 1) / threads * threads, (rows + rowsPerGroup - 1) / rowsPerGroup, 1},
		threadGroup: [3]int{threads, 1, 1},
	})
	if !ok {
		return nil
	}
	return out[0]
}

var loopControllerMix = &gpuKernel{
	name:    "loop_controller_mix",
	inputs:  []string{"input", "message", "compressed", "weight"},
	outputs: []string{"out"},
	metal: gpuSource{source: `
uint rows = compressed_shape[0];
for (uint i = 1; i + 1 < compressed_ndim; ++i) rows *= compressed_shape[i];
uint first = threadgroup_position_in_grid.y * RowsPerGroup;
threadgroup float shared[RowsPerGroup * Rank];
for (uint i = thread_index_in_threadgroup; i < RowsPerGroup * Rank; i += threads_per_threadgroup.x) {
    uint r = first + i / Rank;
    shared[i] = r < rows ? compressed[r * Rank + i % Rank] : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint c = thread_position_in_grid.x;
if (c >= Width) return;
const device float* wg = weight + size_t(c) * Rank;
const device float* wf = weight + size_t(Width + c) * Rank;
float gain[RowsPerGroup] = {}, forget[RowsPerGroup] = {};
for (uint k = 0; k < Rank; ++k) {
    float a = wg[k], b = wf[k];
    for (uint r = 0; r < RowsPerGroup; ++r) {
        gain[r] = fma(shared[r * Rank + k], a, gain[r]);
        forget[r] = fma(shared[r * Rank + k], b, forget[r]);
    }
}
for (uint r = 0; r < RowsPerGroup && first + r < rows; ++r) {
    size_t at = size_t(first + r) * Width + c;
    float keep = 1.0f - metal::precise::tanh(forget[r]);
    out[at] = float(input[at]) * keep + float(message[at]) * metal::precise::tanh(gain[r]);
}
`},
}
