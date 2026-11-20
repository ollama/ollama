package mlx

// #include "generated.h"
import "C"

import "math"

var matvecBF16Weights = &gpuKernel{
	name:    "matvec_bf16_weights",
	inputs:  []string{"x", "w"},
	outputs: []string{"out"},
	metal: gpuSource{source: `
uint lane = thread_index_in_simdgroup;
uint row = thread_position_in_grid.x / 32;
if (row >= ROWS) return;
float4 part(0);
uint base = row * WIDTH + lane;
for (uint i = 0; i < WIDTH; i += 128) {
    part.x = fma(float(w[base + i]), x[lane + i], part.x);
    part.y = fma(float(w[base + i + 32]), x[lane + i + 32], part.y);
    part.z = fma(float(w[base + i + 64]), x[lane + i + 64], part.z);
    part.w = fma(float(w[base + i + 96]), x[lane + i + 96], part.w);
}
float answer = simd_sum((part.x + part.y) + (part.z + part.w));
if (lane == 0) out[row] = answer;
`},
	cuda: gpuSource{source: `
unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int row = thread / 32;
if (row >= ROWS) return;
unsigned int lane = threadIdx.x % 32;
const auto* weights = w + row * WIDTH + lane;
float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
for (unsigned int i = 0; i < WIDTH; i += 128) {
    a = fmaf(static_cast<float>(weights[i]), x[lane + i], a);
    b = fmaf(static_cast<float>(weights[i + 32]), x[lane + i + 32], b);
    c = fmaf(static_cast<float>(weights[i + 64]), x[lane + i + 64], c);
    d = fmaf(static_cast<float>(weights[i + 96]), x[lane + i + 96], d);
}
float sum = (a + b) + (c + d);
for (int step = 16; step > 0; step /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, step);
}
if (lane == 0) out[row] = sum;
`},
}

// TryMatvecBF16Weights projects a single FP32 input row through the transpose
// of a BF16 weight matrix. GPU kernels accumulate and return FP32.
// It returns false if the shape or backend is unsupported or the native
// graph cannot be constructed. Evaluation follows the usual MLX semantics.
func TryMatvecBF16Weights(x, weight *Array) (*Array, bool) {
	if weight.NumDims() != 2 || weight.DType() != DTypeBFloat16 || x.NumDims() == 0 || x.DType() != DTypeFloat32 {
		return nil, false
	}
	rows, width := weight.Dim(0), weight.Dim(1)
	const lanes, tileWidth = 32, 128
	if rows == 0 || width == 0 || width%tileWidth != 0 ||
		x.Dim(-1) != width || x.Size() != width ||
		rows > math.MaxInt32/lanes || int64(rows)*int64(width) > math.MaxUint32 {
		return nil, false
	}
	// Backend availability alone does not tell us whether the caller selected
	// a GPU stream. A CPU call must not disable the GPU kernel for later calls.
	var deviceType C.mlx_device_type
	if err := mlxError(C.mlx_device_get_type(&deviceType, DefaultDevice().ctx)); err != nil || deviceType != C.MLX_GPU {
		return nil, false
	}

	shape := make([]int32, x.NumDims())
	for i := range shape {
		shape[i] = int32(x.Dim(i))
	}
	shape[len(shape)-1] = int32(rows)
	launch := gpuLaunch{
		inputs:      []*Array{x, weight},
		ints:        []gpuIntArg{{"ROWS", rows}, {"WIDTH", width}},
		grid:        [3]int{rows * lanes, 1, 1},
		threadGroup: [3]int{tileWidth, 1, 1},
		outputs:     []gpuOutputSpec{{"OUT", shape, DTypeFloat32}},
	}
	if outs, ok := matvecBF16Weights.applyCUDA(launch); ok {
		return outs[0], true
	}
	if outs, ok := matvecBF16Weights.applyMetal(launch); ok {
		return outs[0], true
	}
	return nil, false
}
