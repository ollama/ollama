package mlx

import (
	"math"
	"sync"
)

// MLX derives its NVFP4 scale from the block maximum divided by 6. The 4-over-6
// variant also tries a divisor of 4 and compares both reconstruction errors in one pass.
var nvfp4Activation = &gpuKernel{
	name:    "nvfp4_activation",
	inputs:  []string{"x"},
	outputs: []string{"inner", "amax"},
	cuda:    nvfp4ActivationCUDA,
	metal: gpuSource{
		header: `
#include <metal_stdlib>
using namespace metal;

inline float round_e4m3(float x) {
    // Nonnegative E4M3FN with round-to-nearest-even.
    uint exponent = (as_type<uint>(x) >> 23) & 255u;
    float spacing = as_type<float>((max(exponent, 121u) - 3u) << 23);
    return rint(x / spacing) * spacing;
}

inline float round_e2m1(float x) {
    float m = abs(x);
    float q = m > 0.25f ? 0.5f : 0.0f;
    if (m >= 0.75f) q = 1.0f;
    if (m > 1.25f) q = 1.5f;
    if (m >= 1.75f) q = 2.0f;
    if (m > 2.5f) q = 3.0f;
    if (m >= 3.5f) q = 4.0f;
    if (m > 5.0f) q = 6.0f;
    return x < 0.0f ? -q : q;
}
`,
		source: `
#pragma clang fp contract(off) reassociate(off)
uint row = threadgroup_position_in_grid.y;
uint worker = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint group = simdgroup_index_in_threadgroup;
uint start = row * W + worker * 16;
bool active = worker < W / 16;
float data[16];
float block_max = 0.0f;
for (uint i = 0; i < 16; ++i) {
    data[i] = active ? float(x[start + i]) : 0.0f;
    block_max = max(block_max, abs(data[i]));
}

threadgroup float partial[G / 32];
float simd_amax = simd_max(block_max);
if (lane == 0) partial[group] = simd_amax;
threadgroup_barrier(mem_flags::mem_threadgroup);
float row_max = simd_max(lane < G / 32 ? partial[lane] : 0.0f);
if (worker == 0) amax[row] = row_max;
if (!active) return;

float normalizer = min(float(MaxProduct) / row_max, FLT_MAX);
float decoder = 1.0f / normalizer;
float outer = row_max * (1.0f / float(MaxProduct));
float scale_a = round_e4m3(min(block_max * (normalizer * (1.0f / 6.0f)), 448.0f));
float scale_b = round_e4m3(min(block_max * (normalizer * 0.25f), 448.0f));
float inv_a = min(1.0f / (scale_a * decoder), FLT_MAX);
float inv_b = min(1.0f / (scale_b * decoder), FLT_MAX);
float candidate_a[16], candidate_b[16];
float error_a = 0.0f, error_b = 0.0f;
for (uint i = 0; i < 16; ++i) {
    candidate_a[i] = round_e2m1(data[i] * inv_a) * scale_a;
    candidate_b[i] = round_e2m1(data[i] * inv_b) * scale_b;
    error_a += abs(data[i] - candidate_a[i] * outer);
    error_b += abs(data[i] - candidate_b[i] * outer);
}
for (uint i = 0; i < 16; ++i) {
    inner[start + i] = error_b < error_a ? candidate_b[i] : candidate_a[i];
}
`,
	},
}

// QuantizeNVFP4Activation4Over6 returns FP32 reconstructions of the inner E2M1
// values times their E4M3 block scales, and one FP32 absolute maximum per row.
// Multiply the reconstructed values by maxima / maxProduct to restore the
// input scale. For each block of 16, the scale selected from the 4 and 6
// ranges minimizes total absolute error. Ties select the 6 range.
func QuantizeNVFP4Activation4Over6(x *Array, maxProduct int) (values, maxima *Array) {
	if x == nil || x.NumDims() == 0 || x.Dim(-1) == 0 || x.Dim(-1)%16 != 0 {
		panic("mlx: NVFP4 activations require rows of complete 16-value blocks")
	}
	if maxProduct < 1 || maxProduct > Nvfp4MaxProduct {
		panic("mlx: NVFP4 activation normalization exceeds the FP4 × FP8 range")
	}
	if values, maxima := fastNVFP4Activation(x, maxProduct); values != nil {
		return values, maxima
	}
	shape := x.Dims()
	width := shape[len(shape)-1]
	blocks := x.Reshape(x.Size()/width, width/16, 16)
	denominator := FromValue(float32(maxProduct))
	inverse := FromValue(1 / float32(maxProduct))
	if MetalIsAvailable() {
		result := compiledNVFP4Blocks(blocks, denominator, inverse)
		values, maxima = result[0], result[1]
	} else {
		values, maxima = quantizeNVFP4Blocks(blocks, denominator, inverse)
	}
	values = values.Reshape(shape...)
	shape[len(shape)-1] = 1
	return values, maxima.Reshape(shape...)
}

// One SIMD lane handles 16 channels, with up to 256 lanes per threadgroup.
func fastNVFP4Activation(x *Array, maxProduct int) (values, maxima *Array) {
	if x == nil || x.NumDims() == 0 || x.DType() != DTypeBFloat16 {
		return nil, nil
	}
	width, size := x.Dim(-1), x.Size()
	if width < 16 || width%16 != 0 || width > 16*256 || size == 0 || uint64(size) > math.MaxUint32 {
		return nil, nil
	}
	useCUDA := !MetalIsAvailable()
	if useCUDA && !nvfp4ActivationCUDAEnabled() {
		return nil, nil
	}
	workers := (width/16 + 31) / 32 * 32
	if useCUDA && size/width > math.MaxInt32/workers {
		return nil, nil
	}
	shape := make([]int32, x.NumDims())
	for i := range shape {
		shape[i] = int32(x.Dim(i))
	}
	maxShape := append([]int32(nil), shape...)
	maxShape[len(maxShape)-1] = 1
	launch := gpuLaunch{
		ints:    []gpuIntArg{{"W", width}, {"G", workers}, {"MaxProduct", maxProduct}},
		outputs: []gpuOutputSpec{{"NVFP4Values", shape, DTypeFloat32}, {"NVFP4Maxima", maxShape, DTypeFloat32}},
		grid:    [3]int{workers, size / width, 1}, threadGroup: [3]int{workers, 1, 1},
		inputs: []*Array{x},
	}
	apply := nvfp4Activation.applyMetal
	if useCUDA {
		launch.grid = [3]int{workers * (size / width), 1, 1}
		apply = nvfp4Activation.applyCUDA
	}
	output, ok := apply(launch)
	if !ok {
		return nil, nil
	}
	return output[0], output[1]
}

// The MLX graph materializes both 4-over-6 candidates. Compare their errors in
// registers using MLX's SM120 sum order so close decisions choose the same scale.
var nvfp4ActivationCUDA = gpuSource{
	header: `
__device__ float nvfp4_round_e4m3(float input) {
    unsigned exponent = __float_as_uint(input) >> 23;
    int spacing_exponent = int(exponent > 121 ? exponent : 121) - 130;
    float spacing = ldexpf(1.0f, spacing_exponent);
    return nearbyintf(__fdiv_rn(input, spacing)) * spacing;
}

__device__ float nvfp4_round_e2m1(float input) {
    float magnitude = fabsf(input);
    float value = magnitude > 0.25f ? 0.5f : 0.0f;
    if (magnitude >= 0.75f) value = 1.0f;
    if (magnitude > 1.25f) value = 1.5f;
    if (magnitude >= 1.75f) value = 2.0f;
    if (magnitude > 2.5f) value = 3.0f;
    if (magnitude >= 3.5f) value = 4.0f;
    if (magnitude > 5.0f) value = 6.0f;
    return input < 0.0f ? -value : value;
}
`,
	source: `
unsigned token = blockIdx.x;
unsigned worker = threadIdx.x;
unsigned lane = worker % 32;
unsigned start = token * W + worker * 16;
bool active = worker < W / 16;
float data[16];
float block_max = 0.0f;
for (int i = 0; i < 16; ++i) {
    data[i] = active ? float(x[start + i]) : 0.0f;
    block_max = fmaxf(block_max, fabsf(data[i]));
}

__shared__ float warp_maxima[G / 32];
float token_max = block_max;
for (int stride = 16; stride; stride >>= 1) {
    token_max = fmaxf(token_max, __shfl_down_sync(0xffffffff, token_max, stride));
}
if (lane == 0) warp_maxima[worker / 32] = token_max;
__syncthreads();
token_max = lane < G / 32 ? warp_maxima[lane] : 0.0f;
for (int stride = 16; stride; stride >>= 1) {
    token_max = fmaxf(token_max, __shfl_down_sync(0xffffffff, token_max, stride));
}
token_max = __shfl_sync(0xffffffff, token_max, 0);
if (worker == 0) amax[token] = token_max;
if (!active) return;

float float_max = __uint_as_float(0x7f7fffff);
float normalizer = fminf(__fdiv_rn(float(MaxProduct), token_max), float_max);
float decoder = __fdiv_rn(1.0f, normalizer);
float outer = __fmul_rn(token_max, 1.0f / float(MaxProduct));
float values[2][16], errors[2];
for (int candidate = 0; candidate < 2; ++candidate) {
    float reciprocal = candidate == 0 ? 1.0f / 6.0f : 0.25f;
    float limit = __fmul_rn(block_max, __fmul_rn(normalizer, reciprocal));
    float scale = nvfp4_round_e4m3(fminf(limit, 448.0f));
    float inverse = fminf(__fdiv_rn(1.0f, __fmul_rn(scale, decoder)), float_max);

    // The CUDA reduction sums adjacent sets of four, then the warp
    // combines its partials. Preserve separate multiplies and subtraction.
    float partial[4] = {};
    for (int i = 0; i < 16; ++i) {
        float quantized = nvfp4_round_e2m1(__fmul_rn(data[i], inverse));
        values[candidate][i] = __fmul_rn(quantized, scale);
        float restored = __fmul_rn(values[candidate][i], outer);
        partial[i / 4] += fabsf(__fsub_rn(data[i], restored));
    }
    errors[candidate] = (partial[0] + partial[2]) + (partial[1] + partial[3]);
}
int selected = errors[1] < errors[0];
for (int i = 0; i < 16; ++i) inner[start + i] = values[selected][i];
`,
}

// The CUDA error reduction matches MLX's SM120 order.
var nvfp4ActivationCUDAEnabled = sync.OnceValue(func() bool {
	if !CUDAIsAvailable() {
		return false
	}
	_, arch := kernelDevice()
	return arch == "sm_120"
})

var compiledNVFP4Blocks = Compile("NVFP4ActivationBlocks", func(inputs ...*Array) []*Array {
	values, maxima := quantizeNVFP4Blocks(inputs[0], inputs[1], inputs[2])
	return []*Array{values, maxima}
}, Shapeless())

func quantizeNVFP4Blocks(blocks, maxProduct, inverseMaxProduct *Array) (values, maxima *Array) {
	blocks = blocks.AsType(DTypeFloat32)
	blockMax := blocks.Abs().MaxAxis(-1, true)
	maxima = blockMax.MaxAxis(-2, true)
	one := FromValue(float32(1))
	limit := FromValue(float32(math.MaxFloat32))
	normalizer := Minimum(Div(maxProduct, maxima), limit)
	decoder := Div(one, normalizer)
	outer := Mul(maxima, inverseMaxProduct)
	candidate := func(limitValue float32) (values, errorSum *Array) {
		scale := Mul(blockMax, MulScalar(normalizer, 1/limitValue))
		scale = Minimum(scale, FromValue(float32(448)))
		scale = FromFP8(ToFP8(scale), DTypeFloat32)
		inverse := Minimum(Div(one, Mul(scale, decoder)), limit)
		values = Mul(roundE2M1(Mul(blocks, inverse)), scale)
		errorSum = Sum(Sub(blocks, Mul(values, outer)).Abs(), -1, true)
		return values, errorSum
	}
	full, fullError := candidate(6)
	alternate, alternateError := candidate(4)
	return Where(alternateError.Less(fullError), alternate, full), maxima
}

func roundE2M1(x *Array) *Array {
	zero := FromValue(float32(0))
	magnitude := x.Abs()
	result := zero
	levels := [...]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}
	for index := 1; index < len(levels); index++ {
		midpoint := FromValue((levels[index-1] + levels[index]) / 2)
		selectLevel := magnitude.Greater(midpoint)
		if index%2 == 0 {
			selectLevel = midpoint.LessEqual(magnitude)
		}
		result = Where(selectLevel, FromValue(levels[index]), result)
	}
	return Where(x.Less(zero), Neg(result), result)
}
