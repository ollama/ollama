package mlx

import "math"

// expertSwiGLUMetal is the softcapped SwiGLU of one rounded gate and up pair
// (a cap of zero or less leaves its input uncapped), and the expert epilogue
// around it: the outer scale and both BF16 roundings.
const expertSwiGLUMetal = `
inline float softcapped_swiglu(float g, float u, float gate_cap, float up_cap) {
#pragma clang fp contract(off) reassociate(off)
    float sigmoid = 1.0f / (1.0f + metal::precise::exp(-g));
    float gate = gate_cap > 0.0f ? metal::precise::tanh(g / gate_cap) * gate_cap : g;
    float up = up_cap > 0.0f ? metal::precise::tanh(u / up_cap) * up_cap : u;
    return (gate * sigmoid) * up;
}
inline float expert_swiglu(float gate, float up, float scale, float score, float gate_cap, float up_cap) {
#pragma clang fp contract(off) reassociate(off)
    float g = float(bfloat(gate * scale));
    float u = float(bfloat(up * scale));
    return float(bfloat(softcapped_swiglu(g, u, gate_cap, up_cap) * score));
}
`

// NVFP4SwiGLUQuantize finishes an NVFP4 gate/up projection and quantizes the
// down projection's input. inner [routes, 2*width] holds FP32 or BF16 products
// with gate and up interleaved. Each route's outer scale is
// xmax[route/xmaxStride] * maxima[ids[route]] / (denominator * maxProduct).
// The scaled gate and up round to BF16, pass through the softcapped SwiGLU
// (a cap of zero or less leaves its input uncapped) and the route's score,
// and round to BF16 again before the 4-over-6
// quantization of QuantizeNVFP4Activation4Over6. It returns the reconstructed
// values [routes, width] in dtype and their FP32 row maxima [routes], or nils
// when the Metal kernel cannot run.
func NVFP4SwiGLUQuantize(inner, xmax, maxima, denominator, ids, scores *Array, xmaxStride int,
	gateCap, upCap float32, maxProduct int, dtype DType,
) (values, amax *Array) {
	if (!MetalIsAvailable() && !CUDAIsAvailable()) || inner == nil || inner.NumDims() != 2 || ids == nil || scores == nil {
		return nil, nil
	}
	routes, width := inner.Dim(0), inner.Dim(1)/2
	if routes < 1 || width < 16 || width%16 != 0 || width > 16*1024 || inner.Dim(1) != 2*width ||
		ids.Size() != routes || scores.Size() != routes || xmaxStride < 1 || xmax.Size()*xmaxStride != routes ||
		maxima.DType() != DTypeFloat32 || denominator.DType() != DTypeFloat32 || scores.DType() != DTypeFloat32 ||
		(dtype != DTypeFloat32 && dtype != DTypeBFloat16) || uint64(inner.Size()) > math.MaxInt32 {
		return nil, nil
	}
	per := 16
	if MetalIsAvailable() {
		per = tuneNVFP4Quantize(routes, width, kernelArchitecture())
	}
	workers := (width/per + 31) / 32 * 32
	// Kernels index every input; a 0-d array would arrive as a scalar.
	denominator = denominator.Reshape(1)
	apply := nvfp4SwiGLUQuantize.applyMetal
	if !MetalIsAvailable() {
		apply = nvfp4SwiGLUQuantize.applyCUDA
	}
	out, ok := apply(gpuLaunch{
		inputs: []*Array{inner, xmax, maxima, denominator, ids, scores},
		dtypes: []gpuDTypeArg{{"OutT", dtype}},
		ints: []gpuIntArg{
			{"Width", width},
			{"Per", per},
			{"Workers", workers},
			{"XmaxStride", xmaxStride},
			{"MaxProduct", maxProduct},
			{"GateCapBits", int(math.Float32bits(gateCap))},
			{"UpCapBits", int(math.Float32bits(upCap))},
		},
		outputs: []gpuOutputSpec{
			{"nvfp4_swiglu_values", []int32{int32(routes), int32(width)}, dtype},
			{"nvfp4_swiglu_amax", []int32{int32(routes)}, DTypeFloat32},
		},
		grid:        [3]int{workers, routes, 1},
		threadGroup: [3]int{workers, 1, 1},
	})
	if !ok {
		return nil, nil
	}
	return out[0], out[1]
}

var nvfp4SwiGLUQuantize = &gpuKernel{
	name:    "nvfp4_swiglu_quantize",
	inputs:  []string{"inner", "xmax", "maxima", "denominator", "ids", "scores"},
	outputs: []string{"values", "amax"},
	metal: gpuSource{header: nvfp4RoundingMetal + nvfp4SpreadMetal + expertSwiGLUMetal, source: `
#pragma clang fp contract(off) reassociate(off)
uint route = threadgroup_position_in_grid.y;
uint worker = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup, group = simdgroup_index_in_threadgroup;
// Blocks of 16 never straddle the active edge, so whole blocks are idle.
bool active = worker < Width / Per;
float gateCap = as_type<float>(GateCapBits), upCap = as_type<float>(UpCapBits);
float scale = (xmax[route / XmaxStride] * maxima[ids[route]]) / (denominator[0] * float(MaxProduct));
float score = scores[route];
float data[Per];
float block_max = 0.0f;
for (uint i = 0; i < Per; ++i) {
    float value = 0.0f;
    if (active) {
        size_t at = size_t(route) * (2 * Width) + 2 * (worker * Per + i);
        value = expert_swiglu(float(inner[at]), float(inner[at + 1]), scale, score, gateCap, upCap);
    }
    data[i] = value;
    block_max = max(block_max, abs(value));
}
for (uint offset = 1; offset < 16 / Per; offset *= 2) {
    block_max = max(block_max, simd_shuffle_xor(block_max, ushort(offset)));
}

threadgroup float partial[Workers / 32];
float simd_amax = simd_max(block_max);
if (lane == 0) partial[group] = simd_amax;
threadgroup_barrier(mem_flags::mem_threadgroup);
float row_max = simd_max(lane < Workers / 32 ? partial[lane] : 0.0f);
if (worker == 0) amax[route] = row_max;

float quantized[Per];
nvfp4_4over6_lanes<Per>(data, block_max, row_max, float(MaxProduct), lane, quantized);
if (active) {
    size_t start = size_t(route) * Width + worker * Per;
    for (uint i = 0; i < Per; ++i) values[start + i] = OutT(quantized[i]);
}
`},
	cuda: gpuSource{
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
__device__ float softcapped_swiglu(float g, float u, float gate_cap, float up_cap) {
    float sigmoid = 1.0f / (1.0f + expf(-g));
    float gate = gate_cap > 0.0f ? tanhf(g / gate_cap) * gate_cap : g;
    float up = up_cap > 0.0f ? tanhf(u / up_cap) * up_cap : u;
    return (gate * sigmoid) * up;
}
__device__ float expert_swiglu(float gate, float up, float scale, float score, float gate_cap, float up_cap) {
    float g = (float)(__nv_bfloat16)(gate * scale);
    float u = (float)(__nv_bfloat16)(up * scale);
    return (float)(__nv_bfloat16)(softcapped_swiglu(g, u, gate_cap, up_cap) * score);
}
`,
		source: `
const int route = blockIdx.y;
const int worker = threadIdx.x;
const int lane = threadIdx.x % 32;
const int warp = threadIdx.x / 32;
bool active = worker < Width / Per;
float gateCap = __uint_as_float(GateCapBits), upCap = __uint_as_float(UpCapBits);
float scale = (xmax[route / XmaxStride] * maxima[(int)ids[route]]) / (denominator[0] * (float)MaxProduct);
float score = scores[route];
float data[Per];
float block_max = 0.0f;
for (int i = 0; i < Per; ++i) {
    float value = 0.0f;
    if (active) {
        size_t at = (size_t)route * (2 * Width) + 2 * (worker * Per + i);
        value = expert_swiglu((float)inner[at], (float)inner[at + 1], scale, score, gateCap, upCap);
    }
    data[i] = value;
    block_max = fmaxf(block_max, fabsf(value));
}
for (int offset = 1; offset < 16 / Per; offset *= 2) {
    block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, offset));
}
__shared__ float partial[Workers / 32];
float warp_amax = block_max;
for (int stride = 16; stride; stride >>= 1) warp_amax = fmaxf(warp_amax, __shfl_down_sync(0xffffffff, warp_amax, stride));
if (lane == 0) partial[warp] = warp_amax;
__syncthreads();
float row_max = lane < Workers / 32 ? partial[lane] : 0.0f;
for (int stride = 16; stride; stride >>= 1) row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, stride));
row_max = __shfl_sync(0xffffffff, row_max, 0);
if (worker == 0) amax[route] = row_max;
if (!active) return;
float float_max = __uint_as_float(0x7f7fffff);
float normalizer = fminf(__fdiv_rn((float)MaxProduct, row_max), float_max);
float decoder = __fdiv_rn(1.0f, normalizer);
float outer = __fmul_rn(row_max, 1.0f / (float)MaxProduct);
float cand[2][Per], errors[2];
for (int candidate = 0; candidate < 2; ++candidate) {
    float reciprocal = candidate == 0 ? 1.0f / 6.0f : 0.25f;
    float limit = __fmul_rn(block_max, __fmul_rn(normalizer, reciprocal));
    float qscale = nvfp4_round_e4m3(fminf(limit, 448.0f));
    float inverse = fminf(__fdiv_rn(1.0f, __fmul_rn(qscale, decoder)), float_max);
    float partial4[4] = {};
    for (int i = 0; i < Per; ++i) {
        float quantized = nvfp4_round_e2m1(__fmul_rn(data[i], inverse));
        cand[candidate][i] = __fmul_rn(quantized, qscale);
        float restored = __fmul_rn(cand[candidate][i], outer);
        partial4[i / 4] += fabsf(__fsub_rn(data[i], restored));
    }
    errors[candidate] = (partial4[0] + partial4[2]) + (partial4[1] + partial4[3]);
}
int selected = errors[1] < errors[0];
size_t start = (size_t)route * Width + worker * Per;
for (int i = 0; i < Per; ++i) values[start + i] = (OutT)cand[selected][i];
`,
	},
}

// MXFP8SwiGLUQuantize finishes an MXFP8 gate/up projection. inner
// [routes, 2*width] holds the products with gate and up interleaved. Gate and
// up round to BF16, pass through the softcapped SwiGLU (a cap of zero or less
// leaves its input uncapped) and the route's score, round to BF16 again and
// are quantized to MXFP8 for the down projection.
// It returns the reconstructed values [routes, width] in dtype, or nil when
// the Metal kernel cannot run.
func MXFP8SwiGLUQuantize(inner, scores *Array, gateCap, upCap float32, dtype DType) *Array {
	if !MetalIsAvailable() || inner == nil || inner.NumDims() != 2 || scores == nil || scores.DType() != DTypeFloat32 {
		return nil
	}
	routes, width := inner.Dim(0), inner.Dim(1)/2
	if routes < 1 || width%32 != 0 || inner.Dim(1) != 2*width || scores.Size() != routes ||
		uint64(inner.Size()) > math.MaxInt32 || (dtype != DTypeFloat32 && dtype != DTypeBFloat16) {
		return nil
	}
	out, ok := mxfp8SwiGLUQuantize.applyMetal(gpuLaunch{
		inputs: []*Array{inner, scores},
		dtypes: []gpuDTypeArg{{"OutT", dtype}},
		ints: []gpuIntArg{
			{"Width", width},
			{"GateCapBits", int(math.Float32bits(gateCap))},
			{"UpCapBits", int(math.Float32bits(upCap))},
		},
		outputs:     []gpuOutputSpec{{"mxfp8_swiglu_values", []int32{int32(routes), int32(width)}, dtype}},
		grid:        [3]int{width, routes, 1},
		threadGroup: [3]int{min(width, 256), 1, 1},
	})
	if !ok {
		return nil
	}
	return out[0]
}

var mxfp8SwiGLUQuantize = &gpuKernel{
	name:    "mxfp8_swiglu_quantize",
	inputs:  []string{"inner", "scores"},
	outputs: []string{"values"},
	metal: gpuSource{header: mxfp8Metal + expertSwiGLUMetal, source: `
#pragma clang fp contract(off) reassociate(off)
uint c = thread_position_in_grid.x, route = thread_position_in_grid.y;
float gateCap = as_type<float>(GateCapBits), upCap = as_type<float>(UpCapBits);
size_t at = size_t(route) * (2 * Width) + 2 * c;
float g = float(bfloat(float(inner[at])));
float u = float(bfloat(float(inner[at + 1])));
float v = float(bfloat(softcapped_swiglu(g, u, gateCap, upCap) * scores[route]));
float block_max = simd_max(abs(v));
values[size_t(route) * Width + c] = OutT(mxfp8_value(v, block_max));
`},
}

// ExpertSum sums each token's topK routed expert outputs. inner is
// [routes, width]; route k of token t is row t*topK+k of inner, or row
// order[t*topK+k] when order is given. With NVFP4 maxima, each route is first
// scaled by amax[route] * maxima[ids[route]] / (denominator * maxProduct).
// Each route rounds to BF16, the sum is FP32 and the result is BF16.
// It returns nil when the Metal kernel cannot run.
func ExpertSum(inner, amax, maxima, denominator, ids, order *Array, topK, maxProduct int) *Array {
	if (!MetalIsAvailable() && !CUDAIsAvailable()) || inner == nil || inner.NumDims() != 2 || topK < 1 || inner.Dim(0)%topK != 0 ||
		uint64(inner.Size()) > math.MaxInt32 {
		return nil
	}
	routes, width := inner.Dim(0), inner.Dim(1)
	scaled := 0
	if maxima != nil {
		if amax.Size() != routes || ids.Size() != routes || maxima.DType() != DTypeFloat32 || denominator.DType() != DTypeFloat32 {
			return nil
		}
		scaled = 1
		denominator = denominator.Reshape(1)
	} else {
		amax, maxima, denominator, ids = inner, inner, inner, inner
	}
	reorder := 1
	if order == nil {
		order, reorder = inner, 0
	} else if order.Size() != routes {
		return nil
	}
	apply := expertSum.applyMetal
	if !MetalIsAvailable() {
		apply = expertSum.applyCUDA
	}
	out, ok := apply(gpuLaunch{
		inputs: []*Array{inner, amax, maxima, denominator, ids, order},
		dtypes: []gpuDTypeArg{{"T", DTypeBFloat16}},
		ints: []gpuIntArg{
			{"Width", width},
			{"TopK", topK},
			{"Reorder", reorder},
			{"Scaled", scaled},
			{"MaxProduct", maxProduct},
		},
		outputs:     []gpuOutputSpec{{"expert_sum", []int32{int32(routes / topK), int32(width)}, DTypeBFloat16}},
		grid:        [3]int{width, routes / topK, 1},
		threadGroup: [3]int{min(width, 256), 1, 1},
	})
	if !ok {
		return nil
	}
	return out[0]
}

var expertSum = &gpuKernel{
	name:    "expert_sum",
	inputs:  []string{"inner", "amax", "maxima", "denominator", "ids", "order"},
	outputs: []string{"out"},
	metal: gpuSource{source: `
#pragma clang fp contract(off) reassociate(off)
uint c = thread_position_in_grid.x, token = thread_position_in_grid.y;
if (c >= Width) return;
float sum = 0.0f;
for (uint k = 0; k < TopK; ++k) {
    uint route = token * TopK + k;
    if (Reorder) route = uint(order[route]);
    float value = float(inner[size_t(route) * Width + c]);
    if (Scaled) {
        value *= (float(amax[route]) * float(maxima[uint(ids[route])])) / (float(denominator[0]) * float(MaxProduct));
    }
    value = float(bfloat(value));
    sum = k == 0 ? value : sum + value;
}
out[size_t(token) * Width + c] = bfloat(sum);
`},
	cuda: gpuSource{source: `
const int c = blockIdx.x * blockDim.x + threadIdx.x;
const int token = blockIdx.y;
if (c >= Width) return;
float sum = 0.0f;
for (int k = 0; k < TopK; ++k) {
    int route = token * TopK + k;
    if (Reorder) route = (int)order[route];
    float value = (float)inner[(size_t)route * Width + c];
    if (Scaled) {
        value *= ((float)amax[route] * (float)maxima[(int)ids[route]]) / ((float)denominator[0] * (float)MaxProduct);
    }
    value = (float)(__nv_bfloat16)value;
    sum = k == 0 ? value : sum + value;
}
out[(size_t)token * Width + c] = static_cast<T>(sum);
`},
}
