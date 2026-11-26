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
	if !MetalIsAvailable() || inner == nil || inner.NumDims() != 2 || ids == nil || scores == nil {
		return nil, nil
	}
	routes, width := inner.Dim(0), inner.Dim(1)/2
	if routes < 1 || width < 16 || width%16 != 0 || width > 16*1024 || inner.Dim(1) != 2*width ||
		ids.Size() != routes || scores.Size() != routes || xmaxStride < 1 || xmax.Size()*xmaxStride != routes ||
		maxima.DType() != DTypeFloat32 || denominator.DType() != DTypeFloat32 || scores.DType() != DTypeFloat32 ||
		(dtype != DTypeFloat32 && dtype != DTypeBFloat16) || uint64(inner.Size()) > math.MaxInt32 {
		return nil, nil
	}
	arch := kernelArchitecture()
	per := tuneNVFP4Quantize(routes, width, arch)
	workers := (width/per + 31) / 32 * 32
	// Kernels index every input; a 0-d array would arrive as a scalar.
	denominator = denominator.Reshape(1)
	out, ok := nvfp4SwiGLUQuantize.applyMetal(gpuLaunch{
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
	if !MetalIsAvailable() || inner == nil || inner.NumDims() != 2 || topK < 1 || inner.Dim(0)%topK != 0 ||
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
	out, ok := expertSum.applyMetal(gpuLaunch{
		inputs: []*Array{inner, amax, maxima, denominator, ids, order},
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
}
