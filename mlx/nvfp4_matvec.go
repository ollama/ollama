package mlx

import "math"

// Stock short GatherQMM decodes the same weights for each route to an expert.
// A verification batch can share those reads across tokens that select the same expert.

const nvfp4DecodeMetal = `
#include <metal_stdlib>
using namespace metal;

// Shift E2M1 into the low exponent range of binary16. The half-to-float
// conversion handles its zero and subnormal values before rescaling.
inline float nvfp4_e2m1(uint bits) {
    ushort fp16 = ushort(((bits & 8u) << 12) | ((bits & 7u) << 9));
    return float(as_type<half>(fp16)) * 16384.0f;
}

inline float nvfp4_e4m3(uchar bits) {
    uint magnitude = bits & 127u;
    ushort sign = ushort(bits & 128u) << 8;
    if (magnitude == 127u) return float(as_type<half>(ushort(sign | 0x7f80u)));
    return float(as_type<half>(ushort(sign | (magnitude << 7)))) * 256.0f;
}

`

// Keep GatherQMM's eight-element FP32 partial sums when accumulating the shared tile.
var gatherNVFP4QMV = &gpuKernel{
	name:    "gather_nvfp4_qmv",
	inputs:  []string{"activation", "packed", "scales", "experts"},
	outputs: []string{"result"},
	metal: gpuSource{
		header: nvfp4DecodeMetal + `template<uint Tokens, uint Width, uint Channels, uint Outputs, uint RowStep>
inline void nvfp4_gather_dot(
    const device float* activation, const device uint* packed,
    const device uchar* scales, device float* result,
    thread const uint* routes, uint expert, uint first, uint lane) {
    float sums[Outputs][Tokens] = {};
    for (uint col = lane * 8; col < Width; col += 256) {
        float4 left[Tokens], right[Tokens];
        for (uint j = 0; j < Tokens; ++j) {
            uint row = routes[j] / RowStep;
            const device float4* x = (const device float4*)(activation + row * Width + col);
            left[j] = x[0];
            right[j] = x[1];
        }
        for (uint out = 0; out < Outputs; ++out) {
            if (Channels % Outputs && first + out >= Channels) break;
            uint row = expert * Channels + first + out;
            uint word = packed[row * (Width / 8) + col / 8];
            float scale = nvfp4_e4m3(scales[row * (Width / 16) + col / 16]);
            float4 low, high;
            for (uint k = 0; k < 4; ++k) {
                low[k] = nvfp4_e2m1(word >> (4 * k));
                high[k] = nvfp4_e2m1(word >> (4 * k + 16));
            }
            for (uint j = 0; j < Tokens; ++j) {
                float4 x = left[j], y = right[j];
                float a = x.x * low.x + x.y * low.y + x.z * low.z + x.w * low.w;
                float b = y.x * high.x + y.y * high.y + y.z * high.z + y.w * high.w;
                sums[out][j] += (a + b) * scale;
            }
        }
    }
    for (uint out = 0; out < Outputs; ++out) {
        if (Channels % Outputs && first + out >= Channels) break;
        for (uint j = 0; j < Tokens; ++j) {
            float value = simd_sum(sums[out][j]);
            if (lane == 0) result[routes[j] * Channels + first + out] = value;
        }
    }
}
`,
		source: `
uint route = threadgroup_position_in_grid.y;
uint lane = thread_index_in_simdgroup;
uint first = (threadgroup_position_in_grid.x * (Workers / 32) + simdgroup_index_in_threadgroup) * Outputs;
if (Channels % ((Workers / 32) * Outputs) && first >= Channels) return;
uint expert = experts[route];
if (expert >= Experts) {
    if (lane == 0) {
        for (uint out = 0; out < Outputs && first + out < Channels; ++out)
            result[route * Channels + first + out] = NAN;
    }
    return;
}
// The current route always contributes a bit, including on partial SIMDs.
uint matching = simd_or(lane < Routes && uint(experts[lane]) == expert ? (1u << lane) : 0u);
uint routes[4] = {route, 0, 0, 0};
uint count = 1;
// The register tile fits up to four routes. Larger runs compute one dot per route.
if (popcount(matching) <= 4) {
    if (ctz(matching) != route) return;
    matching &= matching - 1;
    while (matching) {
        routes[count++] = ctz(matching);
        matching &= matching - 1;
    }
}
switch (count) {
case 1: nvfp4_gather_dot<1, Width, Channels, Outputs, RowStep>(activation, packed, scales, result, routes, expert, first, lane); break;
case 2: nvfp4_gather_dot<2, Width, Channels, Outputs, RowStep>(activation, packed, scales, result, routes, expert, first, lane); break;
case 3: nvfp4_gather_dot<3, Width, Channels, Outputs, RowStep>(activation, packed, scales, result, routes, expert, first, lane); break;
case 4: nvfp4_gather_dot<4, Width, Channels, Outputs, RowStep>(activation, packed, scales, result, routes, expert, first, lane); break;
}
`,
	},
}

// GatherNVFP4QMV multiplies FP32 vectors by selected transposed NVFP4
// matrices. For indices [B,T], x may be [B,1,1,K] (one vector per row) or
// [B,T,1,K] (one per route). Weights are [E,N,K/8] packed E2M1, with one
// E4M3FN scale per 16 elements. The FP32 output has shape [B,T,1,N].
// Up to 32 routes fit in one SIMD ballot. Out-of-range IDs produce NaNs.
// It returns nil when the device, layout, or native reduction is unsupported.
func GatherNVFP4QMV(x, weight, scales, indices *Array) *Array {
	if x == nil || x.NumDims() != 4 || x.DType() != DTypeFloat32 || x.Dim(2) != 1 ||
		!validNVFP4RouteIndices(indices) || indices.Dim(0) != x.Dim(0) {
		return nil
	}
	rows, topK, width := indices.Dim(0), indices.Dim(1), x.Dim(3)
	if rows < 1 || topK < 1 || rows > 32/topK ||
		(x.Dim(1) != 1 && x.Dim(1) != topK) || !validNVFP4Bank(width, weight, scales) {
		return nil
	}
	routes, experts, channels := rows*topK, weight.Dim(0), weight.Dim(1)
	// GatherQMM's aligned QMV kernel reduces 16 elements per lane. This
	// kernel matches the eight-element reduction of the other QMV shapes.
	if (width%512 == 0 && channels%8 == 0) ||
		uint64(x.Size()) > math.MaxUint32 || uint64(routes)*uint64(channels) > math.MaxUint32 {
		return nil
	}
	routesPerVector, outputs, workers := topK, 2, 64
	if x.Dim(1) != 1 {
		routesPerVector, outputs, workers = 1, 4, 128
	}
	tile := outputs * (workers / 32)
	groups := 1 + (channels-1)/tile
	if groups > math.MaxInt32/workers || !UsesM5MatrixKernels() {
		return nil
	}
	result, ok := gatherNVFP4QMV.applyMetal(gpuLaunch{
		ints: []gpuIntArg{
			{"Width", width},
			{"Channels", channels},
			{"Experts", experts},
			{"Routes", routes},
			{"RowStep", routesPerVector},
			{"Outputs", outputs},
			{"Workers", workers},
		},
		inputs:      []*Array{x, weight, scales, indices},
		outputs:     []gpuOutputSpec{{"GatherNVFP4QMV", []int32{int32(rows), int32(topK), 1, int32(channels)}, DTypeFloat32}},
		grid:        [3]int{groups * workers, routes, 1},
		threadGroup: [3]int{workers, 1, 1},
	})
	if !ok {
		return nil
	}
	return result[0]
}

func validNVFP4RouteIndices(indices *Array) bool {
	return indices != nil && indices.NumDims() == 2 &&
		(indices.DType() == DTypeInt32 || indices.DType() == DTypeUint32)
}

func validNVFP4Bank(width int, weight, scales *Array) bool {
	if width <= 0 || width%16 != 0 || weight == nil || scales == nil ||
		weight.NumDims() != 3 || weight.DType() != DTypeUint32 ||
		scales.NumDims() != 3 || scales.DType() != DTypeUint8 {
		return false
	}
	experts, channels := weight.Dim(0), weight.Dim(1)
	// The shaders address both packed arrays with 32-bit element offsets.
	return experts > 0 && channels > 0 && weight.Dim(2) == width/8 &&
		scales.Dim(0) == experts && scales.Dim(1) == channels && scales.Dim(2) == width/16 &&
		uint64(experts)*uint64(channels) <= math.MaxUint32/uint64(width/8)
}
