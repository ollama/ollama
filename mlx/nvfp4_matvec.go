package mlx

import "math"

// Stock short GatherQMM decodes the same weights for each route to an expert.
// A verification batch can share those reads across tokens that select the same expert.

// Keep GatherQMM's eight-element FP32 partial sums when accumulating the shared tile.
var gatherNVFP4QMV = &gpuKernel{
	name:    "gather_nvfp4_qmv",
	inputs:  []string{"activation", "packed", "scales", "experts"},
	outputs: []string{"result"},
	cuda: gpuSource{
		header: `
// E2M1 shifted into the low exponent range of binary16. The half-to-float
// conversion handles its zero and subnormal values before rescaling. These
// manual decodes reproduce the Metal float(as_type<half>(...)) results.
__device__ float short_decode_e2m1(unsigned int bits) {
    unsigned int sign = (bits >> 3) & 1u;
    unsigned int exp = (bits >> 1) & 3u;
    unsigned int mant = bits & 1u;
    float half = exp == 0
        ? ldexpf((float)(mant << 9), -24)
        : ldexpf(1.0f + (float)(mant << 9) * (1.0f / 1024.0f), (int)exp - 15);
    return (sign ? -half : half) * 16384.0f;
}

__device__ float short_decode_e4m3(unsigned char bits) {
    unsigned int magnitude = bits & 127u;
    unsigned int sign = (bits >> 7) & 1u;
    if (magnitude == 127u) return sign ? __uint_as_float(0xff800000u) : __uint_as_float(0x7f800000u);
    unsigned int exp = magnitude >> 3;
    unsigned int mant = (magnitude & 7u) << 7;
    float half = exp == 0
        ? ldexpf((float)mant, -24)
        : ldexpf(1.0f + (float)mant * (1.0f / 1024.0f), (int)exp - 15);
    return (sign ? -half : half) * 256.0f;
}

__device__ float short_warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return __shfl_sync(0xffffffff, v, 0);
}

// Collect the routes that select one expert. Every warp ballots all routes;
// the routes of an expert form runs of Chunk in route order, and only the
// first route of a run computes it.
template <unsigned int Chunk, typename Indices>
__device__ unsigned int expert_run(const Indices* indices, unsigned int total, unsigned int route,
                                   unsigned int expert, unsigned int lane, unsigned int (&routes)[Chunk]) {
    unsigned int rank = 0, count = 0;
    for (unsigned int window = 0; window < total; window += 32) {
        unsigned int peer = window + lane;
        bool same = peer < total && static_cast<unsigned int>(indices[peer]) == expert;
        unsigned int matching = __ballot_sync(0xffffffff, same);
        if (window + 32 <= route) {
            rank += __popc(matching);
            continue;
        }
        if (window <= route) {
            unsigned int before = (1u << (route - window)) - 1u;
            if ((rank + __popc(matching & before)) % Chunk != 0) return 0;
            matching &= ~before;
        }
        while (matching != 0 && count < Chunk) {
            routes[count++] = window + (__ffs(matching) - 1);
            matching &= matching - 1;
        }
        if (count == Chunk) break;
    }
    return count;
}

// Select a register by per-lane indices without dynamic indexing.
template <unsigned int Outputs, unsigned int Chunk>
__device__ float pick_dot(const float (&dots)[Outputs][Chunk], unsigned int out, unsigned int j) {
    float value = 0.0f;
    #pragma unroll
    for (unsigned int o = 0; o < Outputs; ++o) {
        #pragma unroll
        for (unsigned int t = 0; t < Chunk; ++t) value = o == out && t == j ? dots[o][t] : value;
    }
    return value;
}

template <unsigned int Chunk>
__device__ unsigned int pick_route(const unsigned int (&routes)[Chunk], unsigned int j) {
    unsigned int route = routes[0];
    #pragma unroll
    for (unsigned int t = 1; t < Chunk; ++t) route = t == j ? routes[t] : route;
    return route;
}

// GatherQMM's NVFP4 vector kernel gives each lane 8 consecutive values of
// every 256. A lane sums each half of its 8 products, scales their sum by the
// block's E4M3 scale and adds it to the row total; the warp then reduces the
// lanes.
template <unsigned int Tokens, unsigned int Outputs, unsigned int Chunk, unsigned int Width, unsigned int RowStep>
__device__ void nvfp4_dots(const float* activation, const unsigned int* packed, const unsigned char* scales,
                           const unsigned int (&routes)[Chunk], unsigned int lane, float (&dots)[Outputs][Chunk]) {
    float sums[Outputs][Tokens] = {};
    for (unsigned int col = lane * 8; col < Width; col += 256) {
        float left[Tokens][4], right[Tokens][4];
        #pragma unroll
        for (unsigned int j = 0; j < Tokens; ++j) {
            unsigned int row = routes[j] / RowStep;
            const float* v = activation + (unsigned long long)row * Width + col;
            left[j][0] = v[0]; left[j][1] = v[1]; left[j][2] = v[2]; left[j][3] = v[3];
            right[j][0] = v[4]; right[j][1] = v[5]; right[j][2] = v[6]; right[j][3] = v[7];
        }
        #pragma unroll
        for (unsigned int out = 0; out < Outputs; ++out) {
            unsigned int word = packed[out * (Width / 8) + col / 8];
            float scale = short_decode_e4m3(scales[out * (Width / 16) + col / 16]);
            float low[4] = {short_decode_e2m1(word), short_decode_e2m1(word >> 4),
                            short_decode_e2m1(word >> 8), short_decode_e2m1(word >> 12)};
            float high[4] = {short_decode_e2m1(word >> 16), short_decode_e2m1(word >> 20),
                             short_decode_e2m1(word >> 24), short_decode_e2m1(word >> 28)};
            #pragma unroll
            for (unsigned int j = 0; j < Tokens; ++j) {
                float p = left[j][0] * low[0] + left[j][1] * low[1] + left[j][2] * low[2] + left[j][3] * low[3];
                float q = right[j][0] * high[0] + right[j][1] * high[1] + right[j][2] * high[2] + right[j][3] * high[3];
                sums[out][j] += (p + q) * scale;
            }
        }
    }
    #pragma unroll
    for (unsigned int out = 0; out < Outputs; ++out)
        #pragma unroll
        for (unsigned int j = 0; j < Tokens; ++j) dots[out][j] = short_warp_sum(sums[out][j]);
}

// Size the register tile to the run.
template <unsigned int Tokens, unsigned int Outputs, unsigned int Chunk, unsigned int Width, unsigned int RowStep>
__device__ void gather_dots(unsigned int count, const float* activation, const unsigned int* packed,
                            const unsigned char* scales, unsigned long long row,
                            const unsigned int (&routes)[Chunk], unsigned int lane, float (&dots)[Outputs][Chunk]) {
    if (count == Tokens) {
        nvfp4_dots<Tokens, Outputs, Chunk, Width, RowStep>(activation, packed + row * (Width / 8), scales + row * (Width / 16), routes, lane, dots);
    } else if constexpr (Tokens > 1) {
        gather_dots<Tokens - 1, Outputs, Chunk, Width, RowStep>(count, activation, packed, scales, row, routes, lane, dots);
    }
}
`,
		source: `
const unsigned int route = blockIdx.y;
const unsigned int lane = threadIdx.x % 32;
const unsigned int warp = threadIdx.x / 32;
const unsigned int first = (blockIdx.x * Simdgroups + warp) * Outputs;
const unsigned int expert = static_cast<unsigned int>(experts[route]);
if (expert >= static_cast<unsigned int>(Experts)) {
    if (lane < Outputs) result[route * Channels + first + lane] = __uint_as_float(0x7fc00000u);
    return;
}
unsigned int routes[Chunk];
const unsigned int count = expert_run<Chunk>(experts, Routes, route, expert, lane, routes);
if (count == 0) return;
float dots[Outputs][Chunk] = {};
gather_dots<Chunk, Outputs, Chunk, Width, RowStep>(count, activation, packed, scales,
                                                   (unsigned long long)expert * Channels + first, routes, lane, dots);
const unsigned int j = lane / Outputs, out = lane % Outputs;
if (j < count) result[(unsigned long long)pick_route<Chunk>(routes, j) * Channels + first + out] = pick_dot<Outputs, Chunk>(dots, out, j);
`,
	},
}

// GatherNVFP4QMV multiplies FP32 vectors by selected transposed NVFP4
// matrices. For indices [B,T], x may be [B,1,1,K] (one vector per row) or
// [B,T,1,K] (one per route). Weights are [E,N,K/8] packed E2M1, with one
// E4M3FN scale per 16 elements. The FP32 output has shape [B,T,1,N].
// Routes that select the same expert form runs and read that expert's weights
// once. Metal handles up to 32 routes in one SIMD ballot; CUDA handles up to
// 128 routes with a multi-window ballot. Out-of-range IDs produce NaNs. It
// returns nil when the device, layout, or native reduction is unsupported.
func GatherNVFP4QMV(x, weight, scales, indices *Array) *Array {
	if x == nil || x.NumDims() != 4 || x.DType() != DTypeFloat32 || x.Dim(2) != 1 ||
		!validNVFP4RouteIndices(indices) || indices.Dim(0) != x.Dim(0) {
		return nil
	}
	rows, topK, width := indices.Dim(0), indices.Dim(1), x.Dim(3)
	if rows < 1 || topK < 1 ||
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
	if CUDAIsAvailable() {
		return gatherNVFP4QMVShortCUDA(x, weight, scales, indices, rows, topK, width, routes, experts, channels)
	}
	return nil
}

// gatherNVFP4QMVShortCUDA runs the deduplicating gather on CUDA for up to 128
// routes. It mirrors the Mac ShortGatherQMM NVFP4 tile geometry: the gate/up
// product reads one vector per row and the down product one per route.
func gatherNVFP4QMVShortCUDA(x, weight, scales, indices *Array, rows, topK, width, routes, experts, channels int) *Array {
	if routes < 1 || routes > 128 {
		return nil
	}
	outputs, simdgroups, chunk := 2, 2, min(rows, 4)
	rowStep := topK
	if x.Dim(1) != 1 {
		outputs, simdgroups, rowStep = 4, 4, 1
	}
	if channels%(outputs*simdgroups) != 0 {
		return nil
	}
	if channels/outputs > math.MaxInt32/32 {
		return nil
	}
	result, ok := gatherNVFP4QMV.applyCUDA(gpuLaunch{
		ints: []gpuIntArg{
			{"Width", width},
			{"RowStep", rowStep},
			{"Outputs", outputs},
			{"Simdgroups", simdgroups},
			{"Chunk", chunk},
			{"Experts", experts},
			{"Routes", routes},
			{"Channels", channels},
		},
		inputs:      []*Array{x, weight, scales, indices},
		outputs:     []gpuOutputSpec{{"GatherNVFP4QMV", []int32{int32(rows), int32(topK), 1, int32(channels)}, DTypeFloat32}},
		grid:        [3]int{channels / outputs * 32, routes, 1},
		threadGroup: [3]int{32 * simdgroups, 1, 1},
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
