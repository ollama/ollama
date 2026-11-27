package mlx

import "math"

// ShortGatherQMM returns GatherQMM(x, weight, scales, nil, nil, indices, true,
// group, bits, mode, nil, false) for NVFP4 (16, 4) or MXFP8 (32, 8) expert
// banks, reading each selected expert's weights once for all the routes that
// select it instead of once per route. indices [rows, topK] hold at most 128
// routes. x is FP32 [rows, 1, 1, K], one vector per row, or
// [rows, topK, 1, K], one per route. The FP32 result is [rows, topK, 1, N].
// Each route keeps the lane order of GatherQMM's vector kernels, so its
// result does not depend on the routes that share its expert. With
// quantized activations, as Inkling's are, at its expert widths on M5 it
// matches GatherQMM bit for bit. Out-of-range indices give NaN. It returns
// nil when the Metal kernel cannot run.
func ShortGatherQMM(x, weight, scales, indices *Array, mode string) *Array {
	plan, ok := planShortGather(x, weight, scales, indices, mode)
	if !ok {
		return nil
	}
	out, ok := plan.kernel.applyMetal(plan.launch([]*Array{x, weight, scales, indices}, nil,
		gpuOutputSpec{"short_gather_qmm", []int32{int32(plan.rows), int32(plan.topK), 1, int32(plan.channels)}, DTypeFloat32}))
	if !ok {
		return nil
	}
	return out[0]
}

// ShortGatherNVFP4SwiGLU is ShortGatherQMM for an NVFP4 gate/up bank with
// gate and up channels interleaved and one vector per row, finished as
// NVFP4SwiGLUQuantize finishes the products before it quantizes them. Each
// route's gate and up scale by
// xmax[route/xmaxStride] * maxima[ids[route]] / (denominator * maxProduct)
// and round to BF16; the softcapped SwiGLU (a cap of zero or less leaves its
// input uncapped) times the route's score rounds to BF16 again. It returns those values [rows*topK, N/2] in BF16, or nil when
// the Metal kernel cannot run.
func ShortGatherNVFP4SwiGLU(x, weight, scales, indices, xmax, maxima, denominator, scores *Array, xmaxStride int,
	gateCap, upCap float32, maxProduct int,
) *Array {
	plan, ok := planShortGather(x, weight, scales, indices, "nvfp4")
	if !ok || !plan.epilogue || x.Dim(1) != 1 || xmax == nil || maxima == nil ||
		denominator == nil || scores == nil || maxima.DType() != DTypeFloat32 || denominator.DType() != DTypeFloat32 ||
		scores.DType() != DTypeFloat32 || xmax.DType() != DTypeFloat32 || maxima.Size() != weight.Dim(0) ||
		scores.Size() != plan.routes || xmaxStride < 1 || xmax.Size()*xmaxStride != plan.routes {
		return nil
	}
	// Kernels index every input; a 0-d array would arrive as a scalar.
	denominator = denominator.Reshape(1)
	out, ok := shortGatherNVFP4SwiGLU.applyMetal(plan.launch([]*Array{x, weight, scales, indices, xmax, maxima, denominator, scores},
		[]gpuIntArg{
			{"XmaxStride", xmaxStride},
			{"MaxProduct", maxProduct},
			{"GateCapBits", int(math.Float32bits(gateCap))},
			{"UpCapBits", int(math.Float32bits(upCap))},
		},
		gpuOutputSpec{"short_gather_swiglu", []int32{int32(plan.routes), int32(plan.channels / 2)}, DTypeBFloat16}))
	if !ok {
		return nil
	}
	return out[0]
}

// shortGatherTuning is one short gather's launch geometry: each SIMD group
// computes outputs channels for a run of up to chunk routes that share an
// expert, simdgroups to a threadgroup.
type shortGatherTuning struct {
	outputs, simdgroups, chunk int
	// epilogue allows the NVFP4 gate/up gather to finish the SwiGLU itself.
	epilogue bool
}

// tuneShortGather chooses the geometry for rows of routes on a device
// architecture, or false to keep the stock GatherQMM. perRoute is true for
// one activation vector per route (the down product). Top-k routing picks an
// expert at most once per row, so at most rows routes share one.
func tuneShortGather(mode string, rows int, perRoute bool, architecture string) (shortGatherTuning, bool) {
	generation, _ := appleGPU(architecture)
	if rows == 1 && generation < 15 {
		// One row shares no expert, and before M3 the stock kernel is faster.
		return shortGatherTuning{}, false
	}
	if mode == "mxfp8" {
		return shortGatherTuning{outputs: 4, simdgroups: 2, chunk: min(rows, 8)}, true
	}
	// NVFP4 runs stop at four routes, past which the larger tile costs more
	// than the repeated weight reads it saves.
	t := shortGatherTuning{outputs: 2, simdgroups: 2, chunk: min(rows, 4)}
	if perRoute {
		t.outputs, t.simdgroups = 4, 4
	}
	if generation >= 17 {
		// M5: a down product of up to four rows is faster with 8 channels per
		// threadgroup, and a gate/up product of more rows with 16.
		if perRoute && rows <= 4 {
			t.outputs, t.simdgroups = 2, 4
		} else if !perRoute && rows > 4 {
			t.outputs, t.simdgroups = 4, 4
		}
		// Before M5 the epilogue gather made forwards of 8 and 16 rows 1-2%
		// slower (M2 Ultra, M4 Pro).
		t.epilogue = true
	}
	return t, true
}

// shortGatherPlan is one validated ShortGatherQMM launch.
type shortGatherPlan struct {
	shortGatherTuning
	kernel                    *gpuKernel
	rows, topK, routes, width int
	channels, rowStep, lanes  int
}

func planShortGather(x, weight, scales, indices *Array, mode string) (shortGatherPlan, bool) {
	if !MetalIsAvailable() || x == nil || weight == nil || scales == nil || indices == nil ||
		x.NumDims() != 4 || x.DType() != DTypeFloat32 || x.Dim(2) != 1 ||
		indices.NumDims() != 2 || (indices.DType() != DTypeUint32 && indices.DType() != DTypeInt32) ||
		weight.NumDims() != 3 || weight.DType() != DTypeUint32 || scales.NumDims() != 3 || scales.DType() != DTypeUint8 {
		return shortGatherPlan{}, false
	}
	p := shortGatherPlan{rows: indices.Dim(0), topK: indices.Dim(1), width: x.Dim(3), channels: weight.Dim(1), lanes: 8}
	experts := weight.Dim(0)
	p.routes = p.rows * p.topK
	if x.Dim(0) != p.rows || (x.Dim(1) != 1 && x.Dim(1) != p.topK) || p.routes < 1 || p.routes > 128 ||
		experts < 1 || scales.Dim(0) != experts || scales.Dim(1) != p.channels {
		return shortGatherPlan{}, false
	}
	switch mode {
	case "nvfp4":
		if p.width%16 != 0 || weight.Dim(2) != p.width/8 || scales.Dim(2) != p.width/16 {
			return shortGatherPlan{}, false
		}
		p.kernel = shortGatherNVFP4
	case "mxfp8":
		if p.width%32 != 0 || weight.Dim(2) != p.width/4 || scales.Dim(2) != p.width/32 {
			return shortGatherPlan{}, false
		}
		p.kernel = shortGatherMXFP8
		if p.width%256 != 0 {
			p.lanes = 4
		}
	default:
		return shortGatherPlan{}, false
	}
	arch := kernelArchitecture()
	tuning, ok := tuneShortGather(mode, p.rows, x.Dim(1) != 1, arch)
	if !ok || p.channels%(tuning.outputs*tuning.simdgroups) != 0 {
		return shortGatherPlan{}, false
	}
	p.shortGatherTuning = tuning
	p.rowStep = p.topK
	if x.Dim(1) != 1 {
		p.rowStep = 1
	}
	return p, true
}

func (p shortGatherPlan) launch(inputs []*Array, ints []gpuIntArg, output gpuOutputSpec) gpuLaunch {
	return gpuLaunch{
		inputs: inputs,
		ints: append([]gpuIntArg{
			{"Width", p.width},
			{"RowStep", p.rowStep},
			{"Outputs", p.outputs},
			{"Simdgroups", p.simdgroups},
			{"Lanes", p.lanes},
			{"Chunk", p.chunk},
		}, ints...),
		outputs:     []gpuOutputSpec{output},
		grid:        [3]int{p.channels / p.outputs * 32, p.routes, 1},
		threadGroup: [3]int{32 * p.simdgroups, 1, 1},
	}
}

// shortGatherHeader collects the routes that select one expert. Every
// threadgroup ballots all routes; the routes of an expert form runs of Chunk
// in route order, and only the first route of a run computes it.
const shortGatherHeader = `
#include <metal_stdlib>
using namespace metal;

// Fewer than 8 indices arrive in constant memory.
template <uint Chunk, typename Indices>
inline uint expert_run(Indices indices, uint total, uint route, uint expert, uint lane, thread uint (&routes)[Chunk]) {
    uint rank = 0, count = 0;
    for (uint window = 0; window < total; window += 32) {
        uint peer = window + lane;
        bool same = peer < total && uint(indices[peer]) == expert;
        uint matching = uint(static_cast<simd_vote::vote_t>(simd_ballot(same)));
        if (window + 32 <= route) {
            rank += popcount(matching);
            continue;
        }
        if (window <= route) {
            uint before = (1u << (route - window)) - 1u;
            if ((rank + popcount(matching & before)) % Chunk != 0) return 0;
            matching &= ~before;
        }
        while (matching != 0 && count < Chunk) {
            routes[count++] = window + ctz(matching);
            matching &= matching - 1;
        }
        if (count == Chunk) break;
    }
    return count;
}

// Select a register by per-lane indices without dynamic indexing.
template <uint Outputs, uint Chunk>
inline float pick_dot(thread const float (&dots)[Outputs][Chunk], uint out, uint j) {
    float value = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (uint o = 0; o < Outputs; ++o) {
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < Chunk; ++t) value = o == out && t == j ? dots[o][t] : value;
    }
    return value;
}

template <uint Chunk>
inline uint pick_route(thread const uint (&routes)[Chunk], uint j) {
    uint route = routes[0];
    _Pragma("clang loop unroll(full)")
    for (uint t = 1; t < Chunk; ++t) route = t == j ? routes[t] : route;
    return route;
}

inline float decode_e4m3(uchar bits) {
    uint magnitude = bits & 127u;
    ushort sign = ushort(bits & 128u) << 8;
    if (magnitude == 127u) return float(as_type<half>(ushort(sign | 0x7f80u)));
    return float(as_type<half>(ushort(sign | (magnitude << 7)))) * 256.0f;
}
`

// shortGatherSource is the kernel body shared by both formats. Each format's
// gather_dots fills dots[out][j] in every lane for channel first + out of
// route routes[j], j < count; lanes then store one value each.
const shortGatherSource = `
const uint channels = weight_shape[1];
const uint route = threadgroup_position_in_grid.y, lane = thread_index_in_simdgroup;
const uint first = (threadgroup_position_in_grid.x * Simdgroups + simdgroup_index_in_threadgroup) * Outputs;
const uint expert = uint(indices[route]);
if (expert >= uint(weight_shape[0])) {
    if (lane < Outputs) y[ulong(route) * channels + first + lane] = NAN;
    return;
}
uint routes[Chunk];
const uint count = expert_run<Chunk>(indices, indices_shape[0] * indices_shape[1], route, expert, lane, routes);
if (count == 0) return;
float dots[Outputs][Chunk] = {};
gather_dots<Chunk, Outputs, Chunk, Width, RowStep, Lanes>(count, x, weight, scales, ulong(expert) * channels + first,
                                                          routes, lane, dots);
const uint j = lane / Outputs, out = lane % Outputs;
if (j < count) y[ulong(pick_route(routes, j)) * channels + first + out] = pick_dot(dots, out, j);
`

// shortGatherSwiGLUSource is shortGatherSource for NVFP4 gate/up pairs
// (Outputs is even): lane pair * count + j finishes route j's SwiGLU of a pair.
const shortGatherSwiGLUSource = `
const uint channels = weight_shape[1];
const uint route = threadgroup_position_in_grid.y, lane = thread_index_in_simdgroup;
const uint first = (threadgroup_position_in_grid.x * Simdgroups + simdgroup_index_in_threadgroup) * Outputs;
const uint expert = uint(indices[route]);
if (expert >= uint(weight_shape[0])) {
    if (lane < Outputs / 2) y[ulong(route) * (channels / 2) + first / 2 + lane] = bfloat(NAN);
    return;
}
uint routes[Chunk];
const uint count = expert_run<Chunk>(indices, indices_shape[0] * indices_shape[1], route, expert, lane, routes);
if (count == 0) return;
float dots[Outputs][Chunk] = {};
gather_dots<Chunk, Outputs, Chunk, Width, RowStep, Lanes>(count, x, weight, scales, ulong(expert) * channels + first,
                                                          routes, lane, dots);
if (lane < count * (Outputs / 2)) {
    const uint j = lane % count, pair = lane / count;
    const uint r = pick_route(routes, j);
    const float value = swiglu_epilogue(pick_dot(dots, 2 * pair, j), pick_dot(dots, 2 * pair + 1, j),
                                        xmax[r / XmaxStride], maxima[expert], denominator[0], float(MaxProduct),
                                        scores[r], as_type<float>(GateCapBits), as_type<float>(UpCapBits));
    y[ulong(r) * (channels / 2) + first / 2 + pair] = bfloat(value);
}
`

// GatherQMM's NVFP4 vector kernel gives each lane 8 consecutive values of
// every 256. A lane sums each half of its 8 products, scales their sum by the
// block's E4M3 scale and adds it to the row total; simd_sum then reduces the
// lanes.
var shortGatherNVFP4 = &gpuKernel{
	name:    "short_gather_nvfp4",
	inputs:  []string{"x", "weight", "scales", "indices"},
	outputs: []string{"y"},
	metal:   gpuSource{source: shortGatherSource, header: shortGatherHeader + shortGatherNVFP4Header},
}

var shortGatherNVFP4SwiGLU = &gpuKernel{
	name:    "short_gather_nvfp4_swiglu",
	inputs:  []string{"x", "weight", "scales", "indices", "xmax", "maxima", "denominator", "scores"},
	outputs: []string{"y"},
	metal: gpuSource{source: shortGatherSwiGLUSource, header: shortGatherHeader + shortGatherNVFP4Header + expertSwiGLUMetal + `
// NVFP4SwiGLUQuantize's arithmetic before it quantizes.
inline float swiglu_epilogue(float gate, float up, float xmax, float wmax, float denominator, float max_product,
                             float score, float gate_cap, float up_cap) {
#pragma clang fp contract(off) reassociate(off)
    return expert_swiglu(gate, up, (xmax * wmax) / (denominator * max_product), score, gate_cap, up_cap);
}
`},
}

const shortGatherNVFP4Header = `
// E2M1 shifted into the low exponent range of binary16, rescaled after the
// conversion handles its zero and subnormal codes.
inline float decode_e2m1(uint bits) {
    return float(as_type<half>(ushort(((bits & 8u) << 12) | ((bits & 7u) << 9)))) * 16384.0f;
}

template <uint Tokens, uint Outputs, uint Chunk, uint Width, uint RowStep>
inline void nvfp4_dots(const device float* x, const device uint* packed, const device uchar* scales,
                       thread const uint (&routes)[Chunk], uint lane, thread float (&dots)[Outputs][Chunk]) {
    float sums[Outputs][Tokens] = {};
    for (uint col = lane * 8; col < Width; col += 256) {
        float4 left[Tokens], right[Tokens];
        for (uint j = 0; j < Tokens; ++j) {
            const device float4* v = (const device float4*)(x + ulong(routes[j] / RowStep) * Width + col);
            left[j] = v[0];
            right[j] = v[1];
        }
        for (uint out = 0; out < Outputs; ++out) {
            uint word = packed[out * (Width / 8) + col / 8];
            float scale = decode_e4m3(scales[out * (Width / 16) + col / 16]);
            float4 low, high;
            for (uint k = 0; k < 4; ++k) {
                low[k] = decode_e2m1(word >> (4 * k));
                high[k] = decode_e2m1(word >> (4 * k + 16));
            }
            for (uint j = 0; j < Tokens; ++j) {
                float4 a = left[j], b = right[j];
                float p = a.x * low.x + a.y * low.y + a.z * low.z + a.w * low.w;
                float q = b.x * high.x + b.y * high.y + b.z * high.z + b.w * high.w;
                sums[out][j] += (p + q) * scale;
            }
        }
    }
    for (uint out = 0; out < Outputs; ++out)
        for (uint j = 0; j < Tokens; ++j) dots[out][j] = simd_sum(sums[out][j]);
}

// Size the register tile to the run.
template <uint Tokens, uint Outputs, uint Chunk, uint Width, uint RowStep, uint Lanes>
inline void gather_dots(uint count, const device float* x, const device uint* weight, const device uchar* scales,
                        ulong row, thread const uint (&routes)[Chunk], uint lane, thread float (&dots)[Outputs][Chunk]) {
    if (count == Tokens) {
        nvfp4_dots<Tokens, Outputs, Chunk, Width, RowStep>(x, weight + row * (Width / 8), scales + row * (Width / 16),
                                                           routes, lane, dots);
    } else if constexpr (Tokens > 1) {
        gather_dots<Tokens - 1, Outputs, Chunk, Width, RowStep, Lanes>(count, x, weight, scales, row, routes, lane, dots);
    }
}
`

// GatherQMM's MXFP8 vector kernels give each lane Lanes consecutive values
// of every 32*Lanes: 8 when K is a multiple of 256, otherwise 4. A lane sums
// its products in order, scales the sum by the block's E8M0 scale and adds it
// to the row total; simd_sum then reduces the lanes.
var shortGatherMXFP8 = &gpuKernel{
	name:    "short_gather_mxfp8",
	inputs:  []string{"x", "weight", "scales", "indices"},
	outputs: []string{"y"},
	metal: gpuSource{source: shortGatherSource, header: shortGatherHeader + `
// E8M0 as GatherQMM decodes it: 0 is 2^-127 and 255 is NaN.
inline float decode_e8m0(uchar bits) {
    if (bits == 0) return as_type<float>(0x00400000u);
    if (bits == 255) return as_type<float>(0x7fc00000u);
    return as_type<float>(uint(bits) << 23);
}

template <uint Lanes>
inline void mxfp8_weights(const device uchar* row, thread float (&w)[Lanes]) {
    if constexpr (Lanes == 8) {
        uint2 bits = *(const device uint2*)row;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 4; ++i) {
            w[i] = decode_e4m3(uchar(bits.x >> (8 * i)));
            w[i + 4] = decode_e4m3(uchar(bits.y >> (8 * i)));
        }
    } else {
        uint bits = *(const device uint*)row;
        _Pragma("clang loop unroll(full)")
        for (uint i = 0; i < 4; ++i) w[i] = decode_e4m3(uchar(bits >> (8 * i)));
    }
}

template <uint Tokens, uint Outputs, uint Chunk, uint Width, uint RowStep, uint Lanes>
inline void mxfp8_dots(const device float* activation, const device uchar* weights, const device uchar* scales,
                       thread const uint (&routes)[Chunk], uint lane, thread float (&dots)[Outputs][Chunk]) {
    const device float* x[Tokens];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0; j < Tokens; ++j) x[j] = activation + ulong(routes[j] / RowStep) * Width;
    float sums[Outputs][Tokens] = {};
    for (uint col = lane * Lanes; col < Width; col += 32 * Lanes) {
        float w[Outputs][Lanes];
        float scale[Outputs];
        _Pragma("clang loop unroll(full)")
        for (uint out = 0; out < Outputs; ++out) {
            mxfp8_weights<Lanes>(weights + out * Width + col, w[out]);
            scale[out] = decode_e8m0(scales[out * (Width / 32) + col / 32]);
        }
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < Tokens; ++j) {
            float v[Lanes];
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < Lanes; i += 4) {
                float4 part = *(const device float4*)(x[j] + col + i);
                v[i] = part.x;
                v[i + 1] = part.y;
                v[i + 2] = part.z;
                v[i + 3] = part.w;
            }
            _Pragma("clang loop unroll(full)")
            for (uint out = 0; out < Outputs; ++out) {
                float accum = 0.0f;
                _Pragma("clang loop unroll(full)")
                for (uint i = 0; i < Lanes; ++i) accum += v[i] * w[out][i];
                float partial = scale[out] * accum;
                sums[out][j] += partial;
            }
        }
    }
    _Pragma("clang loop unroll(full)")
    for (uint out = 0; out < Outputs; ++out) {
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < Tokens; ++j) dots[out][j] = simd_sum(sums[out][j]);
    }
}

// Size the register tile to the run.
template <uint Tokens, uint Outputs, uint Chunk, uint Width, uint RowStep, uint Lanes>
inline void gather_dots(uint count, const device float* x, const device uint* weight, const device uchar* scales,
                        ulong row, thread const uint (&routes)[Chunk], uint lane, thread float (&dots)[Outputs][Chunk]) {
    if (count == Tokens) {
        mxfp8_dots<Tokens, Outputs, Chunk, Width, RowStep, Lanes>(x, (const device uchar*)weight + row * Width,
                                                                  scales + row * (Width / 32), routes, lane, dots);
    } else if constexpr (Tokens > 1) {
        gather_dots<Tokens - 1, Outputs, Chunk, Width, RowStep, Lanes>(count, x, weight, scales, row, routes, lane, dots);
    }
}
`},
}
