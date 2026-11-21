package mlx

import "math"

// Accelerating stock GatherQMM on M5 requires matrix tiles confined to one expert.
// Build runs of up to 32 rows from sorted IDs on the GPU to avoid a host readback.
var nvfp4RouteRuns = &gpuKernel{
	name:    "nvfp4_route_runs",
	inputs:  []string{"indices"},
	outputs: []string{"runs"},
	metal: gpuSource{
		source: `
uint rows = indices_shape[0];
uint capacity = (rows + 31) / 32 + Experts - 1;
constexpr uint tileRows = 32;
uint expert = thread_index_in_threadgroup;
uint low = 0, high = rows;
while (low < high) {
    uint middle = (low + high) / 2;
    if (uint(indices[middle]) < expert)
        low = middle + 1;
    else
        high = middle;
}
uint start = low;
high = rows;
while (low < high) {
    uint middle = (low + high) / 2;
    if (uint(indices[middle]) <= expert)
        low = middle + 1;
    else
        high = middle;
}
uint end = low, count = (end - start + tileRows - 1) / tileRows;
threadgroup uint prefix[Experts];
prefix[expert] = count;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride = 1; stride < Experts; stride *= 2) {
    uint left = expert >= stride ? prefix[expert - stride] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix[expert] += left;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Prefix sums place each expert’s runs contiguously in the output.
uint begin = prefix[expert] - count;
for (uint i = 0; i < count; ++i) {
    uint index = (begin + i) * 3, row = start + i * tileRows;
    runs[index] = row;
    runs[index + 1] = min(uint(tileRows), end - row);
    runs[index + 2] = expert;
}
// Empty records pad the grid to its host-known upper bound.
for (uint i = prefix[Experts - 1] + expert; i < capacity; i += Experts) {
    runs[i * 3] = 0;
    runs[i * 3 + 1] = 0;
    runs[i * 3 + 2] = 0;
}
`,
	},
}

// A nonzero finite float is an integer multiple of 2^quantum.
const nvfp4QuantumMetal = `
inline int nvfp4_binary_quantum(float f) {
    uint raw = as_type<uint>(abs(f));
    if (raw == 0)
        return 127;
    return int(raw >> 23) - 150 + int(ctz((raw & 0x7fffffu) | 0x800000u));
}
// Bound exact accumulation by 0.999 * 2^(quantum + 24). Adjusting the
// exponent of 0.999 avoids ldexp in the tile loop. Zero forces native
// replay if an intermediate could be subnormal; exponents above 104
// put the bound beyond the largest finite FP32 value.
inline float nvfp4_exact_limit(int quantum) {
    uint bits = as_type<uint>(0.999f) + (uint(quantum + 24) << 23);
    return quantum < -126 ? 0.f : quantum > 104 ? INFINITY : as_type<float>(bits);
}

`

// Matrix accumulation can change FP32 rounding. Compute each input block's peak
// and binary quantum once so all output tiles can reuse the same exactness bound.
var nvfp4ActivationBounds = &gpuKernel{
	name:    "nvfp4_activation_bounds",
	inputs:  []string{"values"},
	outputs: []string{"quantum", "maxima"},
	metal: gpuSource{
		header: nvfp4DecodeMetal + nvfp4QuantumMetal,
		source: `
constexpr uint blockSize = 32;

uint block = thread_position_in_grid.x;
int exponent = 127;
float peak = 0;
for (uint i = 0; i < blockSize; ++i) {
    float v = float(values[block * blockSize + i]);
    exponent = min(exponent, nvfp4_binary_quantum(v));
    peak = max(peak, abs(v));
}
quantum[block] = exponent;
maxima[block] = peak;
`,
	},
}

// Stock FP32 GatherQMM is slower on long runs routed to one expert. Reuse decoded
// weights on M5 matrix tiles and replay ambiguous sums in native FP32 order.
var gatherNVFP4QMM = &gpuKernel{
	name:    "gather_nvfp4_qmm",
	inputs:  []string{"activation", "weights", "scales", "runs", "quantum", "maxima"},
	outputs: []string{"result"},
	metal: gpuSource{
		header: nvfp4DecodeMetal + nvfp4QuantumMetal + `
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_simdgroup_matrix>
`,
		source: `
constexpr uint BM = 32, BN = 64, BK = 32, Workers = 128;

uint worker = thread_index_in_threadgroup, lane = thread_index_in_simdgroup,
     simd = simdgroup_index_in_threadgroup;
uint tile = threadgroup_position_in_grid.y, start = runs[tile * 3], rows = runs[tile * 3 + 1],
     expert = runs[tile * 3 + 2];
if (rows == 0)
    return;
uint col0 = threadgroup_position_in_grid.x * BN;
uint tileRow = (simd / (BN / 32)) * 16, tileColumn = (simd % (BN / 32)) * 32;
threadgroup bfloat decodedWeights[BN * BK];
threadgroup float final[BM * BN];
using Shape = metal::dextents<int32_t, 2>;
using Strides = metal::array<int32_t, 2>;
using ActivationTile = metal::tensor<device bfloat, Shape, metal::tensor_inline>;
using WeightTile = metal::tensor<threadgroup bfloat, Shape, metal::tensor_inline>;
using ResultTile = metal::tensor<threadgroup float, Shape, metal::tensor_inline>;
constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, BK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> product;
auto accumulator =
    product.template get_destination_cooperative_tensor<ActivationTile, WeightTile, float>();
for (uint i = 0; i < accumulator.get_capacity(); ++i)
    accumulator[i] = 0;
// The minimum binary quantum of all products so far also divides every
// possible FP32 prefix, including prefixes rounded to a coarser power of two.
int smallest = 127;
for (uint offset = 0; offset < Width; offset += BK) {
    for (uint i = worker; i < BN * (BK / 8); i += Workers) {
        uint channel = col0 + i / (BK / 8), at = offset + (i % (BK / 8)) * 8;
        uint wr = expert * Channels + channel;
        uint word = 0;
        float gain = 0;
        if (Channels % BN == 0 || channel < Channels) {
            word = weights[wr * (Width / 8) + at / 8];
            gain = nvfp4_e4m3(scales[wr * (Width / 16) + at / 16]);
        }
        for (uint j = 0; j < 8; ++j)
            decodedWeights[i * 8 + j] = bfloat(nvfp4_e2m1(word >> (4 * j)) * gain);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tileRow < rows) {
        int valid = int(min(16u, rows - tileRow));
        uint block = (start + tileRow + min(lane, 15u)) * (Width / BK) + offset / BK;
        float rowPeak = lane < valid ? maxima[block] : 0.f;
        float inputPeak = simd_max(rowPeak);
        int rowQuantum = lane < valid ? quantum[block] : 127;
        int minInputQuantum = simd_min(rowQuantum);
        int weightQuantum = 127;
        float weightPeak = 0;
        for (uint h = 0; h < BK / 16; ++h) {
            uint channel = col0 + tileColumn + lane;
            float gain = 0;
            if (Channels % BN == 0 || channel < Channels)
                gain = nvfp4_e4m3(scales[(expert * Channels + channel) * (Width / 16) + offset / 16 + h]);
            weightQuantum = min(weightQuantum, nvfp4_binary_quantum(gain) - 1);
            weightPeak = max(weightPeak, abs(gain) * 6.f);
        }
        smallest = min(smallest, minInputQuantum + simd_min(weightQuantum));
        float columnPeak = weightPeak;
        int columnQuantum = weightQuantum;
        weightPeak = simd_max(weightPeak);
        float partialPeak = 0;
        for (uint i = 0; i < accumulator.get_capacity(); ++i)
            partialPeak = max(partialPeak, abs(accumulator[i]));
        partialPeak = simd_max(partialPeak);
        // Every add is exact if every possible intermediate sum is an
        // integer smaller than 2^24 in this binary quantum. The maximum
        // absolute prefix plus the block L1 bound covers both reduction orders.
        float upper = partialPeak + float(BK) * inputPeak * weightPeak;
        // Native FP32 arithmetic flushes subnormal intermediates; the
        // matrix unit can preserve them. Replay if a prefix can be subnormal.
        bool safe = upper < nvfp4_exact_limit(smallest);
        float before[16] = {};
        uint replay = 0;
        // A tile can fail the common bound while individual entries pass.
        if (!safe) {
            for (uint i = 0; i < accumulator.get_capacity(); ++i) {
                auto loc = accumulator.get_multidimensional_index(i);
                uint row = loc[1], column = loc[0];
                float lhs = simd_shuffle(rowPeak, row), rhs = simd_shuffle(columnPeak, column);
                int productQuantum =
                    simd_shuffle(rowQuantum, row) + simd_shuffle(columnQuantum, column);
                float prev = accumulator[i];
                before[i] = prev;
                float bound = abs(prev) + float(BK) * lhs * rhs;
                int quantum = min(nvfp4_binary_quantum(prev), productQuantum);
                float limit = nvfp4_exact_limit(quantum);
                if (row < uint(valid) && !(bound < limit))
                    replay |= 1u << i;
            }
        }
        auto lhs = ActivationTile((device bfloat *)activation + (start + tileRow) * Width + offset,
                                  Shape(BK, valid), Strides{1, Width});
        auto rhs = WeightTile(decodedWeights + tileColumn * BK, Shape(BK, 32), Strides{1, BK});
        product.run(lhs, rhs, accumulator);
        // Reproduce the native FP32 reduction on the ambiguous outputs.
        while (replay) {
            uint i = ctz(replay);
            replay &= replay - 1;
            auto loc = accumulator.get_multidimensional_index(i);
            uint row = loc[1], column = loc[0];
            float value = before[i];
            for (uint h = 0; h < BK; ++h) {
                value = fma(float(activation[(start + tileRow + row) * Width + offset + h]),
                            float(decodedWeights[(tileColumn + column) * BK + h]), value);
            }
            accumulator[i] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
accumulator.store(ResultTile(final + tileRow * BN + tileColumn, Shape(32, 16), Strides{1, BN}));
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = worker; i < rows * BN; i += Workers) {
    uint channel = col0 + i % BN;
    if (Channels % BN == 0 || channel < Channels)
        result[(start + i / BN) * Channels + channel] = final[i];
}
`,
	},
}

// PrefersCompactNVFP4QMM reports when compact tiles beat native sorted FP32
// GatherQMM. On M5 the compatibility reduction benefits from compact tiles.
// Native grouped work is faster alongside MPP attention on the M4 Pro.
func PrefersCompactNVFP4QMM() bool {
	return hasMetalTensorOps() && UsesM5MatrixKernels() && metalMatmul() == metalMatmulScalar
}

// GatherNVFP4QMM multiplies sorted BF16 vectors [R,1,1,K] by transposed
// NVFP4 matrices [E,N,K/8] with E4M3FN scales [E,N,K/16]. Its FP32 output
// [R,1,1,N] matches grouped GatherQMM on the FP32 reconstruction of the
// vectors. Reconstructed NVFP4 activations fit exactly in BF16.
// The caller supplies one valid, nondecreasing expert ID per row in [R,1].
// It returns nil for unsupported devices or shapes.
func GatherNVFP4QMM(values, weight, scales, indices *Array) *Array {
	if !hasMetalTensorOps() || values == nil || values.NumDims() != 4 || values.DType() != DTypeBFloat16 ||
		values.Dim(1) != 1 || values.Dim(2) != 1 || !validNVFP4RouteIndices(indices) {
		return nil
	}
	rows, width := values.Dim(0), values.Dim(3)
	if indices.Dim(0) != rows || indices.Dim(1) != 1 || width%32 != 0 ||
		!validNVFP4Bank(width, weight, scales) {
		return nil
	}
	experts, channels := weight.Dim(0), weight.Dim(1)
	// The prefix scan uses one hardware thread per expert. Native GatherQMM
	// uses the sorted matrix reduction for at least 16 routes and 4 per expert.
	if experts > 1024 || rows < 16 || rows/experts < 4 ||
		uint64(rows)*uint64(width) > math.MaxUint32 || uint64(rows)*uint64(channels) > math.MaxUint32 {
		return nil
	}
	columnTiles := 1 + (channels-1)/64
	if columnTiles > math.MaxInt32/128 {
		return nil
	}
	switch metalMatmul() {
	case metalMatmulUnsupported:
		return nil
	case metalMatmulTensor:
		return GatherQMM(values.AsType(DTypeFloat32), weight, scales, nil, nil, indices, true, 16, 4, "nvfp4", nil, true)
	}
	// Each expert after the first contributes at most one extra partial run.
	capacity := (rows+31)/32 + experts - 1
	runs, ok := nvfp4RouteRuns.applyMetal(gpuLaunch{
		ints:        []gpuIntArg{{"Experts", experts}},
		inputs:      []*Array{indices},
		outputs:     []gpuOutputSpec{{"NVFP4RouteRuns", []int32{int32(capacity), 3}, DTypeUint32}},
		grid:        [3]int{experts, 1, 1},
		threadGroup: [3]int{experts, 1, 1},
	})
	if !ok {
		return nil
	}

	metaShape := []int32{int32(rows), int32(width / 32)}
	bounds, ok := nvfp4ActivationBounds.applyMetal(gpuLaunch{
		inputs: []*Array{values},
		outputs: []gpuOutputSpec{
			{"NVFP4Quantum", metaShape, DTypeInt32},
			{"NVFP4Maxima", metaShape, DTypeFloat32},
		},
		grid:        [3]int{rows * (width / 32), 1, 1},
		threadGroup: [3]int{128, 1, 1},
	})
	if !ok {
		return nil
	}

	output, ok := gatherNVFP4QMM.applyMetal(gpuLaunch{
		ints:        []gpuIntArg{{"Width", width}, {"Channels", channels}},
		inputs:      []*Array{values, weight, scales, runs[0], bounds[0], bounds[1]},
		outputs:     []gpuOutputSpec{{"GatherNVFP4QMM", []int32{int32(rows), 1, 1, int32(channels)}, DTypeFloat32}},
		grid:        [3]int{columnTiles * 128, capacity, 1},
		threadGroup: [3]int{128, 1, 1},
	})
	if !ok {
		return nil
	}
	return output[0]
}
