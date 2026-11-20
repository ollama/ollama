package mlx

import (
	"math"
	"slices"
)

func supportsRelativeAttentionPrefill(name, arch string) bool {
	return isM5MatrixDevice(name, arch) ||
		name == "Apple M2 Pro" && arch == "applegpu_g14s" ||
		name == "Apple M2 Ultra" && arch == "applegpu_g14d" ||
		name == "Apple M3 Ultra" && arch == "applegpu_g15d" ||
		name == "Apple M4 Pro" && arch == "applegpu_g16s"
}

// Stock SDPA takes a dense bias for learned relative positions. Tiling the queries
// limits both the bias allocation and the attention work to the keys each tile can see.
func relativeAttentionPrefill(q, k, v, table, sink *Array, window int, scale float32) *Array {
	// Native matrix attention uses the same key-tile alignment for these widths.
	width := q.Dim(3)
	if (width != 64 && width != 128) || !MetalIsAvailable() || !supportsRelativeAttentionPrefill(kernelDevice()) {
		return nil
	}
	heads, queries, keys := q.Dim(1), q.Dim(2), k.Dim(2)
	const tile = 256
	cut := func(x *Array, start, end int) *Array { return x.Slice(Slice(), Slice(), Slice(start, end), Slice()) }
	sinks := Zeros(q.DType(), heads)
	var pieces []*Array
	for begin := 0; begin < queries; {
		end := min(queries, begin+tile)
		// Preserve the native matrix attention path on the final partial tile.
		if queries-end < 64 {
			end = queries
		}
		lastKey := keys - queries + end
		firstKey := 0
		if window > 0 {
			// Keep the first retained key on the original reduction boundary.
			firstKey = max(0, keys-queries+begin+1-window) / 64 * 64
		}
		n, span := end-begin, lastKey-firstKey
		if uint64(heads)*uint64(n)*uint64(span) > math.MaxUint32 {
			return nil
		}
		// Pass query-major tables so the mask kernel reads adjacent heads per query.
		bias, ok := relativeAttentionMask.applyMetal(gpuLaunch{
			inputs:  []*Array{cut(table, begin, end).Transpose(0, 2, 1, 3), cut(sink, begin, end).Transpose(0, 2, 1, 3), FromValue(n), FromValue(span), FromValue(window)},
			dtypes:  []gpuDTypeArg{{"T", q.DType()}},
			ints:    []gpuIntArg{{"Heads", heads}, {"Distances", table.Dim(3)}},
			outputs: []gpuOutputSpec{{"relative_bias", []int32{1, int32(heads), int32(n), int32(span)}, q.DType()}},
			grid:    [3]int{span, n, heads}, threadGroup: [3]int{256, 1, 1},
		})
		if !ok {
			return nil
		}
		out := FastScaledDotProductAttentionWithSinks(cut(q, begin, end), cut(k, firstKey, lastKey), cut(v, firstKey, lastKey), scale, FastSDPANoWindow, "", bias[0], sinks)
		if begin == 0 && end == queries {
			return out
		}
		pieces = append(pieces, out)
		begin = end
	}
	return Concatenate(pieces, 2)
}

// RelativePositionDecodeBias reuses the prefill mask shader for a single final query.
// All keys are causal for this row; the caller applies any sliding-window mask.
func RelativePositionDecodeBias(table, sink *Array, keys int) *Array {
	if table == nil || sink == nil || keys < 1 || keys > math.MaxInt32 ||
		!MetalIsAvailable() || !supportsRelativeAttentionPrefill(kernelDevice()) {
		return nil
	}
	if table.NumDims() != 4 || table.Dim(0) != 1 || table.Dim(2) != 1 ||
		table.DType() != DTypeBFloat16 || sink.DType() != DTypeBFloat16 {
		return nil
	}
	heads, distances := table.Dim(1), table.Dim(3)
	if heads == 0 || distances == 0 || uint64(heads)*uint64(max(keys, distances)) > math.MaxUint32 ||
		sink.NumDims() != 4 || sink.Dim(0) != 1 || sink.Dim(1) != heads || sink.Dim(2) != 1 || sink.Dim(3) != 1 {
		return nil
	}
	result, ok := relativeAttentionMask.applyMetal(gpuLaunch{
		inputs:  []*Array{table, sink, FromValue(1), FromValue(keys), FromValue(0)},
		dtypes:  []gpuDTypeArg{{"T", DTypeFloat32}},
		ints:    []gpuIntArg{{"Heads", heads}, {"Distances", distances}},
		outputs: []gpuOutputSpec{{"relative_bias", []int32{1, int32(heads), 1, int32(keys)}, DTypeFloat32}},
		grid:    [3]int{keys, 1, heads}, threadGroup: [3]int{256, 1, 1},
	})
	if !ok {
		return nil
	}
	return result[0]
}

// The stock graph gathers distances, shifts by the sink and applies masks separately.
// Write the final bias for a visible query/key tile in one pass.
var relativeAttentionMask = &gpuKernel{
	name:    "relative_attention_mask",
	inputs:  []string{"table", "sinks", "queries", "keys", "window"},
	outputs: []string{"bias"},
	metal: gpuSource{source: `
uint3 pos = thread_position_in_grid;
int distance = int(keys) - int(queries) + int(pos.y) - int(pos.x);
uint row = pos.y * Heads + pos.z;
float relative = (distance >= 0 && distance < Distances) ? float(table[row * Distances + distance]) : 0.0f;
float shifted = relative - float(sinks[row]);
bool masked = distance < 0 || (window > 0 && distance >= window);
bias[(pos.z * queries + pos.y) * keys + pos.x] = T(masked ? -INFINITY : shifted);
`},
}

// RelativePositionAttentionBF16Probabilities computes causal attention in reverse 128-key
// blocks. It expects BF16 query values widened to FP32. The tensor operations accumulate
// in FP32 and round the unnormalized probabilities to BF16 for the value product.
// firstKey is the absolute token position of the first retained key, including after eviction.
func RelativePositionAttentionBF16Probabilities(q, k, v, table, sink *Array, firstKey, window int, scale float32) *Array {
	if firstKey < 0 || window < 1 || window > math.MaxInt32 {
		return nil
	}
	for i, a := range []*Array{q, k, v, table, sink} {
		if a == nil || a.NumDims() != 4 || a.Dim(0) != 1 || a.Size() < 1 || a.Size() > math.MaxInt32 {
			return nil
		}
		if (i == 0 && a.DType() != DTypeFloat32) || (i != 0 && a.DType() != DTypeBFloat16) {
			return nil
		}
	}
	const width, tileRows, groups = 128, 16, 4
	heads, queries, kvHeads, keys := q.Dim(1), q.Dim(2), k.Dim(1), k.Dim(2)
	if queries <= 8 || queries > keys || heads%kvHeads != 0 || q.Dim(3) != width || k.Dim(3) != width ||
		!slices.Equal(k.Dims(), v.Dims()) || !slices.Equal(table.Dims()[:3], q.Dims()[:3]) ||
		!slices.Equal(sink.Dims()[:3], q.Dims()[:3]) || sink.Dim(3) != 1 {
		return nil
	}
	if !hasMetalTensorOps() {
		return nil
	}
	leading := firstKey % width
	paddedKeys := (leading + keys + width - 1) / width * width
	paddedQueries := (queries + tileRows - 1) / tileRows * tileRows
	if paddedKeys > math.MaxInt32/kvHeads/width || paddedQueries > math.MaxInt32/heads/width {
		return nil
	}
	// Align physical key blocks to their original token positions, including the
	// first partial block of a sliding window. The shader masks both padded borders.
	if leading != 0 || paddedKeys != keys {
		trailing := paddedKeys - leading - keys
		k = PadConstant(k, []int{2}, []int{leading}, []int{trailing})
		v = PadConstant(v, []int{2}, []int{leading}, []int{trailing})
	}
	if paddedQueries != queries {
		q = PadConstant(q, []int{2}, []int{0}, []int{paddedQueries - queries})
	}
	shape := FromValues([]int32{int32(queries), int32(keys), int32(table.Dim(3)), int32(window), int32(leading), int32(heads / kvHeads), int32(paddedQueries), int32(paddedKeys)}, 8)
	result, ok := relativeAttentionBF16Probabilities.applyMetal(gpuLaunch{
		inputs:  []*Array{q, k, v, table, sink, FromValue(scale), shape},
		ints:    []gpuIntArg{{"Groups", groups}},
		outputs: []gpuOutputSpec{{"attention", []int32{1, int32(heads), int32(queries), width}, DTypeFloat32}},
		grid:    [3]int{paddedQueries / tileRows * groups * 32, heads, 1}, threadGroup: [3]int{groups * 32, 1, 1},
	})
	if !ok {
		return nil
	}
	return result[0]
}

// Stock FP32 SDPA retains FP32 probabilities in the value product. Round each
// reverse tile's unnormalized probabilities to BF16 while preserving FP32 normalization.
var relativeAttentionBF16Probabilities = &gpuKernel{
	name:    "relative_attention_bf16_probabilities",
	inputs:  []string{"query", "key", "value", "table", "sink", "scale", "shape"},
	outputs: []string{"out"},
	metal: gpuSource{header: `
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
`, source: `
using namespace mpp::tensor_ops;
using dims = dextents<int, 2>;
using strides = array<int, 2>;
constexpr int rows = 16, width = 128, channels = width / Groups;
constexpr matmul2d_descriptor dotShape(rows, channels, width, false, true, true);
constexpr matmul2d_descriptor valueShape(rows, channels, width, false, false, true);
matmul2d<dotShape, execution_simdgroup> dot;
matmul2d<valueShape, execution_simdgroup> weightedValue;

const int Queries=shape[0], Keys=shape[1], Distances=shape[2], Window=shape[3];
const int Leading=shape[4], Sharing=shape[5], QPad=shape[6], KPad=shape[7];
const int head = threadgroup_position_in_grid.y;
const int rowStart = threadgroup_position_in_grid.x * rows;
const int group = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
const int columnStart = group * channels;
const int queryStart = Leading + Keys - Queries + rowStart;
const int newest = (queryStart + min(rows, Queries - rowStart) - 1) / width;
const int oldest = max(Leading, queryStart + 1 - Window) / width;

threadgroup float tile[rows * width], scalePrior[rows], denominator[rows], sinkWeight[rows];
threadgroup bfloat probabilities[rows * width];
float accumulator[rows][channels / 32] = {};
float peaks[rows / Groups], denominators[rows / Groups] = {};
for (int i = 0; i < rows / Groups; ++i) peaks[i] = -INFINITY;
tensor<device float, dims, tensor_inline> queries((device float*)query + (head * QPad + rowStart) * width,
    dims(width, rows), strides{1, width});
tensor<threadgroup float, dims, tensor_inline> output(tile + columnStart, dims(channels, rows), strides{1, width});
tensor<threadgroup bfloat, dims, tensor_inline> p(probabilities, dims(width, rows), strides{1, width});

for (int block = newest; block >= oldest; --block) {
    const int base = ((head / Sharing) * KPad + block * width) * width;
    tensor<device bfloat, dims, tensor_inline> keys((device bfloat*)key + base + columnStart * width,
        dims(width, channels), strides{1, width});
    dot.run(queries, keys, output);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each SIMD group normalizes complete rows using the running 128-key maximum.
    for (int r = group; r < rows; r += Groups) {
        const int queryRow = rowStart + r, row = head * Queries + queryRow;
        float score[4], localMax = -INFINITY;
        for (int j = 0; j < 4; ++j) {
            const int col = lane + j * 32, keyIndex = block * width + col;
            const int distance = queryStart + r - keyIndex;
            score[j] = -INFINITY;
            if (queryRow < Queries && keyIndex >= Leading && distance >= 0 && distance < Window) {
                const float relative = distance < Distances ? float(table[row * Distances + distance]) : 0;
                score[j] = fma(tile[r * width + col], float(scale), relative);
            }
            localMax = max(localMax, score[j]);
        }
        const int i = r / Groups;
        const float peak = max(peaks[i], simd_max(localMax));
        const float previous = isfinite(peaks[i]) ? fast::exp(peaks[i] - peak) : 0;
        float localSum = 0;
        for (int j = 0; j < 4; ++j) {
            const float weight = isfinite(score[j]) ? fast::exp(score[j] - peak) : 0;
            probabilities[r * width + lane + j * 32] = bfloat(weight);
            localSum += weight;
        }
        denominators[i] = fma(denominators[i], previous, simd_sum(localSum));
        peaks[i] = peak;
        if (lane == 0) {
            scalePrior[r] = previous;
            if (block == oldest && queryRow < Queries) {
                // Include the sink in the softmax denominator with a single reciprocal.
                const float gap = float(sink[row]) - peak;
                const float inverse = 1 / (denominators[i] + fast::exp(gap));
                denominator[r] = 1;
                sinkWeight[r] = inverse;
                if (inverse < FLT_MIN) {
                    // A large sink can flush the reciprocal even when the gated
                    // mean is normal. Normalize V before applying the sink there.
                    denominator[r] = denominators[i];
                    sinkWeight[r] = 1 / (1 + fast::exp(gap - fast::log(denominators[i])));
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<device bfloat, dims, tensor_inline> values((device bfloat*)value + base + columnStart,
        dims(channels, width), strides{1, width});
    weightedValue.run(p, values, output);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < channels / 32; ++j) {
            accumulator[r][j] = fma(accumulator[r][j], scalePrior[r], tile[r * width + columnStart + lane + j*32]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (int r = 0; r < min(rows, Queries - rowStart); ++r) {
    const int offset = (head * Queries + rowStart + r) * width + columnStart + lane;
    for (int j = 0; j < channels / 32; ++j) out[offset + j * 32] = (accumulator[r][j] / denominator[r]) * sinkWeight[r];
}
`},
}
