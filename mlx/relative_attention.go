package mlx

import (
	"math"
	"os"
	"sync"
)

// MLX selects split vector attention for a single query with 1024 or more visible
// keys on these devices. The fused bias path must match its number of partial sums.
var relativeAttentionParts = sync.OnceValue(func() int {
	if !MetalIsAvailable() || os.Getenv("MLX_SDPA_BLOCKS") != "" {
		return 0
	}
	return relativeAttentionSplitParts(kernelDevice())
})

func relativeAttentionSplitParts(name, arch string) int {
	switch {
	case name == "Apple M2 Ultra" && arch == "applegpu_g14d",
		name == "Apple M3 Ultra" && arch == "applegpu_g15d",
		name == "Apple M5 Ultra" && arch == "applegpu_g17d":
		return 128
	case name == "Apple M4 Pro" && arch == "applegpu_g16s",
		isM5Pro(name, arch), name == "Apple M5 Max" && arch == "applegpu_g17s":
		return 64
	default:
		return 0
	}
}

// RelativePositionAttention applies causal attention to one ordered sequence of
// BF16 keys and values. Q has shape [1, heads, queries, width], and K and V
// have shape [1, kvHeads, keys, width]. The queries cover the final positions
// of K. Each group of heads/kvHeads query heads shares one KV head.
//
// For a query and key separated by d positions, table[0, head, query, d]
// adds a score bias. Distances past the end of the table contribute zero.
// Each query also has a learned sink of shape [1, heads, queries, 1] with
// zero value. The query dtype (BF16 or FP32) controls the shifted bias, partial
// numerators and output. K, V, the bias table and sink stay in BF16.
// A positive window limits history to that many keys including the query's
// own position. Zero allows its entire history. Unsupported inputs return nil.
func RelativePositionAttention(q, k, v, table, sink *Array, window int, scale float32) *Array {
	for i, a := range []*Array{q, k, v, table, sink} {
		if a == nil || a.NumDims() != 4 || a.Dim(0) != 1 {
			return nil
		}
		if dtype := a.DType(); dtype != DTypeBFloat16 && !(i == 0 && dtype == DTypeFloat32) {
			return nil
		}
		if size := a.Size(); size == 0 || uint64(size) > math.MaxUint32 {
			return nil
		}
	}
	heads, queries, width := q.Dim(1), q.Dim(2), q.Dim(3)
	kvHeads, keys := k.Dim(1), k.Dim(2)
	if keys < queries || heads%kvHeads != 0 || window < 0 || window > math.MaxInt32 ||
		k.Dim(3) != width || v.Dim(1) != kvHeads || v.Dim(2) != keys || v.Dim(3) != width ||
		table.Dim(1) != heads || table.Dim(2) != queries || sink.Dim(1) != heads || sink.Dim(2) != queries || sink.Dim(3) != 1 {
		return nil
	}
	if queries > 8 {
		return relativeAttentionPrefill(q, k, v, table, sink, window, scale)
	}
	// Match single-query native SDPA on long visible histories.
	if keys-queries+1 < 1024 || (window > 0 && window < 1024) {
		return nil
	}
	parts := relativeAttentionParts()
	if parts == 0 {
		return nil
	}
	sharing := heads / kvHeads
	maxVisible := keys
	if window > 0 {
		maxVisible = min(keys, window)
	}
	// The Metal shader assigns four channels to each SIMD lane. Larger
	// groups, and small groups with long histories on Ultra, can select a
	// different native partition count and change the rounding of each part.
	if width != 128 || sharing > 4 || (parts == 128 && sharing <= 2 && maxVisible > 8192) {
		return nil
	}
	rows := heads * queries
	if uint64(rows) > math.MaxUint32/uint64(parts*width) {
		return nil
	}
	partials, ok := relativeAttentionPart.applyMetal(gpuLaunch{
		inputs: []*Array{q, k, v, table, sink, FromValue(keys), FromValue(window), FromValue(scale)},
		dtypes: []gpuDTypeArg{{"T", q.DType()}},
		ints:   []gpuIntArg{{"Queries", queries}, {"Distances", table.Dim(3)}, {"Sharing", heads / k.Dim(1)}, {"Parts", parts}},
		outputs: []gpuOutputSpec{
			{"relative_numerators", []int32{int32(rows), int32(parts), 128}, q.DType()},
			{"relative_statistics", []int32{int32(rows), int32(parts), 2}, DTypeFloat32},
		},
		grid: [3]int{32 * parts, rows, 1}, threadGroup: [3]int{128, 1, 1},
	})
	if !ok {
		return nil
	}
	output, ok := relativeAttentionMerge.applyMetal(gpuLaunch{
		inputs:  partials,
		dtypes:  []gpuDTypeArg{{"T", q.DType()}},
		ints:    []gpuIntArg{{"Parts", parts}},
		outputs: []gpuOutputSpec{{"relative_result", []int32{1, int32(heads), int32(queries), 128}, q.DType()}},
		grid:    [3]int{1024, rows, 1}, threadGroup: [3]int{128, 1, 1},
	})
	if !ok {
		return nil
	}
	return output[0]
}

// Stock vector SDPA takes a dense mask for learned distances and per-query sinks.
// Apply the bias while reading each key and round the partial numerator as MLX does.
var relativeAttentionPart = &gpuKernel{
	name:    "relative_attention_part",
	inputs:  []string{"query", "keys", "values", "relative", "sink", "extent", "window", "scale"},
	outputs: []string{"numerators", "statistics"},
	metal: gpuSource{source: `
uint row = thread_position_in_grid.y;
uint lane = thread_index_in_simdgroup;
uint shard = thread_position_in_grid.x / 32;
uint proposal = row % Queries;
uint head = row / Queries;
uint end = uint(extent) - Queries + proposal + 1;
uint visible = window ? min(end, uint(window)) : end;
uint offset = (head / Sharing) * uint(extent) * 128;
uint component = lane * 4;
float4 q;
for (uint j = 0; j < 4; j++) {
	q[j] = float(query[row * 128 + component + j]) * scale;
}
bool with_sink = shard == 0;
float peak = with_sink ? 0.0f : -INFINITY;
float weight = with_sink ? 1.0f : 0.0f;
float4 value(0.0f);
for (uint i = shard; i < visible; i += Parts) {
	uint base = offset + (i + end - visible) * 128 + component;
	float dot = 0.0f;
	for (uint j = 0; j < 4; j++) {
		dot = fma(q[j], float(keys[base + j]), dot);
	}
	uint distance = visible - i - 1;
	float bias = distance < Distances ? float(relative[row * Distances + distance]) : 0.0f;
	bias = float(T(bias - float(sink[row])));
	float score = simd_sum(dot) + bias;
	float next = max(peak, score);
	float prior = fast::exp(peak - next);
	float prob = fast::exp(score - next);
	for (uint j = 0; j < 4; j++) {
		value[j] = fma(value[j], prior, prob * float(values[base + j]));
	}
	weight = fma(weight, prior, prob);
	peak = next;
}
uint idx = row * Parts + shard;
if (lane == 0) {
	statistics[2 * idx] = peak;
	statistics[2 * idx + 1] = weight;
}
for (uint j = 0; j < 4; j++) {
	numerators[idx * 128 + component + j] = T(value[j]);
}
`},
}

// MLX exposes only the final SDPA output. Merge the partial softmax sums and
// numerators in its native reduction order to reproduce the same output rounding.
var relativeAttentionMerge = &gpuKernel{
	name:    "relative_attention_merge",
	inputs:  []string{"numerators", "statistics"},
	outputs: []string{"output"},
	metal: gpuSource{source: `
uint row = thread_position_in_grid.y;
uint lane = thread_index_in_simdgroup;
uint component = (thread_position_in_grid.x / 32) * 4;
uint first = row * Parts;
float peak = -INFINITY;
for (uint p = lane; p < Parts; p += 32) {
	peak = max(peak, statistics[2 * (first + p)]);
}
peak = simd_max(peak);
float z = 0.0f;
float4 value(0.0f);
for (uint p = lane; p < Parts; p += 32) {
	uint idx = first + p;
	float factor = fast::exp(statistics[2 * idx] - peak);
	z = fma(statistics[2 * idx + 1], factor, z);
	for (uint j = 0; j < 4; j++) {
		value[j] = fma(float(numerators[idx * 128 + component + j]), factor, value[j]);
	}
}
z = simd_sum(z);
for (uint j = 0; j < 4; j++) {
	float answer = simd_sum(value[j]) / z;
	if (lane == 0) {
		uint idx = row * 128 + component + j;
		output[idx] = T(answer);
	}
}
`},
}
