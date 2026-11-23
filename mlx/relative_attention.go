package mlx

import (
	"math"
	"sync"
)

var relativeAttentionParts = sync.OnceValue(func() int {
	if !MetalIsAvailable() {
		return 0
	}
	return relativeAttentionSplitParts(kernelArchitecture())
})

// relativeAttentionSplitParts is how many partial sums the split decode kernel
// reduces over the keys, or 0 to keep the stock graph. Larger GPUs split
// wider, as MLX's two-pass vector SDPA does.
func relativeAttentionSplitParts(architecture string) int {
	generation, size := appleGPU(architecture)
	switch {
	case generation == 0:
		return 0
	case size == 'd':
		return 128
	case size == 's':
		return 64
	default:
		return 32
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
//
// The kernel reads its inputs in place. It is fastest when K and V are windows
// of a cache buffer that share one layout of contiguous 128-channel rows with a
// head stride of whole rows, and Q, the table and the sink are views with a
// contiguous last axis. Other layouts are read through their strides, one key
// at a time.
func RelativePositionAttention(q, k, v, table, sink *Array, window int, scale float32) *Array {
	if !relativeAttentionShapes(q, k, v, table, sink, window) {
		return nil
	}
	queries := q.Dim(2)
	if queries > 8 {
		return relativeAttentionPrefill(q, k, v, table, sink, window, scale)
	}
	if m := relativeBlockSplit(queries, k.Dim(2), window); m > 0 {
		// The first m queries see histories that single-token decode splits
		// into a different number of partitions; give them their own launch.
		query := func(x *Array, from, to int) *Array { return x.Slice(Slice(), Slice(), Slice(from, to), Slice()) }
		keys := k.Dim(2) - (queries - m)
		early := RelativePositionAttention(query(q, 0, m), query(k, 0, keys), query(v, 0, keys), query(table, 0, m), query(sink, 0, m), window, scale)
		late := RelativePositionAttention(query(q, m, queries), k, v, query(table, m, queries), query(sink, m, queries), window, scale)
		if early == nil || late == nil {
			return nil
		}
		return Concatenate([]*Array{early, late}, 2)
	}
	kernel, parts := relativeSplitFor(queries, k, window)
	return relativeAttentionSplitMerge(kernel, parts, q, k, v, table, sink, window, scale)
}

// relativeBlockSplit returns how many leading queries of a block over keys take
// a different partition count than its last query when each is decoded alone,
// or 0 when the whole block agrees. Partials depend only on a query's own keys
// and the partition count, so a block that agrees matches single-token decode.
func relativeBlockSplit(queries, keys, window int) int {
	visible := func(n int) int {
		if window > 0 {
			return min(n, window)
		}
		return n
	}
	_, last := relativeSplit(1, visible(keys))
	for j := range queries - 1 {
		if _, parts := relativeSplit(1, visible(keys-(queries-1-j))); parts == last {
			return j
		}
	}
	return queries - 1
}

// relativeAttentionSplitMerge runs one split-pass kernel over parts partitions
// and merges the partial sums.
func relativeAttentionSplitMerge(kernel *gpuKernel, parts int, q, k, v, table, sink *Array, window int, scale float32) *Array {
	heads, queries := q.Dim(1), q.Dim(2)
	partials := relativeAttentionSplitPass(kernel, q, k, v, table, sink, window, scale, parts)
	if partials == nil {
		return nil
	}
	output, ok := relativeAttentionMerge.applyMetal(gpuLaunch{
		inputs:  partials,
		dtypes:  []gpuDTypeArg{{"T", q.DType()}},
		ints:    []gpuIntArg{{"Parts", parts}},
		outputs: []gpuOutputSpec{{"relative_result", []int32{1, int32(heads), int32(queries), 128}, q.DType()}},
		grid:    [3]int{1024, heads * queries, 1}, threadGroup: [3]int{128, 1, 1},
	})
	if !ok {
		return nil
	}
	return output[0]
}

// RelativePositionAttentionGated runs RelativePositionAttention for one or two
// requests of up to eight queries (element i of q, k, v, table and sink) and
// applies the sigmoid output gate as it merges the partial sums. gate is BF16
// [requests, queries, heads*128] with a contiguous last axis; it may be a column
// view of a wider projection. The BF16 result has the gate's shape and holds
// attention * sigmoid(gate), with each request's queries before its heads.
// It returns nil when RelativePositionAttention would decline any request, or
// would split a request's block at a partition-count change.
func RelativePositionAttentionGated(q, k, v, table, sink []*Array, gate *Array, window int, scale float32) *Array {
	requests := len(q)
	if requests < 1 || requests > 2 || len(k) != requests || len(v) != requests || len(table) != requests ||
		len(sink) != requests || gate == nil || gate.DType() != DTypeBFloat16 || gate.NumDims() != 3 ||
		gate.Dim(0) != requests || uint64(gate.Size()) > math.MaxUint32 {
		return nil
	}
	heads, queries := q[0].Dim(1), q[0].Dim(2)
	if gate.Dim(1) != queries || gate.Dim(2) != heads*128 {
		return nil
	}
	for i := range requests {
		if !relativeAttentionShapes(q[i], k[i], v[i], table[i], sink[i], window) || q[i].Dim(2) > 8 ||
			q[i].DType() != q[0].DType() || q[i].Dim(1) != q[0].Dim(1) || q[i].Dim(2) != q[0].Dim(2) {
			return nil
		}
		// A block that straddles a partition-count change takes the separate
		// route, which splits it as single-token decode would.
		if relativeBlockSplit(queries, k[i].Dim(2), window) > 0 {
			return nil
		}
	}
	// Each request splits as it would alone, so its sums do not depend on the
	// other request's history length.
	var inputs []*Array
	parts := make([]int, 2)
	for i := range requests {
		kernel, p := relativeSplitFor(queries, k[i], window)
		partials := relativeAttentionSplitPass(kernel, q[i], k[i], v[i], table[i], sink[i], window, scale, p)
		if partials == nil {
			return nil
		}
		inputs = append(inputs, partials...)
		parts[i] = p
	}
	if requests == 1 {
		// The kernel binds a second request's partials but never reads them.
		inputs = append(inputs, inputs...)
		parts[1] = parts[0]
	}
	output, ok := relativeAttentionMergeGated.applyMetal(gpuLaunch{
		inputs:  append(inputs, gate),
		dtypes:  []gpuDTypeArg{{"T", q[0].DType()}},
		ints:    []gpuIntArg{{"Parts", parts[0]}, {"Parts1", parts[1]}, {"Queries", queries}, {"Heads", heads}},
		outputs: []gpuOutputSpec{{"relative_gated", []int32{int32(requests), int32(queries), int32(heads * 128)}, DTypeBFloat16}},
		grid:    [3]int{1024, requests * heads * queries, 1}, threadGroup: [3]int{128, 1, 1},
	})
	if !ok {
		return nil
	}
	return output[0]
}

// relativeAttentionShapes checks the shapes and dtypes every relative
// attention route requires.
func relativeAttentionShapes(q, k, v, table, sink *Array, window int) bool {
	for i, a := range []*Array{q, k, v, table, sink} {
		if a == nil || a.NumDims() != 4 || a.Dim(0) != 1 {
			return false
		}
		if dtype := a.DType(); dtype != DTypeBFloat16 && !(i == 0 && dtype == DTypeFloat32) {
			return false
		}
		if size := a.Size(); size == 0 || uint64(size) > math.MaxUint32 {
			return false
		}
	}
	heads, queries, width := q.Dim(1), q.Dim(2), q.Dim(3)
	kvHeads, keys := k.Dim(1), k.Dim(2)
	return keys >= queries && heads%kvHeads == 0 && window >= 0 && window <= math.MaxInt32 &&
		k.Dim(3) == width && v.Dim(1) == kvHeads && v.Dim(2) == keys && v.Dim(3) == width &&
		table.Dim(1) == heads && table.Dim(2) == queries && sink.Dim(1) == heads && sink.Dim(2) == queries && sink.Dim(3) == 1
}

// relativeAttentionVisible returns the most keys any query of k can see.
func relativeAttentionVisible(k *Array, window int) int {
	if window > 0 {
		return min(k.Dim(2), window)
	}
	return k.Dim(2)
}

// relativeSplit returns this device's split-pass kernel and partition count
// for a block of queries over visible keys (nil and zero where it has none).
func relativeSplit(queries, visible int) (*gpuKernel, int) {
	base := relativeAttentionParts()
	if base == 0 {
		return nil, 0
	}
	return relativeSplitConfig(base, queries, visible, kernelArchitecture())
}

// relativeSplitFor picks the split for a block of queries over k. The shared
// kernel numbers key blocks from the start of the view the queries share, which
// matches each query's single-token decode only without a window.
func relativeSplitFor(queries int, k *Array, window int) (*gpuKernel, int) {
	kernel, parts := relativeSplit(queries, relativeAttentionVisible(k, window))
	if window > 0 && kernel == relativeAttentionSharedPart {
		kernel = relativeAttentionBlockPart
	}
	return kernel, parts
}

// relativeSplitConfig tunes the split pass by architecture generation, then
// size class, then shape. base is the device's native partition count. The
// partition count follows the block's visible keys, not its number of queries,
// so a query's sums match single-token decode; RelativePositionAttention splits
// a block that crosses a history length where the count changes.
func relativeSplitConfig(base, queries, visible int, architecture string) (*gpuKernel, int) {
	generation, size := appleGPU(architecture)
	if generation >= 16 && size == 's' {
		// Pro and Max GPUs from M4: on M4 Pro and M5 Pro the blocked kernels
		// were faster at every width and history. Staging keys for a block of
		// queries paid off past 4K keys.
		kernel := relativeAttentionBlockPart
		if queries > 1 && visible > 4096 {
			kernel = relativeAttentionSharedPart
		}
		// Fastest on M5 Pro: half the base count up to 4K keys (sliding windows,
		// short contexts), the base count up to 16K, twice it beyond.
		if visible <= 4096 {
			return kernel, base / 2
		} else if visible <= 16384 {
			return kernel, base
		}
		return kernel, 2 * base
	}
	// M2 Ultra: the per-key kernel was faster. M3 Ultra was mixed with an earlier
	// blocked version. Base GPUs and Pro/Max GPUs before M4 take the per-key
	// kernel too, unmeasured.
	return relativeAttentionPart, base
}

func relativeAttentionSplitPass(kernel *gpuKernel, q, k, v, table, sink *Array, window int, scale float32, parts int) []*Array {
	heads, queries, width := q.Dim(1), q.Dim(2), q.Dim(3)
	kvHeads, keys := k.Dim(1), k.Dim(2)
	// Histories under 1024 keys take the stock graph, whose vector SDPA reads
	// them in one pass; the split kernels serve longer ones.
	if keys-queries+1 < 1024 || (window > 0 && window < 1024) || kernel == nil || parts <= 0 {
		return nil
	}
	sharing := heads / kvHeads
	// The Metal shader assigns four channels to each SIMD lane. Larger
	// groups, and on Ultra small groups with long histories, are where MLX's
	// vector SDPA uses a partition count other than relativeAttentionParts();
	// they take the stock graph, so the per-key kernel's sums equal MLX's
	// wherever it runs.
	if width != 128 || sharing > 4 || (relativeAttentionParts() == 128 && sharing <= 2 && relativeAttentionVisible(k, window) > 8192) {
		return nil
	}
	rows := heads * queries
	if uint64(rows) > math.MaxUint32/uint64(parts*width) {
		return nil
	}
	launch := gpuLaunch{
		inputs: []*Array{q, k, v, table, sink, FromValue(keys), FromValue(window), FromValue(scale)},
		dtypes: []gpuDTypeArg{{"T", q.DType()}},
		ints:   []gpuIntArg{{"Queries", queries}, {"Distances", table.Dim(3)}, {"Sharing", heads / k.Dim(1)}, {"Parts", parts}},
		outputs: []gpuOutputSpec{
			{"relative_numerators", []int32{int32(rows), int32(parts), 128}, q.DType()},
			{"relative_statistics", []int32{int32(rows), int32(parts), 2}, DTypeFloat32},
		},
		grid: [3]int{32 * parts, kvHeads * queries, 1}, threadGroup: [3]int{128, 1, 1},
	}
	if kernel == relativeAttentionSharedPart {
		// One threadgroup of at most 256 threads per partition and KV head,
		// one SIMD group per query.
		launch.grid, launch.threadGroup = [3]int{32 * queries * parts, kvHeads, 1}, [3]int{32 * queries, 1, 1}
	}
	partials, ok := kernel.applyMetal(launch)
	if !ok {
		return nil
	}
	return partials
}

// Stock vector SDPA takes a dense mask for learned distances and per-query sinks.
// Apply the bias while reading each key and round the partial numerator as MLX does.
// Inputs keep their layout, so cache windows and transposed tables need no copy.
// One SIMD group serves the query heads that share a KV head: it loads each key
// and value row once and runs every head's reduction over it in the same order.
var relativeAttentionPart = &gpuKernel{
	name:    "relative_attention_part",
	inputs:  []string{"query", "keys", "values", "relative", "sink", "extent", "window", "scale"},
	outputs: []string{"numerators", "statistics"},
	strided: true,
	metal: gpuSource{source: `
uint lane = thread_index_in_simdgroup;
uint shard = thread_position_in_grid.x / 32;
uint proposal = thread_position_in_grid.y % Queries;
uint group = thread_position_in_grid.y / Queries;
uint end = uint(extent) - Queries + proposal + 1;
uint visible = window ? min(end, uint(window)) : end;
uint component = lane * 4;
// K and V share whole 128-channel rows; the head stride is a whole number of
// rows, which keeps each lane's four channels aligned for combined loads.
long head_stride = keys_strides[1];
long last = group * Sharing + Sharing - 1;
long limit = long(0xFFFFFFFFu);
bool supported = head_stride >= 0 && head_stride % 128 == 0 &&
	keys_strides[2] == 128 && keys_strides[3] == 1 && values_strides[1] == head_stride &&
	values_strides[2] == 128 && values_strides[3] == 1 &&
	relative_strides[1] >= 0 && relative_strides[2] >= 0 && relative_strides[3] == 1 &&
	sink_strides[1] >= 0 && sink_strides[2] >= 0 &&
	query_strides[1] >= 0 && query_strides[2] >= 0 && query_strides[3] == 1 &&
	group * head_stride + long(extent) * 128 <= limit &&
	last * relative_strides[1] + proposal * relative_strides[2] + Distances <= limit &&
	last * sink_strides[1] + proposal * sink_strides[2] < limit &&
	last * query_strides[1] + proposal * query_strides[2] + 128 <= limit;
` + relativeSplitUnsupported + `
uint offset = group * (uint(head_stride) / 128) * 128;
` + relativeSplitQuery + `
for (uint i = shard; i < visible; i += Parts) {
	uint base = offset + (i + end - visible) * 128 + component;
	float4 k, v;
	for (uint j = 0; j < 4; j++) {
		k[j] = float(keys[base + j]);
		v[j] = float(values[base + j]);
	}
	uint distance = visible - i - 1;
	for (uint s = 0; s < Sharing; s++) {
		float dot = 0.0f;
		for (uint j = 0; j < 4; j++) {
			dot = fma(q[s][j], k[j], dot);
		}
		float bias = distance < Distances ? float(relative[bias_row[s] + distance]) : 0.0f;
		bias = float(T(bias - shift[s]));
		float score = simd_sum(dot) + bias;
		float next = max(peak[s], score);
		float prior = fast::exp(peak[s] - next);
		float prob = fast::exp(score - next);
		for (uint j = 0; j < 4; j++) {
			value[s][j] = fma(value[s][j], prior, prob * v[j]);
		}
		weight[s] = fma(weight[s], prior, prob);
		peak[s] = next;
	}
}
` + relativeSplitEpilogue},
}

// The split pass scores keys in blocks of eight, numbered by key position;
// block b belongs to partition b mod Parts, so each query's partial sums depend
// only on its own keys and the partition count, not on how many queries share
// the launch. A lane accumulates its four channels of all eight dot products,
// and one transposing reduction leaves lane l with the score of key
// (l >> 2) & 7, so the SIMD group updates the softmax for eight keys at once
// instead of reducing and updating once per key.

// relativeSplitUnsupported computes one query's partial sums when the including
// kernel's layout check fails: it reads every input through its strides, one key
// at a time, and takes the keys numbered shard mod Parts. Each key still belongs
// to one partition, so the merge is unchanged. The check depends only on the
// launch, so a whole threadgroup takes this path before any barrier.
const relativeSplitUnsupported = `
if (!supported) {
	float4 q[Sharing], value[Sharing];
	float shift[Sharing], peak[Sharing], weight[Sharing];
	long bias_row[Sharing];
	for (uint s = 0; s < Sharing; s++) {
		long head = group * Sharing + s;
		bias_row[s] = head * relative_strides[1] + long(proposal) * relative_strides[2];
		shift[s] = float(sink[head * sink_strides[1] + long(proposal) * sink_strides[2]]);
		long query_row = head * query_strides[1] + long(proposal) * query_strides[2];
		for (uint j = 0; j < 4; j++) {
			q[s][j] = float(query[query_row + long(component + j) * query_strides[3]]) * scale;
		}
		peak[s] = shard == 0 ? 0.0f : -INFINITY;
		weight[s] = shard == 0 ? 1.0f : 0.0f;
		value[s] = float4(0.0f);
	}
	for (uint i = shard; i < visible; i += Parts) {
		long key = long(i + end - visible);
		float4 k, v;
		for (uint j = 0; j < 4; j++) {
			long channel = long(component + j);
			k[j] = float(keys[group * keys_strides[1] + key * keys_strides[2] + channel * keys_strides[3]]);
			v[j] = float(values[group * values_strides[1] + key * values_strides[2] + channel * values_strides[3]]);
		}
		uint distance = visible - i - 1;
		for (uint s = 0; s < Sharing; s++) {
			float dot = 0.0f;
			for (uint j = 0; j < 4; j++) {
				dot = fma(q[s][j], k[j], dot);
			}
			float bias = distance < Distances ? float(relative[bias_row[s] + long(distance) * relative_strides[3]]) : 0.0f;
			bias = float(T(bias - shift[s]));
			float score = simd_sum(dot) + bias;
			float next = max(peak[s], score);
			float prior = fast::exp(peak[s] - next);
			float prob = fast::exp(score - next);
			for (uint j = 0; j < 4; j++) {
				value[s][j] = fma(value[s][j], prior, prob * v[j]);
			}
			weight[s] = fma(weight[s], prior, prob);
			peak[s] = next;
		}
	}
` + relativeSplitEpilogue + `
	return;
}
`

// relativeSplitQuery loads one query's heads, scaled, with their bias rows and
// sinks, and starts each head's running softmax.
const relativeSplitQuery = `
float4 q[Sharing], value[Sharing];
float shift[Sharing], peak[Sharing], weight[Sharing];
uint bias_row[Sharing];
for (uint s = 0; s < Sharing; s++) {
	uint head = group * Sharing + s;
	bias_row[s] = head * uint(relative_strides[1]) + proposal * uint(relative_strides[2]);
	shift[s] = float(sink[head * uint(sink_strides[1]) + proposal * uint(sink_strides[2])]);
	uint query_row = head * uint(query_strides[1]) + proposal * uint(query_strides[2]) + component;
	for (uint j = 0; j < 4; j++) {
		q[s][j] = float(query[query_row + j]) * scale;
	}
	// Partition 0 also carries the sink: a zero logit with no value.
	peak[s] = shard == 0 ? 0.0f : -INFINITY;
	weight[s] = shard == 0 ? 1.0f : 0.0f;
	value[s] = float4(0.0f);
}
`

// relativeSplitPrologue checks the layout and loads one query's heads. The
// including kernel defines lane, shard, proposal and group. Query-dependent
// bounds use the last query, so the check agrees across a threadgroup.
const relativeSplitPrologue = `
constexpr uint Row = 128;
uint end = uint(extent) - Queries + proposal + 1;
uint visible = window ? min(end, uint(window)) : end;
uint begin = end - visible;
uint first_end = uint(extent) - Queries + 1;
uint lo = window ? first_end - min(first_end, uint(window)) : 0;
uint hi = uint(extent);
uint component = lane * 4;
long head_stride = keys_strides[1];
long last_head = group * Sharing + Sharing - 1;
long last_query = Queries - 1;
long limit = long(0xFFFFFFFFu);
bool supported = head_stride >= 0 && head_stride % Row == 0 &&
	keys_strides[2] == Row && keys_strides[3] == 1 && values_strides[1] == head_stride &&
	values_strides[2] == Row && values_strides[3] == 1 &&
	relative_strides[1] >= 0 && relative_strides[2] >= 0 && relative_strides[3] == 1 &&
	sink_strides[1] >= 0 && sink_strides[2] >= 0 &&
	query_strides[1] >= 0 && query_strides[2] >= 0 && query_strides[3] == 1 &&
	group * head_stride + long(extent) * Row <= limit &&
	last_head * relative_strides[1] + last_query * relative_strides[2] + Distances <= limit &&
	last_head * sink_strides[1] + last_query * sink_strides[2] < limit &&
	last_head * query_strides[1] + last_query * query_strides[2] + 128 <= limit;
` + relativeSplitUnsupported + relativeSplitQuery + `
uint newest = hi - 1;
uint slot = (lane >> 2) & 7;
bool upper16 = (lane & 16) != 0, upper8 = (lane & 8) != 0, upper4 = (lane & 4) != 0;
uint first_block = lo / 8;
uint start_block = first_block + (shard + Parts - first_block % Parts) % Parts;
`

// relativeSplitBlock folds the eight keys from start into one query's sums.
// The including kernel defines RELATIVE_KEY(t) and RELATIVE_VALUE(t), which
// return this lane's four channels of key start + t as float4.
const relativeSplitBlock = `
if (start < end && start + 8 > begin) {
	float dots[Sharing][8];
	for (uint t = 0; t < 8; t++) {
		float4 k = RELATIVE_KEY(t);
		for (uint s = 0; s < Sharing; s++) {
			dots[s][t] = dot(q[s], k);
		}
	}
	uint index = start + slot;
	bool live = index >= begin && index < end;
	uint distance = end - 1 - index;
	float prob[Sharing];
	for (uint s = 0; s < Sharing; s++) {
		// Halve the eight partial dots across lane bits 4, 3 and 2, then
		// finish each sum across lane bits 0 and 1.
		float four[4], two[2];
		for (uint j = 0; j < 4; j++) {
			float keep = upper16 ? dots[s][j + 4] : dots[s][j];
			float give = upper16 ? dots[s][j] : dots[s][j + 4];
			four[j] = keep + simd_shuffle_xor(give, 16);
		}
		for (uint j = 0; j < 2; j++) {
			float keep = upper8 ? four[j + 2] : four[j];
			float give = upper8 ? four[j] : four[j + 2];
			two[j] = keep + simd_shuffle_xor(give, 8);
		}
		float score = (upper4 ? two[1] : two[0]) + simd_shuffle_xor(upper4 ? two[0] : two[1], 4);
		score += simd_shuffle_xor(score, 1);
		score += simd_shuffle_xor(score, 2);
		float bias = live && distance < Distances ? float(relative[bias_row[s] + distance]) : 0.0f;
		bias = float(T(bias - shift[s]));
		score = live ? score + bias : -INFINITY;
		float top = max(score, simd_shuffle_xor(score, 4));
		top = max(top, simd_shuffle_xor(top, 8));
		top = max(top, simd_shuffle_xor(top, 16));
		float next = max(peak[s], top);
		float prior = fast::exp(peak[s] - next);
		float p = fast::exp(score - next);
		float total = p + simd_shuffle_xor(p, 4);
		total += simd_shuffle_xor(total, 8);
		total += simd_shuffle_xor(total, 16);
		weight[s] = fma(weight[s], prior, total);
		peak[s] = next;
		value[s] *= prior;
		prob[s] = p;
	}
	for (uint t = 0; t < 8; t++) {
		float4 v = RELATIVE_VALUE(t);
		for (uint s = 0; s < Sharing; s++) {
			value[s] = fma(float4(simd_shuffle(prob[s], ushort(t * 4))), v, value[s]);
		}
	}
}
`

// relativeSplitEpilogue writes one query's partial sums.
const relativeSplitEpilogue = `
for (uint s = 0; s < Sharing; s++) {
	uint idx = ((group * Sharing + s) * Queries + proposal) * Parts + shard;
	if (lane == 0) {
		statistics[2 * idx] = peak[s];
		statistics[2 * idx + 1] = weight[s];
	}
	for (uint j = 0; j < 4; j++) {
		numerators[idx * 128 + component + j] = T(value[s][j]);
	}
}
`

// relativeAttentionBlockPart reads keys and values from device memory. Each
// SIMD group serves one query of one KV head and partition.
var relativeAttentionBlockPart = &gpuKernel{
	name:    "relative_attention_block_part",
	inputs:  []string{"query", "keys", "values", "relative", "sink", "extent", "window", "scale"},
	outputs: []string{"numerators", "statistics"},
	strided: true,
	metal: gpuSource{source: `
uint lane = thread_index_in_simdgroup;
uint shard = thread_position_in_grid.x / 32;
uint proposal = thread_position_in_grid.y % Queries;
uint group = thread_position_in_grid.y / Queries;
` + relativeSplitPrologue + `
uint row_base = group * uint(head_stride) + component;
#define RELATIVE_LOAD(source, t) float4(float(source[row_base + min(start + (t), newest) * Row]), \
	float(source[row_base + min(start + (t), newest) * Row + 1]), \
	float(source[row_base + min(start + (t), newest) * Row + 2]), \
	float(source[row_base + min(start + (t), newest) * Row + 3]))
#define RELATIVE_KEY(t) RELATIVE_LOAD(keys, t)
#define RELATIVE_VALUE(t) RELATIVE_LOAD(values, t)
// Blocks start at this query's first visible key, as in single-token decode,
// so a windowed query's sums do not depend on where the block's view begins.
for (uint start = begin + shard * 8; start < end; start += Parts * 8) {
` + relativeSplitBlock + `
}
` + relativeSplitEpilogue},
}

// relativeAttentionSharedPart computes the same sums for blocks of queries over
// long histories, where reading the history once per query made the pass grow
// with the block width. One threadgroup per partition and KV head stages four
// of its blocks of keys and values at a time in threadgroup memory, and its
// SIMD groups, one per query, all score the staged blocks.
var relativeAttentionSharedPart = &gpuKernel{
	name:    "relative_attention_shared_part",
	inputs:  []string{"query", "keys", "values", "relative", "sink", "extent", "window", "scale"},
	outputs: []string{"numerators", "statistics"},
	strided: true,
	metal: gpuSource{source: `
uint lane = thread_index_in_simdgroup;
uint proposal = simdgroup_index_in_threadgroup;
uint shard = threadgroup_position_in_grid.x;
uint group = threadgroup_position_in_grid.y;
uint tid = thread_index_in_threadgroup;
` + relativeSplitPrologue + `
// Words per staged row: 16-byte chunks of a BF16 row.
constexpr uint Words = Row / 8;
threadgroup uint4 tile_keys[32 * 16], tile_values[32 * 16];
const device uint4* key_words = (const device uint4*)(keys + group * head_stride);
const device uint4* value_words = (const device uint4*)(values + group * head_stride);
const threadgroup bfloat16_t* staged_keys = (const threadgroup bfloat16_t*)tile_keys;
const threadgroup bfloat16_t* staged_values = (const threadgroup bfloat16_t*)tile_values;
#define RELATIVE_STAGED(source, t) float4(float(source[(sub + (t)) * Row + component]), \
	float(source[(sub + (t)) * Row + component + 1]), \
	float(source[(sub + (t)) * Row + component + 2]), \
	float(source[(sub + (t)) * Row + component + 3]))
#define RELATIVE_KEY(t) RELATIVE_STAGED(staged_keys, t)
#define RELATIVE_VALUE(t) RELATIVE_STAGED(staged_values, t)
for (uint batch = start_block; batch * 8 < hi; batch += 4 * Parts) {
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint c = tid; c < 32 * Words; c += 32 * Queries) {
		uint slot_row = c / Words;
		uint row = min((batch + (slot_row / 8) * Parts) * 8 + slot_row % 8, newest);
		tile_keys[c] = key_words[row * Words + c % Words];
		tile_values[c] = value_words[row * Words + c % Words];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for (uint sub = 0; sub < 32; sub += 8) {
		uint start = (batch + (sub / 8) * Parts) * 8;
` + relativeSplitBlock + `
	}
}
` + relativeSplitEpilogue},
}

// relativeMergeSums reduces one row's partial softmax sums. The including
// kernel defines row, lane, component, parts, and nums and stats pointing at
// that request's partials. It leaves the denominator z (summed across the SIMD
// group) and each lane's share of four numerator channels from component.
const relativeMergeSums = `
uint first = row * parts;
float peak = -INFINITY;
for (uint p = lane; p < parts; p += 32) {
	peak = max(peak, stats[2 * (first + p)]);
}
peak = simd_max(peak);
float z = 0.0f;
float4 value(0.0f);
for (uint p = lane; p < parts; p += 32) {
	uint idx = first + p;
	float factor = fast::exp(stats[2 * idx] - peak);
	z = fma(stats[2 * idx + 1], factor, z);
	for (uint j = 0; j < 4; j++) {
		value[j] = fma(float(nums[idx * 128 + component + j]), factor, value[j]);
	}
}
z = simd_sum(z);
`

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
auto nums = numerators;
auto stats = statistics;
constexpr uint parts = Parts;
` + relativeMergeSums + `
for (uint j = 0; j < 4; j++) {
	float answer = simd_sum(value[j]) / z;
	if (lane == 0) {
		uint idx = row * 128 + component + j;
		output[idx] = T(answer);
	}
}
`},
}

// relativeAttentionMergeGated merges one or two requests' partial sums and
// applies the attention output gate, so the gate and joining the requests need
// no launches of their own. The attention rounds to T before the gate, as the
// separate merge output does, and the sigmoid is evaluated the way MLX's
// elementwise sigmoid evaluates it, so the result matches the separate gate.
// The gate is read in place through its strides.
var relativeAttentionMergeGated = &gpuKernel{
	name:    "relative_attention_merge_gated",
	inputs:  []string{"numerators", "statistics", "numerators1", "statistics1", "gate"},
	outputs: []string{"output"},
	strided: true,
	metal: gpuSource{source: `
uint request = thread_position_in_grid.y / (Heads * Queries);
uint row = thread_position_in_grid.y % (Heads * Queries);
uint lane = thread_index_in_simdgroup;
uint component = (thread_position_in_grid.x / 32) * 4;
uint head = row / Queries;
uint query = row % Queries;
// Evaluate the gate before the reduction so its load latency overlaps the
// merge. The sigmoid of -|g| cannot overflow; reflect it for positive g.
long gate_at = long(request) * gate_strides[0] + long(query) * gate_strides[1] +
	long(head * 128 + component) * gate_strides[2];
float4 opening;
for (uint j = 0; j < 4; j++) {
	float g = float(gate[gate_at + long(j) * gate_strides[2]]);
	float closed = 1 / (1 + metal::precise::exp(metal::abs(g)));
	opening[j] = g < 0 ? closed : 1 - closed;
}
auto nums = request == 0 ? numerators : numerators1;
auto stats = request == 0 ? statistics : statistics1;
uint parts = request == 0 ? Parts : Parts1;
` + relativeMergeSums + `
uint at = ((request * Queries + query) * Heads + head) * 128 + component;
for (uint j = 0; j < 4; j++) {
	float answer = float(T(simd_sum(value[j]) / z));
	if (lane == 0) {
		output[at + j] = bfloat16_t(answer * opening[j]);
	}
}
`},
}
