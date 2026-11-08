package mlx

import (
	"fmt"
	"math"
)

const indexedBlockAttentionMetalSource = `
uint row = threadgroup_position_in_grid.y;
uint simd_group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;

constexpr int SIMDGroups = 32;
constexpr int SIMDWidth = 32;
constexpr int ValuesPerLane = D / SIMDWidth;

thread float query[ValuesPerLane];
thread float values[ValuesPerLane];
threadgroup float outputs[SIMDGroups * SIMDWidth];
threadgroup float max_scores[SIMDGroups];
threadgroup float sum_scores[SIMDGroups];

int query_index = int(row % QueryLength);
int batch_head = int(row / QueryLength);
int batch = batch_head / QueryHeads;
int query_head = batch_head - batch * QueryHeads;
int kv_head = query_head / (QueryHeads / KVHeads);
int query_end = query_ends[batch * QueryLength + query_index];
int key_count = int(key_length);

auto query_ptr = q + ((batch_head * QueryLength + query_index) * D) + lane * ValuesPerLane;
auto output_ptr = out + ((batch_head * QueryLength + query_index) * D) + simd_group * ValuesPerLane;

for (int i = 0; i < ValuesPerLane; ++i) {
  query[i] = float(scale) * static_cast<float>(query_ptr[i]);
  values[i] = 0.0f;
}

int blocks_offset = (batch * QueryLength + query_index) * TopKBlocks;
float max_score = -3.402823466e+38F;
float sum_score = 0.0f;

// Each SIMD group owns an interleaved subset of the selected tokens and
// accumulates an independent online-softmax state.
for (int selected = int(simd_group); selected < TopKBlocks * BlockSize; selected += SIMDGroups) {
  int block_slot = selected / BlockSize;
  int block_offset = selected - block_slot * BlockSize;
  int block = block_indices[blocks_offset + block_slot];
  int key_index = block * BlockSize + block_offset;
  bool valid = block >= 0 && key_index < key_count && key_index < query_end;

  float score = -INFINITY;
  if (valid) {
    auto key_ptr = k + (((batch * KVHeads + kv_head) * key_count + key_index) * D) + lane * ValuesPerLane;
    score = 0.0f;
    for (int i = 0; i < ValuesPerLane; ++i) {
      score += query[i] * static_cast<float>(key_ptr[i]);
    }
    score = simd_sum(score);
  }

  float next_max = metal::max(max_score, score);
  float previous_factor = fast::exp(max_score - next_max);
  float weight = valid ? fast::exp(score - next_max) : 0.0f;
  max_score = next_max;
  sum_score = sum_score * previous_factor + weight;

  if (valid) {
    auto value_ptr = v + (((batch * KVHeads + kv_head) * key_count + key_index) * D) + lane * ValuesPerLane;
    for (int i = 0; i < ValuesPerLane; ++i) {
      values[i] = values[i] * previous_factor + weight * static_cast<float>(value_ptr[i]);
    }
  } else {
    for (int i = 0; i < ValuesPerLane; ++i) {
      values[i] *= previous_factor;
    }
  }
}

// QSA always attends the incomplete compression block after the selected
// complete blocks. At most BlockSize-1 SIMD groups have a valid tail token.
int tail_start = (query_end / BlockSize) * BlockSize;
int tail_index = tail_start + int(simd_group);
bool valid_tail = tail_index < query_end && tail_index < key_count;
if (valid_tail) {
  auto key_ptr = k + (((batch * KVHeads + kv_head) * key_count + tail_index) * D) + lane * ValuesPerLane;
  float score = 0.0f;
  for (int i = 0; i < ValuesPerLane; ++i) {
    score += query[i] * static_cast<float>(key_ptr[i]);
  }
  score = simd_sum(score);

  float next_max = metal::max(max_score, score);
  float previous_factor = fast::exp(max_score - next_max);
  float weight = fast::exp(score - next_max);
  max_score = next_max;
  sum_score = sum_score * previous_factor + weight;

  auto value_ptr = v + (((batch * KVHeads + kv_head) * key_count + tail_index) * D) + lane * ValuesPerLane;
  for (int i = 0; i < ValuesPerLane; ++i) {
    values[i] = values[i] * previous_factor + weight * static_cast<float>(value_ptr[i]);
  }
}

if (lane == 0) {
  max_scores[simd_group] = max_score;
  sum_scores[simd_group] = sum_score;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

max_score = max_scores[lane];
float row_max = simd_max(max_score);
float row_factor = fast::exp(max_score - row_max);
float row_sum = simd_sum(sum_scores[lane] * row_factor);

// Transpose the lane-by-SIMD-group accumulators through threadgroup memory.
// Each SIMD group then reduces and writes one contiguous output segment.
for (int i = 0; i < ValuesPerLane; ++i) {
  outputs[lane * SIMDWidth + simd_group] = values[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  values[i] = simd_sum(outputs[simd_group * SIMDWidth + lane] * row_factor);
  values[i] /= row_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (lane == 0) {
  for (int i = 0; i < ValuesPerLane; ++i) {
    output_ptr[i] = static_cast<InT>(values[i]);
  }
}
`

var indexedBlockAttention = &gpuKernel{
	name:    "indexed_block_scaled_dot_product_attention",
	inputs:  []string{"q", "k", "v", "block_indices", "query_ends", "scale", "key_length", "block_size"},
	outputs: []string{"out"},
	metal: gpuSource{
		source: indexedBlockAttentionMetalSource,
		header: "#include <metal_simdgroup>\nusing namespace metal;\n",
	},
	fallback: func(launch gpuLaunch) []*Array {
		in := launch.inputs
		return []*Array{indexedBlockScaledDotProductAttentionGraph(
			in[0], in[1], in[2], in[3], in[4], int(in[7].Int()), in[5].Float(),
		)}
	},
}

// The graph path is faster for small query batches. At 1024 tokens the
// block-indexed kernel pulls ahead and avoids the graph's context-dependent
// intermediate growth.
const indexedBlockAttentionMinQueryLength = 1024

func indexedTokenScaledDotProductAttentionGraph(q, k, v, indices, valid *Array, scale float32) *Array {
	outputType := q.DType()
	B, queryHeads, L, D := int32(q.Dim(0)), int32(q.Dim(1)), int32(q.Dim(2)), int32(q.Dim(3))
	kvHeads, K := int32(k.Dim(1)), int32(k.Dim(2))
	repeats := queryHeads / kvHeads

	offsets := make([]int32, B)
	for i := range offsets {
		offsets[i] = int32(i) * K
	}
	logical := Add(indices, FromValues(offsets, int(B), 1, 1))
	gather := func(history *Array) *Array {
		flattened := Reshape(Transpose(history, 1, 0, 2, 3), kvHeads, B*K, D)
		return Transpose(Take(flattened, logical, 1), 1, 0, 2, 3, 4)
	}
	selectedK, selectedV := gather(k), gather(v)

	qr := Reshape(q, B, kvHeads, repeats, L, 1, D)
	kr := Transpose(ExpandDims(selectedK, 2), 0, 1, 2, 3, 5, 4)
	scores := Squeeze(Matmul(qr.AsType(DTypeFloat32), kr.AsType(DTypeFloat32)), 4)
	scores = MulScalar(scores, scale)
	mask := ExpandDims(ExpandDims(valid, 1), 1)
	fill := AddScalar(Zeros(scores.DType(), scores.Dims()...), -float32(math.MaxFloat32))
	scores = Where(mask, scores, fill)
	probs := SoftmaxAxis(scores, -1, true)

	vr := ExpandDims(selectedV.AsType(DTypeFloat32), 2)
	out := Matmul(ExpandDims(probs, 4), vr)
	out = Squeeze(out, 4)
	return Reshape(out, B, queryHeads, L, D).AsType(outputType)
}

func indexedBlockScaledDotProductAttentionGraph(q, k, v, blockIndices, queryEnds *Array, blockSize int, scale float32) *Array {
	B, L, topK := int32(blockIndices.Dim(0)), int32(blockIndices.Dim(1)), int32(blockIndices.Dim(2))
	keyLength := k.Dim(2)

	offsets := Reshape(Arange(0, float64(blockSize), 1, DTypeInt32), 1, 1, 1, int32(blockSize))
	indices := Add(MulScalar(ExpandDims(blockIndices, -1), float32(blockSize)), offsets)
	validBlocks := ExpandDims(blockIndices.Greater(FromValue(-1)), -1)
	queryLimit := ExpandDims(ExpandDims(queryEnds, -1), -1)
	valid := Mul(validBlocks.AsType(DTypeInt32), indices.Less(queryLimit).AsType(DTypeInt32))
	valid = Mul(valid, indices.Less(FromValue(keyLength)).AsType(DTypeInt32))
	indices = Reshape(indices, B, L, topK*int32(blockSize))
	valid = Reshape(valid, B, L, topK*int32(blockSize))

	tailWidth := blockSize - 1
	if tailWidth > 0 {
		tailOffsets := Reshape(Arange(0, float64(tailWidth), 1, DTypeInt32), 1, 1, int32(tailWidth))
		tailStart := MulScalar(ExpandDims(FloorDivideScalar(queryEnds, int32(blockSize)), -1), float32(blockSize))
		tail := Add(tailStart, tailOffsets)
		tailValid := Mul(tail.Less(ExpandDims(queryEnds, -1)).AsType(DTypeInt32), tail.Less(FromValue(keyLength)).AsType(DTypeInt32))
		indices = Concatenate([]*Array{indices, tail}, -1)
		valid = Concatenate([]*Array{valid, tailValid}, -1)
	}

	validMask := valid.Greater(FromValue(0))
	indices = Where(validMask, indices, Zeros(DTypeInt32, indices.Dims()...)).AsType(DTypeInt32)
	return indexedTokenScaledDotProductAttentionGraph(q, k, v, indices, validMask, scale)
}

// IndexedBlockScaledDotProductAttention attends to a per-query list of
// complete K/V blocks followed by that query's incomplete causal block.
// q is [B, QH, L, D], k and v are [B, KVH, K, D], blockIndices is
// [B, L, TopK], and queryEnds is [B, L]. Negative block indices are ignored.
// A finite initial online-softmax maximum keeps SIMD groups with no valid
// selected token from evaluating -Inf - -Inf.
func IndexedBlockScaledDotProductAttention(q, k, v, blockIndices, queryEnds *Array, blockSize int, scale float32) *Array {
	if q == nil || k == nil || v == nil || blockIndices == nil || queryEnds == nil ||
		q.NumDims() != 4 || k.NumDims() != 4 || v.NumDims() != 4 ||
		blockIndices.NumDims() != 3 || queryEnds.NumDims() != 2 || blockSize <= 0 {
		panic("mlx.IndexedBlockScaledDotProductAttention: invalid inputs")
	}
	B, queryHeads, L, D := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3)
	kvHeads, keyLength := k.Dim(1), k.Dim(2)
	if k.Dim(0) != B || v.Dim(0) != B || v.Dim(1) != kvHeads || v.Dim(2) != keyLength ||
		k.Dim(3) != D || v.Dim(3) != D || queryHeads%kvHeads != 0 ||
		blockIndices.Dim(0) != B || blockIndices.Dim(1) != L ||
		queryEnds.Dim(0) != B || queryEnds.Dim(1) != L {
		panic(fmt.Sprintf(
			"mlx.IndexedBlockScaledDotProductAttention: incompatible shapes q=%v k=%v v=%v blocks=%v queryEnds=%v",
			q.Dims(), k.Dims(), v.Dims(), blockIndices.Dims(), queryEnds.Dims(),
		))
	}
	if !MetalIsAvailable() || L < indexedBlockAttentionMinQueryLength || blockSize > 32 || D%32 != 0 ||
		q.DType() != k.DType() || q.DType() != v.DType() ||
		(q.DType() != DTypeBFloat16 && q.DType() != DTypeFloat16) ||
		blockIndices.DType() != DTypeInt32 || queryEnds.DType() != DTypeInt32 {
		return indexedBlockScaledDotProductAttentionGraph(q, k, v, blockIndices, queryEnds, blockSize, scale)
	}
	return indexedBlockScaledDotProductAttentionMetal(q, k, v, blockIndices, queryEnds, blockSize, scale)
}

func indexedBlockScaledDotProductAttentionMetal(q, k, v, blockIndices, queryEnds *Array, blockSize int, scale float32) *Array {
	B, queryHeads, L, D := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3)
	kvHeads, keyLength, topK := k.Dim(1), k.Dim(2), blockIndices.Dim(2)
	outs := indexedBlockAttention.run(gpuLaunch{
		dtypes: []gpuDTypeArg{{"InT", q.DType()}},
		ints: []gpuIntArg{
			{"D", D},
			{"QueryLength", L},
			{"QueryHeads", queryHeads},
			{"KVHeads", kvHeads},
			{"BlockSize", blockSize},
			{"TopKBlocks", topK},
		},
		outputs: []gpuOutputSpec{{
			"INDEXED_BLOCK_SCALED_DOT_PRODUCT_ATTENTION",
			[]int32{int32(B), int32(queryHeads), int32(L), int32(D)},
			q.DType(),
		}},
		grid:        [3]int{1024, B * queryHeads * L, 1},
		threadGroup: [3]int{1024, 1, 1},
		inputs: []*Array{
			q, k, v, blockIndices, queryEnds,
			FromValue(scale), FromValue(keyLength), FromValue(blockSize),
		},
	})
	return outs[0]
}
