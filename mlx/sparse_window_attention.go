package mlx

// Each SIMD group scores a subset of the raw window and selected compressed
// keys, then the groups combine their online softmax states. This avoids a
// per-query copy of the selected keys.
var sparseWindowAttention = &gpuKernel{
	name: "sparse_window_attention",
	inputs: []string{
		"q", "raw", "compressed", "selection", "sink", "query_start", "key_start", "scale",
	},
	outputs: []string{"out"},
	metal: gpuSource{header: "#include <metal_simdgroup>\nusing namespace metal;\n", source: `
uint row = threadgroup_position_in_grid.y;
uint group = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
constexpr int Groups = G;
constexpr int LaneWidth = 32;
constexpr int PerLane = D / LaneWidth;

thread float query[PerLane];
thread float weighted[PerLane];
threadgroup float partial[(Groups == 16 || Groups == 32) ? Groups * LaneWidth : Groups * D];
threadgroup float group_maximum[Groups];
threadgroup float group_sum[Groups];

int length = int(q_shape[1]);
int head = int(row % H);
int token = int(row / H) % length;
int batch = int(row / H) / length;
int absolute = query_start + token;
int raw_count = int(raw_shape[1]);
int compressed_count = int(compressed_shape[1]);
int top_k = int(selection_shape[2]);
int first = max(absolute - W + 1, key_start);
int last = min(absolute + 1, key_start + raw_count);

auto q_ptr = q + (((batch * length + token) * H + head) * D) + lane * PerLane;
auto out_ptr = out + (((batch * length + token) * H + head) * D);
for (int i = 0; i < PerLane; ++i) {
  query[i] = float(q_ptr[i]) * scale;
  weighted[i] = 0.0f;
}
float maximum = -3.402823466e+38F;
float sum = 0.0f;

for (int slot = int(group); slot < W + top_k; slot += Groups) {
  bool valid = false;
  bool is_raw = slot < W;
  int index = 0;
  if (is_raw) {
    int position = first + slot;
    valid = position < last;
    index = position - key_start;
  } else {
    index = selection[(batch * length + token) * top_k + slot - W];
    valid = index >= 0 && index < compressed_count;
  }
  float score = -INFINITY;
  float key_values[PerLane];
  if (valid) {
    if (is_raw) {
      auto ptr = raw + ((batch * raw_count + index) * D) + lane * PerLane;
      for (int i = 0; i < PerLane; ++i) key_values[i] = float(ptr[i]);
    } else {
      auto ptr = compressed + ((batch * compressed_count + index) * D) + lane * PerLane;
      for (int i = 0; i < PerLane; ++i) key_values[i] = float(ptr[i]);
    }
    score = 0.0f;
    for (int i = 0; i < PerLane; ++i) score += query[i] * key_values[i];
    score = simd_sum(score);
  }
  float next_maximum = metal::max(maximum, score);
  float previous_factor = fast::exp(maximum - next_maximum);
  float weight = valid ? fast::exp(score - next_maximum) : 0.0f;
  maximum = next_maximum;
  sum = sum * previous_factor + weight;
  for (int i = 0; i < PerLane; ++i) {
    weighted[i] *= previous_factor;
    if (valid) weighted[i] += weight * key_values[i];
  }
}

if (group == 0) {
  float score = sink[head];
  float next_maximum = metal::max(maximum, score);
  float previous_factor = fast::exp(maximum - next_maximum);
  maximum = next_maximum;
  sum = sum * previous_factor + fast::exp(score - next_maximum);
  for (int i = 0; i < PerLane; ++i) weighted[i] *= previous_factor;
}

if (lane == 0) {
  group_maximum[group] = maximum;
  group_sum[group] = sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float source_maximum = lane < Groups ? group_maximum[lane] : -3.402823466e+38F;
float row_maximum = simd_max(source_maximum);
float factor = lane < Groups ? fast::exp(source_maximum - row_maximum) : 0.0f;
float denominator = simd_sum((lane < Groups ? group_sum[lane] : 0.0f) * factor);
if (Groups == 16 || Groups == 32) {
  for (int i = 0; i < PerLane; ++i) {
    partial[lane * Groups + group] = weighted[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int output_group = int(group); output_group < LaneWidth; output_group += Groups) {
      float source = lane < Groups ? partial[output_group * Groups + lane] : 0.0f;
      float value = simd_sum(source * factor) / denominator;
      if (lane == 0) out_ptr[output_group * PerLane + i] = static_cast<QT>(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
} else {
  for (int i = 0; i < PerLane; ++i) {
    partial[group * D + lane * PerLane + i] = weighted[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (group == 0) {
    float factors[Groups];
    for (int j = 0; j < Groups; ++j) factors[j] = fast::exp(group_maximum[j] - row_maximum);
    for (int i = 0; i < PerLane; ++i) {
      int col = lane * PerLane + i;
      float value = 0.0f;
      for (int j = 0; j < Groups; ++j) value += partial[j * D + col] * factors[j];
      out_ptr[col] = static_cast<QT>(value / denominator);
    }
  }
}
`},
	cuda: gpuSource{source: `
constexpr int Groups = 8;
constexpr int LaneWidth = 32;
constexpr int PerLane = D / LaneWidth;
int row = int(blockIdx.x);
int group = int(threadIdx.x / LaneWidth);
int lane = int(threadIdx.x % LaneWidth);
int length = int(q_shape[1]);
int head = row % H;
int token = (row / H) % length;
int batch = (row / H) / length;
int absolute = *query_start + token;
int raw_count = int(raw_shape[1]);
int compressed_count = int(compressed_shape[1]);
int top_k = int(selection_shape[2]);
int first = absolute - W + 1 > *key_start ? absolute - W + 1 : *key_start;
int last = absolute + 1 < *key_start + raw_count ? absolute + 1 : *key_start + raw_count;

float query[PerLane];
float weighted[PerLane];
auto q_ptr = q + row * D + lane * PerLane;
for (int i = 0; i < PerLane; ++i) {
  query[i] = float(q_ptr[i]) * *scale;
  weighted[i] = 0.0f;
}
float maximum = -3.402823466e+38F;
float sum = 0.0f;

for (int slot = group; slot < W + top_k; slot += Groups) {
  bool is_raw = slot < W;
  int index;
  bool valid;
  if (is_raw) {
    int position = first + slot;
    valid = position < last;
    index = position - *key_start;
  } else {
    index = selection[(batch * length + token) * top_k + slot - W];
    valid = index >= 0 && index < compressed_count;
  }
  float score = -INFINITY;
  float key_values[PerLane];
  if (valid) {
    if (is_raw) {
      auto ptr = raw + (batch * raw_count + index) * D + lane * PerLane;
      for (int i = 0; i < PerLane; ++i) key_values[i] = float(ptr[i]);
    } else {
      auto ptr = compressed + (batch * compressed_count + index) * D + lane * PerLane;
      for (int i = 0; i < PerLane; ++i) key_values[i] = float(ptr[i]);
    }
    score = 0.0f;
    for (int i = 0; i < PerLane; ++i) score += query[i] * key_values[i];
    for (int offset = 16; offset > 0; offset /= 2) score += __shfl_down_sync(0xffffffff, score, offset);
    score = __shfl_sync(0xffffffff, score, 0);
  }
  float next_maximum = fmaxf(maximum, score);
  float previous_factor = __expf(maximum - next_maximum);
  float weight = valid ? __expf(score - next_maximum) : 0.0f;
  maximum = next_maximum;
  sum = sum * previous_factor + weight;
  for (int i = 0; i < PerLane; ++i) {
    weighted[i] *= previous_factor;
    if (valid) weighted[i] += weight * key_values[i];
  }
}

if (group == 0) {
  float score = sink[head];
  float next_maximum = fmaxf(maximum, score);
  float previous_factor = __expf(maximum - next_maximum);
  maximum = next_maximum;
  sum = sum * previous_factor + __expf(score - next_maximum);
  for (int i = 0; i < PerLane; ++i) weighted[i] *= previous_factor;
}

__shared__ float group_maximum[Groups];
__shared__ float group_sum[Groups];
__shared__ float partial[Groups * D];
__shared__ float row_maximum;
__shared__ float denominator;
if (lane == 0) {
  group_maximum[group] = maximum;
  group_sum[group] = sum;
}
for (int i = 0; i < PerLane; ++i) partial[group * D + lane * PerLane + i] = weighted[i];
__syncthreads();
if (threadIdx.x == 0) {
  float high = group_maximum[0];
  for (int i = 1; i < Groups; ++i) high = fmaxf(high, group_maximum[i]);
  float total = 0.0f;
  for (int i = 0; i < Groups; ++i) total += group_sum[i] * __expf(group_maximum[i] - high);
  row_maximum = high;
  denominator = total;
}
__syncthreads();
for (int col = int(threadIdx.x); col < D; col += int(blockDim.x)) {
  float value = 0.0f;
  for (int i = 0; i < Groups; ++i) {
    value += partial[i * D + col] * __expf(group_maximum[i] - row_maximum);
  }
  out[row * D + col] = static_cast<QT>(value / denominator);
}
`},
}

// SparseWindowAttention attends to one raw sliding window and a per-query
// compressed-key selection, with one sink logit per query head. A false result
// means the GPU kernel is unavailable and the caller should use its graph
// path. q is [B,L,H,D], raw and compressed are [B,K,D], and selection is
// [B,L,TopK] with negative entries masked out.
func SparseWindowAttention(q, raw, compressed, selection, sink *Array, queryStart, keyStart, window int, scale float32) (*Array, bool) {
	if q == nil || raw == nil || compressed == nil || selection == nil || sink == nil ||
		q.NumDims() != 4 || raw.NumDims() != 3 || compressed.NumDims() != 3 ||
		selection.NumDims() != 3 || sink.NumDims() != 1 {
		panic("mlx: SparseWindowAttention requires q [B,L,H,D], raw and compressed [B,K,D], selection [B,L,TopK], sink [H]")
	}
	B, L, H, D := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3)
	if raw.Dim(0) != B || raw.Dim(2) != D || compressed.Dim(0) != B || compressed.Dim(2) != D ||
		selection.Dim(0) != B || selection.Dim(1) != L || sink.Dim(0) != H {
		panic("mlx: SparseWindowAttention has incompatible dimensions")
	}
	if (!MetalIsAvailable() && !CUDAIsAvailable()) || B == 0 || L == 0 || H == 0 || D == 0 || D > 512 || D%32 != 0 ||
		raw.Dim(1) == 0 || compressed.Dim(1) == 0 || selection.Dim(2) == 0 ||
		window <= 0 || window > 128 || selection.Dim(2) > 512 ||
		(q.DType() != DTypeBFloat16 && q.DType() != DTypeFloat32) ||
		(raw.DType() != DTypeBFloat16 && raw.DType() != DTypeFloat32) ||
		compressed.DType() != DTypeFloat32 || selection.DType() != DTypeInt32 || sink.DType() != DTypeFloat32 {
		return nil, false
	}
	// Prompt chunks of at least 32 queries benefit from fewer groups. Metal
	// decode uses 16 groups to fit devices with a 640-thread limit.
	groups := 32
	if MetalIsAvailable() {
		if L >= 32 {
			groups = 4
		} else {
			groups = 16
		}
	}
	launch := gpuLaunch{
		// CUDA caches compiled kernels by template arguments, so raw precision
		// must be explicit even though the source reads raw through its input.
		dtypes:      []gpuDTypeArg{{"QT", q.DType()}, {"RawT", raw.DType()}},
		ints:        []gpuIntArg{{"D", D}, {"H", H}, {"W", window}, {"G", groups}},
		outputs:     []gpuOutputSpec{{"SPARSE_WINDOW_ATTENTION", []int32{int32(B), int32(L), int32(H), int32(D)}, q.DType()}},
		grid:        [3]int{groups * 32, B * L * H, 1},
		threadGroup: [3]int{groups * 32, 1, 1},
		inputs:      []*Array{q, raw, compressed, selection, sink, FromValue(queryStart), FromValue(keyStart), FromValue(scale)},
	}
	var outputs []*Array
	var ok bool
	if MetalIsAvailable() {
		outputs, ok = sparseWindowAttention.applyMetal(launch)
	} else {
		launch.grid = [3]int{256 * B * L * H, 1, 1}
		launch.threadGroup = [3]int{256, 1, 1}
		outputs, ok = sparseWindowAttention.applyCUDA(launch)
	}
	if !ok {
		return nil, false
	}
	return outputs[0], true
}
