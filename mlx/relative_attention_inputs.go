package mlx

import "math"

// RelativeAttentionInputs prepares one layer's attention inputs from its
// projections. source [rows, tokens, stride] holds the query (heads*128),
// relative (heads*relDim), gate and sink (heads) columns in that order; k and
// v are [rows, tokens, kvHeads*128].
//
// The query and relative rows are scaled per head, clipped to an L2 norm and
// multiplied by the optional per-token temperature [rows, tokens]. Keys are
// clipped and values RMS normalized without a weight. Each result rounds to
// the source dtype. The relative rows then project through projection
// [relDim, distances] into a bias table.
//
// It returns q [rows, heads, tokens, 128] in FP32, k and v
// [rows, kvHeads, tokens, 128], table [rows, heads, tokens, distances] and
// sink [rows, heads, tokens, 1], or nils when the Metal kernel cannot run.
func RelativeAttentionInputs(source, k, v, queryScale, relativeScale, projection, temperature *Array,
	heads, kvHeads int, queryClip, relativeClip, epsilon float32,
) (q, keys, values, table, sink *Array) {
	const width = 128
	if !MetalIsAvailable() || source == nil || k == nil || v == nil || source.NumDims() != 3 ||
		source.DType() != DTypeBFloat16 || k.DType() != source.DType() || v.DType() != source.DType() {
		return nil, nil, nil, nil, nil
	}
	rows, tokens, stride := source.Dim(0), source.Dim(1), source.Dim(2)
	relDim, distances := projection.Dim(0), projection.Dim(1)
	if heads < kvHeads || kvHeads < 1 || relDim > 32 || distances%width != 0 ||
		stride < heads*(2*width+relDim+1) || k.Size() != rows*tokens*kvHeads*width || v.Size() != k.Size() ||
		queryScale.Size() != heads*width || relativeScale.Size() != heads*relDim ||
		uint64(rows)*uint64(tokens)*uint64(heads)*uint64(distances) > math.MaxInt32 {
		return nil, nil, nil, nil, nil
	}
	hasTemperature := 0
	if temperature != nil {
		if temperature.Size() != rows*tokens || temperature.DType() != DTypeFloat32 {
			return nil, nil, nil, nil, nil
		}
		hasTemperature = 1
	} else {
		temperature = queryScale
	}
	shape := func(h, w int) []int32 { return []int32{int32(rows), int32(h), int32(tokens), int32(w)} }
	out, ok := relativeAttentionPrologue.applyMetal(gpuLaunch{
		inputs: []*Array{source, k, v, queryScale, relativeScale, projection, temperature},
		ints: []gpuIntArg{
			{"Stride", stride},
			{"Heads", heads},
			{"KVHeads", kvHeads},
			{"RelDim", relDim},
			{"Distances", distances},
			{"HasTemperature", hasTemperature},
			{"QueryClipBits", int(math.Float32bits(queryClip))},
			{"RelativeClipBits", int(math.Float32bits(relativeClip))},
			{"EpsBits", int(math.Float32bits(epsilon))},
		},
		outputs: []gpuOutputSpec{
			{"relative_q", shape(heads, width), DTypeFloat32},
			{"relative_k", shape(kvHeads, width), source.DType()},
			{"relative_v", shape(kvHeads, width), source.DType()},
			{"relative_table", shape(heads, distances), source.DType()},
			{"relative_sink", shape(heads, 1), source.DType()},
		},
		grid:        [3]int{width, heads, rows * tokens},
		threadGroup: [3]int{width, 1, 1},
	})
	if !ok {
		return nil, nil, nil, nil, nil
	}
	return out[0], out[1], out[2], out[3], out[4]
}

// relativePrologueHeader holds the prologue's threadgroup reductions as plain
// functions: Metal before version 4 (macOS 14 and 15) rejects lambdas.
const relativePrologueHeader = `
// threadgroup_total sums x across a threadgroup of Groups SIMD groups.
template <uint Groups>
inline float threadgroup_total(float x, threadgroup float* partial, uint lane, uint group) {
    float s = simd_sum(x);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partial[group] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint g = 0; g < Groups; ++g) sum += partial[g];
    return sum;
}

// clip_norm scales x so a vector with this sum of squares has at most the
// threshold's norm.
inline float clip_norm(float x, float sum, float threshold) {
    float factor = threshold / (metal::precise::sqrt(sum) + 1e-6f);
    return x * min(factor, 1.0f);
}
`

var relativeAttentionPrologue = &gpuKernel{
	name:    "relative_attention_prologue",
	inputs:  []string{"source", "k", "v", "query_scale", "relative_scale", "projection", "temperature"},
	outputs: []string{"q_out", "k_out", "v_out", "table", "sink"},
	metal: gpuSource{header: relativePrologueHeader, source: `
constexpr uint Width = 128, Groups = Width / 32;
const uint i = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, group = simdgroup_index_in_threadgroup;
const uint h = threadgroup_position_in_grid.y, rt = threadgroup_position_in_grid.z;
const uint tokens = source_shape[1];
const uint row = rt / tokens, t = rt % tokens;
const float tau = HasTemperature ? temperature[rt] : 1.0f;
threadgroup float partial[Groups];
threadgroup float relative[32];
const device auto* src = source + size_t(rt) * Stride;
// Query.
float qv = float(src[h * Width + i]) * float(query_scale[h * Width + i]);
qv = clip_norm(qv, threadgroup_total<Groups>(qv * qv, partial, lane, group), as_type<float>(QueryClipBits));
if (HasTemperature) qv = qv * tau;
q_out[((size_t(row) * Heads + h) * tokens + t) * Width + i] = float(bfloat(qv));
// Relative profile and its bias table.
float rv = 0.0f;
if (i < RelDim) rv = float(src[Heads * Width + h * RelDim + i]) * float(relative_scale[h * RelDim + i]);
rv = clip_norm(rv, threadgroup_total<Groups>(rv * rv, partial, lane, group), as_type<float>(RelativeClipBits));
if (HasTemperature) rv = rv * tau;
if (i < RelDim) relative[i] = float(bfloat(rv));
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint d = i; d < Distances; d += Width) {
    float acc = 0.0f;
    for (uint j = 0; j < RelDim; ++j) acc = fma(relative[j], float(projection[j * Distances + d]), acc);
    table[((size_t(row) * Heads + h) * tokens + t) * Distances + d] = bfloat(acc);
}
if (i == 0) sink[(size_t(row) * Heads + h) * tokens + t] = src[2 * Heads * Width + Heads * RelDim + h];
if (h >= KVHeads) return;
// Key and value.
size_t kv = size_t(rt) * KVHeads * Width + h * Width + i;
size_t out = ((size_t(row) * KVHeads + h) * tokens + t) * Width + i;
float kx = float(k[kv]);
k_out[out] = bfloat(clip_norm(kx, threadgroup_total<Groups>(kx * kx, partial, lane, group), as_type<float>(QueryClipBits)));
float vx = float(v[kv]);
float inv = metal::precise::rsqrt(threadgroup_total<Groups>(vx * vx, partial, lane, group) / float(Width) + as_type<float>(EpsBits));
v_out[out] = bfloat(vx * inv);
`},
}
