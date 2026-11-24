package mlx

import "math"

// RMSNormRoundBF16 normalizes the last axis of x in FP32, applies an optional
// FP32 weight and rounds once to BF16. It matches RMSNormFn on an FP32 copy
// of x followed by a cast, without materializing either intermediate.
// It returns nil when the Metal kernel cannot run.
func RMSNormRoundBF16(x, weight *Array, eps float32) *Array {
	return rmsNormRound(x, nil, weight, eps)
}

// RMSNormPairRoundBF16 normalizes a and b without weights and returns them
// side by side, [..., 2*width] in BF16.
func RMSNormPairRoundBF16(a, b *Array, eps float32) *Array {
	if b == nil {
		return nil
	}
	return rmsNormRound(a, b, nil, eps)
}

func rmsNormRound(a, b, weight *Array, eps float32) *Array {
	if !MetalIsAvailable() || a == nil || a.NumDims() == 0 {
		return nil
	}
	width := a.Dim(-1)
	if width < 8 || width%8 != 0 || width > 8*1024 || a.Size() == 0 || uint64(a.Size()) > math.MaxInt32/2 {
		return nil
	}
	supported := func(x *Array) bool {
		return x.DType() == DTypeFloat32 || x.DType() == DTypeBFloat16 || x.DType() == DTypeFloat16
	}
	if !supported(a) {
		return nil
	}
	sources := 1
	second := a
	if b != nil {
		if !supported(b) || b.Size() != a.Size() || b.Dim(-1) != width {
			return nil
		}
		sources, second = 2, b
	}
	hasWeight := 0
	w := a
	if weight != nil {
		if weight.DType() != DTypeFloat32 || weight.Size() != width {
			return nil
		}
		hasWeight, w = 1, weight
	}
	rows := a.Size() / width
	shape := make([]int32, a.NumDims())
	for i := range shape {
		shape[i] = int32(a.Dim(i))
	}
	shape[len(shape)-1] = int32(sources * width)
	threads := (width/8 + 31) / 32 * 32
	out, ok := rmsNormRoundKernel.applyMetal(gpuLaunch{
		inputs:  []*Array{a, second, w},
		ints:    []gpuIntArg{{"Width", width}, {"Sources", sources}, {"HasWeight", hasWeight}, {"EpsBits", int(math.Float32bits(eps))}},
		outputs: []gpuOutputSpec{{"rms_norm_round", shape, DTypeBFloat16}},
		grid:    [3]int{threads, rows, sources}, threadGroup: [3]int{threads, 1, 1},
	})
	if !ok {
		return nil
	}
	return out[0]
}

var rmsNormRoundKernel = &gpuKernel{
	name:    "rms_norm_round_bf16",
	inputs:  []string{"x0", "x1", "weight"},
	outputs: []string{"out"},
	metal: gpuSource{source: `
constexpr uint Groups = (Width / 8 + 31) / 32;
uint row = threadgroup_position_in_grid.y, source = threadgroup_position_in_grid.z;
uint t = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup, group = simdgroup_index_in_threadgroup;
bool active = t * 8 < Width;
size_t at = size_t(row) * Width + t * 8;
float v[8];
float sum = 0.0f;
for (uint i = 0; i < 8; ++i) {
    v[i] = !active ? 0.0f : source == 0 ? float(x0[at + i]) : float(x1[at + i]);
    sum += v[i] * v[i];
}
threadgroup float partial[Groups];
sum = simd_sum(sum);
if (lane == 0) partial[group] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
float total = 0.0f;
for (uint g = 0; g < Groups; ++g) total += partial[g];
float inv = metal::precise::rsqrt(total / float(Width) + as_type<float>(EpsBits));
if (!active) return;
size_t dst = (size_t(row) * Sources + source) * Width + t * 8;
for (uint i = 0; i < 8; ++i) {
    float y = v[i] * inv;
    if (HasWeight) y = float(weight[t * 8 + i]) * y;
    out[dst + i] = bfloat(y);
}
`},
}
