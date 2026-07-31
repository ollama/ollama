package mlx

const mambaGatedRMSNormMetalKernelHeader = `
#include <metal_stdlib>
using namespace metal;
`

var mambaGatedGroupRMSNorm = &gpuKernel{
	name:    "mamba_gated_group_rmsnorm",
	inputs:  []string{"x", "gate", "weight"},
	outputs: []string{"out"},
	metal: gpuSource{
		source: mambaGatedRMSNormMetalKernelSource,
		header: mambaGatedRMSNormMetalKernelHeader,
	},
}

const mambaGatedRMSNormMetalKernelSource = `
constexpr int Threads = 256;

auto group_idx = int(threadgroup_position_in_grid.x);
auto tid = int(thread_index_in_threadgroup);
auto g = group_idx % Groups;
auto token = group_idx / Groups;
auto base = token * Inner + g * GroupSize;
constexpr float eps_val = float(EpsNano) * 1.0e-9f;

threadgroup float partial[Threads];
float sum = 0.0f;
for (int i = tid; i < GroupSize; i += Threads) {
  float gate_val = static_cast<float>(gate[base + i]);
  float silu = gate_val / (1.0f + exp(-gate_val));
  float v = static_cast<float>(x[base + i]) * silu;
  sum += v * v;
}
partial[tid] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int offset = Threads / 2; offset > 0; offset >>= 1) {
  if (tid < offset) {
    partial[tid] += partial[tid + offset];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

float inv = rsqrt(partial[0] / float(GroupSize) + eps_val);
for (int i = tid; i < GroupSize; i += Threads) {
  float gate_val = static_cast<float>(gate[base + i]);
  float silu = gate_val / (1.0f + exp(-gate_val));
  float v = static_cast<float>(x[base + i]) * silu;
  out[base + i] = static_cast<OutT>(v * inv * static_cast<float>(weight[g * GroupSize + i]));
}
`

func mambaGatedRMSNormValidate(x, gate, weight *Array, groups int, eps float32, outDType DType) (B, L, inner, groupSize int, ok bool) {
	if x == nil || gate == nil || weight == nil || groups <= 0 || eps < 0 {
		return 0, 0, 0, 0, false
	}
	if x.DType() != DTypeFloat32 || !mambaGatedRMSNormSupportedDType(gate.DType()) || !mambaGatedRMSNormSupportedDType(weight.DType()) || !mambaGatedRMSNormSupportedDType(outDType) {
		return 0, 0, 0, 0, false
	}
	xd := x.Dims()
	gd := gate.Dims()
	wd := weight.Dims()
	if len(xd) != 3 || len(gd) != 3 || len(wd) != 1 {
		return 0, 0, 0, 0, false
	}
	B, L, inner = xd[0], xd[1], xd[2]
	if B <= 0 || L <= 0 || inner <= 0 || inner%groups != 0 {
		return 0, 0, 0, 0, false
	}
	if gd[0] != B || gd[1] != L || gd[2] != inner || wd[0] != inner {
		return 0, 0, 0, 0, false
	}
	groupSize = inner / groups
	if groupSize <= 0 {
		return 0, 0, 0, 0, false
	}
	return B, L, inner, groupSize, true
}

func mambaGatedRMSNormSupportedDType(dtype DType) bool {
	return dtype == DTypeFloat32 || dtype == DTypeFloat16 || dtype == DTypeBFloat16
}

func mambaGatedRMSNormTemplateArgs(groups, inner, groupSize int, eps float32) []gpuIntArg {
	epsNano := int(eps*1.0e9 + 0.5)
	return []gpuIntArg{
		{"Groups", groups},
		{"Inner", inner},
		{"GroupSize", groupSize},
		{"EpsNano", epsNano},
	}
}

// FastMambaGatedGroupRMSNorm computes RMSNorm(y * SiLU(gate), groups) * weight
// for Nemotron-H Mamba2. It returns ok=false for unsupported backends or shapes
// so callers can use the backend-neutral MLX expression.
func FastMambaGatedGroupRMSNorm(x, gate, weight *Array, groups int, eps float32, outDType DType) (out *Array, ok bool) {
	B, L, inner, groupSize, ok := mambaGatedRMSNormValidate(x, gate, weight, groups, eps, outDType)
	if !ok {
		return nil, false
	}

	const threads = 256
	outs, ok := mambaGatedGroupRMSNorm.applyMetal(gpuLaunch{
		dtypes: []gpuDTypeArg{{"OutT", outDType}},
		ints:   mambaGatedRMSNormTemplateArgs(groups, inner, groupSize, eps),
		outputs: []gpuOutputSpec{
			{"MAMBA_GATED_GROUP_RMSNORM", []int32{int32(B), int32(L), int32(inner)}, outDType},
		},
		grid:        [3]int{B * L * groups * threads, 1, 1},
		threadGroup: [3]int{threads, 1, 1},
		inputs:      []*Array{x, gate, weight},
	})
	if !ok {
		return nil, false
	}
	return outs[0], true
}
