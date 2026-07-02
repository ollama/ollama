package mlx

// #include <stdlib.h>
// #include "generated.h"
import "C"

import (
	"sync"
	"unsafe"
)

var (
	mambaGatedRMSNormMetalKernelOnce sync.Once
	mambaGatedRMSNormMetalKernel     C.mlx_fast_metal_kernel
	mambaGatedRMSNormMetalDisabled   bool
)

const mambaGatedRMSNormMetalKernelHeader = `
#include <metal_stdlib>
using namespace metal;
`

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

func initMambaGatedRMSNormMetalKernel() {
	inputs, freeInputs, ok := cStringVector([]string{"x", "gate", "weight"})
	if !ok {
		mambaGatedRMSNormMetalDisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"out"})
	if !ok {
		mambaGatedRMSNormMetalDisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("mamba_gated_group_rmsnorm")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(mambaGatedRMSNormMetalKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString(mambaGatedRMSNormMetalKernelHeader)
	defer C.free(unsafe.Pointer(cHeader))

	mambaGatedRMSNormMetalKernel = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(false),
	)
}

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

func addMambaGatedRMSNormTemplateArgs(cfg C.mlx_fast_metal_kernel_config, groups, inner, groupSize int, eps float32, outDType DType) bool {
	epsNano := int(eps*1.0e9 + 0.5)
	for _, tpl := range []struct {
		name  string
		value int
	}{
		{name: "Groups", value: groups},
		{name: "Inner", value: inner},
		{name: "GroupSize", value: groupSize},
		{name: "EpsNano", value: epsNano},
	} {
		cn := C.CString(tpl.name)
		rc := C.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, cn, C.int(tpl.value))
		C.free(unsafe.Pointer(cn))
		if rc != 0 {
			return false
		}
	}

	cn := C.CString("OutT")
	rc := C.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, cn, C.mlx_dtype(outDType))
	C.free(unsafe.Pointer(cn))
	return rc == 0
}

// FastMambaGatedGroupRMSNorm computes RMSNorm(y * SiLU(gate), groups) * weight
// for Nemotron-H Mamba2. It returns ok=false for unsupported backends or shapes
// so callers can use the backend-neutral MLX expression.
func FastMambaGatedGroupRMSNorm(x, gate, weight *Array, groups int, eps float32, outDType DType) (out *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	if mambaGatedRMSNormMetalDisabled {
		return nil, false
	}
	B, L, inner, groupSize, ok := mambaGatedRMSNormValidate(x, gate, weight, groups, eps, outDType)
	if !ok {
		return nil, false
	}

	mambaGatedRMSNormMetalKernelOnce.Do(initMambaGatedRMSNormMetalKernel)
	if mambaGatedRMSNormMetalDisabled {
		return nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addMambaGatedRMSNormTemplateArgs(cfg, groups, inner, groupSize, eps, outDType) {
		return nil, false
	}

	outShape := []C.int{C.int(B), C.int(L), C.int(inner)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(outShape), C.size_t(len(outShape)), C.mlx_dtype(outDType)) != 0 {
		return nil, false
	}

	const threads = 256
	gridX := B * L * groups * threads
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, C.int(gridX), 1, 1) != 0 {
		return nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1) != 0 {
		return nil, false
	}

	inputs := []C.mlx_array{x.ctx, gate.ctx, weight.ctx}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, mambaGatedRMSNormMetalKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		return nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 1 {
		return nil, false
	}

	out = New("MAMBA_GATED_GROUP_RMSNORM")
	C.mlx_vector_array_get(&out.ctx, outVec, 0)
	return out, true
}
