package mlx

// #include <stdlib.h>
// #include "generated.h"
import "C"

// This gathered SwiGLU gate-scale kernel is currently only used by Laguna. As
// part of the retained Laguna fast path, the Apple M5 Max p2048/g128 run was
// about 24% faster on prompt and 7% faster on generate than the GGML target.
// No other model has shown a kept win from this path yet.
// TODO: consider moving this implementation under x/models/laguna once the mlx
// package has a refined custom-kernel API that can safely support model-local
// kernels without exposing low-level MLX internals.
// TODO(cuda): add CUDA equivalents for these custom Metal kernels before
// enabling the same fast path on CUDA-backed MLX; non-Metal backends must keep
// using the generic fallback.

import (
	"sync"
	"unsafe"
)

var (
	swiGLUGatheredGateScaleMetalKernelOnce sync.Once
	swiGLUGatheredGateScaleMetalKernel     C.mlx_fast_metal_kernel
	swiGLUGatheredGateScaleMetalDisabled   bool
)

const swiGLUGatheredGateScaleMetalKernelSource = `
auto elem = thread_position_in_grid.x;
auto total = Rows * H;
if (elem >= total) {
  return;
}

auto row = elem / H;
auto expert = (ScaleCount == 1) ? 0 : int(indices[row]);
float g = static_cast<float>(gate[elem]) * static_cast<float>(scales[expert]);
float u = static_cast<float>(up[elem]);
float s = 1.0f / (1.0f + exp(-g));
y[elem] = static_cast<OutT>((g * s) * u);
`

func initSwiGLUGatheredGateScaleMetalKernel() {
	inputs, freeInputs, ok := cStringVector([]string{"gate", "up", "scales", "indices"})
	if !ok {
		swiGLUGatheredGateScaleMetalDisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"y"})
	if !ok {
		swiGLUGatheredGateScaleMetalDisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("swiglu_gathered_gate_scale")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(swiGLUGatheredGateScaleMetalKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))

	swiGLUGatheredGateScaleMetalKernel = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(false),
	)
}

func swiGLUGatheredGateScaleValidate(gate, up, scales, indices *Array) (dims []int, rows, H, scaleCount int, outDType DType, ok bool) {
	if gate == nil || up == nil || scales == nil || indices == nil {
		return nil, 0, 0, 0, 0, false
	}
	outDType = gate.DType()
	if !swiGLUGatheredGateScaleSupportedDType(outDType) || up.DType() != outDType || !swiGLUGatheredGateScaleSupportedDType(scales.DType()) || indices.DType() != DTypeInt32 {
		return nil, 0, 0, 0, 0, false
	}
	dims = gate.Dims()
	ud := up.Dims()
	sd := scales.Dims()
	id := indices.Dims()
	if len(dims) != 4 || len(ud) != 4 || len(sd) != 1 || len(id) != 2 {
		return nil, 0, 0, 0, 0, false
	}
	for i := range dims {
		if dims[i] <= 0 || dims[i] != ud[i] {
			return nil, 0, 0, 0, 0, false
		}
	}
	if dims[2] != 1 || id[0] != dims[0] || id[1] != dims[1] || sd[0] <= 0 {
		return nil, 0, 0, 0, 0, false
	}
	H = dims[3]
	rows = dims[0] * dims[1]
	scaleCount = sd[0]
	if rows <= 0 || H <= 0 {
		return nil, 0, 0, 0, 0, false
	}
	return dims, rows, H, scaleCount, outDType, true
}

func swiGLUGatheredGateScaleSupportedDType(dtype DType) bool {
	return dtype == DTypeFloat32 || dtype == DTypeFloat16 || dtype == DTypeBFloat16
}

func addSwiGLUGatheredGateScaleTemplateArgs(cfg C.mlx_fast_metal_kernel_config, rows, H, scaleCount int, outDType DType) bool {
	for _, tpl := range []struct {
		name  string
		value int
	}{
		{name: "Rows", value: rows},
		{name: "H", value: H},
		{name: "ScaleCount", value: scaleCount},
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

// FastSwiGLUGatheredGateScale computes SiLU(gate * scales[indices]) * up for
// gate/up shaped [tokens, topK, 1, hidden]. It returns ok=false for unsupported
// shapes, dtypes, or backends so callers can use generic MLX ops.
func FastSwiGLUGatheredGateScale(gate, up, scales, indices *Array) (y *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	if swiGLUGatheredGateScaleMetalDisabled {
		return nil, false
	}
	dims, rows, H, scaleCount, outDType, ok := swiGLUGatheredGateScaleValidate(gate, up, scales, indices)
	if !ok {
		return nil, false
	}

	swiGLUGatheredGateScaleMetalKernelOnce.Do(initSwiGLUGatheredGateScaleMetalKernel)
	if swiGLUGatheredGateScaleMetalDisabled {
		return nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addSwiGLUGatheredGateScaleTemplateArgs(cfg, rows, H, scaleCount, outDType) {
		return nil, false
	}

	outShape := make([]C.int, len(dims))
	for i, dim := range dims {
		outShape[i] = C.int(dim)
	}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(outShape), C.size_t(len(outShape)), C.mlx_dtype(outDType)) != 0 {
		return nil, false
	}

	total := rows * H
	gridX := ((total + 127) / 128) * 128
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, C.int(gridX), 1, 1) != 0 {
		return nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1) != 0 {
		return nil, false
	}

	inputs := []C.mlx_array{gate.ctx, up.ctx, scales.ctx, indices.ctx}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, swiGLUGatheredGateScaleMetalKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		return nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 1 {
		return nil, false
	}

	y = New("SWIGLU_GATHERED_GATE_SCALE")
	C.mlx_vector_array_get(&y.ctx, outVec, 0)
	return y, true
}
