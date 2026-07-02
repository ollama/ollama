package mlx

// #include <stdlib.h>
// #include "generated.h"
import "C"

import (
	"sync"
	"unsafe"
)

var (
	moeWeightedSumMetalKernelOnce sync.Once
	moeWeightedSumMetalKernel     C.mlx_fast_metal_kernel
	moeWeightedSumMetalDisabled   bool
)

const moeWeightedSumMetalKernelSource = `
auto elem = thread_position_in_grid.x;
auto total = B * L * H;
if (elem >= total) {
  return;
}

auto h = elem % H;
auto token = elem / H;
auto expert_base = token * TopK * H + h;
auto score_base = token * TopK;

float acc = 0.0f;
for (int k = 0; k < TopK; ++k) {
  acc += static_cast<float>(expert[expert_base + k * H]) * static_cast<float>(scores[score_base + k]);
}
y[elem] = static_cast<OutT>(acc);
`

func initMoEWeightedSumMetalKernel() {
	inputs, freeInputs, ok := cStringVector([]string{"expert", "scores"})
	if !ok {
		moeWeightedSumMetalDisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"y"})
	if !ok {
		moeWeightedSumMetalDisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("moe_weighted_sum")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(moeWeightedSumMetalKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))

	moeWeightedSumMetalKernel = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(false),
	)
}

func moeWeightedSumValidate(expert, scores *Array, outDType DType) (B, L, topK, H int, ok bool) {
	if expert == nil || scores == nil {
		return 0, 0, 0, 0, false
	}
	if !moeWeightedSumSupportedDType(expert.DType()) || !moeWeightedSumSupportedDType(scores.DType()) || !moeWeightedSumSupportedDType(outDType) {
		return 0, 0, 0, 0, false
	}
	ed := expert.Dims()
	sd := scores.Dims()
	if len(ed) != 4 || len(sd) != 3 {
		return 0, 0, 0, 0, false
	}
	B, L, topK, H = ed[0], ed[1], ed[2], ed[3]
	if B <= 0 || L <= 0 || topK <= 0 || H <= 0 {
		return 0, 0, 0, 0, false
	}
	if sd[0] != B || sd[1] != L || sd[2] != topK {
		return 0, 0, 0, 0, false
	}
	return B, L, topK, H, true
}

func moeWeightedSumSupportedDType(dtype DType) bool {
	return dtype == DTypeFloat32 || dtype == DTypeFloat16 || dtype == DTypeBFloat16
}

func addMoEWeightedSumTemplateArgs(cfg C.mlx_fast_metal_kernel_config, B, L, topK, H int, outDType DType) bool {
	for _, tpl := range []struct {
		name  string
		value int
	}{
		{name: "B", value: B},
		{name: "L", value: L},
		{name: "TopK", value: topK},
		{name: "H", value: H},
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

// FastMoEWeightedSum computes sum_k expert[B,L,k,H] * scores[B,L,k] and
// returns [B,L,H]. It returns ok=false for unsupported backends or shapes so
// callers can use a backend-neutral fallback.
func FastMoEWeightedSum(expert, scores *Array, outDType DType) (y *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	if moeWeightedSumMetalDisabled {
		return nil, false
	}
	B, L, topK, H, ok := moeWeightedSumValidate(expert, scores, outDType)
	if !ok {
		return nil, false
	}

	moeWeightedSumMetalKernelOnce.Do(initMoEWeightedSumMetalKernel)
	if moeWeightedSumMetalDisabled {
		return nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addMoEWeightedSumTemplateArgs(cfg, B, L, topK, H, outDType) {
		return nil, false
	}

	yShape := []C.int{C.int(B), C.int(L), C.int(H)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(yShape), C.size_t(len(yShape)), C.mlx_dtype(outDType)) != 0 {
		return nil, false
	}

	total := B * L * H
	gridX := ((total + 255) / 256) * 256
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, C.int(gridX), 1, 1) != 0 {
		return nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1) != 0 {
		return nil, false
	}

	inputs := []C.mlx_array{expert.ctx, scores.ctx}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, moeWeightedSumMetalKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		return nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 1 {
		return nil, false
	}

	y = New("MOE_WEIGHTED_SUM")
	C.mlx_vector_array_get(&y.ctx, outVec, 0)
	return y, true
}
