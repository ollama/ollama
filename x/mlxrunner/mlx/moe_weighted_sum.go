package mlx

// #include <stdlib.h>
// #include "generated.h"
import "C"

// These fused MoE weighted-sum kernels are currently only used by Laguna. As
// part of the retained Laguna fast path, the Apple M5 Max p2048/g128 run was
// about 24% faster on prompt and 7% faster on generate than the GGML target.
// Gemma and Qwen weighted-sum trials were neutral or slightly negative.
// TODO: consider moving this implementation under x/models/laguna once the mlx
// package has a refined custom-kernel API that can safely support model-local
// kernels without exposing low-level MLX internals.
// TODO(cuda): add CUDA equivalents for these custom Metal kernels before
// enabling the same fast path on CUDA-backed MLX; non-Metal backends must keep
// using the generic fallback.

import (
	"math"
	"sync"
	"unsafe"
)

var (
	moeWeightedSumMetalKernelOnce sync.Once
	moeWeightedSumMetalKernel     C.mlx_fast_metal_kernel
	moeWeightedSumMetalDisabled   bool

	moeWeightedSumAddMetalKernelOnce sync.Once
	moeWeightedSumAddMetalKernel     C.mlx_fast_metal_kernel
	moeWeightedSumAddMetalDisabled   bool

	moeWeightedSumAdd2MetalKernelOnce sync.Once
	moeWeightedSumAdd2MetalKernel     C.mlx_fast_metal_kernel
	moeWeightedSumAdd2MetalDisabled   bool
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
y[elem] = static_cast<OutT>(acc * (float(ScaleNum) / float(ScaleDen)));
`

const moeWeightedSumAddMetalKernelSource = `
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
y[elem] = static_cast<OutT>(static_cast<float>(shared[elem]) + acc * (float(ScaleNum) / float(ScaleDen)));
`

const moeWeightedSumAdd2MetalKernelSource = `
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
y[elem] = static_cast<OutT>(static_cast<float>(add_a[elem]) + static_cast<float>(add_b[elem]) + acc * (float(ScaleNum) / float(ScaleDen)));
`

func initMoEWeightedSumKernel(target *C.mlx_fast_metal_kernel, disabled *bool, name string, inputsList []string, source string) {
	inputs, freeInputs, ok := cStringVector(inputsList)
	if !ok {
		*disabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"y"})
	if !ok {
		*disabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(source)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))

	*target = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(false),
	)
}

func initMoEWeightedSumMetalKernel() {
	initMoEWeightedSumKernel(&moeWeightedSumMetalKernel, &moeWeightedSumMetalDisabled, "moe_weighted_sum", []string{"expert", "scores"}, moeWeightedSumMetalKernelSource)
}

func initMoEWeightedSumAddMetalKernel() {
	initMoEWeightedSumKernel(&moeWeightedSumAddMetalKernel, &moeWeightedSumAddMetalDisabled, "moe_weighted_sum_add", []string{"expert", "scores", "shared"}, moeWeightedSumAddMetalKernelSource)
}

func initMoEWeightedSumAdd2MetalKernel() {
	initMoEWeightedSumKernel(&moeWeightedSumAdd2MetalKernel, &moeWeightedSumAdd2MetalDisabled, "moe_weighted_sum_add2", []string{"expert", "scores", "add_a", "add_b"}, moeWeightedSumAdd2MetalKernelSource)
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

func moeWeightedSumAddValidate(expert, scores, shared *Array, outDType DType) (B, L, topK, H int, ok bool) {
	B, L, topK, H, ok = moeWeightedSumValidate(expert, scores, outDType)
	if !ok || shared == nil || !moeWeightedSumSupportedDType(shared.DType()) {
		return 0, 0, 0, 0, false
	}
	sd := shared.Dims()
	if len(sd) != 3 || sd[0] != B || sd[1] != L || sd[2] != H {
		return 0, 0, 0, 0, false
	}
	return B, L, topK, H, true
}

func moeWeightedSumAdd2Validate(expert, scores, addA, addB *Array, outDType DType) (B, L, topK, H int, ok bool) {
	B, L, topK, H, ok = moeWeightedSumAddValidate(expert, scores, addA, outDType)
	if !ok || addB == nil || !moeWeightedSumSupportedDType(addB.DType()) {
		return 0, 0, 0, 0, false
	}
	dims := addB.Dims()
	if len(dims) != 3 || dims[0] != B || dims[1] != L || dims[2] != H {
		return 0, 0, 0, 0, false
	}
	return B, L, topK, H, true
}

func moeWeightedSumSupportedDType(dtype DType) bool {
	return dtype == DTypeFloat32 || dtype == DTypeFloat16 || dtype == DTypeBFloat16
}

func addMoEWeightedSumTemplateArgs(cfg C.mlx_fast_metal_kernel_config, B, L, topK, H int, outDType DType, scaleNum, scaleDen int) bool {
	for _, tpl := range []struct {
		name  string
		value int
	}{
		{name: "B", value: B},
		{name: "L", value: L},
		{name: "TopK", value: topK},
		{name: "H", value: H},
		{name: "ScaleNum", value: scaleNum},
		{name: "ScaleDen", value: scaleDen},
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

func moeWeightedSumScaleRatio(scale float32) (num, den int, ok bool) {
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return 0, 0, false
	}
	if scale == 1 {
		return 1, 1, true
	}

	const fixedDen = 65536
	n := math.Round(float64(scale) * fixedDen)
	if n < math.MinInt32 || n > math.MaxInt32 {
		return 0, 0, false
	}
	return int(n), fixedDen, true
}

// FastMoEWeightedSum computes addA + addB + scale *
// sum_k expert[B,L,k,H] * scores[B,L,k] and returns [B,L,H]. addA and addB are
// optional. It owns fused-variant dispatch and returns ok=false for unsupported
// backends or shapes so callers can use a backend-neutral fallback.
func FastMoEWeightedSum(expert, scores, addA, addB *Array, outDType DType, scale float32) (y *Array, ok bool) {
	switch {
	case addA == nil && addB == nil:
		return fastMoEWeightedSumScaled(expert, scores, outDType, scale)
	case addA != nil && addB == nil:
		return fastMoEWeightedSumScaledAdd(expert, scores, addA, outDType, scale)
	case addA != nil && addB != nil:
		return fastMoEWeightedSumScaledAdd2(expert, scores, addA, addB, outDType, scale)
	default:
		return nil, false
	}
}

func fastMoEWeightedSumScaled(expert, scores *Array, outDType DType, scale float32) (y *Array, ok bool) {
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
	scaleNum, scaleDen, ok := moeWeightedSumScaleRatio(scale)
	if !ok {
		return nil, false
	}

	moeWeightedSumMetalKernelOnce.Do(initMoEWeightedSumMetalKernel)
	if moeWeightedSumMetalDisabled {
		return nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addMoEWeightedSumTemplateArgs(cfg, B, L, topK, H, outDType, scaleNum, scaleDen) {
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

func fastMoEWeightedSumScaledAdd(expert, scores, shared *Array, outDType DType, scale float32) (y *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	if moeWeightedSumAddMetalDisabled {
		return nil, false
	}
	B, L, topK, H, ok := moeWeightedSumAddValidate(expert, scores, shared, outDType)
	if !ok {
		return nil, false
	}
	scaleNum, scaleDen, ok := moeWeightedSumScaleRatio(scale)
	if !ok {
		return nil, false
	}

	moeWeightedSumAddMetalKernelOnce.Do(initMoEWeightedSumAddMetalKernel)
	if moeWeightedSumAddMetalDisabled {
		return nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addMoEWeightedSumTemplateArgs(cfg, B, L, topK, H, outDType, scaleNum, scaleDen) {
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

	inputs := []C.mlx_array{expert.ctx, scores.ctx, shared.ctx}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, moeWeightedSumAddMetalKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		return nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 1 {
		return nil, false
	}

	y = New("MOE_WEIGHTED_SUM_ADD")
	C.mlx_vector_array_get(&y.ctx, outVec, 0)
	return y, true
}

func fastMoEWeightedSumScaledAdd2(expert, scores, addA, addB *Array, outDType DType, scale float32) (y *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	if moeWeightedSumAdd2MetalDisabled {
		return nil, false
	}
	B, L, topK, H, ok := moeWeightedSumAdd2Validate(expert, scores, addA, addB, outDType)
	if !ok {
		return nil, false
	}
	scaleNum, scaleDen, ok := moeWeightedSumScaleRatio(scale)
	if !ok {
		return nil, false
	}

	moeWeightedSumAdd2MetalKernelOnce.Do(initMoEWeightedSumAdd2MetalKernel)
	if moeWeightedSumAdd2MetalDisabled {
		return nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addMoEWeightedSumTemplateArgs(cfg, B, L, topK, H, outDType, scaleNum, scaleDen) {
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

	inputs := []C.mlx_array{expert.ctx, scores.ctx, addA.ctx, addB.ctx}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, moeWeightedSumAdd2MetalKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		return nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 1 {
		return nil, false
	}

	y = New("MOE_WEIGHTED_SUM_ADD2")
	C.mlx_vector_array_get(&y.ctx, outVec, 0)
	return y, true
}
