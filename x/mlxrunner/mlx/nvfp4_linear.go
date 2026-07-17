package mlx

// #include <stdlib.h>
// #include "generated.h"
import "C"

// These fast dense quantized-linear kernels are currently only used by Laguna.
// As part of the retained Laguna fast path, the Apple M5 Max p2048/g128 run was
// about 24% faster on prompt and 7% faster on generate than the GGML target.
// No other model has shown a kept win from this path yet.
// TODO: consider moving this implementation under x/models/laguna once the mlx
// package has a refined custom-kernel API that can safely support model-local
// kernels without exposing low-level MLX internals.
// TODO(cuda): add CUDA equivalents for these custom Metal kernels before
// enabling the same fast path on CUDA-backed MLX; non-Metal backends must keep
// using the generic fallback.

import (
	"strings"
	"sync"
	"unsafe"
)

var (
	nvfp4LinearMetalKernelOnce sync.Once
	nvfp4LinearMetalKernel     C.mlx_fast_metal_kernel
	nvfp4LinearMetalDisabled   bool

	nvfp4LinearScaledMetalKernelOnce sync.Once
	nvfp4LinearScaledMetalKernel     C.mlx_fast_metal_kernel
	nvfp4LinearScaledMetalDisabled   bool

	mxfp8LinearMetalKernelOnce sync.Once
	mxfp8LinearMetalKernel     C.mlx_fast_metal_kernel
	mxfp8LinearMetalDisabled   bool
)

const nvfp4LinearMetalKernelSource = `
constexpr int NR0 = 64;
constexpr int NR1 = 32;
constexpr int NK = 32;
constexpr int NL0 = NK / 16;
constexpr int NL1 = NK / 8;

const int r1 = int(threadgroup_position_in_grid.x) * NR1;
const int r0 = int(threadgroup_position_in_grid.y) * NR0;
if (r1 >= T || r0 >= N) {
  return;
}

const short tiitg = short(thread_index_in_threadgroup);
const short tiisg = short(thread_index_in_simdgroup);
const short sgitg = short(simdgroup_index_in_threadgroup);

const short nr0 = short((N - r0 < NR0) ? (N - r0) : NR0);
const short nr1 = short((T - r1 < NR1) ? (T - r1) : NR1);

const short lr0 = short(((tiitg / NL0) < nr0) ? (tiitg / NL0) : (nr0 - 1));
const short lr1 = short(((tiitg / NL1) < nr1) ? (tiitg / NL1) : (nr1 - 1));
const short il0 = short(tiitg % NL0);
const short iy = short(8 * (tiitg % NL1));

threadgroup half sa[NR0 * NK];
threadgroup half sb[NR1 * NK];
threadgroup float temp[NR1 * NR0];

const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w);
const size_t k_bytes = K / 2;
const size_t k_groups = K / 16;

auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK));

mpp::tensor_ops::matmul2d<
  mpp::tensor_ops::matmul2d_descriptor(
    NR1, NR0, NK, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
  execution_simdgroups<4>> mm;

auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

MOE_MAPPED_FOR_UNROLL (uint16_t i = 0; i < cT.get_capacity(); ++i) {
  if (cT.is_valid_element(i)) {
    cT[i] = 0.0f;
  }
}

for (int loop_k = 0; loop_k < K; loop_k += NK) {
  const int col = r0 + lr0;
  const int k0 = loop_k + 16 * il0;
  const size_t w_base = size_t(col) * k_bytes + size_t(k0 / 2);
  const size_t s_base = size_t(col) * k_groups + size_t(k0 / 16);
  const half scale = moe_mapped_nvfp4_scale(scales[s_base]);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 16; i++) {
    const short sx = short(2 * il0 + i / 8);
    const short sy = short((tiitg / NL0) / 8);
    const short lx = short(i % 8);
    const short ly = short((tiitg / NL0) % 8);

    const uint8_t packed = wb[w_base + size_t(i / 2)];
    const uint nibble = ((i & 1) == 0) ? uint(packed & 0x0f) : uint((packed >> 4) & 0x0f);
    *(sa + NK * (8 * sy + ly) + 8 * sx + lx) = scale * moe_mapped_nvfp4_value(nibble);
  }

  const short sx = short(tiitg % NL1);
  const short sy = short((tiitg / NL1) / 8);
  const short ly = short((tiitg / NL1) % 8);
  const size_t x_base = size_t(r1 + lr1) * K + size_t(loop_k + iy);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
    *(sb + NK * (8 * sy + ly) + 8 * sx + i) = half(x[x_base + i]);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto sA = tA.slice(0, 0);
  auto sB = tB.slice(0, 0);
  mm.run(sB, sA, cT);
}

threadgroup_barrier(mem_flags::mem_threadgroup);

auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(temp, dextents<int32_t, 2>(NR0, NR1));
cT.store(tC);

threadgroup_barrier(mem_flags::mem_threadgroup);

for (short j = sgitg; j < nr1; j += 4) {
  device OutT* dst = y + size_t(r1 + j) * N + size_t(r0);
  threadgroup float* src = temp + int(j) * NR0;
  int i = tiisg;
  for (; i < nr0; i += 32) {
    dst[i] = static_cast<OutT>(src[i]);
  }
}
`

var nvfp4LinearScaledMetalKernelSource = strings.NewReplacer(
	"for (short j = sgitg; j < nr1; j += 4) {",
	"const float global = static_cast<float>(global_scale);\n\nfor (short j = sgitg; j < nr1; j += 4) {",
	"    dst[i] = static_cast<OutT>(src[i]);",
	"    OutT unscaled = static_cast<OutT>(src[i]);\n    dst[i] = static_cast<OutT>(static_cast<float>(unscaled) * global);",
).Replace(nvfp4LinearMetalKernelSource)

const mxfp8LinearMetalKernelSource = `
constexpr int NR0 = 64;
constexpr int NR1 = 32;
constexpr int NK = 32;
constexpr int NL0 = NK / 16;
constexpr int NL1 = NK / 8;

const int r1 = int(threadgroup_position_in_grid.x) * NR1;
const int r0 = int(threadgroup_position_in_grid.y) * NR0;
if (r1 >= T || r0 >= N) {
  return;
}

const short tiitg = short(thread_index_in_threadgroup);
const short tiisg = short(thread_index_in_simdgroup);
const short sgitg = short(simdgroup_index_in_threadgroup);

const short nr0 = short((N - r0 < NR0) ? (N - r0) : NR0);
const short nr1 = short((T - r1 < NR1) ? (T - r1) : NR1);

const short lr0 = short(((tiitg / NL0) < nr0) ? (tiitg / NL0) : (nr0 - 1));
const short lr1 = short(((tiitg / NL1) < nr1) ? (tiitg / NL1) : (nr1 - 1));
const short il0 = short(tiitg % NL0);
const short iy = short(8 * (tiitg % NL1));

threadgroup half sa[NR0 * NK];
threadgroup half sb[NR1 * NK];
threadgroup float temp[NR1 * NR0];

const device uint8_t* wb = reinterpret_cast<const device uint8_t*>(w);
const size_t k_bytes = K;
const size_t k_groups = K / 32;

auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK));

mpp::tensor_ops::matmul2d<
  mpp::tensor_ops::matmul2d_descriptor(
    NR1, NR0, NK, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
  execution_simdgroups<4>> mm;

auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

MOE_MAPPED_FOR_UNROLL (uint16_t i = 0; i < cT.get_capacity(); ++i) {
  if (cT.is_valid_element(i)) {
    cT[i] = 0.0f;
  }
}

for (int loop_k = 0; loop_k < K; loop_k += NK) {
  const int col = r0 + lr0;
  const int k0 = loop_k + 16 * il0;
  const size_t w_base = size_t(col) * k_bytes + size_t(k0);
  const size_t s_base = size_t(col) * k_groups + size_t(loop_k / 32);
  const float scale = moe_mapped_mxfp8_scale(scales[s_base]);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 16; i++) {
    const short sx = short(2 * il0 + i / 8);
    const short sy = short((tiitg / NL0) / 8);
    const short lx = short(i % 8);
    const short ly = short((tiitg / NL0) % 8);

    const uint8_t packed = wb[w_base + size_t(i)];
    *(sa + NK * (8 * sy + ly) + 8 * sx + lx) = half(scale * float(moe_mapped_mxfp8_value(packed)));
  }

  const short sx = short(tiitg % NL1);
  const short sy = short((tiitg / NL1) / 8);
  const short ly = short((tiitg / NL1) % 8);
  const size_t x_base = size_t(r1 + lr1) * K + size_t(loop_k + iy);

  MOE_MAPPED_FOR_UNROLL (short i = 0; i < 8; i++) {
    *(sb + NK * (8 * sy + ly) + 8 * sx + i) = half(x[x_base + i]);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto sA = tA.slice(0, 0);
  auto sB = tB.slice(0, 0);
  mm.run(sB, sA, cT);
}

threadgroup_barrier(mem_flags::mem_threadgroup);

auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(temp, dextents<int32_t, 2>(NR0, NR1));
cT.store(tC);

threadgroup_barrier(mem_flags::mem_threadgroup);

for (short j = sgitg; j < nr1; j += 4) {
  device OutT* dst = y + size_t(r1 + j) * N + size_t(r0);
  threadgroup float* src = temp + int(j) * NR0;
  int i = tiisg;
  for (; i < nr0; i += 32) {
    dst[i] = static_cast<OutT>(src[i]);
  }
}
`

func initNVFP4LinearMetalKernel() {
	initNVFP4LinearKernel(&nvfp4LinearMetalKernel, &nvfp4LinearMetalDisabled, "nvfp4_linear", []string{"x", "w", "scales"}, nvfp4LinearMetalKernelSource)
}

func initNVFP4LinearScaledMetalKernel() {
	initNVFP4LinearKernel(&nvfp4LinearScaledMetalKernel, &nvfp4LinearScaledMetalDisabled, "nvfp4_linear_scaled", []string{"x", "w", "scales", "global_scale"}, nvfp4LinearScaledMetalKernelSource)
}

func initNVFP4LinearKernel(target *C.mlx_fast_metal_kernel, disabled *bool, name string, inputsList []string, source string) {
	if !metalTensorOpsAvailable() {
		*disabled = true
		return
	}

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
	cHeader := C.CString(moeMappedMetalKernelHeader)
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

func initMXFP8LinearMetalKernel() {
	if !metalTensorOpsAvailable() {
		mxfp8LinearMetalDisabled = true
		return
	}

	inputs, freeInputs, ok := cStringVector([]string{"x", "w", "scales"})
	if !ok {
		mxfp8LinearMetalDisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"y"})
	if !ok {
		mxfp8LinearMetalDisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("mxfp8_linear")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(mxfp8LinearMetalKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString(moeMappedMetalKernelHeader)
	defer C.free(unsafe.Pointer(cHeader))

	mxfp8LinearMetalKernel = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(false),
	)
}

func addNVFP4LinearTemplateArgs(cfg C.mlx_fast_metal_kernel_config, tokens, N, K int, outDType DType) bool {
	for _, tpl := range []struct {
		name  string
		value int
	}{
		{name: "T", value: tokens},
		{name: "N", value: N},
		{name: "K", value: K},
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

func fpLinearValidate(x, w, scales *Array, weightDiv, scaleDiv int) (tokens, N, K int, outShape []C.int, ok bool) {
	if x == nil || w == nil || scales == nil {
		return 0, 0, 0, nil, false
	}
	if x.DType() != DTypeFloat32 && x.DType() != DTypeFloat16 && x.DType() != DTypeBFloat16 {
		return 0, 0, 0, nil, false
	}
	if w.DType() != DTypeUint32 || scales.DType() != DTypeUint8 {
		return 0, 0, 0, nil, false
	}
	xd := x.Dims()
	wd := w.Dims()
	sd := scales.Dims()
	switch len(xd) {
	case 2:
		tokens, K = xd[0], xd[1]
		outShape = []C.int{C.int(tokens), 0}
	case 3:
		if xd[0] != 1 {
			return 0, 0, 0, nil, false
		}
		tokens, K = xd[1], xd[2]
		outShape = []C.int{C.int(xd[0]), C.int(tokens), 0}
	default:
		return 0, 0, 0, nil, false
	}
	if len(wd) != 2 || len(sd) != 2 {
		return 0, 0, 0, nil, false
	}
	N = wd[0]
	if tokens <= 0 || N <= 0 || K < 512 || K%32 != 0 {
		return 0, 0, 0, nil, false
	}
	if wd[1] != K/weightDiv || sd[0] != N || sd[1] != K/scaleDiv {
		return 0, 0, 0, nil, false
	}
	outShape[len(outShape)-1] = C.int(N)
	return tokens, N, K, outShape, true
}

func nvfp4LinearValidate(x, w, scales *Array) (tokens, N, K int, outShape []C.int, ok bool) {
	return fpLinearValidate(x, w, scales, 8, 16)
}

func nvfp4LinearGlobalScaleValidate(globalScale *Array) bool {
	if globalScale == nil || !globalScale.Valid() {
		return false
	}
	switch globalScale.DType() {
	case DTypeFloat32, DTypeFloat16, DTypeBFloat16:
	default:
		return false
	}
	dims := globalScale.Dims()
	return len(dims) == 0
}

func mxfp8LinearValidate(x, w, scales *Array) (tokens, N, K int, outShape []C.int, ok bool) {
	return fpLinearValidate(x, w, scales, 4, 32)
}

func applyFPLinearKernel(kernel C.mlx_fast_metal_kernel, name string, x, w, scales *Array, tokens, N, K int, yShape []C.int) (y *Array, ok bool) {
	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addNVFP4LinearTemplateArgs(cfg, tokens, N, K, x.DType()) {
		return nil, false
	}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(yShape), C.size_t(len(yShape)), C.mlx_dtype(x.DType())) != 0 {
		return nil, false
	}
	gridX := ((tokens + 31) / 32) * 128
	gridY := (N + 63) / 64
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, C.int(gridX), C.int(gridY), 1) != 0 {
		return nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1) != 0 {
		return nil, false
	}

	inputs := []C.mlx_array{x.ctx, w.ctx, scales.ctx}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, kernel, inVec, cfg, DefaultStream().ctx) != 0 {
		return nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 1 {
		return nil, false
	}

	y = New(name)
	C.mlx_vector_array_get(&y.ctx, outVec, 0)
	return y, true
}

func fastNVFP4Linear(x, w, scales *Array) (y *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	if nvfp4LinearMetalDisabled {
		return nil, false
	}
	tokens, N, K, yShape, ok := nvfp4LinearValidate(x, w, scales)
	if !ok {
		return nil, false
	}

	nvfp4LinearMetalKernelOnce.Do(initNVFP4LinearMetalKernel)
	if nvfp4LinearMetalDisabled {
		return nil, false
	}
	return applyFPLinearKernel(nvfp4LinearMetalKernel, "NVFP4_LINEAR", x, w, scales, tokens, N, K, yShape)
}

// fastNVFP4LinearScaled is equivalent to fastNVFP4Linear followed by a scalar
// global-scale multiply cast back to the input dtype.
func fastNVFP4LinearScaled(x, w, scales, globalScale *Array) (y *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	if nvfp4LinearScaledMetalDisabled {
		return nil, false
	}
	tokens, N, K, yShape, ok := nvfp4LinearValidate(x, w, scales)
	if !ok || !nvfp4LinearGlobalScaleValidate(globalScale) {
		return nil, false
	}

	nvfp4LinearScaledMetalKernelOnce.Do(initNVFP4LinearScaledMetalKernel)
	if nvfp4LinearScaledMetalDisabled {
		return nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addNVFP4LinearTemplateArgs(cfg, tokens, N, K, x.DType()) {
		return nil, false
	}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(yShape), C.size_t(len(yShape)), C.mlx_dtype(x.DType())) != 0 {
		return nil, false
	}
	gridX := ((tokens + 31) / 32) * 128
	gridY := (N + 63) / 64
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, C.int(gridX), C.int(gridY), 1) != 0 {
		return nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1) != 0 {
		return nil, false
	}

	inputs := []C.mlx_array{x.ctx, w.ctx, scales.ctx, globalScale.ctx}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, nvfp4LinearScaledMetalKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		return nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 1 {
		return nil, false
	}

	y = New("NVFP4_LINEAR_SCALED")
	C.mlx_vector_array_get(&y.ctx, outVec, 0)
	return y, true
}

func fastMXFP8Linear(x, w, scales *Array) (y *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	if mxfp8LinearMetalDisabled {
		return nil, false
	}
	tokens, N, K, yShape, ok := mxfp8LinearValidate(x, w, scales)
	if !ok {
		return nil, false
	}

	mxfp8LinearMetalKernelOnce.Do(initMXFP8LinearMetalKernel)
	if mxfp8LinearMetalDisabled {
		return nil, false
	}
	return applyFPLinearKernel(mxfp8LinearMetalKernel, "MXFP8_LINEAR", x, w, scales, tokens, N, K, yShape)
}

// FastQuantizedLinear computes x @ w.T for supported block-quantized weights.
// It owns dtype dispatch so model code only needs to provide the quantization
// metadata it already carries. Unsupported modes, shapes, or global-scale forms
// return ok=false and should use the generic QuantizedLinear fallback.
func FastQuantizedLinear(x, w, scales, globalScale *Array, groupSize, bits int, mode string) (y *Array, ok bool) {
	switch {
	case mode == "nvfp4" && bits == 4 && groupSize == 16:
		if globalScale != nil {
			return fastNVFP4LinearScaled(x, w, scales, globalScale)
		}
		return fastNVFP4Linear(x, w, scales)
	case mode == "mxfp8" && bits == 8 && groupSize == 32:
		if globalScale != nil {
			return nil, false
		}
		return fastMXFP8Linear(x, w, scales)
	default:
		return nil, false
	}
}
