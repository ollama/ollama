package mlx

// #include <stdlib.h>
// #include "generated.h"
import "C"

import (
	"sync"
	"unsafe"
)

var (
	mambaConvSiLUMetalKernelOnce sync.Once
	mambaConvSiLUMetalKernel     C.mlx_fast_metal_kernel
	mambaConvSiLUMetalDisabled   bool
)

const mambaConvSiLUMetalKernelSource = `
auto elem = thread_position_in_grid.x;
auto total = B * T * C;
if (elem >= total) {
  return;
}

auto c = elem % C;
auto t = (elem / C) % T;
auto b = elem / (T * C);
auto in_base = (b * (T + K - 1) + t) * C + c;

float acc = 0.0f;
for (int i = 0; i < K; ++i) {
  acc += static_cast<float>(x[in_base + i * C]) * static_cast<float>(w[c * K + i]);
}
acc += static_cast<float>(bias[c]);

float sigmoid = 1.0f / (1.0f + exp(-acc));
y[elem] = acc * sigmoid;
`

func initMambaConvSiLUMetalKernel() {
	inputs, freeInputs, ok := cStringVector([]string{"x", "w", "bias"})
	if !ok {
		mambaConvSiLUMetalDisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"y"})
	if !ok {
		mambaConvSiLUMetalDisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("mamba_depthwise_conv_silu")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(mambaConvSiLUMetalKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))

	mambaConvSiLUMetalKernel = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(false),
	)
}

func mambaConvSiLUValidate(x, w, bias *Array, outLen int) (B, T, Cdim, K int, ok bool) {
	if x == nil || w == nil || bias == nil || outLen <= 0 {
		return 0, 0, 0, 0, false
	}
	if x.DType() != DTypeFloat32 || w.DType() != DTypeFloat32 || bias.DType() != DTypeFloat32 {
		return 0, 0, 0, 0, false
	}
	xd := x.Dims()
	wd := w.Dims()
	bd := bias.Dims()
	if len(xd) != 3 || len(wd) != 2 || len(bd) != 1 {
		return 0, 0, 0, 0, false
	}
	B, T, Cdim, K = xd[0], outLen, xd[2], wd[1]
	if B <= 0 || T <= 0 || Cdim <= 0 || K <= 0 || xd[1] < T+K-1 {
		return 0, 0, 0, 0, false
	}
	if wd[0] != Cdim || bd[0] != Cdim {
		return 0, 0, 0, 0, false
	}
	return B, T, Cdim, K, true
}

func addMambaConvSiLUTemplateArgs(cfg C.mlx_fast_metal_kernel_config, B, T, Cdim, K int) bool {
	for _, tpl := range []struct {
		name  string
		value int
	}{
		{name: "B", value: B},
		{name: "T", value: T},
		{name: "C", value: Cdim},
		{name: "K", value: K},
	} {
		cn := C.CString(tpl.name)
		rc := C.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, cn, C.int(tpl.value))
		C.free(unsafe.Pointer(cn))
		if rc != 0 {
			return false
		}
	}
	return true
}

// FastMambaDepthwiseConvSiLU computes SiLU(depthwise_causal_conv1d(x, w) + bias)
// for float32 x [B,T+K-1,C], w [C,K], and bias [C]. It returns ok=false for
// unsupported backends or shapes so callers can use a backend-neutral fallback.
func FastMambaDepthwiseConvSiLU(x, w, bias *Array, outLen int) (y *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, false
	}
	if mambaConvSiLUMetalDisabled {
		return nil, false
	}
	B, T, Cdim, K, ok := mambaConvSiLUValidate(x, w, bias, outLen)
	if !ok {
		return nil, false
	}

	mambaConvSiLUMetalKernelOnce.Do(initMambaConvSiLUMetalKernel)
	if mambaConvSiLUMetalDisabled {
		return nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addMambaConvSiLUTemplateArgs(cfg, B, T, Cdim, K) {
		return nil, false
	}

	yShape := []C.int{C.int(B), C.int(T), C.int(Cdim)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(yShape), C.size_t(len(yShape)), C.mlx_dtype(DTypeFloat32)) != 0 {
		return nil, false
	}

	total := B * T * Cdim
	gridX := ((total + 255) / 256) * 256
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, C.int(gridX), 1, 1) != 0 {
		return nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1) != 0 {
		return nil, false
	}

	inputs := []C.mlx_array{x.ctx, w.ctx, bias.ctx}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, mambaConvSiLUMetalKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		return nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 1 {
		return nil, false
	}

	y = New("MAMBA_DEPTHWISE_CONV_SILU")
	C.mlx_vector_array_get(&y.ctx, outVec, 0)
	return y, true
}
