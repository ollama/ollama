package mlx

// #include <stdlib.h>
// #include "generated.h"
import "C"

// These fused sigmoid/top-k router kernels are currently only used by Laguna.
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
	"sync"
	"unsafe"
)

var (
	sigmoidTopKRouterMetalKernelOnce sync.Once
	sigmoidTopKRouterMetalKernel     C.mlx_fast_metal_kernel
	sigmoidTopKRouterMetalDisabled   bool

	scaledSigmoidTopKRouterMetalKernelOnce sync.Once
	scaledSigmoidTopKRouterMetalKernel     C.mlx_fast_metal_kernel
	scaledSigmoidTopKRouterMetalDisabled   bool
)

const sigmoidTopKRouterMetalKernelSource = `
constexpr int Threads = 128;
constexpr int SIMDWidth = 32;
constexpr int Groups = Threads / SIMDWidth;

auto token = threadgroup_position_in_grid.x;
if (token >= T) {
  return;
}
auto tid = thread_index_in_threadgroup;
auto lane = tid % SIMDWidth;
auto group = tid / SIMDWidth;

threadgroup float adjusted_values[E];
threadgroup float prob_values[E];
threadgroup float group_adjusted[Groups];
threadgroup float group_prob[Groups];
threadgroup int group_index[Groups];
threadgroup float top_prob[TopK];
threadgroup int top_index[TopK];

auto row = gates + size_t(token) * E;
for (int e = int(tid); e < E; e += Threads) {
  float g = static_cast<float>(row[e]);
  float p = 1.0f / (1.0f + exp(-g));
  prob_values[e] = p;
  adjusted_values[e] = p + static_cast<float>(bias[e]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int k = 0; k < TopK; ++k) {
  float best_adjusted = -INFINITY;
  float best_prob = 0.0f;
  int best_index = 0;
  for (int e = int(tid); e < E; e += Threads) {
    float adjusted = adjusted_values[e];
    if (adjusted > best_adjusted) {
      best_adjusted = adjusted;
      best_prob = prob_values[e];
      best_index = e;
    }
  }

  for (ushort offset = SIMDWidth / 2; offset > 0; offset >>= 1) {
    float other_adjusted = simd_shuffle_down(best_adjusted, offset);
    float other_prob = simd_shuffle_down(best_prob, offset);
    int other_index = simd_shuffle_down(best_index, offset);
    if (lane + offset < SIMDWidth && other_adjusted > best_adjusted) {
      best_adjusted = other_adjusted;
      best_prob = other_prob;
      best_index = other_index;
    }
  }

  if (lane == 0) {
    group_adjusted[group] = best_adjusted;
    group_prob[group] = best_prob;
    group_index[group] = best_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (group == 0) {
    best_adjusted = (lane < Groups) ? group_adjusted[lane] : -INFINITY;
    best_prob = (lane < Groups) ? group_prob[lane] : 0.0f;
    best_index = (lane < Groups) ? group_index[lane] : 0;

    for (ushort offset = SIMDWidth / 2; offset > 0; offset >>= 1) {
      float other_adjusted = simd_shuffle_down(best_adjusted, offset);
      float other_prob = simd_shuffle_down(best_prob, offset);
      int other_index = simd_shuffle_down(best_index, offset);
      if (lane + offset < SIMDWidth && other_adjusted > best_adjusted) {
        best_adjusted = other_adjusted;
        best_prob = other_prob;
        best_index = other_index;
      }
    }

    if (lane == 0) {
      top_prob[k] = best_prob;
      top_index[k] = best_index;
      adjusted_values[best_index] = -INFINITY;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
  float denom = 1.0f;
  if (Normalize != 0) {
    denom = 0.0f;
    for (int k = 0; k < TopK; ++k) {
      denom += top_prob[k];
    }
  }

  auto score_row = scores + size_t(token) * TopK;
  auto index_row = indices + size_t(token) * TopK;
  for (int k = 0; k < TopK; ++k) {
    score_row[k] = top_prob[k] / denom;
    index_row[k] = top_index[k];
  }
}
`

const scaledSigmoidTopKRouterMetalKernelSource = `
constexpr int Threads = 128;
constexpr int SIMDWidth = 32;
constexpr int Groups = Threads / SIMDWidth;

auto token = threadgroup_position_in_grid.x;
if (token >= T) {
  return;
}
auto tid = thread_index_in_threadgroup;
auto lane = tid % SIMDWidth;
auto group = tid / SIMDWidth;

threadgroup float adjusted_values[E];
threadgroup float prob_values[E];
threadgroup float group_adjusted[Groups];
threadgroup float group_prob[Groups];
threadgroup int group_index[Groups];
threadgroup float top_prob[TopK];
threadgroup int top_index[TopK];

auto row = gates + size_t(token) * E;
for (int e = int(tid); e < E; e += Threads) {
  float g = static_cast<float>(row[e]);
  float p = 1.0f / (1.0f + exp(-g));
  prob_values[e] = p;
  adjusted_values[e] = p + static_cast<float>(bias[e]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int k = 0; k < TopK; ++k) {
  float best_adjusted = -INFINITY;
  float best_prob = 0.0f;
  int best_index = 0;
  for (int e = int(tid); e < E; e += Threads) {
    float adjusted = adjusted_values[e];
    if (adjusted > best_adjusted) {
      best_adjusted = adjusted;
      best_prob = prob_values[e];
      best_index = e;
    }
  }

  for (ushort offset = SIMDWidth / 2; offset > 0; offset >>= 1) {
    float other_adjusted = simd_shuffle_down(best_adjusted, offset);
    float other_prob = simd_shuffle_down(best_prob, offset);
    int other_index = simd_shuffle_down(best_index, offset);
    if (lane + offset < SIMDWidth && other_adjusted > best_adjusted) {
      best_adjusted = other_adjusted;
      best_prob = other_prob;
      best_index = other_index;
    }
  }

  if (lane == 0) {
    group_adjusted[group] = best_adjusted;
    group_prob[group] = best_prob;
    group_index[group] = best_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (group == 0) {
    best_adjusted = (lane < Groups) ? group_adjusted[lane] : -INFINITY;
    best_prob = (lane < Groups) ? group_prob[lane] : 0.0f;
    best_index = (lane < Groups) ? group_index[lane] : 0;

    for (ushort offset = SIMDWidth / 2; offset > 0; offset >>= 1) {
      float other_adjusted = simd_shuffle_down(best_adjusted, offset);
      float other_prob = simd_shuffle_down(best_prob, offset);
      int other_index = simd_shuffle_down(best_index, offset);
      if (lane + offset < SIMDWidth && other_adjusted > best_adjusted) {
        best_adjusted = other_adjusted;
        best_prob = other_prob;
        best_index = other_index;
      }
    }

    if (lane == 0) {
      top_prob[k] = best_prob;
      top_index[k] = best_index;
      adjusted_values[best_index] = -INFINITY;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0) {
  float denom = 1.0f;
  if (Normalize != 0) {
    denom = 0.0f;
    for (int k = 0; k < TopK; ++k) {
      denom += top_prob[k];
    }
  }

  auto score_row = scores + size_t(token) * TopK;
  auto index_row = indices + size_t(token) * TopK;
  for (int k = 0; k < TopK; ++k) {
    int expert = top_index[k];
    float scale = static_cast<float>(scale_a[expert]) * static_cast<float>(scale_b[expert]);
    score_row[k] = (top_prob[k] / denom) * scale;
    index_row[k] = expert;
  }
}
`

func initSigmoidTopKRouterMetalKernel() {
	inputs, freeInputs, ok := cStringVector([]string{"gates", "bias"})
	if !ok {
		sigmoidTopKRouterMetalDisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"scores", "indices"})
	if !ok {
		sigmoidTopKRouterMetalDisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("sigmoid_topk_router")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(sigmoidTopKRouterMetalKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))

	sigmoidTopKRouterMetalKernel = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(false),
	)
}

func initScaledSigmoidTopKRouterMetalKernel() {
	inputs, freeInputs, ok := cStringVector([]string{"gates", "bias", "scale_a", "scale_b"})
	if !ok {
		scaledSigmoidTopKRouterMetalDisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"scores", "indices"})
	if !ok {
		scaledSigmoidTopKRouterMetalDisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("scaled_sigmoid_topk_router")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(scaledSigmoidTopKRouterMetalKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))

	scaledSigmoidTopKRouterMetalKernel = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(false),
	)
}

func sigmoidTopKRouterValidate(gates, bias *Array, topK int) (tokens, experts int, ok bool) {
	if gates == nil || bias == nil || topK <= 0 {
		return 0, 0, false
	}
	if !sigmoidTopKRouterSupportedDType(gates.DType()) || !sigmoidTopKRouterSupportedDType(bias.DType()) {
		return 0, 0, false
	}
	gd := gates.Dims()
	bd := bias.Dims()
	if len(gd) != 2 || len(bd) != 1 {
		return 0, 0, false
	}
	tokens, experts = gd[0], gd[1]
	if tokens <= 0 || experts <= 0 || experts > 1024 || topK > experts || bd[0] != experts {
		return 0, 0, false
	}
	return tokens, experts, true
}

func scaledSigmoidTopKRouterValidate(gates, bias, scaleA, scaleB *Array, topK int) (tokens, experts int, ok bool) {
	tokens, experts, ok = sigmoidTopKRouterValidate(gates, bias, topK)
	if !ok || scaleA == nil || scaleB == nil {
		return 0, 0, false
	}
	if !sigmoidTopKRouterSupportedDType(scaleA.DType()) || !sigmoidTopKRouterSupportedDType(scaleB.DType()) {
		return 0, 0, false
	}
	ad := scaleA.Dims()
	bd := scaleB.Dims()
	if len(ad) != 1 || len(bd) != 1 || ad[0] != experts || bd[0] != experts {
		return 0, 0, false
	}
	return tokens, experts, true
}

func sigmoidTopKRouterSupportedDType(dtype DType) bool {
	return dtype == DTypeFloat32 || dtype == DTypeFloat16 || dtype == DTypeBFloat16
}

func addSigmoidTopKRouterTemplateArgs(cfg C.mlx_fast_metal_kernel_config, tokens, experts, topK int, normalize bool) bool {
	for _, tpl := range []struct {
		name  string
		value int
	}{
		{name: "T", value: tokens},
		{name: "E", value: experts},
		{name: "TopK", value: topK},
	} {
		cn := C.CString(tpl.name)
		rc := C.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, cn, C.int(tpl.value))
		C.free(unsafe.Pointer(cn))
		if rc != 0 {
			return false
		}
	}

	normalizeValue := 0
	if normalize {
		normalizeValue = 1
	}
	cn := C.CString("Normalize")
	rc := C.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, cn, C.int(normalizeValue))
	C.free(unsafe.Pointer(cn))
	return rc == 0
}

// FastSigmoidTopKRouter selects top-k experts by sigmoid(gates)+bias and
// returns the corresponding unbiased sigmoid scores plus int32 expert indices.
// It returns ok=false for unsupported backends or shapes so callers can use a
// backend-neutral fallback.
func FastSigmoidTopKRouter(gates, bias *Array, topK int, normalize bool) (scores, indices *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, nil, false
	}
	if sigmoidTopKRouterMetalDisabled {
		return nil, nil, false
	}
	tokens, experts, ok := sigmoidTopKRouterValidate(gates, bias, topK)
	if !ok {
		return nil, nil, false
	}

	sigmoidTopKRouterMetalKernelOnce.Do(initSigmoidTopKRouterMetalKernel)
	if sigmoidTopKRouterMetalDisabled {
		return nil, nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addSigmoidTopKRouterTemplateArgs(cfg, tokens, experts, topK, normalize) {
		return nil, nil, false
	}

	outShape := []C.int{C.int(tokens), C.int(topK)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(outShape), C.size_t(len(outShape)), C.mlx_dtype(DTypeFloat32)) != 0 {
		return nil, nil, false
	}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(outShape), C.size_t(len(outShape)), C.mlx_dtype(DTypeInt32)) != 0 {
		return nil, nil, false
	}

	gridX := tokens * 128
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, C.int(gridX), 1, 1) != 0 {
		return nil, nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1) != 0 {
		return nil, nil, false
	}

	inputs := []C.mlx_array{gates.ctx, bias.ctx}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, sigmoidTopKRouterMetalKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		return nil, nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 2 {
		return nil, nil, false
	}

	scores = New("SIGMOID_TOPK_ROUTER_SCORES")
	C.mlx_vector_array_get(&scores.ctx, outVec, 0)
	indices = New("SIGMOID_TOPK_ROUTER_INDICES")
	C.mlx_vector_array_get(&indices.ctx, outVec, 1)
	return scores, indices, true
}

// FastSigmoidTopKRouterScaled is like FastSigmoidTopKRouter, but also folds
// scaleA[index] * scaleB[index] into each returned score after optional top-k
// normalization.
func FastSigmoidTopKRouterScaled(gates, bias, scaleA, scaleB *Array, topK int, normalize bool) (scores, indices *Array, ok bool) {
	if !MetalIsAvailable() {
		return nil, nil, false
	}
	if scaledSigmoidTopKRouterMetalDisabled {
		return nil, nil, false
	}
	tokens, experts, ok := scaledSigmoidTopKRouterValidate(gates, bias, scaleA, scaleB, topK)
	if !ok {
		return nil, nil, false
	}

	scaledSigmoidTopKRouterMetalKernelOnce.Do(initScaledSigmoidTopKRouterMetalKernel)
	if scaledSigmoidTopKRouterMetalDisabled {
		return nil, nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	if !addSigmoidTopKRouterTemplateArgs(cfg, tokens, experts, topK, normalize) {
		return nil, nil, false
	}

	outShape := []C.int{C.int(tokens), C.int(topK)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(outShape), C.size_t(len(outShape)), C.mlx_dtype(DTypeFloat32)) != 0 {
		return nil, nil, false
	}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(outShape), C.size_t(len(outShape)), C.mlx_dtype(DTypeInt32)) != 0 {
		return nil, nil, false
	}

	gridX := tokens * 128
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, C.int(gridX), 1, 1) != 0 {
		return nil, nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1) != 0 {
		return nil, nil, false
	}

	inputs := []C.mlx_array{gates.ctx, bias.ctx, scaleA.ctx, scaleB.ctx}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, scaledSigmoidTopKRouterMetalKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		return nil, nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 2 {
		return nil, nil, false
	}

	scores = New("SCALED_SIGMOID_TOPK_ROUTER_SCORES")
	C.mlx_vector_array_get(&scores.ctx, outVec, 0)
	indices = New("SCALED_SIGMOID_TOPK_ROUTER_INDICES")
	C.mlx_vector_array_get(&indices.ctx, outVec, 1)
	return scores, indices, true
}
