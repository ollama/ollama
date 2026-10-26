package mlx

// #include "generated.h"
import "C"

import (
	"unsafe"
)

// FastSDPANoWindow disables sliding-window restriction for SDPA - each
// query attends to the full key context.
const FastSDPANoWindow = -1

func FastScaledDotProductAttention(q, k, v *Array, scale float32, windowSize int, mode string, mask *Array) *Array {
	return FastScaledDotProductAttentionWithSinks(q, k, v, scale, windowSize, mode, mask, nil)
}

// FastScaledDotProductAttentionWithSinks adds one learned softmax denominator
// term per query head. A sink has no corresponding value vector.
//
// windowSize restricts each query to the most recent windowSize keys
// (sliding window attention); pass FastSDPANoWindow for full context.
func FastScaledDotProductAttentionWithSinks(q, k, v *Array, scale float32, windowSize int, mode string, mask, sinks *Array) *Array {
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))

	var maskCtx C.mlx_array
	if mask != nil {
		maskCtx = mask.ctx
	} else {
		empty := New("")
		maskCtx = empty.ctx
	}
	var sinksCtx C.mlx_array
	if sinks != nil {
		sinksCtx = sinks.ctx
	} else {
		empty := New("")
		sinksCtx = empty.ctx
	}

	out := New("FAST_SDPA")
	mlxCheck(C.mlx_fast_scaled_dot_product_attention(&out.ctx, q.ctx, k.ctx, v.ctx, C.float(scale), C.int(windowSize), cMode, maskCtx, sinksCtx, C.bool(false), DefaultStream().ctx))
	return out
}

type LayerNorm struct {
	Weight *Array `weight:"weight"`
	Bias   *Array `weight:"bias"`
}

// fastGatedDeltaUpdate applies MLX's gated-delta recurrence and returns its
// per-token outputs and final float32 state. state and mask may be nil.
func fastGatedDeltaUpdate(q, k, v, gates, beta, state, mask *Array) (y, nextState *Array) {
	outVec := mlxCheck(C.mlx_vector_array_new())
	defer freeVectorArray(outVec)

	var stateCtx, maskCtx C.mlx_array
	if state != nil {
		stateCtx = state.ctx
	}
	if mask != nil {
		maskCtx = mask.ctx
	}

	mlxCheck(C.mlx_fast_gated_delta_update(
		&outVec,
		q.ctx,
		k.ctx,
		v.ctx,
		gates.ctx,
		beta.ctx,
		stateCtx,
		maskCtx,
		DefaultStream().ctx))

	y = New("FAST_GATED_DELTA_Y")
	nextState = New("FAST_GATED_DELTA_STATE")
	mlxCheck(C.mlx_vector_array_get(&y.ctx, outVec, C.size_t(0)))
	mlxCheck(C.mlx_vector_array_get(&nextState.ctx, outVec, C.size_t(1)))
	return y, nextState
}

func (r *LayerNorm) Forward(x *Array, eps float32) *Array {
	out := New("FAST_LAYERNORM")
	mlxCheck(C.mlx_fast_layer_norm(&out.ctx, x.ctx, r.Weight.ctx, r.Bias.ctx, C.float(eps), DefaultStream().ctx))
	return out
}

type RMSNorm struct {
	Weight *Array `weight:"weight"`
}

func (r *RMSNorm) Forward(x *Array, eps float32) *Array {
	out := New("FAST_RMSNORM")
	mlxCheck(C.mlx_fast_rms_norm(&out.ctx, x.ctx, r.Weight.ctx, C.float(eps), DefaultStream().ctx))
	return out
}
