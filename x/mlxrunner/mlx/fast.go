//go:build mlx

package mlx

// #include "generated.h"
import "C"

import (
	"unsafe"
)

func ScaledDotProductAttention(query, key, value, mask *Array, scale float32) *Array {
	if mask == nil {
		mask = New("")
	}

	sinks := New("")

	mode := "causal"
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))

	out := New("FAST_SDPA", query, key, value, mask, sinks)
	C.mlx_fast_scaled_dot_product_attention(&out.ctx, query.ctx, key.ctx, value.ctx, C.float(scale), cMode, mask.ctx, sinks.ctx, DefaultStream().ctx)
	return out
}

type LayerNorm struct {
	Weight Array `weight:"weight"`
	Bias   Array `weight:"bias"`
}

func (r *LayerNorm) Forward(x *Array, eps float32) *Array {
	out := New("FAST_LAYERNORM", x)
	C.mlx_fast_layer_norm(&out.ctx, x.ctx, r.Weight.ctx, r.Bias.ctx, C.float(eps), DefaultStream().ctx)
	return out
}

type RMSNorm struct {
	Weight Array `weight:"weight"`
}

func (r RMSNorm) Forward(x *Array, eps float32) *Array {
	out := New("FAST_RMSNORM", x)
	C.mlx_fast_rms_norm(&out.ctx, x.ctx, r.Weight.ctx, C.float(eps), DefaultStream().ctx)
	return out
}

type RoPE struct {
	Dims        int
	Traditional bool
	Base        float32 `json:"rope_theta"`
	Scale       float32
}

func (r RoPE) Forward(t *Array, offset int) *Array {
	freqs := New("")
	out := New("FAST_ROPE", t, freqs)
	C.mlx_fast_rope(
		&out.ctx,
		t.ctx,
		C.int(r.Dims),
		C._Bool(r.Traditional),
		C.mlx_optional_float{
			value:     C.float(r.Base),
			has_value: C._Bool(func() bool { return r.Base != 0 }()),
		},
		C.float(r.Scale),
		C.int(offset),
		freqs.ctx,
		DefaultStream().ctx,
	)
	return out
}
