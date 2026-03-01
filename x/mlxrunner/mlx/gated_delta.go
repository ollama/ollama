//go:build mlx

package mlx

// #include <stdlib.h>
// #include "generated.h"
import "C"

import (
	"sync"
	"unsafe"
)

var (
	gatedDeltaMetalKernelOnce sync.Once
	gatedDeltaMetalKernel     C.mlx_fast_metal_kernel
	gatedDeltaMetalDisabled   bool
)

const gatedDeltaMetalKernelSource = `
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;

// q, k: [B, T, Hk, Dk]
auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

// v, y: [B, T, Hv, Dv]
auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
y += b_idx * T * Hv * Dv + hv_idx * Dv;

auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;

// state_in, state_out: [B, Hv, Dv, Dk]
auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}

// g: [B, T, Hv]
auto g_ = g + b_idx * T * Hv;
auto beta_ = beta + b_idx * T * Hv;

for (int t = 0; t < T; ++t) {
  float kv_mem = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] * g_[hv_idx];
    kv_mem += state[i] * k_[s_idx];
  }
  kv_mem = simd_sum(kv_mem);

  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

  float out = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] + k_[s_idx] * delta;
    out += state[i] * q_[s_idx];
  }
  out = simd_sum(out);
  if (thread_index_in_simdgroup == 0) {
    y[dv_idx] = static_cast<InT>(out);
  }

  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
  g_ += Hv;
  beta_ += Hv;
}

for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = static_cast<InT>(state[i]);
}
`

func cStringVector(values []string) (C.mlx_vector_string, func(), bool) {
	vec := C.mlx_vector_string_new()
	ok := true
	for _, s := range values {
		cs := C.CString(s)
		if C.mlx_vector_string_append_value(vec, cs) != 0 {
			ok = false
		}
		C.free(unsafe.Pointer(cs))
		if !ok {
			break
		}
	}
	cleanup := func() {
		C.mlx_vector_string_free(vec)
	}
	return vec, cleanup, ok
}

func initGatedDeltaMetalKernel() {
	inputs, freeInputs, ok := cStringVector([]string{"q", "k", "v", "g", "beta", "state_in", "T"})
	if !ok {
		gatedDeltaMetalDisabled = true
		freeInputs()
		return
	}
	defer freeInputs()

	outputs, freeOutputs, ok := cStringVector([]string{"y", "state_out"})
	if !ok {
		gatedDeltaMetalDisabled = true
		freeOutputs()
		return
	}
	defer freeOutputs()

	cName := C.CString("gated_delta_step")
	defer C.free(unsafe.Pointer(cName))
	cSource := C.CString(gatedDeltaMetalKernelSource)
	defer C.free(unsafe.Pointer(cSource))
	cHeader := C.CString("")
	defer C.free(unsafe.Pointer(cHeader))

	gatedDeltaMetalKernel = C.mlx_fast_metal_kernel_new(
		cName,
		inputs,
		outputs,
		cSource,
		cHeader,
		C.bool(true),
		C.bool(false),
	)
}

// GatedDeltaKernel runs a fused Metal kernel for the qwen3.5 recurrent update.
// It returns ok=false on unsupported shapes/devices or kernel setup/apply failure.
func GatedDeltaKernel(q, k, v, g, beta, state *Array) (y, nextState *Array, ok bool) {
	if gatedDeltaMetalDisabled {
		return nil, nil, false
	}
	if q == nil || k == nil || v == nil || g == nil || beta == nil || state == nil {
		return nil, nil, false
	}
	if !q.Valid() || !k.Valid() || !v.Valid() || !g.Valid() || !beta.Valid() || !state.Valid() {
		return nil, nil, false
	}

	qd := q.Dims()
	kd := k.Dims()
	vd := v.Dims()
	gd := g.Dims()
	bd := beta.Dims()
	sd := state.Dims()
	if len(qd) != 4 || len(kd) != 4 || len(vd) != 4 || len(gd) != 3 || len(bd) != 3 || len(sd) != 4 {
		return nil, nil, false
	}

	B, T, Hk, Dk := qd[0], qd[1], qd[2], qd[3]
	if T <= 0 || Hk <= 0 || Dk <= 0 || Dk%32 != 0 {
		return nil, nil, false
	}
	if kd[0] != B || kd[1] != T || kd[2] != Hk || kd[3] != Dk {
		return nil, nil, false
	}
	Hv, Dv := vd[2], vd[3]
	if vd[0] != B || vd[1] != T || Hv <= 0 || Dv <= 0 || Hv%Hk != 0 {
		return nil, nil, false
	}
	if gd[0] != B || gd[1] != T || gd[2] != Hv {
		return nil, nil, false
	}
	if bd[0] != B || bd[1] != T || bd[2] != Hv {
		return nil, nil, false
	}
	if sd[0] != B || sd[1] != Hv || sd[2] != Dv || sd[3] != Dk {
		return nil, nil, false
	}

	dtype := q.DType()
	if k.DType() != dtype || v.DType() != dtype || g.DType() != dtype || beta.DType() != dtype || state.DType() != dtype {
		return nil, nil, false
	}

	gatedDeltaMetalKernelOnce.Do(initGatedDeltaMetalKernel)
	if gatedDeltaMetalDisabled {
		return nil, nil, false
	}

	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)

	cInT := C.CString("InT")
	defer C.free(unsafe.Pointer(cInT))
	if C.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, cInT, C.mlx_dtype(dtype)) != 0 {
		gatedDeltaMetalDisabled = true
		return nil, nil, false
	}
	for _, tpl := range []struct {
		name  string
		value int
	}{
		{name: "Dk", value: Dk},
		{name: "Dv", value: Dv},
		{name: "Hk", value: Hk},
		{name: "Hv", value: Hv},
	} {
		cn := C.CString(tpl.name)
		rc := C.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, cn, C.int(tpl.value))
		C.free(unsafe.Pointer(cn))
		if rc != 0 {
			gatedDeltaMetalDisabled = true
			return nil, nil, false
		}
	}

	yShape := []C.int{C.int(B), C.int(T), C.int(Hv), C.int(Dv)}
	stateShape := []C.int{C.int(B), C.int(Hv), C.int(Dv), C.int(Dk)}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(yShape), C.size_t(len(yShape)), C.mlx_dtype(dtype)) != 0 {
		gatedDeltaMetalDisabled = true
		return nil, nil, false
	}
	if C.mlx_fast_metal_kernel_config_add_output_arg(cfg, unsafe.SliceData(stateShape), C.size_t(len(stateShape)), C.mlx_dtype(dtype)) != 0 {
		gatedDeltaMetalDisabled = true
		return nil, nil, false
	}
	if C.mlx_fast_metal_kernel_config_set_grid(cfg, 32, C.int(Dv), C.int(B*Hv)) != 0 {
		gatedDeltaMetalDisabled = true
		return nil, nil, false
	}
	threadY := Dv
	if threadY > 4 {
		threadY = 4
	}
	if C.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, C.int(threadY), 1) != 0 {
		gatedDeltaMetalDisabled = true
		return nil, nil, false
	}

	tScalar := FromValue(T)
	inputs := []C.mlx_array{
		q.ctx,
		k.ctx,
		v.ctx,
		g.ctx,
		beta.ctx,
		state.ctx,
		tScalar.ctx,
	}
	inVec := C.mlx_vector_array_new_data(unsafe.SliceData(inputs), C.size_t(len(inputs)))
	defer C.mlx_vector_array_free(inVec)

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	if C.mlx_fast_metal_kernel_apply(&outVec, gatedDeltaMetalKernel, inVec, cfg, DefaultStream().ctx) != 0 {
		gatedDeltaMetalDisabled = true
		return nil, nil, false
	}
	if int(C.mlx_vector_array_size(outVec)) < 2 {
		return nil, nil, false
	}

	y = New("GATED_DELTA_METAL_Y")
	nextState = New("GATED_DELTA_METAL_STATE")
	C.mlx_vector_array_get(&y.ctx, outVec, 0)
	C.mlx_vector_array_get(&nextState.ctx, outVec, 1)
	return y, nextState, true
}

func repeatHeadsForGatedDelta(x *Array, repeatFactor int) *Array {
	if repeatFactor <= 1 {
		return x
	}
	shape := x.Dims()
	x = ExpandDims(x, 3)
	x = Tile(x, []int32{1, 1, 1, int32(repeatFactor), 1})
	return Reshape(x, int32(shape[0]), int32(shape[1]), int32(shape[2]*repeatFactor), int32(shape[3]))
}

func gatedDeltaFallback(q, k, v, g, beta, state *Array) (y, nextState *Array) {
	if q == nil || k == nil || v == nil || g == nil || beta == nil || state == nil {
		return nil, nil
	}
	if !q.Valid() || !k.Valid() || !v.Valid() || !g.Valid() || !beta.Valid() || !state.Valid() {
		return nil, nil
	}

	qd := q.Dims()
	kd := k.Dims()
	vd := v.Dims()
	gd := g.Dims()
	bd := beta.Dims()
	sd := state.Dims()
	if len(qd) != 4 || len(kd) != 4 || len(vd) != 4 || len(gd) != 3 || len(bd) != 3 || len(sd) != 4 {
		return nil, nil
	}

	B, T, Hk, Dk := int32(qd[0]), int32(qd[1]), int32(qd[2]), int32(qd[3])
	Hv, Dv := int32(vd[2]), int32(vd[3])
	if T <= 0 || Hk <= 0 || Dk <= 0 || Hv <= 0 || Dv <= 0 || Hv%Hk != 0 {
		return nil, nil
	}
	if kd[0] != int(B) || kd[1] != int(T) || kd[2] != int(Hk) || kd[3] != int(Dk) {
		return nil, nil
	}
	if vd[0] != int(B) || vd[1] != int(T) {
		return nil, nil
	}
	if gd[0] != int(B) || gd[1] != int(T) || gd[2] != int(Hv) {
		return nil, nil
	}
	if bd[0] != int(B) || bd[1] != int(T) || bd[2] != int(Hv) {
		return nil, nil
	}
	if sd[0] != int(B) || sd[1] != int(Hv) || sd[2] != int(Dv) || sd[3] != int(Dk) {
		return nil, nil
	}

	repeatFactor := int(Hv / Hk)
	q = repeatHeadsForGatedDelta(q, repeatFactor)
	k = repeatHeadsForGatedDelta(k, repeatFactor)

	nextState = state
	if T == 1 {
		qt := Squeeze(q, 1)
		kt := Squeeze(k, 1)
		vt := Squeeze(v, 1)
		gt := Squeeze(g, 1)
		bt := Squeeze(beta, 1)

		nextState = Mul(nextState, ExpandDims(ExpandDims(gt, -1), -1))
		kvMem := Sum(Mul(nextState, ExpandDims(kt, 2)), -1, false)
		delta := Mul(Sub(vt, kvMem), ExpandDims(bt, -1))
		nextState = Add(nextState, Mul(ExpandDims(kt, 2), ExpandDims(delta, -1)))
		yt := Sum(Mul(nextState, ExpandDims(qt, 2)), -1, false)
		return ExpandDims(yt, 1), nextState
	}

	outs := make([]*Array, 0, T)
	for t := int32(0); t < T; t++ {
		qt := Squeeze(SliceStartStop(q, []int32{0, t, 0, 0}, []int32{B, t + 1, Hv, Dk}), 1)
		kt := Squeeze(SliceStartStop(k, []int32{0, t, 0, 0}, []int32{B, t + 1, Hv, Dk}), 1)
		vt := Squeeze(SliceStartStop(v, []int32{0, t, 0, 0}, []int32{B, t + 1, Hv, Dv}), 1)
		gt := Squeeze(SliceStartStop(g, []int32{0, t, 0}, []int32{B, t + 1, Hv}), 1)
		bt := Squeeze(SliceStartStop(beta, []int32{0, t, 0}, []int32{B, t + 1, Hv}), 1)

		nextState = Mul(nextState, ExpandDims(ExpandDims(gt, -1), -1))
		kvMem := Sum(Mul(nextState, ExpandDims(kt, 2)), -1, false)
		delta := Mul(Sub(vt, kvMem), ExpandDims(bt, -1))
		nextState = Add(nextState, Mul(ExpandDims(kt, 2), ExpandDims(delta, -1)))
		yt := Sum(Mul(nextState, ExpandDims(qt, 2)), -1, false)
		outs = append(outs, ExpandDims(yt, 1))
	}
	return Concatenate(outs, 1), nextState
}

// GatedDelta runs the recurrent update operation.
//
// It uses the fused Metal kernel when available and otherwise falls back to a
// backend-agnostic MLX implementation with identical inputs/outputs.
func GatedDelta(q, k, v, g, beta, state *Array) (y, nextState *Array) {
	if y, nextState, ok := GatedDeltaKernel(q, k, v, g, beta, state); ok {
		return y, nextState
	}
	return gatedDeltaFallback(q, k, v, g, beta, state)
}
