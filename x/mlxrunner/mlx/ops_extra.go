//go:build mlx

package mlx

// #include "generated.h"
import "C"

import (
	"reflect"
	"unsafe"
)

// Quantization operations

func Quantize(w *Array, groupSize, bits int, mode string) (weights, scales, biases *Array) {
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	res := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(res)
	C.mlx_quantize(&res, w.ctx, optGroupSize, optBits, cMode, DefaultStream().ctx)

	vecSize := int(C.mlx_vector_array_size(res))
	w0 := New("QUANTIZE_W")
	C.mlx_vector_array_get(&w0.ctx, res, 0)
	w1 := New("QUANTIZE_S")
	C.mlx_vector_array_get(&w1.ctx, res, 1)
	if vecSize >= 3 {
		w2 := New("QUANTIZE_B")
		C.mlx_vector_array_get(&w2.ctx, res, 2)
		return w0, w1, w2
	}
	return w0, w1, nil
}

func Dequantize(w, scales, biases *Array, groupSize, bits int, mode string) *Array {
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}
	optDtype := C.mlx_optional_dtype{has_value: false}

	inputs := []*Array{w, scales}
	var b C.mlx_array
	if biases != nil {
		b = biases.ctx
		inputs = append(inputs, biases)
	}

	out := New("DEQUANTIZE", inputs...)
	C.mlx_dequantize(&out.ctx, w.ctx, scales.ctx, b, optGroupSize, optBits, cMode, optDtype, DefaultStream().ctx)
	return out
}

func QuantizedMatmul(x, w, scales, biases *Array, transpose bool, groupSize, bits int, mode string) *Array {
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}

	inputs := []*Array{x, w, scales}
	var b C.mlx_array
	if biases != nil {
		b = biases.ctx
		inputs = append(inputs, biases)
	}

	out := New("QUANTIZED_MATMUL", inputs...)
	C.mlx_quantized_matmul(&out.ctx, x.ctx, w.ctx, scales.ctx, b, C.bool(transpose), optGroupSize, optBits, cMode, DefaultStream().ctx)
	return out
}

func GatherQMM(x, w, scales *Array, biases, lhsIndices, rhsIndices *Array, transpose bool, groupSize, bits int, mode string, sortedIndices bool) *Array {
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))
	optGroupSize := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	optBits := C.mlx_optional_int{value: C.int(bits), has_value: true}

	inputs := []*Array{x, w, scales}
	var b, lhs, rhs C.mlx_array
	if biases != nil {
		b = biases.ctx
		inputs = append(inputs, biases)
	}
	if lhsIndices != nil {
		lhs = lhsIndices.ctx
		inputs = append(inputs, lhsIndices)
	}
	if rhsIndices != nil {
		rhs = rhsIndices.ctx
		inputs = append(inputs, rhsIndices)
	}

	out := New("GATHER_QMM", inputs...)
	C.mlx_gather_qmm(&out.ctx, x.ctx, w.ctx, scales.ctx, b, lhs, rhs, C.bool(transpose), optGroupSize, optBits, cMode, C.bool(sortedIndices), DefaultStream().ctx)
	return out
}

// Missing tensor ops

func Tile(a *Array, reps []int32) *Array {
	cReps := make([]C.int, len(reps))
	for i, r := range reps {
		cReps[i] = C.int(r)
	}
	out := New("TILE", a)
	C.mlx_tile(&out.ctx, a.ctx, unsafe.SliceData(cReps), C.size_t(len(reps)), DefaultStream().ctx)
	return out
}

func Tri(n, m int32, k int) *Array {
	out := New("TRI")
	C.mlx_tri(&out.ctx, C.int(n), C.int(m), C.int(k), C.mlx_dtype(DTypeFloat32), DefaultStream().ctx)
	return out
}

func Where(condition, a, b *Array) *Array {
	out := New("WHERE", condition, a, b)
	C.mlx_where(&out.ctx, condition.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Convenience wrappers (function-style for the model code)

func Stack(arrays []*Array, axis int) *Array {
	vectorData := make([]C.mlx_array, len(arrays))
	for i := range arrays {
		vectorData[i] = arrays[i].ctx
	}
	vector := C.mlx_vector_array_new_data(unsafe.SliceData(vectorData), C.size_t(len(vectorData)))
	defer C.mlx_vector_array_free(vector)

	out := New("STACK", arrays...)
	C.mlx_stack_axis(&out.ctx, vector, C.int(axis), DefaultStream().ctx)
	return out
}

func Neg(a *Array) *Array {
	return a.Negative()
}

func Sum(a *Array, axis int, keepDims bool) *Array {
	return a.SumAxis(axis, keepDims)
}

func Argsort(a *Array, axis int) *Array {
	return a.ArgsortAxis(axis)
}

func Take(a *Array, indices *Array, axis int) *Array {
	return a.TakeAxis(indices, axis)
}

func RSqrt(a *Array) *Array {
	out := New("RSQRT", a)
	C.mlx_rsqrt(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

func Mean(a *Array, axis int, keepDims bool) *Array {
	out := New("MEAN_AXIS", a)
	C.mlx_mean_axis(&out.ctx, a.ctx, C.int(axis), C.bool(keepDims), DefaultStream().ctx)
	return out
}

func Argpartition(a *Array, kth int, axis int) *Array {
	return a.ArgpartitionAxis(kth, axis)
}

func TakeAlongAxis(a, indices *Array, axis int) *Array {
	return a.TakeAlongAxis(indices, axis)
}

// Function-style wrappers matching imagegen API

func Add(a, b *Array) *Array {
	return a.Add(b)
}

func Sub(a, b *Array) *Array {
	return a.Subtract(b)
}

func Mul(a, b *Array) *Array {
	return a.Multiply(b)
}

func Div(a, b *Array) *Array {
	return a.Divide(b)
}

func Matmul(a, b *Array) *Array {
	return a.Matmul(b)
}

func Reshape(a *Array, shape ...int32) *Array {
	axes := make([]int, len(shape))
	for i, s := range shape {
		axes[i] = int(s)
	}
	return a.Reshape(axes...)
}

func Transpose(a *Array, axes ...int) *Array {
	return a.Transpose(axes...)
}

func ExpandDims(a *Array, axis int) *Array {
	return a.ExpandDims(axis)
}

func Squeeze(a *Array, axis int) *Array {
	return a.Squeeze(axis)
}

func Flatten(a *Array) *Array {
	return a.Flatten(0, -1)
}

func Concatenate(arrays []*Array, axis int) *Array {
	if len(arrays) == 0 {
		return nil
	}
	return arrays[0].Concatenate(axis, arrays[1:]...)
}

func SliceStartStop(a *Array, start, stop []int32) *Array {
	n := len(start)
	cStart := make([]C.int, n)
	cStop := make([]C.int, n)
	cStrides := make([]C.int, n)
	for i := 0; i < n; i++ {
		cStart[i] = C.int(start[i])
		cStop[i] = C.int(stop[i])
		cStrides[i] = 1
	}
	out := New("SLICE", a)
	C.mlx_slice(&out.ctx, a.ctx, unsafe.SliceData(cStart), C.size_t(n), unsafe.SliceData(cStop), C.size_t(n), unsafe.SliceData(cStrides), C.size_t(n), DefaultStream().ctx)
	return out
}

func GatherMM(a, b *Array, lhsIndices, rhsIndices *Array, sortedIndices bool) *Array {
	if lhsIndices == nil {
		lhsIndices = New("")
	}
	if rhsIndices == nil {
		rhsIndices = New("")
	}
	return a.GatherMM(b, lhsIndices, rhsIndices, sortedIndices)
}

func SiLU(a *Array) *Array {
	sig := a.Sigmoid()
	return a.Multiply(sig)
}

func RoPEWithBase(x *Array, dims int, traditional bool, base, scale float32, offset int) *Array {
	freqs := New("")
	out := New("FAST_ROPE", x, freqs)
	C.mlx_fast_rope(
		&out.ctx,
		x.ctx,
		C.int(dims),
		C.bool(traditional),
		C.mlx_optional_float{
			value:     C.float(base),
			has_value: C.bool(func() bool { return base != 0 }()),
		},
		C.float(scale),
		C.int(offset),
		freqs.ctx,
		DefaultStream().ctx,
	)
	return out
}

func Sigmoid(a *Array) *Array {
	return a.Sigmoid()
}

func ScaledDotProductAttentionCausal(q, k, v *Array, scale float32, causalMask bool) *Array {
	mask := New("")
	sinks := New("")
	mode := ""
	if causalMask {
		mode = "causal"
	}
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))

	out := New("FAST_SDPA", q, k, v, mask, sinks)
	C.mlx_fast_scaled_dot_product_attention(&out.ctx, q.ctx, k.ctx, v.ctx, C.float(scale), cMode, mask.ctx, sinks.ctx, DefaultStream().ctx)
	return out
}

func RMSNormFn(x, weight *Array, eps float32) *Array {
	out := New("FAST_RMSNORM", x)
	C.mlx_fast_rms_norm(&out.ctx, x.ctx, weight.ctx, C.float(eps), DefaultStream().ctx)
	return out
}

func AddMM(c, a, b *Array, alpha, beta float32) *Array {
	return c.Addmm(a, b, alpha, beta)
}

// Scalar helpers

func AddScalar(a *Array, s float32) *Array {
	scalar := FromValue(s)
	return a.Add(scalar)
}

func MulScalar(a *Array, s float32) *Array {
	scalar := FromValue(s)
	return a.Multiply(scalar)
}

func DivScalar(a *Array, s float32) *Array {
	scalar := FromValue(s)
	return a.Divide(scalar)
}

func FloorDivideScalar(a *Array, s int32) *Array {
	scalar := FromValue(int(s))
	return a.FloorDivide(scalar)
}

// Array constructors

func NewArrayInt32(data []int32, shape []int32) *Array {
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	out := New("NEW_ARRAY_INT32")
	out.ctx = C.mlx_array_new_data(unsafe.Pointer(&data[0]), unsafe.SliceData(cShape), C.int(len(shape)), C.mlx_dtype(DTypeInt32))
	return out
}

func NewScalarArray(value float32) *Array {
	out := New("SCALAR")
	out.ctx = C.mlx_array_new_float32(C.float(value))
	return out
}

func ZerosF32(shape []int32) *Array {
	return Zeros(DTypeFloat32, func() []int {
		ints := make([]int, len(shape))
		for i, s := range shape {
			ints[i] = int(s)
		}
		return ints
	}()...)
}

// Utility

func Collect(v any) []*Array {
	var arrays []*Array
	seen := make(map[uintptr]bool)
	collect(reflect.ValueOf(v), &arrays, seen)
	return arrays
}

func collect(v reflect.Value, arrays *[]*Array, seen map[uintptr]bool) {
	if !v.IsValid() {
		return
	}

	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return
		}
		ptr := v.Pointer()
		if seen[ptr] {
			return
		}
		seen[ptr] = true

		if arr, ok := v.Interface().(*Array); ok {
			if arr != nil && arr.Valid() {
				*arrays = append(*arrays, arr)
			}
			return
		}
		collect(v.Elem(), arrays, seen)
		return
	}

	switch v.Kind() {
	case reflect.Struct:
		// Check if this struct IS an Array (not a pointer to one)
		if arr, ok := v.Addr().Interface().(*Array); ok {
			if arr != nil && arr.Valid() {
				*arrays = append(*arrays, arr)
			}
			return
		}
		for i := 0; i < v.NumField(); i++ {
			field := v.Field(i)
			if field.CanInterface() {
				collect(field, arrays, seen)
			}
		}
	case reflect.Slice:
		for i := 0; i < v.Len(); i++ {
			collect(v.Index(i), arrays, seen)
		}
	case reflect.Map:
		for _, key := range v.MapKeys() {
			collect(v.MapIndex(key), arrays, seen)
		}
	case reflect.Interface:
		if !v.IsNil() {
			collect(v.Elem(), arrays, seen)
		}
	}
}

func EnableCompile() {
	C.mlx_enable_compile()
}

func DisableCompile() {
	C.mlx_disable_compile()
}
