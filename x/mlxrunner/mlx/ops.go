//go:build mlx

package mlx

// #include "generated.h"
import "C"

import (
	"unsafe"
)

func (t *Array) Abs() *Array {
	out := New("ABS")
	C.mlx_abs(&out.ctx, t.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) Add(other *Array) *Array {
	out := New("ADD")
	C.mlx_add(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) Addmm(a, b *Array, alpha, beta float32) *Array {
	out := New("ADDMM")
	C.mlx_addmm(&out.ctx, t.ctx, a.ctx, b.ctx, C.float(alpha), C.float(beta), DefaultStream().ctx)
	return out
}

func (t *Array) Argmax(axis int, keepDims bool) *Array {
	out := New("ARGMAX")
	C.mlx_argmax_axis(&out.ctx, t.ctx, C.int(axis), C.bool(keepDims), DefaultStream().ctx)
	return out
}

func (t *Array) ArgpartitionAxis(kth int, axis int) *Array {
	out := New("ARGPARTITION")
	C.mlx_argpartition_axis(&out.ctx, t.ctx, C.int(kth), C.int(axis), DefaultStream().ctx)
	return out
}

func (t *Array) ArgsortAxis(axis int) *Array {
	out := New("ARGSORT_AXIS")
	C.mlx_argsort_axis(&out.ctx, t.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

func (t *Array) AsType(dtype DType) *Array {
	out := New("AS_TYPE")
	C.mlx_astype(&out.ctx, t.ctx, C.mlx_dtype(dtype), DefaultStream().ctx)
	return out
}

func (t *Array) AsStrided(shape []int, strides []int, offset int) *Array {
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}

	cStrides := make([]C.int64_t, len(strides))
	for i, s := range strides {
		cStrides[i] = C.int64_t(s)
	}

	out := New("AS_STRIDED")
	C.mlx_as_strided(
		&out.ctx, t.ctx,
		unsafe.SliceData(cShape), C.size_t(len(shape)),
		unsafe.SliceData(cStrides), C.size_t(len(strides)),
		C.size_t(offset),
		DefaultStream().ctx,
	)
	return out
}

func (t *Array) Concatenate(axis int, others ...*Array) *Array {
	vector := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vector)

	s := append([]*Array{t}, others...)
	for _, other := range s {
		C.mlx_vector_array_append_value(vector, other.ctx)
	}

	out := New("CONCATENATE")
	C.mlx_concatenate_axis(&out.ctx, vector, C.int(axis), DefaultStream().ctx)
	return out
}

func (t *Array) Divide(other *Array) *Array {
	out := New("DIVIDE")
	C.mlx_divide(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) Cumsum(axis int, reverse, inclusive bool) *Array {
	out := New("CUMSUM")
	C.mlx_cumsum(&out.ctx, t.ctx, C.int(axis), C.bool(reverse), C.bool(inclusive), DefaultStream().ctx)
	return out
}

func (t *Array) ExpandDims(axis int) *Array {
	out := New("EXPAND_DIMS")
	C.mlx_expand_dims(&out.ctx, t.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

func (t *Array) Flatten(startAxis, endAxis int) *Array {
	out := New("FLATTEN")
	C.mlx_flatten(&out.ctx, t.ctx, C.int(startAxis), C.int(endAxis), DefaultStream().ctx)
	return out
}

func (t *Array) FloorDivide(other *Array) *Array {
	out := New("FLOOR_DIVIDE")
	C.mlx_floor_divide(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) GatherMM(other, lhs, rhs *Array, sorted bool) *Array {
	if lhs == nil {
		lhs = New("")
	}
	if rhs == nil {
		rhs = New("")
	}
	out := New("GATHER_MM")
	C.mlx_gather_mm(&out.ctx, t.ctx, other.ctx, lhs.ctx, rhs.ctx, C.bool(sorted), DefaultStream().ctx)
	return out
}

func (t *Array) GreaterEqual(other *Array) *Array {
	out := New("GREATER_EQUAL")
	C.mlx_greater_equal(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) Logsumexp(keepDims bool) *Array {
	out := New("LOGSUMEXP")
	C.mlx_logsumexp(&out.ctx, t.ctx, C.bool(keepDims), DefaultStream().ctx)
	return out
}

func (t *Array) Less(other *Array) *Array {
	out := New("LESS")
	C.mlx_less(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) LogicalOr(other *Array) *Array {
	out := New("LOGICAL_OR")
	C.mlx_logical_or(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) Matmul(other *Array) *Array {
	out := New("MATMUL")
	C.mlx_matmul(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) Multiply(other *Array) *Array {
	out := New("MULTIPLY")
	C.mlx_multiply(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) Negative() *Array {
	out := New("NEGATIVE")
	C.mlx_negative(&out.ctx, t.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) Power(exponent *Array) *Array {
	out := New("POWER")
	C.mlx_power(&out.ctx, t.ctx, exponent.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) PutAlongAxis(indices, values *Array, axis int) *Array {
	out := New("PUT_ALONG_AXIS")
	C.mlx_put_along_axis(&out.ctx, t.ctx, indices.ctx, values.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

func (t *Array) Reshape(axes ...int) *Array {
	cAxes := make([]C.int, len(axes))
	for i := range axes {
		cAxes[i] = C.int(axes[i])
	}

	out := New("RESHAPE")
	C.mlx_reshape(&out.ctx, t.ctx, unsafe.SliceData(cAxes), C.size_t(len(cAxes)), DefaultStream().ctx)
	return out
}

func (t *Array) Sigmoid() *Array {
	out := New("SIGMOID")
	C.mlx_sigmoid(&out.ctx, t.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) Sqrt() *Array {
	out := New("SQRT")
	C.mlx_sqrt(&out.ctx, t.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) Squeeze(axis int) *Array {
	out := New("SQUEEZE")
	C.mlx_squeeze_axis(&out.ctx, t.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

func (t *Array) StackAxis(axis int, others ...*Array) *Array {
	vectorData := make([]C.mlx_array, len(others)+1)
	vectorData[0] = t.ctx
	for i := range others {
		vectorData[i+1] = others[i].ctx
	}

	vector := C.mlx_vector_array_new_data(unsafe.SliceData(vectorData), C.size_t(len(vectorData)))
	defer C.mlx_vector_array_free(vector)

	out := New("STACK_AXIS")
	C.mlx_stack_axis(&out.ctx, vector, C.int(axis), DefaultStream().ctx)
	return out
}

func (t *Array) Subtract(other *Array) *Array {
	out := New("SUBTRACT")
	C.mlx_subtract(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) SumAxis(axis int, keepDims bool) *Array {
	out := New("SUM_AXIS")
	C.mlx_sum_axis(&out.ctx, t.ctx, C.int(axis), C.bool(keepDims), DefaultStream().ctx)
	return out
}

func (t *Array) TakeAxis(indices *Array, axis int) *Array {
	out := New("TAKE_AXIS")
	C.mlx_take_axis(&out.ctx, t.ctx, indices.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

func (t *Array) TakeAlongAxis(indices *Array, axis int) *Array {
	out := New("TAKE_ALONG_AXIS")
	C.mlx_take_along_axis(&out.ctx, t.ctx, indices.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

func (t *Array) Tanh() *Array {
	out := New("TANH")
	C.mlx_tanh(&out.ctx, t.ctx, DefaultStream().ctx)
	return out
}

func (t *Array) Transpose(axes ...int) *Array {
	cAxes := make([]C.int, len(axes))
	for i, axis := range axes {
		cAxes[i] = C.int(axis)
	}

	out := New("TRANSPOSE")
	C.mlx_transpose_axes(&out.ctx, t.ctx, unsafe.SliceData(cAxes), C.size_t(len(cAxes)), DefaultStream().ctx)
	return out
}

func Zeros(dtype DType, shape ...int) *Array {
	cAxes := make([]C.int, len(shape))
	for i := range shape {
		cAxes[i] = C.int(shape[i])
	}

	t := New("ZEROS")
	C.mlx_zeros(&t.ctx, unsafe.SliceData(cAxes), C.size_t(len(cAxes)), C.mlx_dtype(dtype), DefaultStream().ctx)
	return t
}
