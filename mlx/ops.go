package mlx

// #include "generated.h"
import "C"

import (
	"unsafe"
)

func (t *Array) Abs() *Array {
	out := New("ABS")
	mlxCheck(C.mlx_abs(&out.ctx, t.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Add(other *Array) *Array {
	out := New("ADD")
	mlxCheck(C.mlx_add(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Addmm(a, b *Array, alpha, beta float32) *Array {
	out := New("ADDMM")
	mlxCheck(C.mlx_addmm(&out.ctx, t.ctx, a.ctx, b.ctx, C.float(alpha), C.float(beta), DefaultStream().ctx))
	return out
}

func (t *Array) Argmax(axis int, keepDims bool) *Array {
	out := New("ARGMAX")
	mlxCheck(C.mlx_argmax_axis(&out.ctx, t.ctx, C.int(axis), C.bool(keepDims), DefaultStream().ctx))
	return out
}

func (t *Array) ArgpartitionAxis(kth int, axis int) *Array {
	out := New("ARGPARTITION")
	mlxCheck(C.mlx_argpartition_axis(&out.ctx, t.ctx, C.int(kth), C.int(axis), DefaultStream().ctx))
	return out
}

func (t *Array) ArgsortAxis(axis int) *Array {
	out := New("ARGSORT_AXIS")
	mlxCheck(C.mlx_argsort_axis(&out.ctx, t.ctx, C.int(axis), DefaultStream().ctx))
	return out
}

func (t *Array) AsType(dtype DType) *Array {
	out := New("AS_TYPE")
	mlxCheck(C.mlx_astype(&out.ctx, t.ctx, C.mlx_dtype(dtype), DefaultStream().ctx))
	return out
}

func (t *Array) BitwiseXor(other *Array) *Array {
	out := New("BITWISE_XOR")
	mlxCheck(C.mlx_bitwise_xor(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
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
	mlxCheck(C.mlx_as_strided(
		&out.ctx, t.ctx,
		unsafe.SliceData(cShape), C.size_t(len(shape)),
		unsafe.SliceData(cStrides), C.size_t(len(strides)),
		C.size_t(offset),
		DefaultStream().ctx,
	))
	return out
}

func (t *Array) BitwiseAnd(other *Array) *Array {
	out := New("BITWISE_AND")
	mlxCheck(C.mlx_bitwise_and(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Concatenate(axis int, others ...*Array) *Array {
	if len(others) == 0 {
		return t.Clone()
	}

	vector := mlxCheck(C.mlx_vector_array_new())
	defer freeVectorArray(vector)

	s := append([]*Array{t}, others...)
	for _, other := range s {
		mlxCheck(C.mlx_vector_array_append_value(vector, other.ctx))
	}

	out := New("CONCATENATE")
	mlxCheck(C.mlx_concatenate_axis(&out.ctx, vector, C.int(axis), DefaultStream().ctx))
	return out
}

func (t *Array) Cumsum(axis int, reverse, inclusive bool) *Array {
	out := New("CUMSUM")
	optDtype := C.mlx_optional_dtype{has_value: false}
	mlxCheck(C.mlx_cumsum_axis(&out.ctx, t.ctx, C.int(axis), C.bool(reverse), C.bool(inclusive), optDtype, DefaultStream().ctx))
	return out
}

func (t *Array) Divide(other *Array) *Array {
	out := New("DIVIDE")
	mlxCheck(C.mlx_divide(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) ExpandDims(axis int) *Array {
	out := New("EXPAND_DIMS")
	mlxCheck(C.mlx_expand_dims(&out.ctx, t.ctx, C.int(axis), DefaultStream().ctx))
	return out
}

func (t *Array) Flatten(startAxis, endAxis int) *Array {
	out := New("FLATTEN")
	mlxCheck(C.mlx_flatten(&out.ctx, t.ctx, C.int(startAxis), C.int(endAxis), DefaultStream().ctx))
	return out
}

func (t *Array) FloorDivide(other *Array) *Array {
	out := New("FLOOR_DIVIDE")
	mlxCheck(C.mlx_floor_divide(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
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
	mlxCheck(C.mlx_gather_mm(&out.ctx, t.ctx, other.ctx, lhs.ctx, rhs.ctx, C.bool(sorted), DefaultStream().ctx))
	return out
}

func (t *Array) LogsumexpAxis(axis int, keepDims bool) *Array {
	out := New("LOGSUMEXP_AXIS")
	mlxCheck(C.mlx_logsumexp_axis(&out.ctx, t.ctx, C.int(axis), C.bool(keepDims), DefaultStream().ctx))
	return out
}

func (t *Array) Equal(other *Array) *Array {
	out := New("EQUAL")
	mlxCheck(C.mlx_equal(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Greater(other *Array) *Array {
	out := New("GREATER")
	mlxCheck(C.mlx_greater(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Less(other *Array) *Array {
	out := New("LESS")
	mlxCheck(C.mlx_less(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) LessEqual(other *Array) *Array {
	out := New("LESS_EQUAL")
	mlxCheck(C.mlx_less_equal(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) MaxAxis(axis int, keepDims bool) *Array {
	out := New("MAX_AXIS")
	mlxCheck(C.mlx_max_axis(&out.ctx, t.ctx, C.int(axis), C.bool(keepDims), DefaultStream().ctx))
	return out
}

func (t *Array) Matmul(other *Array) *Array {
	out := New("MATMUL")
	mlxCheck(C.mlx_matmul(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Multiply(other *Array) *Array {
	out := New("MULTIPLY")
	mlxCheck(C.mlx_multiply(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Negative() *Array {
	out := New("NEGATIVE")
	mlxCheck(C.mlx_negative(&out.ctx, t.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Power(exponent *Array) *Array {
	out := New("POWER")
	mlxCheck(C.mlx_power(&out.ctx, t.ctx, exponent.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) PutAlongAxis(indices, values *Array, axis int) *Array {
	out := New("PUT_ALONG_AXIS")
	mlxCheck(C.mlx_put_along_axis(&out.ctx, t.ctx, indices.ctx, values.ctx, C.int(axis), DefaultStream().ctx))
	return out
}

func (t *Array) ScatterAddAxis(indices, values *Array, axis int) *Array {
	out := New("SCATTER_ADD_AXIS")
	mlxCheck(C.mlx_scatter_add_axis(&out.ctx, t.ctx, indices.ctx, values.ctx, C.int(axis), DefaultStream().ctx))
	return out
}

func (t *Array) Reshape(axes ...int) *Array {
	cAxes := make([]C.int, len(axes))
	for i := range axes {
		cAxes[i] = C.int(axes[i])
	}

	out := New("RESHAPE")
	mlxCheck(C.mlx_reshape(&out.ctx, t.ctx, unsafe.SliceData(cAxes), C.size_t(len(cAxes)), DefaultStream().ctx))
	return out
}

func (t *Array) RightShift(other *Array) *Array {
	out := New("RIGHT_SHIFT")
	mlxCheck(C.mlx_right_shift(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Remainder(other *Array) *Array {
	out := New("REMAINDER")
	mlxCheck(C.mlx_remainder(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Sigmoid() *Array {
	out := New("SIGMOID")
	mlxCheck(C.mlx_sigmoid(&out.ctx, t.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Sign() *Array {
	out := New("SIGN")
	mlxCheck(C.mlx_sign(&out.ctx, t.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Sqrt() *Array {
	out := New("SQRT")
	mlxCheck(C.mlx_sqrt(&out.ctx, t.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Squeeze(axis int) *Array {
	out := New("SQUEEZE")
	mlxCheck(C.mlx_squeeze_axis(&out.ctx, t.ctx, C.int(axis), DefaultStream().ctx))
	return out
}

func (t *Array) StackAxis(axis int, others ...*Array) *Array {
	vectorData := make([]C.mlx_array, len(others)+1)
	vectorData[0] = t.ctx
	for i := range others {
		vectorData[i+1] = others[i].ctx
	}

	vector := mlxCheck(C.mlx_vector_array_new_data(unsafe.SliceData(vectorData), C.size_t(len(vectorData))))
	defer freeVectorArray(vector)

	out := New("STACK_AXIS")
	mlxCheck(C.mlx_stack_axis(&out.ctx, vector, C.int(axis), DefaultStream().ctx))
	return out
}

func (t *Array) Subtract(other *Array) *Array {
	out := New("SUBTRACT")
	mlxCheck(C.mlx_subtract(&out.ctx, t.ctx, other.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) SumAxis(axis int, keepDims bool) *Array {
	out := New("SUM_AXIS")
	mlxCheck(C.mlx_sum_axis(&out.ctx, t.ctx, C.int(axis), C.bool(keepDims), DefaultStream().ctx))
	return out
}

func (t *Array) TakeAxis(indices *Array, axis int) *Array {
	out := New("TAKE_AXIS")
	mlxCheck(C.mlx_take_axis(&out.ctx, t.ctx, indices.ctx, C.int(axis), DefaultStream().ctx))
	return out
}

func (t *Array) TakeAlongAxis(indices *Array, axis int) *Array {
	out := New("TAKE_ALONG_AXIS")
	mlxCheck(C.mlx_take_along_axis(&out.ctx, t.ctx, indices.ctx, C.int(axis), DefaultStream().ctx))
	return out
}

func (t *Array) Tanh() *Array {
	out := New("TANH")
	mlxCheck(C.mlx_tanh(&out.ctx, t.ctx, DefaultStream().ctx))
	return out
}

func (t *Array) Transpose(axes ...int) *Array {
	cAxes := make([]C.int, len(axes))
	for i, axis := range axes {
		cAxes[i] = C.int(axis)
	}

	out := New("TRANSPOSE")
	mlxCheck(C.mlx_transpose_axes(&out.ctx, t.ctx, unsafe.SliceData(cAxes), C.size_t(len(cAxes)), DefaultStream().ctx))
	return out
}

func Zeros(dtype DType, shape ...int) *Array {
	cAxes := make([]C.int, len(shape))
	for i := range shape {
		cAxes[i] = C.int(shape[i])
	}

	t := New("ZEROS")
	mlxCheck(C.mlx_zeros(&t.ctx, unsafe.SliceData(cAxes), C.size_t(len(cAxes)), C.mlx_dtype(dtype), DefaultStream().ctx))
	return t
}
