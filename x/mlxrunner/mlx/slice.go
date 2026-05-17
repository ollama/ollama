package mlx

// #include "generated.h"
import "C"

import (
	"math"
	"unsafe"
)

// End is a sentinel value meaning "to the end of the dimension",
// equivalent to an omitted stop in Python (e.g. a[i:]).
const End = math.MaxInt32

type slice struct {
	args []int
}

func Slice(args ...int) slice {
	return slice{args: args}
}

func resolve(val, dim int) C.int {
	if val == End {
		return C.int(dim)
	}
	if val < 0 {
		return C.int(dim + val)
	}
	return C.int(val)
}

func makeSlices(dims []int, slices ...slice) (starts, stops, strides []C.int) {
	if len(slices) != len(dims) {
		panic("number of slice arguments must match number of tensor dimensions")
	}

	args := [3][]C.int{
		make([]C.int, len(slices)),
		make([]C.int, len(slices)),
		make([]C.int, len(slices)),
	}

	for i, s := range slices {
		dim := dims[i]
		switch len(s.args) {
		case 0:
			// slice[:]
			args[0][i] = C.int(0)
			args[1][i] = C.int(dim)
			args[2][i] = C.int(1)
		case 1:
			// slice[i]
			start := resolve(s.args[0], dim)
			args[0][i] = start
			args[1][i] = start + 1
			args[2][i] = C.int(1)
		case 2:
			// slice[i:j]
			args[0][i] = resolve(s.args[0], dim)
			args[1][i] = resolve(s.args[1], dim)
			args[2][i] = C.int(1)
		case 3:
			// slice[i:j:k]
			args[0][i] = resolve(s.args[0], dim)
			args[1][i] = resolve(s.args[1], dim)
			args[2][i] = C.int(s.args[2])
		default:
			panic("invalid slice arguments")
		}
	}

	return args[0], args[1], args[2]
}

func (t *Array) Slice(slices ...slice) *Array {
	starts, stops, strides := makeSlices(t.Dims(), slices...)
	out := New("SLICE")
	C.mlx_slice(
		&out.ctx, t.ctx,
		unsafe.SliceData(starts), C.size_t(len(starts)),
		unsafe.SliceData(stops), C.size_t(len(stops)),
		unsafe.SliceData(strides), C.size_t(len(strides)),
		DefaultStream().ctx,
	)
	return out
}

func (t *Array) SliceUpdate(other *Array, slices ...slice) *Array {
	starts, stops, strides := makeSlices(t.Dims(), slices...)
	out := New("SLICE_UPDATE")
	C.mlx_slice_update(
		&out.ctx, t.ctx, other.ctx,
		unsafe.SliceData(starts), C.size_t(len(starts)),
		unsafe.SliceData(stops), C.size_t(len(stops)),
		unsafe.SliceData(strides), C.size_t(len(strides)),
		DefaultStream().ctx,
	)
	return out
}
