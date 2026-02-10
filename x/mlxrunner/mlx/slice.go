//go:build mlx

package mlx

// #include "generated.h"
import "C"

import (
	"cmp"
	"unsafe"
)

type slice struct {
	args []int
}

func Slice(args ...int) slice {
	return slice{args: args}
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
		switch len(s.args) {
		case 0:
			// slice[:]
			args[0][i] = C.int(0)
			args[1][i] = C.int(dims[i])
			args[2][i] = C.int(1)
		case 1:
			// slice[i]
			args[0][i] = C.int(s.args[0])
			args[1][i] = C.int(s.args[0] + 1)
			args[2][i] = C.int(1)
		case 2:
			// slice[i:j]
			args[0][i] = C.int(s.args[0])
			args[1][i] = cmp.Or(C.int(s.args[1]), C.int(dims[i]))
			args[2][i] = C.int(1)
		case 3:
			// slice[i:j:k]
			args[0][i] = C.int(s.args[0])
			args[1][i] = cmp.Or(C.int(s.args[1]), C.int(dims[i]))
			args[2][i] = C.int(s.args[2])
		default:
			panic("invalid slice arguments")
		}
	}

	return args[0], args[1], args[2]
}

func (t *Array) Slice(slices ...slice) *Array {
	starts, stops, strides := makeSlices(t.Dims(), slices...)
	out := New("SLICE", t)
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
	out := New("SLICE_UPDATE", t, other)
	C.mlx_slice_update(
		&out.ctx, t.ctx, other.ctx,
		unsafe.SliceData(starts), C.size_t(len(starts)),
		unsafe.SliceData(stops), C.size_t(len(stops)),
		unsafe.SliceData(strides), C.size_t(len(strides)),
		DefaultStream().ctx,
	)
	return out
}
