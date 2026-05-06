package mlx

// #include "generated.h"
import "C"

import "unsafe"

func (t *Array) Categorical(axis int) *Array {
	key := New("")
	out := New("")
	C.mlx_random_categorical(&out.ctx, t.ctx, C.int(axis), key.ctx, DefaultStream().ctx)
	return out
}

func Bernoulli(p *Array) *Array {
	dims := p.Dims()
	shape := make([]C.int, len(dims))
	for i, d := range dims {
		shape[i] = C.int(d)
	}

	key := New("")
	out := New("BERNOULLI")
	C.mlx_random_bernoulli(&out.ctx, p.ctx, unsafe.SliceData(shape), C.size_t(len(shape)), key.ctx, DefaultStream().ctx)
	return out
}
