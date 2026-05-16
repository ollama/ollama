package mlx

// #include "generated.h"
import "C"

import "unsafe"

func RandomKey(seed uint64) *Array {
	out := New("RANDOM_KEY")
	C.mlx_random_key(&out.ctx, C.uint64_t(seed))
	return out
}

func (t *Array) Categorical(axis int) *Array {
	return t.CategoricalWithKey(axis, nil)
}

func (t *Array) CategoricalWithKey(axis int, key *Array) *Array {
	if key == nil {
		key = New("")
	}
	out := New("")
	C.mlx_random_categorical(&out.ctx, t.ctx, C.int(axis), key.ctx, DefaultStream().ctx)
	return out
}

func Bernoulli(p *Array) *Array {
	return BernoulliWithKey(p, nil)
}

func BernoulliWithKey(p *Array, key *Array) *Array {
	dims := p.Dims()
	shape := make([]C.int, len(dims))
	for i, d := range dims {
		shape[i] = C.int(d)
	}

	if key == nil {
		key = New("")
	}
	out := New("BERNOULLI")
	C.mlx_random_bernoulli(&out.ctx, p.ctx, unsafe.SliceData(shape), C.size_t(len(shape)), key.ctx, DefaultStream().ctx)
	return out
}
