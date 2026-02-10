//go:build mlx

package mlx

// #include "generated.h"
import "C"

func (t *Array) Categorical(axis int) *Array {
	key := New("")
	out := New("", t, key)
	C.mlx_random_categorical(&out.ctx, t.ctx, C.int(axis), key.ctx, DefaultStream().ctx)
	return out
}
