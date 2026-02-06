package mlx

// #include "generated.h"
import "C"

import (
	"math"
	"sync"
)

var geluApprox = sync.OnceValue(func() *Closure {
	return Compile(func(inputs []*Array) []*Array {
		input := inputs[0]
		return []*Array{
			input.Multiply(
				FromValue[float32](0.5),
			).Multiply(
				input.Add(
					input.Power(FromValue[float32](3.0)).Multiply(FromValue[float32](0.044715)),
				).Multiply(
					FromValue(float32(math.Sqrt(2 / math.Pi))),
				).Tanh().Add(FromValue[float32](1.0)),
			).AsType(input.DType()),
		}
	}, true)
})

var silu = sync.OnceValue(func() *Closure {
	return Compile(func(inputs []*Array) []*Array {
		input := inputs[0]
		return []*Array{
			input.Multiply(
				input.Sigmoid(),
			).AsType(input.DType()),
		}
	}, true)
})

func GELUApprox(t *Array) *Array {
	return geluApprox().Call([]*Array{t})[0]
}

func SILU(t *Array) *Array {
	return silu().Call([]*Array{t})[0]
}
