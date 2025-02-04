package cache

import (
	"github.com/ollama/ollama/ml"
)

type Options struct {
	Position int
}

type Cache interface {
	Sub(i int) Cache
	Put(ctx ml.Context, key, value ml.Tensor, opts Options) (ml.Tensor, ml.Tensor)
}

type Simple struct {
	DType    ml.DType
	Capacity int

	keys, values []ml.Tensor
}

func (c *Simple) Sub(i int) Cache {
	if i >= len(c.keys) {
		c.keys = append(c.keys, make([]ml.Tensor, i-len(c.keys)+1)...)
		c.values = append(c.values, make([]ml.Tensor, i-len(c.values)+1)...)
	}

	return &Simple{
		keys:     c.keys[i : i+1],
		values:   c.values[i : i+1],
		Capacity: c.Capacity,
		DType:    c.DType,
	}
}

func (c *Simple) Put(ctx ml.Context, key, value ml.Tensor, opts Options) (ml.Tensor, ml.Tensor) {
	if c.keys[0] == nil || c.values[0] == nil {
		c.keys[0] = ctx.Zeros(c.DType, key.Dim(0)*key.Dim(1)*c.Capacity)
		c.values[0] = ctx.Zeros(c.DType, value.Dim(0)*value.Dim(1)*c.Capacity)
	}

	ctx.Forward(key.Copy(ctx, c.keys[0].View(ctx, key.Stride(2)*opts.Position, key.Dim(0)*key.Dim(1)*key.Dim(2))))
	ctx.Forward(value.Copy(ctx, c.values[0].View(ctx, value.Stride(2)*opts.Position, value.Dim(0)*value.Dim(1)*value.Dim(2))))

	n := min(c.Capacity, key.Dim(2)+opts.Position)

	key = c.keys[0].View(ctx, 0,
		key.Dim(0), key.Stride(1),
		key.Dim(1), key.Stride(2),
		n,
	)

	value = c.values[0].View(ctx, 0,
		value.Dim(0), value.Stride(1),
		value.Dim(1), value.Stride(2),
		n,
	)

	// TODO shift context if necessary

	return key, value
}
