package cache

import (
	"github.com/ollama/ollama/ml"
)

type TensorCache struct {
	curLayer int

	cacheCtx     ml.Context
	keys, values []ml.Tensor
}

func NewTensorCache(backend ml.Backend) *TensorCache {
	return &TensorCache{
		cacheCtx: backend.NewContext(),
	}
}

func (c *TensorCache) Close() {
	c.cacheCtx.Close()
}

func (c *TensorCache) Sub(i int) *TensorCache {
	if i >= len(c.keys) {
		c.keys = append(c.keys, make([]ml.Tensor, i-len(c.keys)+1)...)
		c.values = append(c.values, make([]ml.Tensor, i-len(c.values)+1)...)
	}

	c.curLayer = i

	return c
}

func (c *TensorCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	return c.keys[c.curLayer], c.values[c.curLayer], nil
}

func (c *TensorCache) Put(ctx ml.Context, key, value ml.Tensor) {
	if c.keys[c.curLayer] == nil || c.values[c.curLayer] == nil {
		c.keys[c.curLayer] = c.cacheCtx.Zeros(key.DType(), key.Shape()...)
		c.values[c.curLayer] = c.cacheCtx.Zeros(value.DType(), value.Shape()...)
	}

	ctx.Forward(key.Copy(ctx, c.keys[c.curLayer]))
	ctx.Forward(value.Copy(ctx, c.values[c.curLayer]))
}
