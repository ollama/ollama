package kvcache

import (
	"github.com/ollama/ollama/ml"
)

// Encoder cache stores K and V tensors that are position independent
//
// The tensors can be of any shape and will be returned as they were stored
// The mask is currently always nil
//
// Not currently safe for multiple sequences
type EncoderCache struct {
	// ** current forward pass **

	// the active layer for Get and Put
	curLayer int

	// if something is stored during this pass, this
	// will be the position (but there is no guarantee
	// anything will be stored)
	curPos int32

	// ** cache metadata **

	// was something stored in the cache?
	encoderCached bool

	// position of the cached data
	encoderPos int32

	// ** cache data storage **

	cacheCtx     ml.Context
	keys, values []ml.Tensor
}

func NewEncoderCache() *EncoderCache {
	return &EncoderCache{}
}

func (c *EncoderCache) Init(backend ml.Backend, dtype ml.DType, capacity int32) {
	c.cacheCtx = backend.NewContext()
}

func (c *EncoderCache) Close() {
	c.cacheCtx.Close()
}

func (c *EncoderCache) StartForward(ctx ml.Context, positions []int32, seqs []int) error {
	// The image is always in the first position
	c.curPos = positions[0]

	return nil
}

func (c *EncoderCache) SetLayer(layer int) {
	if layer >= len(c.keys) {
		c.keys = append(c.keys, make([]ml.Tensor, layer-len(c.keys)+1)...)
		c.values = append(c.values, make([]ml.Tensor, layer-len(c.values)+1)...)
	}

	c.curLayer = layer
}

func (c *EncoderCache) EncoderCached() bool {
	return c.encoderCached
}

func (c *EncoderCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	return c.keys[c.curLayer], c.values[c.curLayer], nil
}

func (c *EncoderCache) Put(ctx ml.Context, key, value ml.Tensor) {
	c.encoderPos = c.curPos
	c.encoderCached = true

	if c.keys[c.curLayer] == nil || c.values[c.curLayer] == nil {
		c.keys[c.curLayer] = c.cacheCtx.Zeros(key.DType(), key.Shape()...)
		c.values[c.curLayer] = c.cacheCtx.Zeros(value.DType(), value.Shape()...)
	}

	ctx.Forward(
		key.Copy(ctx, c.keys[c.curLayer]),
		value.Copy(ctx, c.values[c.curLayer]),
	)
}

func (c *EncoderCache) CopyPrefix(srcSeq, dstSeq int, len int32) {
	panic("encoder cache does not support multiple sequences")
}

func (c *EncoderCache) Remove(seq int, beginIndex, endIndex int32) error {
	if c.encoderPos >= beginIndex && c.encoderPos < endIndex {
		c.encoderCached = false
	}

	return nil
}
