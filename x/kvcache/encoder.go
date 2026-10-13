package kvcache

// import (
// 	"fmt"

// 	"github.com/ollama/ollama/ml"
// 	"github.com/ollama/ollama/model/input"
// )

// // Encoder cache stores K and V tensors that are position independent
// //
// // The tensors can be of any shape and will be returned as they were stored
// // The mask is currently always nil
// //
// // Not currently safe for multiple sequences
// type EncoderCache struct {
// 	// config controls mostly backend-specific optimizations
// 	config *ml.CacheConfig

// 	// ** current forward pass **

// 	// the active layer for Get and Put
// 	curLayer int

// 	// if something is stored during this pass, this
// 	// will be the position (but there is no guarantee
// 	// anything will be stored)
// 	curPos int32

// 	// curReserve indicates that this forward pass is only for
// 	// memory reservation and we should not update our metadata
// 	// based on it.
// 	curReserve bool

// 	// ** cache metadata **

// 	// was something stored in the cache?
// 	encoderCached bool

// 	// position of the cached data
// 	encoderPos int32

// 	// ** cache data storage **
// 	backend      ml.Backend
// 	ctxs         map[int]ml.Context
// 	keys, values map[int]ml.Tensor
// }

// func NewEncoderCache() *EncoderCache {
// 	return &EncoderCache{
// 		ctxs:   make(map[int]ml.Context),
// 		keys:   make(map[int]ml.Tensor),
// 		values: make(map[int]ml.Tensor),
// 	}
// }

// func (c *EncoderCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
// 	if c.config == nil {
// 		var config ml.CacheConfig
// 		if cc, ok := backend.(ml.BackendCacheConfig); ok {
// 			config = cc.CacheConfig()
// 		}
// 		c.config = &config
// 	}

// 	if maxSequences > 1 {
// 		panic(fmt.Errorf("encoder cache does not support multiple sequences; requested: %v", maxSequences))
// 	}

// 	if c.config.CachePadding != 0 && c.config.CachePadding != 1 {
// 		panic(fmt.Errorf("encoder cache is unable to enforce requested CachePadding (%v)", c.config.CachePadding))
// 	}

// 	c.backend = backend
// }

// func (c *EncoderCache) SetConfig(config ml.CacheConfig) {
// 	if c.config != nil {
// 		panic("config cannot be changed after being previously set, either by the model or backend")
// 	}

// 	c.config = &config
// }

// func (c *EncoderCache) Close() {
// 	for _, ctx := range c.ctxs {
// 		ctx.Close()
// 	}
// }

// func (c *EncoderCache) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
// 	// We work with the most recent image
// 	if len(batch.Multimodal) > 0 {
// 		c.curPos = batch.Positions[batch.Multimodal[len(batch.Multimodal)-1].Index]
// 	}

// 	c.curReserve = reserve

// 	return nil
// }

// func (c *EncoderCache) SetLayer(layer int) {
// 	c.curLayer = layer
// }

// func (c *EncoderCache) EncoderCached() bool {
// 	return c.encoderCached
// }

// func (c *EncoderCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
// 	return c.keys[c.curLayer], c.values[c.curLayer], nil
// }

// func (c *EncoderCache) Put(ctx ml.Context, key, value ml.Tensor) {
// 	if !c.curReserve {
// 		c.encoderPos = c.curPos
// 		c.encoderCached = true
// 	}

// 	if c.config.PermutedV {
// 		value = value.Transpose(ctx, 1, 2, 0, 3)
// 	}

// 	if _, ok := c.ctxs[c.curLayer]; !ok {
// 		c.ctxs[c.curLayer] = c.backend.NewContext().Layer(c.curLayer)
// 	}

// 	if _, ok := c.keys[c.curLayer]; !ok {
// 		c.keys[c.curLayer] = c.ctxs[c.curLayer].Empty(key.DType(), key.Shape()...)
// 	}

// 	if _, ok := c.values[c.curLayer]; !ok {
// 		c.values[c.curLayer] = c.ctxs[c.curLayer].Empty(value.DType(), value.Shape()...)
// 	}

// 	ctx.Forward(
// 		key.Copy(ctx, c.keys[c.curLayer]),
// 		value.Copy(ctx, c.values[c.curLayer]),
// 	)
// }

// func (c *EncoderCache) CopyPrefix(srcSeq, dstSeq int, len int32) {
// 	panic("encoder cache does not support multiple sequences")
// }

// func (c *EncoderCache) CanResume(seq int, pos int32) bool {
// 	return true
// }

// func (c *EncoderCache) Remove(seq int, beginIndex, endIndex int32) error {
// 	if c.encoderPos >= beginIndex && c.encoderPos < endIndex {
// 		c.encoderCached = false
// 	}

// 	return nil
// }
