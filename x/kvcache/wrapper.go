package kvcache

// import (
// 	"math"

// 	"github.com/ollama/ollama/ml"
// 	"github.com/ollama/ollama/model/input"
// )

// // Wrapper cache is a container for multiple types of caches,
// // such as for the encoding and decoding portions of a model.
// type WrapperCache struct {
// 	// caches we are wrapping
// 	caches []Cache

// 	// cache to be used for this layer
// 	curType int
// }

// func NewWrapperCache(caches ...Cache) *WrapperCache {
// 	return &WrapperCache{
// 		caches: caches,
// 	}
// }

// func (c *WrapperCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
// 	for _, cache := range c.caches {
// 		cache.Init(backend, dtype, maxSequences, capacity, maxBatch)
// 	}
// }

// func (c *WrapperCache) SetConfig(config ml.CacheConfig) {
// 	for _, cache := range c.caches {
// 		cache.SetConfig(config)
// 	}
// }

// func (c *WrapperCache) Close() {
// 	for _, cache := range c.caches {
// 		cache.Close()
// 	}
// }

// func (c *WrapperCache) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
// 	for i, cache := range c.caches {
// 		err := cache.StartForward(ctx, batch, reserve)
// 		if err != nil {
// 			// unwind on error - Remove with endIndex set to math.MaxInt32 does not fail
// 			for j := i - 1; j >= 0; j-- {
// 				for k := range batch.Positions {
// 					_ = c.caches[j].Remove(batch.Sequences[k], batch.Positions[k], math.MaxInt32)
// 				}
// 			}
// 			return err
// 		}
// 	}

// 	c.curType = 0
// 	return nil
// }

// func (c *WrapperCache) SetLayer(layer int) {
// 	for _, cache := range c.caches {
// 		cache.SetLayer(layer)
// 	}
// }

// func (c *WrapperCache) SetLayerType(layerType int) {
// 	c.curType = layerType
// }

// func (c *WrapperCache) UnderlyingCache() Cache {
// 	return c.caches[c.curType]
// }

// func (c *WrapperCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
// 	return c.caches[c.curType].Get(ctx)
// }

// func (c *WrapperCache) Put(ctx ml.Context, key, value ml.Tensor) {
// 	c.caches[c.curType].Put(ctx, key, value)
// }

// func (c *WrapperCache) CopyPrefix(srcSeq, dstSeq int, len int32) {
// 	for _, cache := range c.caches {
// 		cache.CopyPrefix(srcSeq, dstSeq, len)
// 	}
// }

// func (c *WrapperCache) CanResume(seq int, pos int32) bool {
// 	for _, cache := range c.caches {
// 		if !cache.CanResume(seq, pos) {
// 			return false
// 		}
// 	}

// 	return true
// }

// func (c *WrapperCache) Remove(seq int, beginIndex, endIndex int32) error {
// 	// If the one of these fails, the caller is supposed to retry with endIndex set to math.MaxInt32, which should not fail
// 	for _, cache := range c.caches {
// 		err := cache.Remove(seq, beginIndex, endIndex)
// 		if err != nil {
// 			return err
// 		}
// 	}

// 	return nil
// }
