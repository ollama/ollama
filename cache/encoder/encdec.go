package encoder

import (
	"math"

	"github.com/ollama/ollama/cache"
	"github.com/ollama/ollama/ml"
)

type EncDecCache struct {
	enc *EncoderCache
	dec cache.Cache
}

func NewEncDecCache(enc *EncoderCache, dec cache.Cache) *EncDecCache {
	return &EncDecCache{
		enc: enc,
		dec: dec,
	}
}

func (c *EncDecCache) Init(backend ml.Backend, dtype ml.DType, capacity int32) {
	c.enc.Init(backend, dtype, capacity)
	c.dec.Init(backend, dtype, capacity)
}

func (c *EncDecCache) Close() {
	c.enc.Close()
	c.dec.Close()
}

func (c *EncDecCache) StartForward(ctx ml.Context, positions []int32, seqs []int) error {
	err := c.enc.StartForward(ctx, positions, seqs)
	if err != nil {
		return err
	}

	err = c.dec.StartForward(ctx, positions, seqs)
	if err != nil {
		for i := range positions {
			_ = c.enc.Remove(seqs[i], positions[i], math.MaxInt32)
		}
		return err
	}

	return nil
}

func (c *EncDecCache) SetLayer(layer int) {
	c.enc.SetLayer(layer)
	c.dec.SetLayer(layer)
}

func (c *EncDecCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	return c.dec.Get(ctx)
}

func (c *EncDecCache) Put(ctx ml.Context, key, value ml.Tensor) {
	c.dec.Put(ctx, key, value)
}

func (c *EncDecCache) EncoderCached() bool {
	return c.enc.EncoderCached()
}

func (c *EncDecCache) GetEnc(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	return c.enc.Get(ctx)
}

func (c *EncDecCache) PutEnc(ctx ml.Context, key, value ml.Tensor) {
	c.enc.Put(ctx, key, value)
}

func (c *EncDecCache) CopyPrefix(srcSeq, dstSeq int, len int32) {
	c.enc.CopyPrefix(srcSeq, dstSeq, len)
	c.dec.CopyPrefix(srcSeq, dstSeq, len)
}

func (c *EncDecCache) Remove(seq int, beginIndex, endIndex int32) error {
	// If the one of these fails, the caller is supposed to retry with endIndex set to math.MaxInt32, which should not fail
	err := c.enc.Remove(seq, beginIndex, endIndex)
	if err != nil {
		return err
	}

	err = c.dec.Remove(seq, beginIndex, endIndex)
	if err != nil {
		return err
	}

	return nil
}
