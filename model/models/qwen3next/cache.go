package qwen3next

import (
	"math"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

var (
	_ kvcache.Cache           = (*HybridCache)(nil)
	_ kvcache.CheckpointCache = (*HybridCache)(nil)
)

// HybridCache adapts the shared recurrent cache base for Qwen3-Next naming.
type HybridCache struct {
	*kvcache.Recurrent
}

func NewHybridCache(
	shift func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error),
	convDim, convChannels, deltaStateSize int,
) *HybridCache {
	base := kvcache.NewRecurrentCache(kvcache.RecurrentConfig{
		Shift:               shift,
		ConvDim:             convDim,
		ConvChannels:        convChannels,
		RecurrentStateSize:  deltaStateSize,
		CheckpointLogPrefix: "qwen3next",
	})
	return &HybridCache{Recurrent: base}
}

// DeltaState returns the delta state for current batch sequences as
// [headVDim, headVDim*numVHeads, nSeqs].
func (c *HybridCache) DeltaState(ctx ml.Context, layer int, headVDim, numVHeads int) (ml.Tensor, error) {
	return c.RecurrentState(ctx, layer, headVDim, headVDim*numVHeads)
}

// UpdateDeltaState writes a new delta state for current batch sequences.
func (c *HybridCache) UpdateDeltaState(ctx ml.Context, layer int, newState ml.Tensor) {
	c.UpdateRecurrentState(ctx, layer, newState)
}

func (c *HybridCache) seqTokens() int {
	return c.SeqTokens()
}

func (c *HybridCache) numSeqs() int {
	return c.NumSeqs()
}

// Keep qwen3next behavior for partial mid-sequence removals.
func (c *HybridCache) Remove(seq int, beginIndex, endIndex int32) error {
	if beginIndex > 0 && endIndex != math.MaxInt32 {
		return kvcache.ErrNotSupported
	}
	return c.Recurrent.Remove(seq, beginIndex, endIndex)
}
