package lfm2

import (
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

var (
	_ kvcache.Cache           = (*HybridCache)(nil)
	_ kvcache.CheckpointCache = (*HybridCache)(nil)
)

// HybridCache adapts the shared recurrent cache for LFM2:
// - KV attention cache is handled by the embedded causal cache
// - shortconv recurrent state uses conv slots [dConv, hiddenSize]
//
// This reuses shared checkpoint/restore logic for prefix mismatch recovery.
type HybridCache struct {
	*kvcache.Recurrent
}

func NewHybridCache(shift func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error), hiddenSize, dConv int) *HybridCache {
	base := kvcache.NewRecurrentCache(kvcache.RecurrentConfig{
		Shift:               shift,
		ConvDim:             dConv,
		ConvChannels:        hiddenSize,
		RecurrentStateSize:  1, // LFM2 uses only conv state; keep a minimal recurrent buffer size.
		CheckpointLogPrefix: "lfm2",
	})

	return &HybridCache{Recurrent: base}
}

func (c *HybridCache) seqTokens() int {
	return c.SeqTokens()
}

func (c *HybridCache) numSeqs() int {
	return c.NumSeqs()
}
