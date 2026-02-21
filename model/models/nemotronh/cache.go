package nemotronh

import (
	"errors"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

// ErrUnsupportedBatchLayout is returned when the batch layout is incompatible
// with the layer requirements.
var ErrUnsupportedBatchLayout = errors.New("nemotronh: unsupported batch layout")

var _ kvcache.Cache = (*HybridCache)(nil)
var _ kvcache.CheckpointCache = (*HybridCache)(nil)

// HybridCache adapts the shared recurrent cache base for Nemotron-H naming.
type HybridCache struct {
	*kvcache.Recurrent
}

func NewHybridCache(convDim, convChannels, ssmStateSize int) *HybridCache {
	base := kvcache.NewRecurrentCache(kvcache.RecurrentConfig{
		ConvDim:             convDim,
		ConvChannels:        convChannels,
		RecurrentStateSize:  ssmStateSize,
		CheckpointLogPrefix: "nemotronh",
	})
	return &HybridCache{Recurrent: base}
}

// SSMState returns the SSM state for current batch sequences.
func (c *HybridCache) SSMState(ctx ml.Context, layer int, dState, headDim, nHead int) (ml.Tensor, error) {
	return c.RecurrentState4D(ctx, layer, dState, headDim, nHead)
}

// UpdateSSMState writes a new SSM state for current batch sequences.
func (c *HybridCache) UpdateSSMState(ctx ml.Context, layer int, newState ml.Tensor) {
	c.UpdateRecurrentState(ctx, layer, newState)
}

func (c *HybridCache) slotsTensor() ml.Tensor {
	return c.SlotsTensor()
}

func (c *HybridCache) seqTokens() int {
	return c.SeqTokens()
}

func (c *HybridCache) numSeqs() int {
	return c.NumSeqs()
}
