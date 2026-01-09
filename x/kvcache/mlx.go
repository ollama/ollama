//go:build mlx

package kvcache

import (
	"github.com/ollama/ollama/x/ml"
	"github.com/ollama/ollama/x/model/input"
)

// Causal cache stores K and V tensors according to their position in the
// sequence. Returns the history and a mask for attending to past tokens
type MLXCausal struct {
	DType ml.DType

	// locations for data storage for this batch
	curLocPut ml.Tensor

	// locations for data storage for this batch
	curLocGet ml.Tensor

	// the active layer for Get and Put
	curLayer int

	capacity int

	offset int

	backend      ml.Backend
	ctxs         map[int]ml.Context
	keys, values map[int]ml.Tensor

	// TODO is this needed per layer, or will it always be consistent?
	kHeadDims, vHeadDims, numKVHeads map[int]int
}

func NewMLXCausalCache() *MLXCausal {
	return &MLXCausal{
		ctxs:       make(map[int]ml.Context),
		keys:       make(map[int]ml.Tensor),
		values:     make(map[int]ml.Tensor),
		kHeadDims:  make(map[int]int),
		vHeadDims:  make(map[int]int),
		numKVHeads: make(map[int]int),
	}
}

func (c *MLXCausal) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	c.DType = dtype
	c.capacity = capacity
	c.backend = backend
}

func (c *MLXCausal) SetConfig(config ml.CacheConfig) {}

func (c *MLXCausal) SetLayer(layer int) {
	c.curLayer = layer
}

func (c *MLXCausal) Close() {
	// slog.Info("XXX MLXCausal.Close called", "number of contexts", len(c.ctxs))
	for _, ctx := range c.ctxs {
		ctx.Close()
	}
}

func (c *MLXCausal) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	locsPut := make([]int32, len(batch.Positions))
	for i := c.offset; i < len(batch.Positions); i++ {
		locsPut[i-c.offset] = int32(i)
	}
	c.offset += len(batch.Positions)
	locsGet := make([]int32, c.offset)
	for i := range c.offset {
		locsGet[i] = int32(i)
	}
	c.curLocGet = ctx.Input().FromInts(locsGet, len(locsGet))
	c.curLocPut = ctx.Input().FromInts(locsPut, len(locsPut))
	// slog.Info("XXX MLXCausal.StartForward", "offset", c.offset, "put", locsPut, "get", locsGet)

	return nil
}
func (c *MLXCausal) Put(ctx ml.Context, key, value ml.Tensor) {
	kHeadDim := key.Dim(3)
	vHeadDim := value.Dim(3)
	numKVHeads := key.Dim(1)
	batchSize := key.Dim(2)
	kCellSize := kHeadDim * numKVHeads
	vCellSize := vHeadDim * numKVHeads
	// slog.Info("XXX Causal.Put", "kHeadDim", kHeadDim, "vHeadDim", vHeadDim, "numKVHeads", numKVHeads, "batchSize", batchSize, "kCellSize", kCellSize, "vCellSize", vCellSize)

	if _, ok := c.ctxs[c.curLayer]; !ok {
		// slog.Info("XXX Causal.Put creating new context", "c.curLayer", c.curLayer)
		c.ctxs[c.curLayer] = c.backend.NewContext().Layer(c.curLayer)
	}

	if _, ok := c.keys[c.curLayer]; !ok {
		// slog.Info("XXX MLXCausal.Put allocating keys and values", "c.curLayer", c.curLayer, "shape", []int{c.capacity, kCellSize})
		c.keys[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, c.capacity, kCellSize)
		c.values[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, c.capacity, vCellSize)
		c.kHeadDims[c.curLayer] = kHeadDim
		c.vHeadDims[c.curLayer] = vHeadDim
		c.numKVHeads[c.curLayer] = numKVHeads
	}
	key = key.Reshape(ctx, batchSize, 1, kCellSize)

	// slog.Info("XXX MLXCausal.Put ", "c.keys[c.curLayer]", c.keys[c.curLayer])
	// slog.Info("XXX MLXCausal.Put ", "c.curLocPut", c.curLocPut)
	// slog.Info("XXX MLXCausal.Put ", "key", key)
	ctx.Forward(c.keys[c.curLayer].Scatter(ctx, []ml.Tensor{c.curLocPut}, key, []int{0}))
	value = value.Reshape(ctx, batchSize, 1, vCellSize)
	ctx.Forward(c.values[c.curLayer].Scatter(ctx, []ml.Tensor{c.curLocPut}, value, []int{0}))

}

func (c *MLXCausal) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	key := c.keys[c.curLayer]
	value := c.values[c.curLayer]

	kHeadDim := c.kHeadDims[c.curLayer]
	vHeadDim := c.vHeadDims[c.curLayer]
	numKVHeads := c.numKVHeads[c.curLayer]
	// rowSize := numKVHeads * c.curBatchSize
	// cachedSize := c.curMask.Dim(1)
	cachedSize := c.curLocGet.Dim(0)
	// kCellSize := kHeadDim * numKVHeads
	// vCellSize := vHeadDim * numKVHeads
	// slog.Info("XXX MLXCausal.Get", "shape", []int{1, numKVHeads, cachedSize, kHeadDim})

	key = key.TakeAxes(ctx, c.curLocGet, 0).Reshape(ctx, 1, numKVHeads, cachedSize, kHeadDim)
	value = value.TakeAxes(ctx, c.curLocGet, 0).Reshape(ctx, 1, numKVHeads, cachedSize, vHeadDim)
	return key, value, nil
}

func (c *MLXCausal) CopyPrefix(srcSeq, dstSeq int, len int32) {
	panic("not implemented")
}

func (c *MLXCausal) CanResume(seq int, pos int32) bool {
	panic("not implemented")
}

func (c *MLXCausal) Remove(seq int, beginIndex, endIndex int32) error {
	panic("not implemented")
}
