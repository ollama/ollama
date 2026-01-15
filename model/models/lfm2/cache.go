package lfm2

import (
	"slices"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// HybridCache stores:
// - a standard causal KV cache for attention layers
// - a per-sequence recurrent conv state for shortconv layers
//
// Conv state shape (per layer, per sequence): [dConv, hiddenSize] where dConv = L_cache - 1.
// Stored internally as a tensor of shape [dConv * hiddenSize, maxSlots].
type HybridCache struct {
	kv *kvcache.Causal

	backend      ml.Backend
	dtype        ml.DType
	maxSequences int

	hiddenSize int
	dConv      int

	// slot mapping for recurrent state
	slotForSeq map[int]int
	refCount   []int
	freeSlots  []int

	// per-layer conv state buffers (allocated lazily)
	convCtxs   map[int]ml.Context
	convStates map[int]ml.Tensor // [dConv*hiddenSize, maxSlots]

	// current forward batch (derived in StartForward)
	curSeqs       []int
	curSlots      []int
	curSlotsInput ml.Tensor
	curSeqTokens  int

	// optimization: track if EnsureWritable has been called for this forward pass
	writableEnsured bool
}

func NewHybridCache(shift func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error), hiddenSize, dConv int) *HybridCache {
	return &HybridCache{
		kv:         kvcache.NewCausalCache(shift),
		hiddenSize: hiddenSize,
		dConv:      dConv,
		slotForSeq: make(map[int]int),
		convCtxs:   make(map[int]ml.Context),
		convStates: make(map[int]ml.Tensor),
	}
}

func (c *HybridCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	c.backend = backend
	c.dtype = dtype
	c.maxSequences = maxSequences

	// initialize slot allocator
	c.refCount = make([]int, maxSequences)
	c.freeSlots = c.freeSlots[:0]
	for i := maxSequences - 1; i >= 0; i-- {
		c.freeSlots = append(c.freeSlots, i)
	}

	c.kv.Init(backend, dtype, maxSequences, capacity, maxBatch)
}

func (c *HybridCache) Close() {
	for _, ctx := range c.convCtxs {
		ctx.Close()
	}
	c.kv.Close()
}

func (c *HybridCache) SetConfig(config ml.CacheConfig) {
	c.kv.SetConfig(config)
}

func (c *HybridCache) SetLayer(layer int) {
	c.kv.SetLayer(layer)
}

func (c *HybridCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	return c.kv.Get(ctx)
}

func (c *HybridCache) Put(ctx ml.Context, key, value ml.Tensor) {
	c.kv.Put(ctx, key, value)
}

func (c *HybridCache) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	if err := c.kv.StartForward(ctx, batch, reserve); err != nil {
		return err
	}

	// Derive equal-length sequence layout for shortconv.
	// LFM2 shortconv assumes tokens form a [seq_tokens, seqs] grid.
	seqCounts := make(map[int]int)
	c.curSeqs = c.curSeqs[:0]
	for _, s := range batch.Sequences {
		if _, ok := seqCounts[s]; !ok {
			c.curSeqs = append(c.curSeqs, s)
		}
		seqCounts[s]++
	}

	if len(c.curSeqs) == 0 {
		return nil
	}

	nTokens := len(batch.Sequences)
	nSeqs := len(c.curSeqs)
	want := nTokens / nSeqs
	for _, s := range c.curSeqs {
		if seqCounts[s] != want {
			return kvcache.ErrNotSupported
		}
	}

	c.curSeqTokens = want

	// Ensure slots exist for sequences in this batch
	c.curSlots = c.curSlots[:0]
	for _, s := range c.curSeqs {
		slot, ok := c.slotForSeq[s]
		if !ok {
			var err error
			slot, err = c.allocSlot()
			if err != nil {
				return err
			}
			c.slotForSeq[s] = slot
			c.refCount[slot] = 1
		}
		c.curSlots = append(c.curSlots, slot)
	}

	// Create a tensor for the current slots
	slots := make([]int32, len(c.curSlots))
	for i, v := range c.curSlots {
		slots[i] = int32(v)
	}
	c.curSlotsInput = ctx.Input().FromInts(slots, len(slots))

	// Reset writable flag for new forward pass
	c.writableEnsured = false

	return nil
}

func (c *HybridCache) allocSlot() (int, error) {
	if len(c.freeSlots) == 0 {
		return 0, kvcache.ErrKvCacheFull
	}
	slot := c.freeSlots[len(c.freeSlots)-1]
	c.freeSlots = c.freeSlots[:len(c.freeSlots)-1]
	return slot, nil
}

func (c *HybridCache) freeSlot(slot int) {
	c.freeSlots = append(c.freeSlots, slot)
}

// EnsureWritable ensures that sequences in the current batch have private (non-shared) conv slots.
// Must be called with a valid ctx during a forward pass.
func (c *HybridCache) EnsureWritable(ctx ml.Context) {
	for _, seq := range c.curSeqs {
		slot := c.slotForSeq[seq]
		if c.refCount[slot] <= 1 {
			continue
		}

		newSlot, err := c.allocSlot()
		if err != nil {
			continue
		}
		c.refCount[slot]--
		c.refCount[newSlot] = 1
		c.slotForSeq[seq] = newSlot

		// Copy existing conv state for all initialized layers
		for _, buf := range c.convStates {
			// buf: [dConv*hiddenSize, maxSlots]
			src := buf.Rows(ctx, ctx.Input().FromInts([]int32{int32(slot)}, 1))
			ctx.Forward(buf.SetRows(ctx, src, ctx.Input().FromInts([]int32{int32(newSlot)}, 1)))
		}
	}

	// Rebuild current slots tensor if any slot changed
	slots := make([]int32, len(c.curSeqs))
	for i, s := range c.curSeqs {
		slots[i] = int32(c.slotForSeq[s])
	}
	c.curSlots = c.curSlots[:0]
	for _, v := range slots {
		c.curSlots = append(c.curSlots, int(v))
	}
	c.curSlotsInput = ctx.Input().FromInts(slots, len(slots))
}

func (c *HybridCache) CopyPrefix(srcSeq, dstSeq int, len int32) {
	// KV cache shares prefix metadata (no copy) which is correct for prefix reuse.
	c.kv.CopyPrefix(srcSeq, dstSeq, len)

	// For shortconv state we implement copy-on-write: dst shares the same slot as src.
	// On the first write to dst, EnsureWritable will create a private slot.
	if dstSlot, ok := c.slotForSeq[dstSeq]; ok {
		// remove dst mapping
		c.refCount[dstSlot]--
		if c.refCount[dstSlot] == 0 {
			delete(c.slotForSeq, dstSeq)
			c.freeSlot(dstSlot)
		}
	}

	srcSlot, ok := c.slotForSeq[srcSeq]
	if !ok {
		// src may not have a slot yet; dst will allocate on demand
		return
	}

	c.slotForSeq[dstSeq] = srcSlot
	c.refCount[srcSlot]++
}

func (c *HybridCache) CanResume(seq int, pos int32) bool {
	return c.kv.CanResume(seq, pos)
}

func (c *HybridCache) Remove(seq int, beginIndex, endIndex int32) error {
	if err := c.kv.Remove(seq, beginIndex, endIndex); err != nil {
		return err
	}

	// Any removal invalidates the recurrent state; drop the slot mapping so it resets on next use.
	if slot, ok := c.slotForSeq[seq]; ok {
		c.refCount[slot]--
		if c.refCount[slot] <= 0 {
			delete(c.slotForSeq, seq)
			c.freeSlot(slot)
		}
	}

	return nil
}

func (c *HybridCache) slotsTensor() ml.Tensor {
	return c.curSlotsInput
}

func (c *HybridCache) seqTokens() int {
	return c.curSeqTokens
}

func (c *HybridCache) numSeqs() int {
	return len(c.curSeqs)
}

func (c *HybridCache) convBuffer(ctx ml.Context, layer int) ml.Tensor {
	if buf, ok := c.convStates[layer]; ok {
		return buf
	}

	if _, ok := c.convCtxs[layer]; !ok {
		c.convCtxs[layer] = c.backend.NewContextSize(1).Layer(layer)
	}

	buf := c.convCtxs[layer].Zeros(c.dtype, c.dConv*c.hiddenSize, c.maxSequences)
	c.convStates[layer] = buf
	return buf
}

// ConvState returns the conv state for current batch sequences as shape [dConv, hiddenSize, nSeqs].
func (c *HybridCache) ConvState(ctx ml.Context, layer int) ml.Tensor {
	if !c.writableEnsured {
		needsWritable := false
		for _, seq := range c.curSeqs {
			slot := c.slotForSeq[seq]
			if slot < len(c.refCount) && c.refCount[slot] > 1 {
				needsWritable = true
				break
			}
		}

		if needsWritable {
			c.EnsureWritable(ctx)
		}
		c.writableEnsured = true
	}

	buf := c.convBuffer(ctx, layer)
	cur := buf.Rows(ctx, c.slotsTensor())
	return cur.Reshape(ctx, c.dConv, c.hiddenSize, c.numSeqs())
}

// UpdateConvState writes a new conv state for current batch sequences.
// newState must have shape [dConv, hiddenSize, nSeqs].
func (c *HybridCache) UpdateConvState(ctx ml.Context, layer int, newState ml.Tensor) {
	buf := c.convBuffer(ctx, layer)
	src := newState.Reshape(ctx, c.dConv*c.hiddenSize, c.numSeqs())
	ctx.Forward(buf.SetRows(ctx, src, c.slotsTensor()))
}

// IsSupportedForBatch returns true if the current batch layout supports shortconv.
func (c *HybridCache) IsSupportedForBatch() bool {
	return c.curSeqTokens > 0 && len(c.curSeqs) > 0
}

// Seqs returns the ordered unique sequences for the current forward pass.
func (c *HybridCache) Seqs() []int {
	return slices.Clone(c.curSeqs)
}
