package lfm2

import (
	"slices"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

var _ kvcache.Cache = (*HybridCache)(nil)

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

	// track if EnsureWritable has been called for this forward pass
	writableEnsured bool
	// track any error from EnsureWritable to propagate later
	writableError error
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

	// When reserving memory for estimation, use fake slot assignments
	// without modifying permanent state (slotForSeq, refCount)
	if reserve {
		c.curSlots = c.curSlots[:0]
		slots := make([]int32, nSeqs)
		for i := range nSeqs {
			c.curSlots = append(c.curSlots, i)
			slots[i] = int32(i)
		}
		c.curSlotsInput = ctx.Input().FromInts(slots, len(slots))
		return nil
	}

	// Ensure slots exist for sequences in this batch
	c.curSlots = c.curSlots[:0]
	var newSlots []int // track newly allocated slots that need zeroing
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
			newSlots = append(newSlots, slot)
		}
		c.curSlots = append(c.curSlots, slot)
	}

	// Zero conv state for newly allocated slots to clear stale data from previous sequences
	if len(newSlots) > 0 {
		c.zeroConvSlots(ctx, newSlots)
	}

	// Create a tensor for the current slots
	slots := make([]int32, len(c.curSlots))
	for i, v := range c.curSlots {
		slots[i] = int32(v)
	}
	c.curSlotsInput = ctx.Input().FromInts(slots, len(slots))

	// Reset writable state for new forward pass
	c.writableEnsured = false
	c.writableError = nil

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
	// Bounds check before freeing
	if slot >= 0 && slot < c.maxSequences {
		c.freeSlots = append(c.freeSlots, slot)
	}
}

// zeroConvSlots zeros the conv state for the given slots across all layers.
// This must be called when recycling slots to prevent stale state from affecting new sequences.
func (c *HybridCache) zeroConvSlots(ctx ml.Context, slots []int) {
	if len(slots) == 0 || len(c.convStates) == 0 {
		return
	}

	// Use input context for creating tensors
	inputCtx := ctx.Input()

	// Create slot indices tensor
	slotIndices := make([]int32, len(slots))
	for i, s := range slots {
		slotIndices[i] = int32(s)
	}
	slotsTensor := inputCtx.FromInts(slotIndices, len(slotIndices))

	// Create zero tensor for the slots (SetRows requires F32 source)
	zeros := inputCtx.Zeros(ml.DTypeF32, c.dConv*c.hiddenSize, len(slots))

	// Zero each layer's conv state for these slots
	for _, buf := range c.convStates {
		ctx.Forward(buf.SetRows(ctx, zeros, slotsTensor))
	}
}

// EnsureWritable ensures that sequences in the current batch have private (non-shared) conv slots.
// Returns an error if slot allocation fails.
func (c *HybridCache) EnsureWritable(ctx ml.Context) error {
	for i, seq := range c.curSeqs {
		slot, ok := c.slotForSeq[seq]
		if !ok {
			continue
		}

		// Bounds check
		if slot < 0 || slot >= len(c.refCount) {
			continue
		}

		if c.refCount[slot] <= 1 {
			continue
		}

		newSlot, err := c.allocSlot()
		if err != nil {
			return err
		}
		c.refCount[slot]--
		c.refCount[newSlot] = 1
		c.slotForSeq[seq] = newSlot
		c.curSlots[i] = newSlot

		// Copy existing conv state for all initialized layers
		for _, buf := range c.convStates {
			// buf: [dConv*hiddenSize, maxSlots]
			src := buf.Rows(ctx, ctx.Input().FromInts([]int32{int32(slot)}, 1))
			// SetRows requires F32 source
			srcF32 := src.Cast(ctx, ml.DTypeF32)
			ctx.Forward(buf.SetRows(ctx, srcF32, ctx.Input().FromInts([]int32{int32(newSlot)}, 1)))
		}
	}

	// Rebuild current slots tensor
	slots := make([]int32, len(c.curSlots))
	for i, v := range c.curSlots {
		slots[i] = int32(v)
	}
	c.curSlotsInput = ctx.Input().FromInts(slots, len(slots))

	return nil
}

func (c *HybridCache) CopyPrefix(srcSeq, dstSeq int, prefixLen int32) {
	// KV cache shares prefix metadata (no copy) which is correct for prefix reuse.
	c.kv.CopyPrefix(srcSeq, dstSeq, prefixLen)

	// For shortconv state we implement copy-on-write: dst shares the same slot as src.
	// On the first write to dst, EnsureWritable will create a private slot.
	if dstSlot, ok := c.slotForSeq[dstSeq]; ok {
		// Bounds check before decrementing
		if dstSlot >= 0 && dstSlot < len(c.refCount) {
			c.refCount[dstSlot]--
			if c.refCount[dstSlot] <= 0 {
				c.refCount[dstSlot] = 0
				c.freeSlot(dstSlot)
			}
		}
		delete(c.slotForSeq, dstSeq)
	}

	srcSlot, ok := c.slotForSeq[srcSeq]
	if !ok {
		// src may not have a slot yet; dst will allocate on demand
		return
	}

	// Bounds check before incrementing
	if srcSlot >= 0 && srcSlot < len(c.refCount) {
		c.slotForSeq[dstSeq] = srcSlot
		c.refCount[srcSlot]++
	}
}

func (c *HybridCache) CanResume(seq int, pos int32) bool {
	return c.kv.CanResume(seq, pos)
}

func (c *HybridCache) Remove(seq int, beginIndex, endIndex int32) error {
	if err := c.kv.Remove(seq, beginIndex, endIndex); err != nil {
		return err
	}

	// For recurrent state, any removal invalidates the state because
	// the state at position N depends on all previous positions.
	// Drop the slot mapping so it resets on next use.
	slot, ok := c.slotForSeq[seq]
	if !ok {
		return nil
	}

	// Bounds check
	if slot < 0 || slot >= len(c.refCount) {
		delete(c.slotForSeq, seq)
		return nil
	}

	c.refCount[slot]--
	if c.refCount[slot] <= 0 {
		c.refCount[slot] = 0
		c.freeSlot(slot)
	}
	delete(c.slotForSeq, seq)

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
// Returns an error if copy-on-write allocation fails.
func (c *HybridCache) ConvState(ctx ml.Context, layer int) (ml.Tensor, error) {
	if !c.writableEnsured {
		needsWritable := false
		for _, seq := range c.curSeqs {
			slot, ok := c.slotForSeq[seq]
			if !ok {
				continue
			}
			if slot >= 0 && slot < len(c.refCount) && c.refCount[slot] > 1 {
				needsWritable = true
				break
			}
		}

		if needsWritable {
			if err := c.EnsureWritable(ctx); err != nil {
				c.writableError = err
			}
		}
		c.writableEnsured = true
	}

	if c.writableError != nil {
		return nil, c.writableError
	}

	buf := c.convBuffer(ctx, layer)
	cur := buf.Rows(ctx, c.slotsTensor())
	return cur.Reshape(ctx, c.dConv, c.hiddenSize, c.numSeqs()), nil
}

// UpdateConvState writes a new conv state for current batch sequences.
// newState must have shape [dConv, hiddenSize, nSeqs].
func (c *HybridCache) UpdateConvState(ctx ml.Context, layer int, newState ml.Tensor) {
	buf := c.convBuffer(ctx, layer)
	src := newState.Reshape(ctx, c.dConv*c.hiddenSize, c.numSeqs())
	// SetRows requires F32 source
	srcF32 := src.Cast(ctx, ml.DTypeF32)
	ctx.Forward(buf.SetRows(ctx, srcF32, c.slotsTensor()))
}

// IsSupportedForBatch returns true if the current batch layout supports shortconv.
func (c *HybridCache) IsSupportedForBatch() bool {
	return c.curSeqTokens > 0 && len(c.curSeqs) > 0
}

// Seqs returns the ordered unique sequences for the current forward pass.
func (c *HybridCache) Seqs() []int {
	return slices.Clone(c.curSeqs)
}
