package qwen3next

import (
	"math"
	"slices"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

var _ kvcache.Cache = (*HybridCache)(nil)

// HybridCache stores:
// - a standard causal KV cache for full attention layers
// - per-sequence conv state for linear attention layers
// - per-sequence delta state for linear attention layers
//
// Conv state shape (per layer, per sequence): [convKernelSize-1, convChannels]
// Delta state shape (per layer, per sequence): [headVDim, headVDim * numVHeads]
type HybridCache struct {
	kv *kvcache.Causal

	backend      ml.Backend
	dtype        ml.DType
	maxSequences int

	// Conv state dimensions
	convDim      int // convKernelSize - 1
	convChannels int // d_inner + 2 * num_k_heads * head_k_dim

	// Delta state dimensions
	deltaStateSize int // headVDim * headVDim * numVHeads

	// slot mapping for recurrent state (copy-on-write)
	slotForSeq map[int]int
	refCount   []int
	freeSlots  []int

	// per-layer conv state buffers (allocated lazily)
	convCtxs   map[int]ml.Context
	convStates map[int]ml.Tensor // [convDim*convChannels, maxSlots]

	// per-layer delta state buffers (allocated lazily)
	deltaCtxs   map[int]ml.Context
	deltaStates map[int]ml.Tensor // [deltaStateSize, maxSlots]

	// recurrent checkpoints (per slot)
	checkpointCount     int
	checkpointMinPos    int32
	checkpointInterval  int32
	checkpointCtxSize   int
	checkpoints         map[int]*slotCheckpointStore
	pendingRestore      map[int]checkpointRestore
	curCheckpointPos    []int32
	curCheckpointSlots  map[int]int
	reserveCheckpoints  bool
	checkpointConvCtxs  map[int]ml.Context
	checkpointDeltaCtxs map[int]ml.Context
	checkpointReserved  map[int]struct{}

	// current forward batch (derived in StartForward)
	curSeqs       []int
	curSlots      []int
	curSlotsInput ml.Tensor
	curSeqTokens  int

	// track if EnsureWritable has been called for this forward pass
	writableEnsured bool
	writableError   error
}

func NewHybridCache(
	shift func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error),
	convDim, convChannels, deltaStateSize int,
) *HybridCache {
	return &HybridCache{
		kv:                  kvcache.NewCausalCache(shift),
		convDim:             convDim,
		convChannels:        convChannels,
		deltaStateSize:      deltaStateSize,
		slotForSeq:          make(map[int]int),
		convCtxs:            make(map[int]ml.Context),
		convStates:          make(map[int]ml.Tensor),
		deltaCtxs:           make(map[int]ml.Context),
		deltaStates:         make(map[int]ml.Tensor),
		checkpointCount:     checkpointCountDefault,
		checkpointMinPos:    checkpointMinPosDefault,
		checkpointInterval:  checkpointIntervalDefault,
		checkpoints:         make(map[int]*slotCheckpointStore),
		pendingRestore:      make(map[int]checkpointRestore),
		curCheckpointSlots:  make(map[int]int),
		checkpointConvCtxs:  make(map[int]ml.Context),
		checkpointDeltaCtxs: make(map[int]ml.Context),
		checkpointReserved:  make(map[int]struct{}),
	}
}

func (c *HybridCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	c.backend = backend
	c.dtype = dtype
	c.maxSequences = maxSequences
	c.checkpoints = make(map[int]*slotCheckpointStore)
	c.pendingRestore = make(map[int]checkpointRestore)
	c.curCheckpointPos = c.curCheckpointPos[:0]
	c.curCheckpointSlots = make(map[int]int)
	c.checkpointReserved = make(map[int]struct{})
	c.checkpointCtxSize = c.checkpointCount * c.maxSequences
	if c.checkpointCtxSize < 8 {
		c.checkpointCtxSize = 8
	}

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
	for _, ctx := range c.deltaCtxs {
		ctx.Close()
	}
	for _, ctx := range c.checkpointConvCtxs {
		ctx.Close()
	}
	for _, ctx := range c.checkpointDeltaCtxs {
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

	// Derive equal-length sequence layout for recurrent layers
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
	if reserve {
		c.curSlots = c.curSlots[:0]
		slots := make([]int32, nSeqs)
		for i := range nSeqs {
			c.curSlots = append(c.curSlots, i)
			slots[i] = int32(i)
		}
		c.curSlotsInput = ctx.Input().FromInts(slots, len(slots))
		c.reserveCheckpoints = true
		c.planCheckpoints(batch)
		return nil
	}

	// Ensure slots exist for sequences in this batch
	c.curSlots = c.curSlots[:0]
	var newSlots []int
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

	// Zero state for newly allocated slots
	if len(newSlots) > 0 {
		c.zeroSlots(ctx, newSlots)
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
	c.reserveCheckpoints = false
	c.planCheckpoints(batch)

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
	if slot >= 0 && slot < c.maxSequences {
		c.freeSlots = append(c.freeSlots, slot)
	}
}

// zeroSlots zeros the recurrent state for the given slots across all layers.
func (c *HybridCache) zeroSlots(ctx ml.Context, slots []int) {
	if len(slots) == 0 {
		return
	}

	inputCtx := ctx.Input()

	slotIndices := make([]int32, len(slots))
	for i, s := range slots {
		slotIndices[i] = int32(s)
	}
	slotsTensor := inputCtx.FromInts(slotIndices, len(slotIndices))

	// Zero conv states
	if len(c.convStates) > 0 {
		zeros := inputCtx.Zeros(ml.DTypeF32, c.convDim*c.convChannels, len(slots))
		for _, buf := range c.convStates {
			ctx.Forward(buf.SetRows(ctx, zeros, slotsTensor))
		}
	}

	// Zero delta states
	if len(c.deltaStates) > 0 {
		zeros := inputCtx.Zeros(ml.DTypeF32, c.deltaStateSize, len(slots))
		for _, buf := range c.deltaStates {
			ctx.Forward(buf.SetRows(ctx, zeros, slotsTensor))
		}
	}
}

// EnsureWritable ensures sequences have private slots (copy-on-write).
func (c *HybridCache) EnsureWritable(ctx ml.Context) error {
	for i, seq := range c.curSeqs {
		slot, ok := c.slotForSeq[seq]
		if !ok {
			continue
		}

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

		c.copyRecurrentState(ctx, slot, newSlot)
		c.copyCheckpoints(ctx, slot, newSlot)
	}

	// Rebuild current slots tensor
	slots := make([]int32, len(c.curSlots))
	for i, v := range c.curSlots {
		slots[i] = int32(v)
	}
	c.curSlotsInput = ctx.Input().FromInts(slots, len(slots))

	return nil
}

func (c *HybridCache) copyRecurrentState(ctx ml.Context, srcSlot, dstSlot int) {
	src := ctx.Input().FromInts([]int32{int32(srcSlot)}, 1)
	dst := ctx.Input().FromInts([]int32{int32(dstSlot)}, 1)

	for _, buf := range c.convStates {
		rows := buf.Rows(ctx, src)
		rowsF32 := rows.Cast(ctx, ml.DTypeF32)
		ctx.Forward(buf.SetRows(ctx, rowsF32, dst))
	}

	for _, buf := range c.deltaStates {
		rows := buf.Rows(ctx, src)
		rowsF32 := rows.Cast(ctx, ml.DTypeF32)
		ctx.Forward(buf.SetRows(ctx, rowsF32, dst))
	}
}

func (c *HybridCache) CopyPrefix(srcSeq, dstSeq int, prefixLen int32) {
	c.kv.CopyPrefix(srcSeq, dstSeq, prefixLen)

	// Copy-on-write for recurrent state
	if dstSlot, ok := c.slotForSeq[dstSeq]; ok {
		if c.validSlot(dstSlot) {
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
		return
	}

	if c.validSlot(srcSlot) {
		c.slotForSeq[dstSeq] = srcSlot
		c.refCount[srcSlot]++
	}
}

func (c *HybridCache) CanResume(seq int, pos int32) bool {
	if !c.kv.CanResume(seq, pos) {
		return false
	}
	if pos == 0 {
		return true
	}
	return c.hasCheckpoint(seq, pos)
}

func (c *HybridCache) Remove(seq int, beginIndex, endIndex int32) error {
	if beginIndex > 0 && endIndex != math.MaxInt32 {
		return kvcache.ErrNotSupported
	}

	if beginIndex > 0 {
		restore, ok := c.pendingRestore[seq]
		if !ok || restore.pos+1 != beginIndex {
			return kvcache.ErrNotSupported
		}
		if !c.restoreComplete(restore) {
			return kvcache.ErrNotSupported
		}
		// If the recurrent slot is shared, detach it before applying a restore.
		if slot, ok := c.slotForSeq[seq]; ok && c.validSlot(slot) && c.refCount[slot] > 1 {
			newSlot, err := c.allocSlot()
			if err != nil {
				return err
			}
			ctx := c.backend.NewContext()
			c.copyRecurrentState(ctx, slot, newSlot)
			c.copyCheckpoints(ctx, slot, newSlot)
			if len(c.convStates) > 0 || len(c.deltaStates) > 0 {
				ctx.Compute()
			}
			ctx.Close()

			c.refCount[slot]--
			c.refCount[newSlot] = 1
			c.slotForSeq[seq] = newSlot

			restore.slot = newSlot
			c.pendingRestore[seq] = restore
		}
	}

	if err := c.kv.Remove(seq, beginIndex, endIndex); err != nil {
		return err
	}

	if beginIndex > 0 {
		restore := c.pendingRestore[seq]
		delete(c.pendingRestore, seq)
		return c.applyCheckpointRestore(restore)
	}

	// Removal invalidates recurrent state
	slot, ok := c.slotForSeq[seq]
	delete(c.pendingRestore, seq)
	if !ok {
		return nil
	}

	if !c.validSlot(slot) {
		delete(c.slotForSeq, seq)
		return nil
	}

	c.refCount[slot]--
	if c.refCount[slot] <= 0 {
		c.refCount[slot] = 0
		c.clearCheckpoints(slot)
		c.freeSlot(slot)
	}
	delete(c.slotForSeq, seq)

	return nil
}

func (c *HybridCache) validSlot(slot int) bool {
	return slot >= 0 && slot < len(c.refCount)
}

func (c *HybridCache) slotsTensor() ml.Tensor {
	return c.curSlotsInput
}

// contiguousSlots returns the starting slot if current slots are contiguous and ordered.
func (c *HybridCache) contiguousSlots() (int, bool) {
	if len(c.curSlots) == 0 {
		return 0, false
	}
	start := c.curSlots[0]
	for i, s := range c.curSlots {
		if s != start+i {
			return 0, false
		}
	}
	return start, true
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

	// Recurrent state must stay in F32 (ssm_conv kernels are F32-only).
	buf := c.convCtxs[layer].Zeros(ml.DTypeF32, c.convDim*c.convChannels, c.maxSequences)
	c.convStates[layer] = buf
	return buf
}

func (c *HybridCache) deltaBuffer(ctx ml.Context, layer int) ml.Tensor {
	if buf, ok := c.deltaStates[layer]; ok {
		return buf
	}

	if _, ok := c.deltaCtxs[layer]; !ok {
		c.deltaCtxs[layer] = c.backend.NewContextSize(1).Layer(layer)
	}

	// Recurrent delta state must stay in F32.
	buf := c.deltaCtxs[layer].Zeros(ml.DTypeF32, c.deltaStateSize, c.maxSequences)
	c.deltaStates[layer] = buf
	return buf
}

func (c *HybridCache) ensureWritableOnce(ctx ml.Context) {
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
}

// ConvState returns the conv state for current batch sequences as [convDim, convChannels, nSeqs].
func (c *HybridCache) ConvState(ctx ml.Context, layer int) (ml.Tensor, error) {
	c.ensureWritableOnce(ctx)

	if c.writableError != nil {
		return nil, c.writableError
	}

	buf := c.convBuffer(ctx, layer)
	cur := buf.Rows(ctx, c.slotsTensor())
	return cur.Reshape(ctx, c.convDim, c.convChannels, c.numSeqs()), nil
}

// UpdateConvState writes a new conv state for current batch sequences.
func (c *HybridCache) UpdateConvState(ctx ml.Context, layer int, newState ml.Tensor) {
	buf := c.convBuffer(ctx, layer)
	src := newState.Reshape(ctx, c.convDim*c.convChannels, c.numSeqs())
	srcF32 := src.Cast(ctx, ml.DTypeF32)
	if start, ok := c.contiguousSlots(); ok {
		// Fast path: contiguous slots allow a single view + copy
		offset := start * buf.Stride(1)
		view := buf.View(ctx, offset, c.convDim*c.convChannels, buf.Stride(1), c.numSeqs())
		ctx.Forward(srcF32.Copy(ctx, view))
	} else {
		ctx.Forward(buf.SetRows(ctx, srcF32, c.slotsTensor()))
	}

	c.captureConvCheckpoint(ctx, layer, srcF32)
}

// DeltaState returns the delta state for current batch sequences as [headVDim, headVDim*numVHeads, nSeqs].
func (c *HybridCache) DeltaState(ctx ml.Context, layer int, headVDim, numVHeads int) (ml.Tensor, error) {
	c.ensureWritableOnce(ctx)

	if c.writableError != nil {
		return nil, c.writableError
	}

	buf := c.deltaBuffer(ctx, layer)
	cur := buf.Rows(ctx, c.slotsTensor())
	return cur.Reshape(ctx, headVDim, headVDim*numVHeads, c.numSeqs()), nil
}

// UpdateDeltaState writes a new delta state for current batch sequences.
func (c *HybridCache) UpdateDeltaState(ctx ml.Context, layer int, newState ml.Tensor) {
	buf := c.deltaBuffer(ctx, layer)
	src := newState.Reshape(ctx, c.deltaStateSize, c.numSeqs())
	srcF32 := src.Cast(ctx, ml.DTypeF32)
	if start, ok := c.contiguousSlots(); ok {
		// Fast path: contiguous slots allow a single view + copy
		offset := start * buf.Stride(1)
		view := buf.View(ctx, offset, c.deltaStateSize, buf.Stride(1), c.numSeqs())
		ctx.Forward(srcF32.Copy(ctx, view))
	} else {
		ctx.Forward(buf.SetRows(ctx, srcF32, c.slotsTensor()))
	}

	c.captureDeltaCheckpoint(ctx, layer, srcF32)
}

// IsSupportedForBatch returns true if the current batch layout supports recurrent layers.
func (c *HybridCache) IsSupportedForBatch() bool {
	return c.curSeqTokens > 0 && len(c.curSeqs) > 0
}

// Seqs returns the ordered unique sequences for the current forward pass.
func (c *HybridCache) Seqs() []int {
	return slices.Clone(c.curSeqs)
}
