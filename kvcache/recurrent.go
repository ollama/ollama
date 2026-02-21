package kvcache

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

const (
	DefaultCheckpointCount    = 32
	DefaultCheckpointMinPos   = int32(16)
	DefaultCheckpointInterval = int32(1280)
)

var (
	ErrInvalidRecurrentShape = errors.New("kvcache: invalid recurrent state shape")
)

// Config configures a shared hybrid recurrent cache.
type RecurrentConfig struct {
	Shift               func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error)
	ConvDim             int
	ConvChannels        int
	RecurrentStateSize  int
	CheckpointLogPrefix string
}

var _ Cache = (*Recurrent)(nil)
var _ CheckpointCache = (*Recurrent)(nil)

// Cache stores:
// - a standard causal KV cache
// - per-sequence conv state for recurrent operators
// - per-sequence recurrent state for recurrent operators
//
// Conv state shape (per layer, per sequence): [convDim, convChannels]
// Recurrent state shape (per layer, per sequence): [recurrentStateSize]
type Recurrent struct {
	kv *Causal

	backend      ml.Backend
	dtype        ml.DType
	maxSequences int

	// Conv state dimensions
	convDim      int
	convChannels int

	// Recurrent state dimensions
	recurrentStateSize int

	logPrefix string

	// slot mapping for recurrent state (copy-on-write)
	slotForSeq  map[int]int
	refCount    []int
	freeSlots   []int
	seqCounts   map[int]int
	slotScratch [1]int32

	// per-layer conv state buffers (allocated lazily)
	convCtxs   map[int]ml.Context
	convStates map[int]ml.Tensor // [convDim*convChannels, maxSlots]

	// per-layer recurrent state buffers (allocated lazily)
	recurrentCtxs   map[int]ml.Context
	recurrentStates map[int]ml.Tensor // [recurrentStateSize, maxSlots]

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
	checkpointRecurCtxs map[int]ml.Context
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

func NewRecurrentCache(config RecurrentConfig) *Recurrent {
	return &Recurrent{
		kv:                  NewCausalCache(config.Shift),
		convDim:             config.ConvDim,
		convChannels:        config.ConvChannels,
		recurrentStateSize:  config.RecurrentStateSize,
		logPrefix:           config.CheckpointLogPrefix,
		slotForSeq:          make(map[int]int),
		seqCounts:           make(map[int]int),
		convCtxs:            make(map[int]ml.Context),
		convStates:          make(map[int]ml.Tensor),
		recurrentCtxs:       make(map[int]ml.Context),
		recurrentStates:     make(map[int]ml.Tensor),
		checkpointCount:     DefaultCheckpointCount,
		checkpointMinPos:    DefaultCheckpointMinPos,
		checkpointInterval:  DefaultCheckpointInterval,
		checkpoints:         make(map[int]*slotCheckpointStore),
		pendingRestore:      make(map[int]checkpointRestore),
		curCheckpointSlots:  make(map[int]int),
		checkpointConvCtxs:  make(map[int]ml.Context),
		checkpointRecurCtxs: make(map[int]ml.Context),
		checkpointReserved:  make(map[int]struct{}),
	}
}

func (c *Recurrent) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
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

func (c *Recurrent) Close() {
	for _, ctx := range c.convCtxs {
		ctx.Close()
	}
	for _, ctx := range c.recurrentCtxs {
		ctx.Close()
	}
	for _, ctx := range c.checkpointConvCtxs {
		ctx.Close()
	}
	for _, ctx := range c.checkpointRecurCtxs {
		ctx.Close()
	}
	c.kv.Close()
}

func (c *Recurrent) SetConfig(config ml.CacheConfig) {
	c.kv.SetConfig(config)
}

func (c *Recurrent) SetLayer(layer int) {
	c.kv.SetLayer(layer)
}

func (c *Recurrent) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	return c.kv.Get(ctx)
}

func (c *Recurrent) Put(ctx ml.Context, key, value ml.Tensor) {
	c.kv.Put(ctx, key, value)
}

func (c *Recurrent) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	if err := c.kv.StartForward(ctx, batch, reserve); err != nil {
		return err
	}

	nTokens := len(batch.Sequences)
	if nTokens == 0 {
		c.curSeqs = c.curSeqs[:0]
		c.curSlots = c.curSlots[:0]
		c.curSlotsInput = nil
		c.curSeqTokens = 0
		return nil
	}

	// Fast path for single-sequence batches (common during decode and prefill).
	firstSeq := batch.Sequences[0]
	singleSeq := true
	for _, s := range batch.Sequences[1:] {
		if s != firstSeq {
			singleSeq = false
			break
		}
	}
	if singleSeq {
		return c.startForwardSingleSeq(ctx, firstSeq, nTokens, batch, reserve)
	}

	// Derive equal-length sequence layout for recurrent layers.
	seqCounts := c.seqCounts
	for s := range seqCounts {
		delete(seqCounts, s)
	}

	c.curSeqs = c.curSeqs[:0]
	for _, s := range batch.Sequences {
		if seqCounts[s] == 0 {
			c.curSeqs = append(c.curSeqs, s)
		}
		seqCounts[s]++
	}

	nSeqs := len(c.curSeqs)
	want := nTokens / nSeqs
	for _, s := range c.curSeqs {
		if seqCounts[s] != want {
			return ErrNotSupported
		}
	}

	c.curSeqTokens = want

	if reserve {
		c.curSlots = c.curSlots[:0]
		for i := range nSeqs {
			c.curSlots = append(c.curSlots, i)
		}
		c.setCurSlotsInput(ctx)
		c.reserveCheckpoints = true
		c.writableEnsured = false
		c.writableError = nil
		c.planCheckpoints(batch)
		return nil
	}

	// Ensure slots exist for sequences in this batch.
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

	if len(newSlots) == 1 {
		c.zeroSlot(ctx, newSlots[0])
	} else if len(newSlots) > 1 {
		c.zeroSlots(ctx, newSlots)
	}

	c.setCurSlotsInput(ctx)

	c.writableEnsured = false
	c.writableError = nil
	c.reserveCheckpoints = false
	c.planCheckpoints(batch)

	return nil
}

func (c *Recurrent) startForwardSingleSeq(ctx ml.Context, seq, seqTokens int, batch input.Batch, reserve bool) error {
	c.curSeqs = append(c.curSeqs[:0], seq)
	c.curSeqTokens = seqTokens

	if reserve {
		c.curSlots = append(c.curSlots[:0], 0)
		c.setCurSlotsInput(ctx)
		c.reserveCheckpoints = true
		c.writableEnsured = false
		c.writableError = nil
		c.planCheckpoints(batch)
		return nil
	}

	slot, ok := c.slotForSeq[seq]
	if !ok {
		var err error
		slot, err = c.allocSlot()
		if err != nil {
			return err
		}

		c.slotForSeq[seq] = slot
		c.refCount[slot] = 1
		c.zeroSlot(ctx, slot)
	}

	c.curSlots = append(c.curSlots[:0], slot)
	c.setCurSlotsInput(ctx)
	c.writableEnsured = false
	c.writableError = nil
	c.reserveCheckpoints = false
	c.planCheckpoints(batch)

	return nil
}

func (c *Recurrent) setCurSlotsInput(ctx ml.Context) {
	switch len(c.curSlots) {
	case 0:
		c.curSlotsInput = nil
	case 1:
		c.slotScratch[0] = int32(c.curSlots[0])
		c.curSlotsInput = ctx.Input().FromInts(c.slotScratch[:], 1)
	default:
		slots := make([]int32, len(c.curSlots))
		for i, v := range c.curSlots {
			slots[i] = int32(v)
		}
		c.curSlotsInput = ctx.Input().FromInts(slots, len(slots))
	}
}

func (c *Recurrent) allocSlot() (int, error) {
	if len(c.freeSlots) == 0 {
		return 0, ErrKvCacheFull
	}
	slot := c.freeSlots[len(c.freeSlots)-1]
	c.freeSlots = c.freeSlots[:len(c.freeSlots)-1]
	return slot, nil
}

func (c *Recurrent) freeSlot(slot int) {
	if slot >= 0 && slot < c.maxSequences {
		c.freeSlots = append(c.freeSlots, slot)
	}
}

func (c *Recurrent) zeroSlot(ctx ml.Context, slot int) {
	inputCtx := ctx.Input()
	c.slotScratch[0] = int32(slot)
	slotTensor := inputCtx.FromInts(c.slotScratch[:], 1)

	if len(c.convStates) > 0 {
		zeros := inputCtx.Zeros(ml.DTypeF32, c.convDim*c.convChannels, 1)
		for _, buf := range c.convStates {
			ctx.Forward(buf.SetRows(ctx, zeros, slotTensor))
		}
	}

	if len(c.recurrentStates) > 0 {
		zeros := inputCtx.Zeros(ml.DTypeF32, c.recurrentStateSize, 1)
		for _, buf := range c.recurrentStates {
			ctx.Forward(buf.SetRows(ctx, zeros, slotTensor))
		}
	}
}

// zeroSlots zeros recurrent state for the given slots across all cached layers.
func (c *Recurrent) zeroSlots(ctx ml.Context, slots []int) {
	if len(slots) == 0 {
		return
	}

	inputCtx := ctx.Input()

	slotIndices := make([]int32, len(slots))
	for i, s := range slots {
		slotIndices[i] = int32(s)
	}
	slotsTensor := inputCtx.FromInts(slotIndices, len(slotIndices))

	if len(c.convStates) > 0 {
		zeros := inputCtx.Zeros(ml.DTypeF32, c.convDim*c.convChannels, len(slots))
		for _, buf := range c.convStates {
			ctx.Forward(buf.SetRows(ctx, zeros, slotsTensor))
		}
	}

	if len(c.recurrentStates) > 0 {
		zeros := inputCtx.Zeros(ml.DTypeF32, c.recurrentStateSize, len(slots))
		for _, buf := range c.recurrentStates {
			ctx.Forward(buf.SetRows(ctx, zeros, slotsTensor))
		}
	}
}

// EnsureWritable ensures sequences have private slots (copy-on-write).
func (c *Recurrent) EnsureWritable(ctx ml.Context) error {
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

	c.setCurSlotsInput(ctx)

	return nil
}

func (c *Recurrent) copyRecurrentState(ctx ml.Context, srcSlot, dstSlot int) {
	src := ctx.Input().FromInts([]int32{int32(srcSlot)}, 1)
	dst := ctx.Input().FromInts([]int32{int32(dstSlot)}, 1)

	for _, buf := range c.convStates {
		rows := buf.Rows(ctx, src)
		if rows.DType() != ml.DTypeF32 {
			rows = rows.Cast(ctx, ml.DTypeF32)
		}
		ctx.Forward(buf.SetRows(ctx, rows, dst))
	}

	for _, buf := range c.recurrentStates {
		rows := buf.Rows(ctx, src)
		if rows.DType() != ml.DTypeF32 {
			rows = rows.Cast(ctx, ml.DTypeF32)
		}
		ctx.Forward(buf.SetRows(ctx, rows, dst))
	}
}

func (c *Recurrent) CopyPrefix(srcSeq, dstSeq int, prefixLen int32) {
	c.kv.CopyPrefix(srcSeq, dstSeq, prefixLen)

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

func (c *Recurrent) CanResume(seq int, pos int32) bool {
	if !c.kv.CanResume(seq, pos) {
		return false
	}
	if pos == 0 {
		return true
	}
	return c.hasCheckpoint(seq, pos)
}

func (c *Recurrent) Remove(seq int, beginIndex, endIndex int32) error {
	if beginIndex > 0 && endIndex != math.MaxInt32 {
		return ErrNotSupported
	}

	if beginIndex > 0 {
		restore, ok := c.pendingRestore[seq]
		if !ok || restore.pos+1 != beginIndex {
			return ErrNotSupported
		}
		if !c.restoreComplete(restore) {
			return ErrNotSupported
		}
		if slot, ok := c.slotForSeq[seq]; ok && c.validSlot(slot) && c.refCount[slot] > 1 {
			newSlot, err := c.allocSlot()
			if err != nil {
				return err
			}
			ctx := c.backend.NewContext()
			c.copyRecurrentState(ctx, slot, newSlot)
			c.copyCheckpoints(ctx, slot, newSlot)
			if len(c.convStates) > 0 || len(c.recurrentStates) > 0 {
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

func (c *Recurrent) validSlot(slot int) bool {
	return slot >= 0 && slot < len(c.refCount)
}

func (c *Recurrent) slotsTensor() ml.Tensor {
	return c.curSlotsInput
}

func (c *Recurrent) SlotsTensor() ml.Tensor {
	return c.slotsTensor()
}

// contiguousSlots returns the starting slot if current slots are contiguous and ordered.
func (c *Recurrent) contiguousSlots() (int, bool) {
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

func (c *Recurrent) seqTokens() int {
	return c.curSeqTokens
}

func (c *Recurrent) SeqTokens() int {
	return c.seqTokens()
}

func (c *Recurrent) numSeqs() int {
	return len(c.curSeqs)
}

func (c *Recurrent) NumSeqs() int {
	return c.numSeqs()
}

func (c *Recurrent) convBuffer(ctx ml.Context, layer int) ml.Tensor {
	if buf, ok := c.convStates[layer]; ok {
		return buf
	}

	if _, ok := c.convCtxs[layer]; !ok {
		c.convCtxs[layer] = c.backend.NewContextSize(1).Layer(layer)
	}

	buf := c.convCtxs[layer].Zeros(ml.DTypeF32, c.convDim*c.convChannels, c.maxSequences)
	c.convStates[layer] = buf
	return buf
}

func (c *Recurrent) recurrentBuffer(ctx ml.Context, layer int) ml.Tensor {
	if buf, ok := c.recurrentStates[layer]; ok {
		return buf
	}

	if _, ok := c.recurrentCtxs[layer]; !ok {
		c.recurrentCtxs[layer] = c.backend.NewContextSize(1).Layer(layer)
	}

	buf := c.recurrentCtxs[layer].Zeros(ml.DTypeF32, c.recurrentStateSize, c.maxSequences)
	c.recurrentStates[layer] = buf
	return buf
}

func (c *Recurrent) ensureWritableOnce(ctx ml.Context) {
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

// ConvState returns conv state for current batch sequences as [convDim, convChannels, nSeqs].
func (c *Recurrent) ConvState(ctx ml.Context, layer int) (ml.Tensor, error) {
	c.ensureWritableOnce(ctx)

	if c.writableError != nil {
		return nil, c.writableError
	}

	buf := c.convBuffer(ctx, layer)
	var cur ml.Tensor
	if start, ok := c.contiguousSlots(); ok {
		offset := start * buf.Stride(1)
		cur = buf.View(ctx, offset, c.convDim*c.convChannels, buf.Stride(1), c.numSeqs())
	} else {
		cur = buf.Rows(ctx, c.slotsTensor())
	}
	return cur.Reshape(ctx, c.convDim, c.convChannels, c.numSeqs()), nil
}

// UpdateConvState writes new conv state for current batch sequences.
func (c *Recurrent) UpdateConvState(ctx ml.Context, layer int, newState ml.Tensor) {
	buf := c.convBuffer(ctx, layer)
	src := newState.Reshape(ctx, c.convDim*c.convChannels, c.numSeqs())
	srcF32 := src
	if src.DType() != ml.DTypeF32 {
		srcF32 = src.Cast(ctx, ml.DTypeF32)
	}
	if start, ok := c.contiguousSlots(); ok {
		offset := start * buf.Stride(1)
		view := buf.View(ctx, offset, c.convDim*c.convChannels, buf.Stride(1), c.numSeqs())
		ctx.Forward(srcF32.Copy(ctx, view))
	} else {
		ctx.Forward(buf.SetRows(ctx, srcF32, c.slotsTensor()))
	}

	c.captureConvCheckpoint(ctx, layer, srcF32)
}

// RecurrentState returns recurrent state for current batch sequences with shape [dims..., nSeqs].
func (c *Recurrent) RecurrentState(ctx ml.Context, layer int, dims ...int) (ml.Tensor, error) {
	c.ensureWritableOnce(ctx)

	if c.writableError != nil {
		return nil, c.writableError
	}
	if len(dims) == 0 {
		return nil, ErrInvalidRecurrentShape
	}

	size := 1
	for _, d := range dims {
		if d <= 0 {
			return nil, ErrInvalidRecurrentShape
		}
		size *= d
	}
	if size != c.recurrentStateSize {
		return nil, fmt.Errorf("%w: got %v (size %d), want size %d", ErrInvalidRecurrentShape, dims, size, c.recurrentStateSize)
	}

	buf := c.recurrentBuffer(ctx, layer)
	var cur ml.Tensor
	if start, ok := c.contiguousSlots(); ok {
		offset := start * buf.Stride(1)
		cur = buf.View(ctx, offset, c.recurrentStateSize, buf.Stride(1), c.numSeqs())
	} else {
		cur = buf.Rows(ctx, c.slotsTensor())
	}
	shape := make([]int, 0, len(dims)+1)
	shape = append(shape, dims...)
	shape = append(shape, c.numSeqs())
	return cur.Reshape(ctx, shape...), nil
}

// RecurrentState4D returns recurrent state as [dim0, dim1, dim2, nSeqs].
func (c *Recurrent) RecurrentState4D(ctx ml.Context, layer int, dim0, dim1, dim2 int) (ml.Tensor, error) {
	c.ensureWritableOnce(ctx)

	if c.writableError != nil {
		return nil, c.writableError
	}
	if dim0 <= 0 || dim1 <= 0 || dim2 <= 0 {
		return nil, ErrInvalidRecurrentShape
	}

	size := dim0 * dim1 * dim2
	if size != c.recurrentStateSize {
		return nil, fmt.Errorf("%w: got [%d %d %d] (size %d), want size %d", ErrInvalidRecurrentShape, dim0, dim1, dim2, size, c.recurrentStateSize)
	}

	buf := c.recurrentBuffer(ctx, layer)
	var cur ml.Tensor
	if start, ok := c.contiguousSlots(); ok {
		offset := start * buf.Stride(1)
		cur = buf.View(ctx, offset, c.recurrentStateSize, buf.Stride(1), c.numSeqs())
	} else {
		cur = buf.Rows(ctx, c.slotsTensor())
	}
	return cur.Reshape(ctx, dim0, dim1, dim2, c.numSeqs()), nil
}

// UpdateRecurrentState writes new recurrent state for current batch sequences.
func (c *Recurrent) UpdateRecurrentState(ctx ml.Context, layer int, newState ml.Tensor) {
	buf := c.recurrentBuffer(ctx, layer)
	src := newState.Reshape(ctx, c.recurrentStateSize, c.numSeqs())
	srcF32 := src
	if src.DType() != ml.DTypeF32 {
		srcF32 = src.Cast(ctx, ml.DTypeF32)
	}
	if start, ok := c.contiguousSlots(); ok {
		offset := start * buf.Stride(1)
		view := buf.View(ctx, offset, c.recurrentStateSize, buf.Stride(1), c.numSeqs())
		ctx.Forward(srcF32.Copy(ctx, view))
	} else {
		ctx.Forward(buf.SetRows(ctx, srcF32, c.slotsTensor()))
	}

	c.captureRecurrentCheckpoint(ctx, layer, srcF32)
}

// IsSupportedForBatch returns true if the current batch layout supports recurrent layers.
func (c *Recurrent) IsSupportedForBatch() bool {
	return c.curSeqTokens > 0 && len(c.curSeqs) > 0
}

// Seqs returns the ordered unique sequences for the current forward pass.
func (c *Recurrent) Seqs() []int {
	return slices.Clone(c.curSeqs)
}
