package qwen3next

import (
	"log/slog"
	"math"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

const (
	checkpointCountDefault    = 32
	checkpointMinPosDefault   = int32(16)
	checkpointIntervalDefault = int32(1280)
)

// TODO(jmorganca): Add byte-serialized host-RAM checkpoints to reduce GPU
// memory usage while preserving prefix reuse for recurrent state.

type checkpointEntry struct {
	pos   int32
	conv  map[int]ml.Tensor
	delta map[int]ml.Tensor
}

type slotCheckpointStore struct {
	entries []checkpointEntry
	size    int
	next    int
	lastPos int32
}

type checkpointRestore struct {
	slot int
	idx  int
	pos  int32
}

func newSlotCheckpointStore(n int) *slotCheckpointStore {
	entries := make([]checkpointEntry, n)
	for i := range entries {
		entries[i].pos = -1
	}
	return &slotCheckpointStore{
		entries: entries,
		lastPos: -1,
	}
}

func (s *slotCheckpointStore) reset() {
	s.size = 0
	s.next = 0
	s.lastPos = -1
	for i := range s.entries {
		s.entries[i].pos = -1
	}
}

func (s *slotCheckpointStore) record(pos int32) int {
	if len(s.entries) == 0 {
		return -1
	}
	idx := s.next
	s.next = (s.next + 1) % len(s.entries)
	if s.size < len(s.entries) {
		s.size++
	}
	s.entries[idx].pos = pos
	s.lastPos = pos
	return idx
}

func (s *slotCheckpointStore) bestIndex(targetPos int32) (int, int32, bool) {
	bestIdx := -1
	bestPos := int32(-1)
	for i := range s.entries {
		pos := s.entries[i].pos
		if pos < 0 || pos >= targetPos {
			continue
		}
		if pos > bestPos {
			bestPos = pos
			bestIdx = i
		}
	}
	if bestIdx < 0 {
		return -1, -1, false
	}
	return bestIdx, bestPos, true
}

func (s *slotCheckpointStore) pruneAfter(pos int32) {
	if len(s.entries) == 0 {
		s.size = 0
		s.next = 0
		s.lastPos = -1
		return
	}

	size := 0
	next := -1
	minPos := int32(math.MaxInt32)
	minIdx := 0
	for i := range s.entries {
		if s.entries[i].pos > pos {
			s.entries[i].pos = -1
		}
		if s.entries[i].pos >= 0 {
			size++
			if s.entries[i].pos < minPos {
				minPos = s.entries[i].pos
				minIdx = i
			}
		} else if next == -1 {
			next = i
		}
	}

	s.size = size
	if size == 0 {
		s.next = 0
		s.lastPos = -1
		return
	}
	if next != -1 {
		s.next = next
	} else {
		// Full ring: overwrite the oldest checkpoint next.
		s.next = minIdx
	}
	s.lastPos = pos
}

func (s *slotCheckpointStore) window() (size int, minPos, maxPos, lastPos int32) {
	minPos = int32(math.MaxInt32)
	maxPos = int32(-1)
	for i := range s.entries {
		pos := s.entries[i].pos
		if pos < 0 {
			continue
		}
		size++
		if pos < minPos {
			minPos = pos
		}
		if pos > maxPos {
			maxPos = pos
		}
	}
	if size == 0 {
		minPos = -1
		maxPos = -1
	}
	return size, minPos, maxPos, s.lastPos
}

func (c *HybridCache) planCheckpoints(batch input.Batch) {
	if c.checkpointCount == 0 || len(c.curSeqs) == 0 {
		c.curCheckpointPos = c.curCheckpointPos[:0]
		for k := range c.curCheckpointSlots {
			delete(c.curCheckpointSlots, k)
		}
		return
	}

	if cap(c.curCheckpointPos) < len(c.curSeqs) {
		c.curCheckpointPos = make([]int32, len(c.curSeqs))
	} else {
		c.curCheckpointPos = c.curCheckpointPos[:len(c.curSeqs)]
	}
	for i := range c.curCheckpointPos {
		c.curCheckpointPos[i] = -1
	}
	for k := range c.curCheckpointSlots {
		delete(c.curCheckpointSlots, k)
	}

	posMax := make(map[int]int32, len(c.curSeqs))
	for i, seq := range batch.Sequences {
		pos := batch.Positions[i]
		if cur, ok := posMax[seq]; !ok || pos > cur {
			posMax[seq] = pos
		}
	}

	for i, seq := range c.curSeqs {
		pos, ok := posMax[seq]
		if !ok {
			continue
		}
		if pos < c.checkpointMinPos {
			continue
		}
		slot := c.curSlots[i]
		store := c.checkpointStore(slot)
		lastPos := store.lastPos
		if lastPos < 0 || pos-lastPos >= c.checkpointInterval {
			c.curCheckpointPos[i] = pos
		}
	}
}

func (c *HybridCache) checkpointStore(slot int) *slotCheckpointStore {
	store, ok := c.checkpoints[slot]
	if ok {
		return store
	}
	store = newSlotCheckpointStore(c.checkpointCount)
	c.checkpoints[slot] = store
	return store
}

func (c *HybridCache) checkpointIndexForSlot(slot int, pos int32) int {
	if c.checkpointCount == 0 {
		return -1
	}
	if idx, ok := c.curCheckpointSlots[slot]; ok {
		return idx
	}
	store := c.checkpointStore(slot)
	idx := store.record(pos)
	if idx >= 0 {
		c.curCheckpointSlots[slot] = idx
	}
	return idx
}

func (c *HybridCache) hasCheckpoint(seq int, pos int32) bool {
	if pos <= 0 {
		return false
	}
	slot, ok := c.slotForSeq[seq]
	if !ok {
		return false
	}
	store, ok := c.checkpoints[slot]
	if !ok {
		return false
	}
	_, _, ok = store.bestIndex(pos)
	return ok
}

func (c *HybridCache) PrepareRestore(seq int, targetPos int32) (int32, bool) {
	if targetPos <= 0 {
		return 0, false
	}
	slot, ok := c.slotForSeq[seq]
	if !ok {
		return 0, false
	}
	store, ok := c.checkpoints[slot]
	if !ok {
		slog.Debug("qwen3next: checkpoint miss", "seq", seq, "slot", slot, "target", targetPos, "size", 0)
		return 0, false
	}
	idx, pos, ok := store.bestIndex(targetPos)
	if !ok {
		size, minPos, maxPos, lastPos := store.window()
		slog.Debug("qwen3next: checkpoint miss", "seq", seq, "slot", slot, "target", targetPos, "size", size,
			"min", minPos, "max", maxPos, "last", lastPos)
		return 0, false
	}
	c.pendingRestore[seq] = checkpointRestore{
		slot: slot,
		idx:  idx,
		pos:  pos,
	}
	return pos + 1, true
}

func (c *HybridCache) applyCheckpointRestore(restore checkpointRestore) error {
	entry, ok := c.restoreEntry(restore)
	if !ok {
		return kvcache.ErrNotSupported
	}

	ctx := c.backend.NewContext()
	defer ctx.Close()

	slotIdx := ctx.Input().FromInts([]int32{int32(restore.slot)}, 1)
	for layer, src := range entry.conv {
		buf := c.convBuffer(ctx, layer)
		ctx.Forward(buf.SetRows(ctx, src, slotIdx))
	}
	for layer, src := range entry.delta {
		buf := c.deltaBuffer(ctx, layer)
		ctx.Forward(buf.SetRows(ctx, src, slotIdx))
	}

	if len(entry.conv) > 0 || len(entry.delta) > 0 {
		ctx.Compute()
	}
	store := c.checkpoints[restore.slot]
	store.pruneAfter(restore.pos)
	return nil
}

func (c *HybridCache) restoreComplete(restore checkpointRestore) bool {
	_, ok := c.restoreEntry(restore)
	return ok
}

func (c *HybridCache) restoreEntry(restore checkpointRestore) (*checkpointEntry, bool) {
	store, ok := c.checkpoints[restore.slot]
	if !ok || restore.idx < 0 || restore.idx >= len(store.entries) {
		return nil, false
	}
	entry := &store.entries[restore.idx]
	if entry.pos < 0 {
		return nil, false
	}
	if !c.entryComplete(entry) {
		return nil, false
	}
	return entry, true
}

func (c *HybridCache) entryComplete(entry *checkpointEntry) bool {
	for layer := range c.convStates {
		if entry.conv == nil || entry.conv[layer] == nil {
			return false
		}
	}
	for layer := range c.deltaStates {
		if entry.delta == nil || entry.delta[layer] == nil {
			return false
		}
	}
	return true
}

func (c *HybridCache) clearCheckpoints(slot int) {
	if store, ok := c.checkpoints[slot]; ok {
		store.reset()
	}
}

func (c *HybridCache) copyCheckpoints(ctx ml.Context, srcSlot, dstSlot int) {
	if c.checkpointCount == 0 {
		return
	}
	srcStore, ok := c.checkpoints[srcSlot]
	if !ok || srcStore.size == 0 {
		return
	}
	dstStore := c.checkpointStore(dstSlot)
	dstStore.size = srcStore.size
	dstStore.next = srcStore.next
	dstStore.lastPos = srcStore.lastPos

	for i := range srcStore.entries {
		srcEntry := &srcStore.entries[i]
		dstEntry := &dstStore.entries[i]
		dstEntry.pos = srcEntry.pos
		if srcEntry.conv != nil {
			if dstEntry.conv == nil {
				dstEntry.conv = make(map[int]ml.Tensor)
			}
			for layer, src := range srcEntry.conv {
				dst := c.ensureCheckpointConv(layer, dstEntry)
				ctx.Forward(src.Copy(ctx, dst))
			}
		}
		if srcEntry.delta != nil {
			if dstEntry.delta == nil {
				dstEntry.delta = make(map[int]ml.Tensor)
			}
			for layer, src := range srcEntry.delta {
				dst := c.ensureCheckpointDelta(layer, dstEntry)
				ctx.Forward(src.Copy(ctx, dst))
			}
		}
	}
}

func (c *HybridCache) captureConvCheckpoint(ctx ml.Context, layer int, src ml.Tensor) {
	if c.checkpointCount == 0 {
		return
	}
	if c.reserveCheckpoints {
		c.reserveCheckpointConv(layer)
		return
	}
	if len(c.curCheckpointPos) == 0 {
		return
	}
	for i, pos := range c.curCheckpointPos {
		if pos < 0 {
			continue
		}
		slot := c.curSlots[i]
		idx := c.checkpointIndexForSlot(slot, pos)
		if idx < 0 {
			continue
		}
		entry := &c.checkpoints[slot].entries[idx]
		dst := c.ensureCheckpointConv(layer, entry)
		seqSlice := src.Slice(ctx, 1, i, i+1, 1)
		ctx.Forward(seqSlice.Copy(ctx, dst))
	}
}

func (c *HybridCache) captureDeltaCheckpoint(ctx ml.Context, layer int, src ml.Tensor) {
	if c.checkpointCount == 0 {
		return
	}
	if c.reserveCheckpoints {
		c.reserveCheckpointDelta(layer)
		return
	}
	if len(c.curCheckpointPos) == 0 {
		return
	}
	for i, pos := range c.curCheckpointPos {
		if pos < 0 {
			continue
		}
		slot := c.curSlots[i]
		idx := c.checkpointIndexForSlot(slot, pos)
		if idx < 0 {
			continue
		}
		entry := &c.checkpoints[slot].entries[idx]
		dst := c.ensureCheckpointDelta(layer, entry)
		seqSlice := src.Slice(ctx, 1, i, i+1, 1)
		ctx.Forward(seqSlice.Copy(ctx, dst))
	}
}

func (c *HybridCache) ensureCheckpointConv(layer int, entry *checkpointEntry) ml.Tensor {
	if entry.conv == nil {
		entry.conv = make(map[int]ml.Tensor)
	}
	if t, ok := entry.conv[layer]; ok {
		return t
	}
	ctx, ok := c.checkpointConvCtxs[layer]
	if !ok {
		ctx = c.backend.NewContextSize(c.checkpointCtxSize).Layer(layer)
		c.checkpointConvCtxs[layer] = ctx
	}
	t := ctx.Zeros(ml.DTypeF32, c.convDim*c.convChannels, 1)
	entry.conv[layer] = t
	return t
}

func (c *HybridCache) ensureCheckpointDelta(layer int, entry *checkpointEntry) ml.Tensor {
	if entry.delta == nil {
		entry.delta = make(map[int]ml.Tensor)
	}
	if t, ok := entry.delta[layer]; ok {
		return t
	}
	ctx, ok := c.checkpointDeltaCtxs[layer]
	if !ok {
		ctx = c.backend.NewContextSize(c.checkpointCtxSize).Layer(layer)
		c.checkpointDeltaCtxs[layer] = ctx
	}
	t := ctx.Zeros(ml.DTypeF32, c.deltaStateSize, 1)
	entry.delta[layer] = t
	return t
}

func (c *HybridCache) reserveCheckpointConv(layer int) {
	key := checkpointReserveKey(layer, 0)
	if _, ok := c.checkpointReserved[key]; ok {
		return
	}
	for slot := range c.maxSequences {
		store := c.checkpointStore(slot)
		for i := range store.entries {
			entry := &store.entries[i]
			_ = c.ensureCheckpointConv(layer, entry)
		}
	}
	c.checkpointReserved[key] = struct{}{}
}

func (c *HybridCache) reserveCheckpointDelta(layer int) {
	key := checkpointReserveKey(layer, 1)
	if _, ok := c.checkpointReserved[key]; ok {
		return
	}
	for slot := range c.maxSequences {
		store := c.checkpointStore(slot)
		for i := range store.entries {
			entry := &store.entries[i]
			_ = c.ensureCheckpointDelta(layer, entry)
		}
	}
	c.checkpointReserved[key] = struct{}{}
}

func checkpointReserveKey(layer int, kind int) int {
	return layer*2 + kind
}
