package main

import (
	"errors"
	"log/slog"
	"reflect"
	"time"

	"github.com/ollama/ollama/llama"
)

type InputCache struct {
	// context window size (per slot)
	numCtx int

	// individual caches
	slots []InputCacheSlot

	lc *llama.Context
}

func NewInputCache(lc *llama.Context, kvSize int, numSlots int) *InputCache {
	slots := make([]InputCacheSlot, numSlots)

	for i := range slots {
		slots[i] = InputCacheSlot{
			id:     i,
			inputs: make([]input, 0),
		}
	}

	return &InputCache{
		numCtx: kvSize / numSlots,
		slots:  slots,
		lc:     lc,
	}
}

// Locking: Operations on InputCacheSlot (including finding one
// through LoadCacheSlot) require a lock to be be held that serializes
// these operations with each other and llama.Decode

type InputCacheSlot struct {
	// Index in the KV cache
	id int

	// inputs that are stored in the KV cache
	inputs []input

	// is this cache actively being processed as part of a sequence?
	inUse bool

	// last time this cache was used (as of start of processing)
	lastUsed time.Time
}

func (c *InputCache) LoadCacheSlot(prompt []input) (*InputCacheSlot, []input, int, error) {
	slot, numPast, err := c.findCacheSlot(prompt)
	if err != nil {
		return nil, nil, 0, err
	}

	slot.inUse = true
	slot.lastUsed = time.Now()

	if numPast == len(prompt) {
		// Leave one input to sample so we can get a response
		numPast--
	}

	if !c.lc.KvCacheSeqRm(slot.id, numPast, -1) {
		// Some models don't support partial erasure
		c.lc.KvCacheSeqRm(slot.id, 0, -1)
		numPast = 0
	}

	slog.Debug("loading cache slot", "id", slot.id, "cache", len(slot.inputs), "prompt", len(prompt),
		"used", numPast, "remaining", len(prompt)-numPast)

	prompt = prompt[numPast:]
	slot.inputs = slot.inputs[:numPast]

	return slot, prompt, numPast, nil
}

func (c *InputCache) findCacheSlot(prompt []input) (*InputCacheSlot, int, error) {
	oldest := time.Now()
	var oldestSlot *InputCacheSlot

	longest := -1
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		count := countCommonPrefix(s.inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}

		if s.lastUsed.Compare(oldest) < 0 && !s.inUse {
			oldest = s.lastUsed
			oldestSlot = &c.slots[i]
		}
	}

	if longest == len(longestSlot.inputs) && !longestSlot.inUse {
		return longestSlot, longest, nil
	}

	if oldestSlot.inUse {
		return nil, 0, errors.New("no available cache slots")
	}

	if len(oldestSlot.inputs) != 0 {
		slog.Debug("evicting cache slot", "id", oldestSlot.id, "inputs", len(oldestSlot.inputs),
			"used", oldestSlot.lastUsed)
	}

	if longest > 0 && longestSlot != oldestSlot {
		slog.Debug("forking cache slot", "src", longestSlot.id, "dst", oldestSlot.id, "inputs", longest, "total",
			len(longestSlot.inputs))
		oldestSlot.inputs = make([]input, longest)
		copy(oldestSlot.inputs, longestSlot.inputs[:longest])
		// This is only nil for unit tests
		if c.lc != nil {
			c.lc.KvCacheSeqRm(oldestSlot.id, 0, -1)
			c.lc.KvCacheSeqCp(longestSlot.id, oldestSlot.id, 0, longest)
		}
	}

	return oldestSlot, longest, nil
}

func countCommonPrefix(a []input, b []input) int {
	var count int

	for i := range a {
		if i >= len(b) {
			break
		}

		if !reflect.DeepEqual(a[i], b[i]) {
			break
		}

		count++
	}

	return count
}

func (c *InputCache) ShiftCacheSlot(slot *InputCacheSlot, numKeep int, numDiscard int, numPast int) {
	// TODO (jessegross): KV cache removal can fail for certain types of models
	// server.cpp doesn't handle this, though we can be more graceful
	c.lc.KvCacheSeqRm(slot.id, numKeep, numKeep+numDiscard)
	c.lc.KvCacheSeqAdd(slot.id, numKeep+numDiscard, numPast, -numDiscard)

	for i := numKeep + numDiscard; i < len(slot.inputs); i++ {
		slot.inputs[i-numDiscard] = slot.inputs[i]
	}
	slot.inputs = slot.inputs[:len(slot.inputs)-numDiscard]
}
