package llamarunner

import (
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"time"

	"github.com/ollama/ollama/llama"
)

type InputCache struct {
	// context window size (per slot)
	numCtx int

	// individual KV caches
	slots []InputCacheSlot

	// optimize cache eviction for multiple users
	multiUserCache bool

	lc *llama.Context
}

func NewInputCache(lc *llama.Context, kvSize int, numSlots int, multiUserCache bool) (*InputCache, error) {
	if kvSize/numSlots < 1 {
		return nil, fmt.Errorf("must have at least one kv cache entry per parallel sequence (kv: %v parallel: %v)", kvSize, numSlots)
	}

	slots := make([]InputCacheSlot, numSlots)

	for i := range slots {
		slots[i] = InputCacheSlot{
			Id:     i,
			Inputs: make([]input, 0),
		}
	}

	return &InputCache{
		numCtx:         kvSize / numSlots,
		slots:          slots,
		multiUserCache: multiUserCache,
		lc:             lc,
	}, nil
}

// Locking: Operations on InputCacheSlot (including finding one
// through LoadCacheSlot) require a lock to be held that serializes
// these operations with each other and llama.Decode

type InputCacheSlot struct {
	// Index in the KV cache
	Id int

	// Inputs that are stored in the KV cache
	Inputs []input

	// is this cache actively being processed as part of a sequence?
	InUse bool

	// last time this cache was used (as of start of processing)
	lastUsed time.Time
}

func (c *InputCache) LoadCacheSlot(prompt []input, cachePrompt bool) (*InputCacheSlot, []input, error) {
	var slot *InputCacheSlot
	var numPast int
	var err error

	// In single-user scenarios, the longest cache slot works fine for getting good input
	// cache hit rates and it reuses the same VRAM over and over again, which is good for
	// GPU performance in situations where we miss the input cache.
	// For multiple users, the "best" cache slot produces better input cache hit rates
	// at the cost of worse performance when we miss the input cache (because it causes
	// GPU L2 cache misses due to spreading out accesses across VRAM).
	if !c.multiUserCache {
		slot, numPast, err = c.findLongestCacheSlot(prompt)
	} else {
		slot, numPast, err = c.findBestCacheSlot(prompt)
	}
	if err != nil {
		return nil, nil, err
	}

	if !cachePrompt {
		numPast = 0
	}

	slot.InUse = true
	slot.lastUsed = time.Now()

	if numPast == len(prompt) {
		// Leave one input to sample so we can get a response
		numPast--
	}

	if !c.lc.KvCacheSeqRm(slot.Id, numPast, -1) {
		// Some models don't support partial erasure
		c.lc.KvCacheSeqRm(slot.Id, 0, -1)
		numPast = 0
	}

	slog.Debug("loading cache slot", "id", slot.Id, "cache", len(slot.Inputs), "prompt", len(prompt),
		"used", numPast, "remaining", len(prompt)-numPast)

	slot.Inputs = prompt[:numPast]
	prompt = prompt[numPast:]

	return slot, prompt, nil
}

func (c *InputCache) findLongestCacheSlot(prompt []input) (*InputCacheSlot, int, error) {
	longest := -1
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		if s.InUse {
			continue
		}

		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}
	}

	if longestSlot == nil {
		return nil, 0, errors.New("no available cache slots")
	}

	return longestSlot, longest, nil
}

func (c *InputCache) findBestCacheSlot(prompt []input) (*InputCacheSlot, int, error) {
	oldest := time.Now()
	var oldestSlot *InputCacheSlot

	longest := -1
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}

		if s.lastUsed.Compare(oldest) < 0 && !s.InUse {
			oldest = s.lastUsed
			oldestSlot = &c.slots[i]
		}
	}

	if longest == len(longestSlot.Inputs) && !longestSlot.InUse {
		return longestSlot, longest, nil
	}

	if oldestSlot.InUse {
		return nil, 0, errors.New("no available cache slots")
	}

	if len(oldestSlot.Inputs) != 0 {
		slog.Debug("evicting cache slot", "id", oldestSlot.Id, "inputs", len(oldestSlot.Inputs),
			"used", oldestSlot.lastUsed)
	}

	if longest > 0 && longestSlot != oldestSlot {
		slog.Debug("forking cache slot", "src", longestSlot.Id, "dst", oldestSlot.Id, "inputs", longest, "total",
			len(longestSlot.Inputs))
		oldestSlot.Inputs = make([]input, longest)
		copy(oldestSlot.Inputs, longestSlot.Inputs[:longest])
		// This is only nil for unit tests
		if c.lc != nil {
			c.lc.KvCacheSeqRm(oldestSlot.Id, 0, -1)
			c.lc.KvCacheSeqCp(longestSlot.Id, oldestSlot.Id, 0, longest)
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

func (c *InputCache) ShiftDiscard(inputLen int, numKeep int) int {
	targetFree := (c.numCtx - numKeep) / 2
	targetFree = max(targetFree, 1)

	currentFree := c.numCtx - inputLen

	return max(targetFree-currentFree, 0)
}

type ErrReprocessInputs struct {
	Inputs []input
}

func (e *ErrReprocessInputs) Error() string {
	return fmt.Sprintf("kv cache shift not supported, inputs need reprocessing (input count: %v)", len(e.Inputs))
}

// ShiftCacheSlot frees up space in the KV cache by deleting the oldest half of history
// and shifting the newest half into that space (saving numKeep inputs at the beginning).
//
// Assumes that at least 1 entry can be freed up by shifting (i.e. numKeep < numCtx)
func (c *InputCache) ShiftCacheSlot(slot *InputCacheSlot, numKeep int) error {
	if numKeep >= c.numCtx {
		return fmt.Errorf("unable to shift context - keep exceeds context (keep: %v context: %v)", numKeep, c.numCtx)
	}

	inputLen := len(slot.Inputs)
	discard := c.ShiftDiscard(inputLen, numKeep)

	if discard <= 0 {
		return nil
	}

	slog.Debug("context limit hit - shifting", "id", slot.Id, "limit", c.numCtx, "input", len(slot.Inputs),
		"keep", numKeep, "discard", discard)

	var shiftFailed bool

	if c.lc.KvCacheCanShift() {
		// For models that support shifting, attempt to shift the KV cache
		if !c.lc.KvCacheSeqRm(slot.Id, numKeep, numKeep+discard) {
			shiftFailed = true
			slog.Debug("kv cache removal not supported, clearing cache and returning inputs for reprocessing", "id", slot.Id)
		} else {
			c.lc.KvCacheSeqAdd(slot.Id, numKeep+discard, inputLen, -discard)
		}
	} else {
		// For models that don't support shifting
		shiftFailed = true
		slog.Debug("kv cache cannot shift, clearing cache and returning inputs for reprocessing", "id", slot.Id)
	}

	if shiftFailed {
		// Create new input slice with preserved tokens (numKeep + remaining tokens after discard)
		newInputs := make([]input, numKeep+inputLen-(numKeep+discard))
		copy(newInputs[:numKeep], slot.Inputs[:numKeep])
		copy(newInputs[numKeep:], slot.Inputs[numKeep+discard:])

		// Clear the entire KV cache
		_ = c.lc.KvCacheSeqRm(slot.Id, 0, -1)
		// Reset the slot inputs since we've cleared the cache
		slot.Inputs = []input{}

		// Return error with inputs that need to be reprocessed
		return &ErrReprocessInputs{Inputs: newInputs}
	}

	// Standard shift succeeded - update input array
	for i := numKeep + discard; i < inputLen; i++ {
		slot.Inputs[i-discard] = slot.Inputs[i]
	}
	slot.Inputs = slot.Inputs[:inputLen-discard]

	return nil
}
