package ollamarunner

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"time"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type InputCache struct {
	// context window size (per slot)
	numCtx int32

	// does the cache store data or do we need to always send the full input?
	// note that when enabled is false the underlying cache may either be nil
	// or a non-nil dummy that doesn't actually store anything
	enabled bool

	// individual KV caches
	slots []InputCacheSlot

	// optimize cache eviction for multiple users
	multiUserCache bool

	cache kvcache.Cache
}

func NewInputCache(model model.Model, kvCacheType string, kvSize int32, numSlots int, batchSize int, multiUserCache bool) (*InputCache, error) {
	numCtx := kvSize / int32(numSlots)

	if numCtx < 1 {
		return nil, fmt.Errorf("must have at least one kv cache entry per parallel sequence (kv: %v parallel: %v)", kvSize, numSlots)
	}

	slots := make([]InputCacheSlot, numSlots)

	for i := range slots {
		slots[i] = InputCacheSlot{Id: i}
	}

	cache := model.Config().Cache
	if cache != nil {
		cache.Init(model.Backend(), kvCacheTypeFromStr(kvCacheType), numSlots, int(numCtx), batchSize)
	}

	return &InputCache{
		numCtx:         numCtx,
		enabled:        cache != nil,
		slots:          slots,
		multiUserCache: multiUserCache,
		cache:          cache,
	}, nil
}

func kvCacheTypeFromStr(s string) ml.DType {
	switch s {
	case "q8_0":
		return ml.DTypeQ80
	case "q4_0":
		return ml.DTypeQ40
	default:
		return ml.DTypeF16
	}
}

func (c *InputCache) Close() {
	c.cache.Close()
}

// Locking: Operations on InputCacheSlot (including finding one
// through LoadCacheSlot) require a lock to be be held that serializes
// these operations with each other and processBatch

type InputCacheSlot struct {
	// Index in the KV cache
	Id int

	// Inputs that are stored in the KV cache
	Inputs []input.Input

	// is this cache actively being processed as part of a sequence?
	InUse bool

	// last time this cache was used (as of start of processing)
	lastUsed time.Time
}

func (c *InputCache) LoadCacheSlot(prompt []input.Input) (*InputCacheSlot, []input.Input, error) {
	var slot *InputCacheSlot
	var numPast int32
	var err error

	// In single-user scenarios, the longest cache slot works fine for getting good input
	// cache hit rates and it keeps the footprint of the cache small, which improves throughput.
	// For multiple users, the "best" cache slot produces better input cache hit rates
	// at the cost of worse performance when we miss the input cache.
	if !c.multiUserCache {
		slot, numPast, err = c.findLongestCacheSlot(prompt)
	} else {
		slot, numPast, err = c.findBestCacheSlot(prompt)
	}
	if err != nil {
		return nil, nil, err
	}

	slot.InUse = true
	slot.lastUsed = time.Now()

	if numPast == int32(len(prompt)) {
		// Leave one input to sample so we can get a response
		numPast--
	}

	if c.cache != nil {
		if numPast > 0 && !c.cache.CanResume(slot.Id, numPast) {
			numPast = 0
		}

		err = c.cache.Remove(slot.Id, numPast, math.MaxInt32)
		if err != nil {
			// Some models don't support partial erasure
			err = c.cache.Remove(slot.Id, 0, math.MaxInt32)
			if err != nil {
				return nil, nil, err
			}
			numPast = 0
		}
	}

	slog.Debug("loading cache slot", "id", slot.Id, "cache", len(slot.Inputs), "prompt", len(prompt),
		"used", numPast, "remaining", int32(len(prompt))-numPast)

	prompt = prompt[numPast:]
	slot.Inputs = slot.Inputs[:numPast]

	return slot, prompt, nil
}

func (c *InputCache) findLongestCacheSlot(prompt []input.Input) (*InputCacheSlot, int32, error) {
	longest := int32(-1)
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

func (c *InputCache) findBestCacheSlot(prompt []input.Input) (*InputCacheSlot, int32, error) {
	oldest := time.Now()
	var oldestSlot *InputCacheSlot

	longest := int32(-1)
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

	if longest == int32(len(longestSlot.Inputs)) && !longestSlot.InUse {
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
		oldestSlot.Inputs = make([]input.Input, longest)
		copy(oldestSlot.Inputs, longestSlot.Inputs[:longest])
		if c.cache != nil {
			c.cache.CopyPrefix(longestSlot.Id, oldestSlot.Id, longest)
		}
	}

	return oldestSlot, longest, nil
}

func countCommonPrefix(a []input.Input, b []input.Input) int32 {
	var count int32

	for i := range a {
		if i >= len(b) {
			break
		}

		if a[i].Token != b[i].Token || a[i].MultimodalHash != b[i].MultimodalHash {
			break
		}

		count++
	}

	return count
}

// TODO(jessegross): If we need to reprocess the inputs we should ensure that
// we don't split up a SameBatch
func (c *InputCache) ShiftDiscard(inputLen int32, numKeep int32) int32 {
	targetFree := (c.numCtx - numKeep) / 2
	targetFree = max(targetFree, 1)

	currentFree := c.numCtx - inputLen
	discard := targetFree - currentFree

	if discard < 0 {
		discard = 0
	}

	return discard
}

type ErrReprocessInputs struct {
	Inputs []input.Input
}

func (e *ErrReprocessInputs) Error() string {
	return fmt.Sprintf("kv cache shift not supported, inputs need reprocessing (input count: %v)", len(e.Inputs))
}

// Frees up space in the KV cache by deleting the oldest half of history and shifting
// the newest half into that space (saving numKeep inputs at the beginning).
//
// Assumes that at least 1 entry can be freed up by shifting (i.e. numKeep < numCtx)
func (c *InputCache) ShiftCacheSlot(slot *InputCacheSlot, numKeep int32) error {
	if numKeep >= c.numCtx {
		return fmt.Errorf("unable to shift context - keep exceeds context (keep: %v context: %v)", numKeep, c.numCtx)
	}

	inputLen := int32(len(slot.Inputs))
	discard := c.ShiftDiscard(inputLen, numKeep)

	if discard <= 0 {
		return nil
	}

	slog.Debug("context limit hit - shifting", "id", slot.Id, "limit", c.numCtx, "input", len(slot.Inputs),
		"keep", numKeep, "discard", discard)

	if c.cache != nil {
		err := c.cache.Remove(slot.Id, numKeep, numKeep+discard)
		if err != nil {
			slog.Debug("kv cache removal unsupported, clearing cache and returning inputs for reprocessing",
				"id", slot.Id, "error", err)

			// Create new input slice with preserved tokens (numKeep + remaining tokens after discard)
			newInputs := make([]input.Input, numKeep+inputLen-(numKeep+discard))
			copy(newInputs[:numKeep], slot.Inputs[:numKeep])
			copy(newInputs[numKeep:], slot.Inputs[numKeep+discard:])

			// Reset the cache
			_ = c.cache.Remove(slot.Id, 0, -1)
			slot.Inputs = []input.Input{}

			// Return error with inputs that need to be reprocessed
			return &ErrReprocessInputs{Inputs: newInputs}
		}
	}

	for i := numKeep + discard; i < inputLen; i++ {
		slot.Inputs[i-discard] = slot.Inputs[i]
	}
	slot.Inputs = slot.Inputs[:inputLen-discard]

	return nil
}
