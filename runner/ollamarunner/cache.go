package ollamarunner

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"reflect"
	"time"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
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

func NewInputCache(model model.Model, kvCacheType string, kvSize int32, numSlots int, multiUserCache bool) (*InputCache, error) {
	if kvSize/int32(numSlots) < 1 {
		return nil, fmt.Errorf("must have at least one kv cache entry per parallel sequence (kv: %v parallel: %v)", kvSize, numSlots)
	}

	slots := make([]InputCacheSlot, numSlots)

	for i := range slots {
		slots[i] = InputCacheSlot{
			Id:     i,
			Inputs: make([]input, 0),
		}
	}

	cache := model.Config().Cache
	if cache != nil {
		cache.Init(model.Backend(), kvCacheTypeFromStr(kvCacheType), kvSize)
	}

	return &InputCache{
		numCtx:         kvSize / int32(numSlots),
		enabled:        cache != nil,
		slots:          slots,
		multiUserCache: multiUserCache,
		cache:          cache,
	}, nil
}

func kvCacheTypeFromStr(s string) ml.DType {
	switch s {
	case "q8_0":
		panic("kv cache quantization not yet implemented")
	case "q4_0":
		panic("kv cache quantization not yet implemented")
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
	Inputs []input

	// is this cache actively being processed as part of a sequence?
	InUse bool

	// last time this cache was used (as of start of processing)
	lastUsed time.Time
}

func (c *InputCache) LoadCacheSlot(prompt []input, cachePrompt bool) (*InputCacheSlot, []input, error) {
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

	if !cachePrompt {
		numPast = 0
	}

	slot.InUse = true
	slot.lastUsed = time.Now()

	if numPast == int32(len(prompt)) {
		// Leave one input to sample so we can get a response
		numPast--
	}

	if c.cache != nil {
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

func (c *InputCache) findLongestCacheSlot(prompt []input) (*InputCacheSlot, int32, error) {
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

func (c *InputCache) findBestCacheSlot(prompt []input) (*InputCacheSlot, int32, error) {
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
		oldestSlot.Inputs = make([]input, longest)
		copy(oldestSlot.Inputs, longestSlot.Inputs[:longest])
		if c.cache != nil {
			c.cache.CopyPrefix(longestSlot.Id, oldestSlot.Id, longest)
		}
	}

	return oldestSlot, longest, nil
}

func countCommonPrefix(a []input, b []input) int32 {
	var count int32

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

	// TODO (jessegross): KV cache removal can fail for certain types of models
	if c.cache != nil {
		err := c.cache.Remove(slot.Id, numKeep, numKeep+discard)
		if err != nil {
			return fmt.Errorf("unable to remove old kv cache entries (id: %v, keep: %v discard: %v): %w", slot.Id, numKeep, discard, err)
		}
	}

	for i := numKeep + discard; i < inputLen; i++ {
		slot.Inputs[i-discard] = slot.Inputs[i]
	}
	slot.Inputs = slot.Inputs[:inputLen-discard]

	return nil
}
