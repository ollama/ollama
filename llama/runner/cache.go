package main

import (
	"errors"
	"hash/maphash"
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

	// cache of images to embeddings
	images    []imageCache
	imageHash maphash.Hash

	lc *llama.Context
}

func NewInputCache(lc *llama.Context, kvSize int, numSlots int, multiUserCache bool) *InputCache {
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
		images:         make([]imageCache, numSlots),
		lc:             lc,
	}
}

// Locking: Operations on InputCacheSlot (including finding one
// through LoadCacheSlot) require a lock to be be held that serializes
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

func (c *InputCache) LoadCacheSlot(prompt []input, cachePrompt bool) (*InputCacheSlot, []input, int, error) {
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
		return nil, nil, 0, err
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

	prompt = prompt[numPast:]
	slot.Inputs = slot.Inputs[:numPast]

	return slot, prompt, numPast, nil
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

func (c *InputCache) ShiftCacheSlot(slot *InputCacheSlot, numKeep int, numDiscard int, numPast int) {
	// TODO (jessegross): KV cache removal can fail for certain types of models
	// server.cpp doesn't handle this, though we can be more graceful
	c.lc.KvCacheSeqRm(slot.Id, numKeep, numKeep+numDiscard)
	c.lc.KvCacheSeqAdd(slot.Id, numKeep+numDiscard, numPast, -numDiscard)

	for i := numKeep + numDiscard; i < len(slot.Inputs); i++ {
		slot.Inputs[i-numDiscard] = slot.Inputs[i]
	}
	slot.Inputs = slot.Inputs[:len(slot.Inputs)-numDiscard]
}

// Locking: Lookup and store operations on imageCache require a lock
// to be held that serializes these with each other. Hash does not
// require a lock nor they need to be serialized with InputCacheSlot.

type imageCache struct {
	key      uint64
	val      [][]float32
	lastUsed time.Time
}

func (c *InputCache) HashImage(image []byte) uint64 {
	c.imageHash.Reset()
	_, _ = c.imageHash.Write(image)
	return c.imageHash.Sum64()
}

var ErrImageNotFound = errors.New("image not found in cache")

func (c *InputCache) FindImage(hash uint64) ([][]float32, error) {
	for i := range c.images {
		if c.images[i].key == hash {
			slog.Debug("loading image embeddings from cache", "entry", i)
			c.images[i].lastUsed = time.Now()
			return c.images[i].val, nil
		}
	}

	return nil, ErrImageNotFound
}

func (c *InputCache) AddImage(hash uint64, embed [][]float32) {
	best := time.Now()
	var bestImage int

	for i := range c.images {
		if c.images[i].key == hash {
			bestImage = i
			break
		}

		if c.images[i].lastUsed.Compare(best) < 0 {
			best = c.images[i].lastUsed
			bestImage = i
		}
	}

	slog.Debug("storing image embeddings in cache", "entry", bestImage, "used", c.images[bestImage].lastUsed)
	c.images[bestImage].key = hash
	c.images[bestImage].val = embed
	c.images[bestImage].lastUsed = time.Now()
}
