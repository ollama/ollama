package main

import (
	"log/slog"
	"time"

	"github.com/ollama/ollama/llama"
)

type TokenCache struct {
	// context window size (per slot)
	numCtx int

	// individual caches
	slots []TokenCacheSlot

	lc *llama.Context
}

func NewTokenCache(lc *llama.Context, kvSize int, numSlots int) *TokenCache {
	slots := make([]TokenCacheSlot, numSlots)

	for i := range slots {
		slots[i] = TokenCacheSlot{
			id:     i,
			tokens: make([]int, 0),
		}
	}

	return &TokenCache{
		numCtx: kvSize / numSlots,
		slots:  slots,
		lc:     lc,
	}
}

// Locking: Operations on TokenCacheSlot (including finding one
// through LoadCacheSlot) require a lock to be be held that serializes
// these operations with each other and llama.Decode

type TokenCacheSlot struct {
	// Index in the KV cache
	id int

	// tokens that are stored in the KV cache
	tokens []int

	// is this cache actively being processed as part of a sequence?
	inUse bool

	// last time this cache was used (as of start of processing)
	lastUsed time.Time
}

func (t *TokenCache) LoadCacheSlot(prompt []int) (*TokenCacheSlot, []int, int) {
	slot, numPast := t.findCacheSlot(prompt)

	slot.inUse = true
	slot.lastUsed = time.Now()

	if numPast == len(prompt) {
		// Leave one token to sample so we can get a response
		numPast--
	}

	if !t.lc.KvCacheSeqRm(slot.id, numPast, -1) {
		// Some models don't support partial erasure
		t.lc.KvCacheSeqRm(slot.id, 0, -1)
		numPast = 0
	}

	slog.Debug("loading cache slot", "id", slot.id, "cache", len(slot.tokens), "prompt", len(prompt),
		"used", numPast, "remaining", len(prompt)-numPast)

	prompt = prompt[numPast:]
	slot.tokens = slot.tokens[:numPast]

	return slot, prompt, numPast
}

func (t *TokenCache) findCacheSlot(prompt []int) (*TokenCacheSlot, int) {
	oldest := time.Now()
	var oldestSlot *TokenCacheSlot

	longest := -1
	var longestSlot *TokenCacheSlot

	for i, s := range t.slots {
		count := countCommonPrefix(s.tokens, prompt)
		if count > longest {
			longest = count
			longestSlot = &t.slots[i]
		}

		if s.lastUsed.Compare(oldest) < 0 && !s.inUse {
			oldest = s.lastUsed
			oldestSlot = &t.slots[i]
		}
	}

	if longest == len(longestSlot.tokens) && !longestSlot.inUse {
		return longestSlot, longest
	}

	if oldestSlot.inUse {
		panic("no available cache slots")
	}

	if len(oldestSlot.tokens) != 0 {
		slog.Debug("evicting cache slot", "id", oldestSlot.id, "tokens", len(oldestSlot.tokens),
			"used", oldestSlot.lastUsed)
	}

	if longest > 0 && longestSlot != oldestSlot {
		slog.Debug("forking cache slot", "src", longestSlot.id, "dst", oldestSlot.id, "tokens", longest, "total",
			len(longestSlot.tokens))
		oldestSlot.tokens = make([]int, longest)
		copy(oldestSlot.tokens, longestSlot.tokens[:longest])
		// This is only nil for unit tests
		if t.lc != nil {
			t.lc.KvCacheSeqCp(longestSlot.id, oldestSlot.id, 0, longest)
		}
	}

	return oldestSlot, longest
}

func countCommonPrefix(a []int, b []int) int {
	var count int

	for i := range a {
		if i >= len(b) {
			break
		}

		if a[i] != b[i] {
			break
		}

		count++
	}

	return count
}

func (t *TokenCache) ShiftCacheSlot(slot *TokenCacheSlot, numKeep int, numDiscard int, numPast int) {
	// TODO (jessegross): KV cache removal can fail for certain types of models
	// server.cpp doesn't handle this, though we can be more graceful
	t.lc.KvCacheSeqRm(slot.id, numKeep, numKeep+numDiscard)
	t.lc.KvCacheSeqAdd(slot.id, numKeep+numDiscard, numPast, -numDiscard)

	for i := numKeep + numDiscard; i < len(slot.tokens); i++ {
		slot.tokens[i-numDiscard] = slot.tokens[i]
	}
	slot.tokens = slot.tokens[:len(slot.tokens)-numDiscard]
}
