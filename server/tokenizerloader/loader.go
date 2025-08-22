// Package tokenizerloader provides a unified interface for model tokenization
// with vocab-only optimization and fallback support.
//
// Cache Policy:
// - Default capacity: 8 models
// - LRU eviction: least-recently used model tokenizer evicted
// - Vocab-only objects have small memory footprint relative to full model loads
// - TTL: 30 minutes (soft expiration)
package tokenizerloader

import (
	"container/list"
	"context"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"time"
)

// Public interface the routes will use.
type Tokenizer interface {
	Tokenize(text string) ([]int, error)
	Detokenize(tokens []int) (string, error)
}

// Factory: returns a Tokenizer for modelName (vocab-only if possible, else fallback).
// The bool return indicates if this is a fallback to scheduler (true = fallback, false = vocab-only).
func Get(ctx context.Context, modelName string) (Tokenizer, bool, error) {
	// try vocab-only first
	if t, err := getVocabOnly(modelName); err == nil && t != nil {
		return t, false, nil // vocab-only success
	}
	// fallback to scheduler-backed adapter
	t, err := newFallbackTokenizer(modelName)
	if err != nil {
		return nil, false, err
	}
	return t, true, nil // fallback success
}

/* -------------------- Vocab-only implementation -------------------- */

// NOTE: Wire this to the upstream llama.cpp bindings / internal API that can
// load *vocab only* without allocating GPU/VRAM for context weights.
//
// Stub interface we'll fill in as the internal hook becomes available.
type vocabOnlyModel struct {
	modelName string
	// add fields required by llama.cpp vocab-only open
}

// Sentinel error for vocab-only unavailability
var errVocabOnlyUnavailable = fmt.Errorf("vocab-only open not available")

// Replace these with real llama.cpp hooks when available.
func openVocabOnly(modelName string) (*vocabOnlyModel, error) {
	// TODO: call llama.cpp vocab-only API once available; see PR #8106 discussion
	return nil, errVocabOnlyUnavailable
}
func (m *vocabOnlyModel) Close() error { return nil }
func (m *vocabOnlyModel) Tokenize(text string) ([]int, error) {
	return nil, fmt.Errorf("vocab-only tokenize not implemented")
}
func (m *vocabOnlyModel) Detokenize(tokens []int) (string, error) {
	return "", fmt.Errorf("vocab-only detokenize not implemented")
}

/* -------------------- LRU cache for vocab-only models -------------------- */

type cacheEntry struct {
	key   string
	model *vocabOnlyModel
}

type lruCache struct {
	mu       sync.Mutex
	ll       *list.List
	items    map[string]*list.Element
	capacity int
	ttl      time.Duration
	// optional timestamps per key if you want TTL eviction
}

var (
	cacheOnce sync.Once
	cacheInst *lruCache
)

func cache() *lruCache {
	cacheOnce.Do(func() {
		cacheInst = &lruCache{
			ll:       list.New(),
			items:    make(map[string]*list.Element),
			capacity: 8,                // small default; adjust if needed
			ttl:      30 * time.Minute, // soft TTL; can be refined
		}
	})
	return cacheInst
}

func (c *lruCache) get(key string) *vocabOnlyModel {
	c.mu.Lock()
	defer c.mu.Unlock()
	if ele, ok := c.items[key]; ok {
		c.ll.MoveToFront(ele)
		return ele.Value.(*cacheEntry).model
	}
	return nil
}

func (c *lruCache) add(key string, m *vocabOnlyModel) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if ele, ok := c.items[key]; ok {
		c.ll.MoveToFront(ele)
		ele.Value.(*cacheEntry).model = m
		return
	}
	ele := c.ll.PushFront(&cacheEntry{key: key, model: m})
	c.items[key] = ele
	if c.ll.Len() > c.capacity {
		c.evictOldest()
	}
}

func (c *lruCache) evictOldest() {
	ele := c.ll.Back()
	if ele == nil {
		return
	}
	c.ll.Remove(ele)
	entry := ele.Value.(*cacheEntry)
	delete(c.items, entry.key)
	// free vocab-only model
	if entry.model != nil {
		_ = entry.model.Close()
	}
}

func getVocabOnly(modelName string) (Tokenizer, error) {
	if m := cache().get(modelName); m != nil {
		if os.Getenv("OLLAMA_TOKENIZER_DEBUG") == "1" {
			slog.Debug("tokenizer: vocab-only cache hit", "model", modelName)
		}
		return m, nil
	}

	if os.Getenv("OLLAMA_TOKENIZER_DEBUG") == "1" {
		slog.Debug("tokenizer: attempting vocab-only load", "model", modelName)
	}

	m, err := openVocabOnly(modelName)
	if err != nil {
		if err == errVocabOnlyUnavailable {
			if os.Getenv("OLLAMA_TOKENIZER_DEBUG") == "1" {
				slog.Debug("tokenizer: vocab-only unavailable, falling back to scheduler", "model", modelName)
			}
			return nil, err
		}
		return nil, err
	}

	if os.Getenv("OLLAMA_TOKENIZER_DEBUG") == "1" {
		slog.Debug("tokenizer: vocab-only load successful", "model", modelName)
	}

	cache().add(modelName, m)
	return m, nil
}

/* -------------------- Fallback (scheduler-backed) -------------------- */

// We keep fallback isolated so routes don't import scheduler types.
type fallbackTokenizer struct {
	modelName string
}

func newFallbackTokenizer(modelName string) (Tokenizer, error) {
	return &fallbackTokenizer{modelName: modelName}, nil
}

// These will be implemented via lightweight hooks you expose from server code
// (e.g., small funcs set at init time by routes.go) to call scheduleRunner.
// This avoids import cycles.
var (
	fallbackTokenizeFn   func(modelName, text string) ([]int, error)
	fallbackDetokenizeFn func(modelName string, tokens []int) (string, error)
)

func RegisterFallbackHooks(
	tokenize func(modelName, text string) ([]int, error),
	detokenize func(modelName string, tokens []int) (string, error),
) {
	fallbackTokenizeFn = tokenize
	fallbackDetokenizeFn = detokenize
}

func (f *fallbackTokenizer) Tokenize(text string) ([]int, error) {
	if fallbackTokenizeFn == nil {
		return nil, fmt.Errorf("fallback tokenize hook not set")
	}
	return fallbackTokenizeFn(f.modelName, text)
}

func (f *fallbackTokenizer) Detokenize(tokens []int) (string, error) {
	if fallbackDetokenizeFn == nil {
		return "", fmt.Errorf("fallback detokenize hook not set")
	}
	return fallbackDetokenizeFn(f.modelName, tokens)
}
