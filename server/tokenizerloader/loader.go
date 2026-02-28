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
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/llama"
)

// Public interface the routes will use.
type Tokenizer interface {
	Tokenize(text string) ([]int, error)
	Detokenize(tokens []int) (string, error)
	Close() error
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
	model *llama.Model
}

// Sentinel error for vocab-only unavailability
var ErrVocabOnlyUnavailable = errors.New("vocab-only unavailable")

// MUST be a var so tests can override it.
var openVocabOnly = func(ctx context.Context, model string) (Tokenizer, error) {
	return nil, ErrVocabOnlyUnavailable
}

func (m *vocabOnlyModel) Close() error {
	if m.model != nil {
		llama.FreeModel(m.model)
		m.model = nil
	}
	return nil
}

func (m *vocabOnlyModel) Tokenize(text string) ([]int, error) {
	if m.model == nil {
		return nil, fmt.Errorf("model not loaded")
	}

	// Use the existing Tokenize method from the llama package
	tokens, err := m.model.Tokenize(text, false, false) // no special tokens, no parsing
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %w", err)
	}

	return tokens, nil
}

func (m *vocabOnlyModel) Detokenize(tokens []int) (string, error) {
	if m.model == nil {
		return "", fmt.Errorf("model not loaded")
	}

	// Use the TokenToPiece method to convert each token back to text
	var result strings.Builder

	for i, token := range tokens {
		// Get the text piece for this token
		piece := m.model.TokenToPiece(token)

		// Handle special cases
		if piece == "" {
			// Fallback: try to reconstruct from the token ID
			if i > 0 {
				result.WriteString(" ")
			}
			result.WriteString(fmt.Sprintf("[token_%d]", token))
		} else {
			// Add space between tokens if needed (most tokenizers add leading space)
			if i > 0 && !strings.HasPrefix(piece, " ") {
				result.WriteString(" ")
			}
			result.WriteString(piece)
		}
	}

	return result.String(), nil
}

// Helper function to find model path
func findModelPath(modelName string) (string, error) {
	// This is a simplified implementation - in practice, you'd want to use
	// the existing model loading infrastructure from the server
	// For now, we'll assume the model is in a standard location

	// Check common model directories
	modelDirs := []string{
		"./models",
		"~/.ollama/models",
		"/usr/local/share/ollama/models",
	}

	for _, dir := range modelDirs {
		// Expand ~ to home directory
		if strings.HasPrefix(dir, "~") {
			home, err := os.UserHomeDir()
			if err != nil {
				continue
			}
			dir = filepath.Join(home, dir[1:])
		}

		// Look for .gguf files
		pattern := filepath.Join(dir, modelName, "*.gguf")
		matches, err := filepath.Glob(pattern)
		if err != nil {
			continue
		}

		if len(matches) > 0 {
			return matches[0], nil
		}
	}

	return "", fmt.Errorf("model %s not found in standard locations", modelName)
}

/* -------------------- LRU cache for vocab-only models -------------------- */

type cacheEntry struct {
	key   string
	model Tokenizer
}

type lruCache struct {
	mu       sync.Mutex
	ll       *list.List
	items    map[string]*list.Element
	capacity int
	ttl      time.Duration
	// optional timestamps per key if you want TTL eviction
}

// ---- cache + reset plumbing (names can match your existing types) ----

var (
	cacheMu sync.Mutex
	cache   *lruCache
	mu      sync.Mutex
)

const defaultCapacity = 8

func init() {
	initCache()
}

func initCache() {
	cacheMu.Lock()
	defer cacheMu.Unlock()
	cache = newLRU(defaultCapacity)
}

// unexported reset that tests can call via a test helper
func reset() {
	// Create a new cache instance without locking to avoid deadlock
	cacheMu.Lock()
	defer cacheMu.Unlock()
	cache = newLRU(defaultCapacity)
}

func newLRU(capacity int) *lruCache {
	return &lruCache{
		ll:       list.New(),
		items:    make(map[string]*list.Element),
		capacity: capacity,
		ttl:      30 * time.Minute, // soft TTL; can be refined
	}
}

func (c *lruCache) get(key string) Tokenizer {
	c.mu.Lock()
	defer c.mu.Unlock()
	if ele, ok := c.items[key]; ok {
		c.ll.MoveToFront(ele)
		return ele.Value.(*cacheEntry).model
	}
	return nil
}

func (c *lruCache) add(key string, m Tokenizer) {
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
	if m := cache.get(modelName); m != nil {
		if os.Getenv("OLLAMA_TOKENIZER_DEBUG") == "1" {
			slog.Debug("tokenizer: vocab-only cache hit", "model", modelName)
		}
		return m, nil
	}

	if os.Getenv("OLLAMA_TOKENIZER_DEBUG") == "1" {
		slog.Debug("tokenizer: attempting vocab-only load", "model", modelName)
	}

	m, err := openVocabOnly(context.Background(), modelName)
	if err != nil {
		if err == ErrVocabOnlyUnavailable {
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

	cache.add(modelName, m)
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

func (f *fallbackTokenizer) Close() error {
	// No cleanup needed for fallback tokenizer
	return nil
}
