package tokenizerloader

import (
	"context"
	"sync"
)

// Keep a copy of the original opener so we can restore it.
var (
	origOpenVocabOnly = openVocabOnly
	testMu            sync.Mutex
)

// ResetForTest clears the LRU cache and restores openVocabOnly.
func ResetForTest() {
	testMu.Lock()
	defer testMu.Unlock()
	reset()
	openVocabOnly = origOpenVocabOnly
}

// SetOpenVocabOnlyForTest overrides the vocab-only opener.
func SetOpenVocabOnlyForTest(fn func(context.Context, string) (Tokenizer, error)) {
	testMu.Lock()
	defer testMu.Unlock()
	openVocabOnly = fn
}
