package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/cache"
)

func TestTrieKeysRoundTrip(t *testing.T) {
	tokens := []int32{1, 2, 3, 4, 5}
	for _, lookahead := range []int{0, 1} {
		keys := tokensToTrieKeys(tokens, lookahead)
		got := trieKeysToTokens(keys, lookahead)
		if len(got) != len(tokens) {
			t.Fatalf("lookahead=%d: len=%d want %d", lookahead, len(got), len(tokens))
		}
		for i := range tokens {
			if got[i] != tokens[i] {
				t.Fatalf("lookahead=%d: token[%d]=%d want %d", lookahead, i, got[i], tokens[i])
			}
		}
	}
}

func TestPrefillPersistEmptyCacheRejected(t *testing.T) {
	c := &prefixCache{
		caches: []cache.Cache{cache.NewKVCache()},
	}
	c.ensureRoot()
	if err := c.saveToPath(t.TempDir()); err == nil {
		t.Fatal("expected save to fail on empty cache")
	}
}
