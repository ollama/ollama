package lfm2

import (
	"testing"

	"github.com/ollama/ollama/kvcache"
)

func TestHybridCache_New(t *testing.T) {
	cache := NewHybridCache(nil, 512, 2)
	if cache == nil {
		t.Fatal("expected cache to be created")
	}

	if cache.Recurrent == nil {
		t.Fatal("expected embedded recurrent cache to be created")
	}
}

func TestHybridCache_ImplementsCheckpointCache(t *testing.T) {
	cache := NewHybridCache(nil, 512, 2)

	if _, ok := any(cache).(kvcache.CheckpointCache); !ok {
		t.Fatal("expected HybridCache to implement CheckpointCache")
	}
}

func TestHybridCache_DefaultBatchState(t *testing.T) {
	cache := NewHybridCache(nil, 512, 2)

	if got := cache.numSeqs(); got != 0 {
		t.Fatalf("expected 0 sequences before StartForward, got %d", got)
	}

	if got := cache.seqTokens(); got != 0 {
		t.Fatalf("expected 0 sequence tokens before StartForward, got %d", got)
	}

	if cache.IsSupportedForBatch() {
		t.Fatal("expected unsupported batch layout before StartForward")
	}
}
