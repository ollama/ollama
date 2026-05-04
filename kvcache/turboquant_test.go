package kvcache

import (
	"testing"

	"github.com/ollama/ollama/turboquant"
)

// TestWrapWithTurboQuantWrapperCache verifies Path C: when wrapping a
// WrapperCache that contains a SWA Causal + a plain Causal, TurboQuant
// replaces only the plain Causal sub-cache and leaves the SWA sub-cache
// untouched. This is the gemma3/gemma4 pattern.
func TestWrapWithTurboQuantWrapperCache(t *testing.T) {
	swa := NewSWAMemCache(1024, 4096, nil)
	global := NewCausalCache(nil)
	wc := NewWrapperCache(swa, global)

	wrapped, active := WrapWithTurboQuant(wc, turboquant.PresetTQ3K)

	if !active {
		t.Fatalf("expected active=true when wrapping a WrapperCache with a plain Causal sub-cache")
	}
	// The same WrapperCache pointer should be returned (mutated in place).
	if wrapped != wc {
		t.Fatalf("expected the same WrapperCache pointer to be returned")
	}
	if len(wc.caches) != 2 {
		t.Fatalf("expected 2 sub-caches, got %d", len(wc.caches))
	}
	// SWA sub-cache untouched.
	if swaSub, ok := wc.caches[0].(*Causal); !ok || swaSub != swa {
		t.Fatalf("expected wc.caches[0] to remain the original SWA Causal, got %T", wc.caches[0])
	}
	// Global sub-cache replaced with *TurboQuantCache wrapping the original.
	tqc, ok := wc.caches[1].(*TurboQuantCache)
	if !ok {
		t.Fatalf("expected wc.caches[1] to be *TurboQuantCache, got %T", wc.caches[1])
	}
	if tqc.meta != global {
		t.Fatalf("expected TurboQuantCache.meta to point at the original inner Causal")
	}
	if tqc.preset.Name != turboquant.PresetTQ3K.Name {
		t.Fatalf("expected preset %q, got %q", turboquant.PresetTQ3K.Name, tqc.preset.Name)
	}
}

// TestWrapWithTurboQuantSWAOnly verifies that wrapping a WrapperCache
// containing only SWA sub-caches yields active=false and leaves the
// cache unchanged.
func TestWrapWithTurboQuantSWAOnly(t *testing.T) {
	swa := NewSWAMemCache(1024, 4096, nil)
	wc := NewWrapperCache(swa)

	wrapped, active := WrapWithTurboQuant(wc, turboquant.PresetTQ3K)

	if active {
		t.Fatalf("expected active=false for a SWA-only WrapperCache")
	}
	if wrapped != wc {
		t.Fatalf("expected the same WrapperCache pointer to be returned")
	}
	// Sub-cache must remain the original SWA Causal.
	if sub, ok := wc.caches[0].(*Causal); !ok || sub != swa {
		t.Fatalf("expected wc.caches[0] unchanged, got %T", wc.caches[0])
	}
}

// TestWrapWithTurboQuantTopLevelSWA verifies that a top-level SWA Causal
// cannot be wrapped — TurboQuant needs full-context Causal semantics.
func TestWrapWithTurboQuantTopLevelSWA(t *testing.T) {
	swa := NewSWAMemCache(1024, 4096, nil)

	wrapped, active := WrapWithTurboQuant(swa, turboquant.PresetTQ3K)

	if active {
		t.Fatalf("expected active=false for a top-level SWA Causal")
	}
	if wrapped != swa {
		t.Fatalf("expected the SWA Causal to be returned unchanged")
	}
}

// TestWrapWithTurboQuantTopLevelCausal verifies the existing top-level
// non-SWA Causal case still works: a new *TurboQuantCache is returned
// wrapping the input.
func TestWrapWithTurboQuantTopLevelCausal(t *testing.T) {
	c := NewCausalCache(nil)

	wrapped, active := WrapWithTurboQuant(c, turboquant.PresetTQ3K)

	if !active {
		t.Fatalf("expected active=true for a top-level plain Causal")
	}
	tqc, ok := wrapped.(*TurboQuantCache)
	if !ok {
		t.Fatalf("expected *TurboQuantCache, got %T", wrapped)
	}
	if tqc.meta != c {
		t.Fatalf("expected TurboQuantCache.meta to point at the input Causal")
	}
}

// TestWrapWithTurboQuantHybridCache verifies that a cache embedding *Recurrent
// (matching the model/models/*/HybridCache shape) has its inner *Causal swapped
// for a *TurboQuantCache in place, and the outer pointer is returned unchanged.
func TestWrapWithTurboQuantHybridCache(t *testing.T) {
	type HybridCache struct{ *Recurrent }
	r := NewRecurrentCache(RecurrentConfig{ConvDim: 4, ConvChannels: 2, RecurrentStateSize: 4})
	hc := &HybridCache{Recurrent: r}

	wrapped, active := WrapWithTurboQuant(hc, turboquant.PresetTQ3K)

	if !active {
		t.Fatal("expected active=true for AttentionKVWrapper")
	}
	if wrapped != hc {
		t.Fatalf("expected same pointer returned, got %T", wrapped)
	}
	tqc, ok := r.kv.(*TurboQuantCache)
	if !ok {
		t.Fatalf("expected inner kv to be *TurboQuantCache, got %T", r.kv)
	}
	if tqc.preset.Name != turboquant.PresetTQ3K.Name {
		t.Fatalf("preset mismatch: got %q", tqc.preset.Name)
	}
}

// TestWrapWithTurboQuantHybridCacheIdempotent verifies that a second wrap attempt
// on an already-wrapped hybrid cache returns (cache, false) rather than double-wrapping.
func TestWrapWithTurboQuantHybridCacheIdempotent(t *testing.T) {
	type HybridCache struct{ *Recurrent }
	r := NewRecurrentCache(RecurrentConfig{ConvDim: 4, ConvChannels: 2, RecurrentStateSize: 4})
	hc := &HybridCache{Recurrent: r}

	_, _ = WrapWithTurboQuant(hc, turboquant.PresetTQ3K)
	wrapped, active := WrapWithTurboQuant(hc, turboquant.PresetTQ3K)

	if active {
		t.Fatal("expected active=false on second wrap (already wrapped)")
	}
	if wrapped != hc {
		t.Fatalf("expected same pointer returned, got %T", wrapped)
	}
}
