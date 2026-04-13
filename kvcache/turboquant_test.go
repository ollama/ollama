package kvcache

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestTurboQuantPrepareRestoreUsesCanResume(t *testing.T) {
	w := NewTurboQuantWrapper(ml.DTypeTQ4)

	// Without Init, CanResume returns false (no sequences loaded)
	gotPos, gotOK := w.PrepareRestore(2, 11)
	if gotOK {
		t.Fatalf("PrepareRestore() = (%d, %v), want (0, false) when no data loaded", gotPos, gotOK)
	}
}

func TestTurboQuantMseBitsFromDType(t *testing.T) {
	tq2 := NewTurboQuantWrapper(ml.DTypeTQ2)
	if tq2.mseBits != 2 {
		t.Errorf("TQ2 mseBits = %d, want 2", tq2.mseBits)
	}

	tq3 := NewTurboQuantWrapper(ml.DTypeTQ3)
	if tq3.mseBits != 3 {
		t.Errorf("TQ3 mseBits = %d, want 3", tq3.mseBits)
	}

	tq4 := NewTurboQuantWrapper(ml.DTypeTQ4)
	if tq4.mseBits != 4 {
		t.Errorf("TQ4 mseBits = %d, want 4", tq4.mseBits)
	}
}

func TestTurboQuantSeedsSeparate(t *testing.T) {
	w := NewTurboQuantWrapper(ml.DTypeTQ4)

	if w.keySeedHi == w.valSeedHi || w.keySeedLo == w.valSeedLo {
		t.Errorf("key and value seeds should differ: key=(%x,%x) val=(%x,%x)",
			w.keySeedHi, w.keySeedLo, w.valSeedHi, w.valSeedLo)
	}
}

func TestTurboQuantSetConfigDisablesPermutedV(t *testing.T) {
	w := NewTurboQuantWrapper(ml.DTypeTQ4)

	config := ml.CacheConfig{PermutedV: true}
	w.SetConfig(config)

	// Internal cache should have PermutedV=false since it stores packed indices
	if w.cache.config != nil && w.cache.config.PermutedV {
		t.Error("cache should have PermutedV=false")
	}
}
