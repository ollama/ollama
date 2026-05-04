package ollamarunner

import (
	"testing"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

// TestSmokeAllKVCacheTypes walks every user-facing OLLAMA_KV_CACHE_TYPE
// value through the full runtime gate chain and asserts every layer
// agrees.
func TestSmokeAllKVCacheTypes(t *testing.T) {
	cases := []struct {
		name        string
		wantDType   ml.DType
		wantPreset  string // empty if not a TQ preset
		requiresFA  bool
		isQuantized bool
	}{
		{"f16", ml.DTypeF16, "", false, false},
		{"q8_0", ml.DTypeQ80, "", true, true},
		{"q4_0", ml.DTypeQ40, "", true, true},
		{"tq2", ml.DTypeTQ2, "tq2", true, true},
		{"tq3", ml.DTypeTQ3, "tq3", true, true},
		{"tq4", ml.DTypeTQ4, "tq4", true, true},
		{"tq2k", ml.DTypeTQ2K, "tq2k", false, true},
		{"tq3k", ml.DTypeTQ3K, "tq3k", false, true},
		{"tq4k", ml.DTypeTQ4K, "tq4k", false, true},
	}

	var g ggml.GGML
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !g.SupportsKVCacheType(tc.name) {
				t.Fatalf("SupportsKVCacheType(%q) = false", tc.name)
			}
			if got := g.KVCacheTypeIsQuantized(tc.name); got != tc.isQuantized {
				t.Errorf("KVCacheTypeIsQuantized(%q) = %v, want %v", tc.name, got, tc.isQuantized)
			}
			if got := g.KVCacheTypeRequiresFlashAttention(tc.name); got != tc.requiresFA {
				t.Errorf("KVCacheTypeRequiresFlashAttention(%q) = %v, want %v", tc.name, got, tc.requiresFA)
			}
			dtype := kvCacheTypeFromStr(tc.name)
			if dtype != tc.wantDType {
				t.Errorf("kvCacheTypeFromStr(%q) = %v, want %v", tc.name, dtype, tc.wantDType)
			}
			preset, isTQ := kvcache.PresetFromDType(dtype)
			if tc.wantPreset == "" {
				if isTQ {
					t.Errorf("PresetFromDType(%v) unexpectedly returned a TQ preset", dtype)
				}
				return
			}
			if !isTQ {
				t.Fatalf("PresetFromDType(%v) returned (_, false) for TQ name %q", dtype, tc.name)
			}
			if preset.Name != tc.wantPreset {
				t.Errorf("PresetFromDType(%v) = %q, want %q", dtype, preset.Name, tc.wantPreset)
			}
			byName, err := turboquant.PresetByName(tc.name)
			if err != nil {
				t.Fatalf("PresetByName(%q): %v", tc.name, err)
			}
			if byName.ID != preset.ID {
				t.Errorf("PresetByName(%q).ID = %d, PresetFromDType.ID = %d — gate-chain disagreement",
					tc.name, byName.ID, preset.ID)
			}
		})
	}
}

// TestSmokeRejectsUnknown verifies we don't accidentally widen acceptance.
func TestSmokeRejectsUnknown(t *testing.T) {
	var g ggml.GGML
	for _, name := range []string{
		"tq3a", "tq3ka", "tq3q", "tq3qa",
		"tq2a", "tq2ka", "tq2q", "tq2qa",
		"tq4a", "tq4ka", "tq4qa",
		"q8k", "q8kv", "q4k", "q4kv",
		"banana",
	} {
		if g.SupportsKVCacheType(name) {
			t.Errorf("SupportsKVCacheType(%q) = true; internal-only / invalid names must reject", name)
		}
		if _, err := turboquant.PresetByName(name); err == nil {
			t.Errorf("PresetByName(%q) accepted; must reject", name)
		}
	}
}
