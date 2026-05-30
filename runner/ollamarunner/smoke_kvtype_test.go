package ollamarunner

import (
	"bytes"
	"os"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

// tqCompatibleGGML returns a real GGML header with a TurboQuant-compatible
// head_dim (128) so SupportsKVCacheType can pass the head_dim gate for tq*
// presets.
func tqCompatibleGGML(t *testing.T) ggml.GGML {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := ggml.WriteGGUF(f, ggml.KV{
		"general.architecture":      "test",
		"test.attention.head_count": uint32(8),
		"test.embedding_length":     uint32(1024), // → head_dim = 1024 / 8 = 128
	}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Seek(0, 0); err != nil {
		t.Fatal(err)
	}
	buf, err := os.ReadFile(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	g, err := ggml.Decode(bytes.NewReader(buf), -1)
	if err != nil {
		t.Fatal(err)
	}
	return *g
}

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
		{"tq2v", ml.DTypeTQ2V, "tq2v", false, true},
		{"tq3v", ml.DTypeTQ3V, "tq3v", false, true},
		{"tq4v", ml.DTypeTQ4V, "tq4v", false, true},
	}

	g := tqCompatibleGGML(t)
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
	g := tqCompatibleGGML(t)
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

// TestSmokeRejectsTQUnsupportedHeadDim verifies the SupportsKVCacheType gate
// refuses tq* requests on models whose head_dim is outside {64,128,256,512}.
// Without this gate, fs/ggml.GraphSize would size the cache at tq* B/elem but
// the runtime would fall back to f16, overrunning the loader's budget.
func TestSmokeRejectsTQUnsupportedHeadDim(t *testing.T) {
	for _, headDim := range []uint32{96, 192, 256 + 1, 576} {
		t.Run(fmtHeadDim(headDim), func(t *testing.T) {
			f, err := os.CreateTemp(t.TempDir(), "*.gguf")
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			// head_count=1, embedding_length=headDim → head_dim = headDim
			if err := ggml.WriteGGUF(f, ggml.KV{
				"general.architecture":      "test",
				"test.attention.head_count": uint32(1),
				"test.embedding_length":     headDim,
			}, nil); err != nil {
				t.Fatal(err)
			}
			buf, err := os.ReadFile(f.Name())
			if err != nil {
				t.Fatal(err)
			}
			g, err := ggml.Decode(bytes.NewReader(buf), -1)
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"tq2", "tq3", "tq4", "tq2k", "tq3k", "tq4k", "tq2v", "tq3v", "tq4v"} {
				if g.SupportsKVCacheType(name) {
					t.Errorf("SupportsKVCacheType(%q) = true at head_dim=%d; must reject (TQ kernels only at 64/128/256/512)",
						name, headDim)
				}
			}
			// f16 / unset must still pass.
			for _, name := range []string{"", "f16"} {
				if !g.SupportsKVCacheType(name) {
					t.Errorf("SupportsKVCacheType(%q) = false at head_dim=%d; non-TQ types unaffected by gate",
						name, headDim)
				}
			}
		})
	}
}

func fmtHeadDim(d uint32) string {
	switch d {
	case 96:
		return "head96"
	case 192:
		return "head192"
	case 257:
		return "head257"
	case 576:
		return "head576_mla"
	}
	return "headX"
}
