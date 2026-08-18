package server

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

func TestGGUFMetadataExtraction(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	big := make([]string, 5000)
	for i := range big {
		big[i] = "tok"
	}
	perLayer := make([]int32, 64)
	for i := range perLayer {
		perLayer[i] = int32(i)
	}

	path, digest := createBinFile(t, ggml.KV{
		"general.architecture":         "bert",
		"bert.attention.softcap":       float32(math.Inf(-1)),
		"general.file_type":            uint32(2),
		"bert.pooling_type":            uint32(1),
		"bert.context_length":          uint32(512),
		"bert.attention.head_count_kv": perLayer,
		"tokenizer.chat_template":      "{{ messages }}",
		"tokenizer.ggml.tokens":        big,
		"tokenizer.ggml.merges":        big,
	}, nil)

	md, err := extractGGUFMetadata(path)
	if err != nil {
		t.Fatal(err)
	}

	if got := md.String("general.architecture"); got != "bert" {
		t.Errorf("architecture = %q", got)
	}
	// Architecture-prefixed lookup, and the key order in the file must not matter.
	if !md.Valid("pooling_type") {
		t.Error("pooling_type not found")
	}
	if got := md.Int("context_length"); got != 512 {
		t.Errorf("context_length = %d, want 512", got)
	}
	if got := md.String("tokenizer.chat_template"); got != "{{ messages }}" {
		t.Errorf("chat_template = %q", got)
	}
	if md.Valid("vision.block_count") {
		t.Error("absent key reported present")
	}

	// A 64-element per-layer array is real metadata and must survive.
	if v, ok := md.KV["bert.attention.head_count_kv"].([]any); !ok || len(v) != 64 {
		t.Errorf("per-layer array = %#v, want 64 elements", md.KV["bert.attention.head_count_kv"])
	}

	// The tokenizer is recorded as omitted, not silently missing.
	want := map[string]bool{"tokenizer.ggml.tokens": true, "tokenizer.ggml.merges": true}
	for _, key := range md.Omitted {
		delete(want, key)
	}
	if len(want) != 0 {
		t.Errorf("not reported omitted: %v (omitted=%v)", want, md.Omitted)
	}
	if _, ok := md.KV["tokenizer.ggml.tokens"]; ok {
		t.Error("oversized array was copied into the metadata file")
	}

	// JSON cannot represent an infinity, and gemma3n ships one. Dropping that
	// key must not cost the rest of the file.
	if !slices.Contains(md.Omitted, "bert.attention.softcap") {
		t.Errorf("non-finite float not omitted; omitted=%v", md.Omitted)
	}

	_ = digest
}

func TestGGUFMetadataFileRoundTrip(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	blob, digest := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
		"llama.block_count":    uint32(32),
	}, nil)

	first, err := readGGUFMetadata(digest)
	if err != nil {
		t.Fatal(err)
	}
	if got := first.Int("block_count"); got != 32 {
		t.Fatalf("block_count = %d, want 32", got)
	}

	// With the blob gone, only the metadata file can answer.
	if err := os.Remove(blob); err != nil {
		t.Fatal(err)
	}
	second, err := readGGUFMetadata(digest)
	if err != nil {
		t.Fatalf("metadata file read failed: %v", err)
	}
	if got := second.Int("block_count"); got != 32 {
		t.Fatalf("block_count from metadata file = %d, want 32", got)
	}

	// And it goes away with the blob.
	removeGGUFMetadata(digest)
	path, err := ggufMetadataPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("metadata file still present after removal: %v", err)
	}
}

// The metadata file is only ever an optimization: anything unusable must be re-extracted.
func TestGGUFMetadataUnusableFile(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	_, digest := createBinFile(t, ggml.KV{"general.architecture": "llama"}, nil)
	if _, err := readGGUFMetadata(digest); err != nil {
		t.Fatal(err)
	}
	path, err := ggufMetadataPath(digest)
	if err != nil {
		t.Fatal(err)
	}

	for _, tt := range []struct {
		name    string
		content []byte
	}{
		{"truncated", []byte(`{"kv":{"general.arch`)},
		{"not json", []byte("\x00\x01\x02")},
		{"empty", []byte{}},
		{"wrong shape", []byte(`[1,2,3]`)},
		{"no kv", []byte(`{"ollama_version":"1.2.3"}`)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := os.WriteFile(path, tt.content, 0o644); err != nil {
				t.Fatal(err)
			}
			if _, ok := loadGGUFMetadata(path); ok {
				t.Fatal("unusable metadata file reported usable")
			}
			md, err := readGGUFMetadata(digest)
			if err != nil {
				t.Fatal(err)
			}
			if got := md.String("general.architecture"); got != "llama" {
				t.Fatalf("architecture = %q, want llama", got)
			}
		})
	}
}

func TestGGUFMetadataRejectsMalformed(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	dir := t.TempDir()

	for _, tt := range []struct {
		name string
		data []byte
	}{
		{"empty file", nil},
		{"bad magic", []byte("NOPE")},
		{"truncated header", []byte("GGUF\x03\x00\x00\x00")},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(dir, tt.name)
			if err := os.WriteFile(path, tt.data, 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := extractGGUFMetadata(path); err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}

// The omission rule decides what never reaches the metadata file, so every array type
// has to agree on it. A miss here silently drops real metadata.
func TestGGUFMetadataOmitRule(t *testing.T) {
	short, long := 64, ggufMetadataMaxArray+1

	for _, tt := range []struct {
		name string
		keep any
		omit any
	}{
		{"string", make([]string, short), make([]string, long)},
		{"int8", make([]int8, short), make([]int8, long)},
		{"int16", make([]int16, short), make([]int16, long)},
		{"int32", make([]int32, short), make([]int32, long)},
		{"int64", make([]int64, short), make([]int64, long)},
		{"uint8", make([]uint8, short), make([]uint8, long)},
		{"uint16", make([]uint16, short), make([]uint16, long)},
		{"uint32", make([]uint32, short), make([]uint32, long)},
		{"uint64", make([]uint64, short), make([]uint64, long)},
		{"float32", make([]float32, short), make([]float32, long)},
		{"float64", make([]float64, short), make([]float64, long)},
		{"bool", make([]bool, short), make([]bool, long)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if omitValue(tt.keep) {
				t.Errorf("%d-element %s array omitted; real per-layer metadata would be lost", short, tt.name)
			}
			if !omitValue(tt.omit) {
				t.Errorf("%d-element %s array kept; the tokenizer would be copied", long, tt.name)
			}
		})
	}

	// Scalars are always kept, except the floats JSON cannot encode.
	for _, keep := range []any{"s", int32(0), uint32(0), float32(1.5), float64(1.5), true} {
		if omitValue(keep) {
			t.Errorf("scalar %#v omitted", keep)
		}
	}
	for _, omit := range []any{
		float32(math.Inf(1)), float32(math.Inf(-1)), float64(math.NaN()),
		[]float32{1, float32(math.Inf(1))},
		[]float64{math.NaN()},
	} {
		if !omitValue(omit) {
			t.Errorf("non-finite %#v kept; json.Marshal would fail the whole file", omit)
		}
	}
}

func TestGGUFMetadataKeysIncludesOmitted(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	big := make([]string, ggufMetadataMaxArray+1)
	path, _ := createBinFile(t, ggml.KV{
		"general.architecture":  "llama",
		"llama.block_count":     uint32(4),
		"tokenizer.ggml.tokens": big,
	}, nil)

	md, err := extractGGUFMetadata(path)
	if err != nil {
		t.Fatal(err)
	}
	keys := md.Keys()
	for _, want := range []string{"general.architecture", "llama.block_count", "tokenizer.ggml.tokens"} {
		if !slices.Contains(keys, want) {
			t.Errorf("Keys() = %v, missing %s", keys, want)
		}
	}
}

// Projector audio detection reads through Keys(), including prefixed forms.
func TestProjectorAudioFromMetadata(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	for _, tt := range []struct {
		name  string
		kv    ggml.KV
		audio bool
	}{
		{"bare key", ggml.KV{"general.architecture": "clip", "has_audio_encoder": true}, true},
		{"prefixed key", ggml.KV{"general.architecture": "clip", "clip.has_audio_encoder": true}, true},
		{"present but false", ggml.KV{"general.architecture": "clip", "has_audio_encoder": false}, false},
		{"absent", ggml.KV{"general.architecture": "clip"}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path, _ := createBinFile(t, tt.kv, nil)
			md, err := extractGGUFMetadata(path)
			if err != nil {
				t.Fatal(err)
			}
			if got := projectorHasAudio(md); got != tt.audio {
				t.Errorf("projectorHasAudio = %v, want %v", got, tt.audio)
			}
		})
	}

	path, _ := createBinFile(t, ggml.KV{
		"general.architecture":       "clip",
		"has_audio_encoder":          true,
		"clip.vision.projector_type": "gemma3nv",
	}, nil)
	md, err := extractGGUFMetadata(path)
	if err != nil {
		t.Fatal(err)
	}
	if !projectorSuppressesAudioCapability(md) {
		t.Error("gemma3nv projector should suppress audio")
	}
}

func TestGGUFMetadataPathRejectsBadDigest(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	hex := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

	for _, digest := range []string{"", "sha256-short", "md5:" + hex, "../escape", "sha256:" + hex + "x"} {
		if _, err := ggufMetadataPath(digest); err == nil {
			t.Errorf("ggufMetadataPath(%q) accepted", digest)
		}
	}
	for _, digest := range []string{"sha256:" + hex, "sha256-" + hex} {
		if _, err := ggufMetadataPath(digest); err != nil {
			t.Errorf("ggufMetadataPath(%q) rejected: %v", digest, err)
		}
	}
}
