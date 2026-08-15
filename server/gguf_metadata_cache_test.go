package server

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/fs/ggml"
)

// writeTestGGUF writes a minimal GGUF file with the given metadata to path.
func writeTestGGUF(t *testing.T, path string, kv ggml.KV) {
	t.Helper()

	if _, ok := kv["general.architecture"]; !ok {
		kv["general.architecture"] = "test"
	}

	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := ggml.WriteGGUF(f, kv, nil); err != nil {
		t.Fatal(err)
	}
}

func TestLoadGGUFMetadataValues(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vision.gguf")
	writeTestGGUF(t, path, ggml.KV{
		"general.architecture":      "gemma4",
		"tokenizer.chat_template":   "{{ .Prompt }}",
		"gemma4.vision.block_count": uint32(12),
	})

	md, err := loadGGUFMetadata(path)
	if err != nil {
		t.Fatal(err)
	}

	if md.architecture != "gemma4" {
		t.Errorf("architecture = %q, want %q", md.architecture, "gemma4")
	}
	if md.chatTemplate != "{{ .Prompt }}" {
		t.Errorf("chatTemplate = %q, want %q", md.chatTemplate, "{{ .Prompt }}")
	}
	if !md.hasVision {
		t.Error("hasVision = false, want true")
	}
	if md.hasPooling {
		t.Error("hasPooling = true, want false")
	}
	if md.hasAudio {
		t.Error("hasAudio = true, want false")
	}
}

func TestLoadGGUFMetadataCacheHit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	writeTestGGUF(t, path, ggml.KV{"general.architecture": "test"})

	first, err := loadGGUFMetadata(path)
	if err != nil {
		t.Fatal(err)
	}

	// A second lookup returns the same *ggufMetadata pointer, confirming the
	// blob was parsed once and served from the cache thereafter.
	second, err := loadGGUFMetadata(path)
	if err != nil {
		t.Fatal(err)
	}

	if first != second {
		t.Errorf("cache miss: got a different *ggufMetadata on the second call")
	}
}

func TestLoadGGUFMetadataInvalidatesOnMtimeAndSize(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	writeTestGGUF(t, path, ggml.KV{
		"general.architecture": "test",
		"pooling_type":         uint32(1),
	})

	first, err := loadGGUFMetadata(path)
	if err != nil {
		t.Fatal(err)
	}
	if !first.hasPooling {
		t.Fatal("expected first blob to report pooling")
	}

	// Rewrite the same path with different content and a later mtime. The
	// key (path|size|mtime) must change, forcing a re-read rather than a
	// stale cache hit.
	writeTestGGUF(t, path, ggml.KV{"general.architecture": "test"})
	if err := os.Chtimes(path, time.Now().Add(time.Hour), time.Now().Add(time.Hour)); err != nil {
		t.Fatal(err)
	}

	second, err := loadGGUFMetadata(path)
	if err != nil {
		t.Fatal(err)
	}

	if first == second {
		t.Error("expected a fresh *ggufMetadata after the file changed")
	}
	if second.hasPooling {
		t.Error("hasPooling = true after rewrite without pooling_type; stale cache")
	}
}

func TestLoadGGUFMetadataMissingFile(t *testing.T) {
	_, err := loadGGUFMetadata(filepath.Join(t.TempDir(), "does-not-exist.gguf"))
	if err == nil {
		t.Error("expected an error for a missing file")
	}
}
