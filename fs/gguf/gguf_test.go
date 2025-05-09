package gguf_test

import (
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/gguf"
)

func blob(tb testing.TB, model string) string {
	tb.Helper()

	home, err := os.UserHomeDir()
	if err != nil {
		tb.Fatal(err)
	}

	models := filepath.Join(home, ".ollama", "models")

	model, tag, found := strings.Cut(model, ":")
	if !found {
		tag = "latest"
	}

	manifest, err := os.Open(filepath.Join(models, "manifests", "registry.ollama.ai", "library", model, tag))
	if err != nil {
		tb.Fatal(err)
	}
	defer manifest.Close()

	var m struct {
		Layers []struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
		} `json:"layers"`
	}

	if err := json.NewDecoder(manifest).Decode(&m); err != nil {
		tb.Fatal(err)
	}

	for _, layer := range m.Layers {
		if layer.MediaType == "application/vnd.ollama.image.model" {
			return filepath.Join(models, "blobs", strings.ReplaceAll(layer.Digest, ":", "-"))
		}
	}

	return ""
}

func TestRead(t *testing.T) {
	f, err := gguf.Open(blob(t, "q25vl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	arch := f.KeyValue("general.architecture")
	t.Log(arch.Key, arch.String())

	blocks := f.KeyValue("block_count")
	t.Log(blocks.Key, blocks.Uint())

	sections := f.KeyValue("rope.mrope_section")
	t.Log(sections.Key, sections.Ints())

	_, r, err := f.TensorReader("blk.0.attn_q.weight")
	if err != nil {
		t.Fatal(err)
	}

	bts := make([]byte, 64)
	if _, err := io.ReadFull(r, bts[:]); err != nil {
		t.Fatal(err)
	}

	t.Log(hex.Dump(bts))
}

func BenchmarkRead(b *testing.B) {
	b.ReportAllocs()
	f, err := gguf.Open(blob(b, "q25vl"))
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()

	b.ResetTimer()
	for range b.N {
		_ = f.KeyValue("general.architecture")
		_ = f.KeyValue("block_count")
		_ = f.TensorInfo("blk.0.attn_q.weight")
		_ = f.TensorInfo("token_embd.weight")
		_ = f.TensorInfo("output.weight")
	}
}
