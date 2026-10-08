package main

import (
	"fmt"
	"os"

	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// resolveGGUF decides whether target names a GGUF model and, if so, returns the
// path to its GGUF file. target may be a direct path to a GGUF file or an
// ollama model name (e.g. "llama3.2:latest") whose manifest points at a GGUF
// blob. MLX model names — whose manifests carry safetensors tensor layers —
// return ("", false) so the caller falls back to the MLX runner.
func resolveGGUF(target string) (string, bool) {
	if isGGUFFile(target) {
		return target, true
	}
	return ggufBlobForModel(target)
}

// isGGUFFile reports whether path is a readable GGUF file. It checks the file's
// header magic via fs/gguf (not the extension), so unsuffixed blob paths work.
func isGGUFFile(path string) bool {
	f, err := gguf.Open(path)
	if err != nil {
		return false
	}
	f.Close()
	return true
}

// ggufBlobForModel resolves an ollama model name to its GGUF blob path when the
// model is GGUF-based. A model is GGUF-based when its manifest carries a model
// layer (application/vnd.ollama.image.model) whose blob has the GGUF magic;
// MLX models instead carry image.tensor layers and yield ("", false).
func ggufBlobForModel(name string) (string, bool) {
	n := model.ParseName(name)
	if !n.IsValid() {
		return "", false
	}
	m, err := manifest.ParseNamedManifest(n)
	if err != nil {
		return "", false
	}

	// bench's llama-server spawn passes only the model GGUF (--mmproj and draft
	// models are not wired), so warn when a model carries them.
	hasExtra := false
	for _, l := range m.Layers {
		if l.MediaType == "application/vnd.ollama.image.projector" || l.MediaType == manifest.MediaTypeImageDraft {
			hasExtra = true
		}
	}

	for _, l := range m.Layers {
		if l.MediaType != "application/vnd.ollama.image.model" {
			continue
		}
		blob, err := manifest.BlobsPath(l.Digest)
		if err != nil || !isGGUFFile(blob) {
			continue
		}
		if hasExtra {
			fmt.Fprintf(os.Stderr, "WARNING: %s has projector/draft layers that bench does not pass to llama-server; vision/speculative profiling is unsupported in -spawn mode\n", name)
		}
		return blob, true
	}
	return "", false
}
