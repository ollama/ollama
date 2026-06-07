package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/gguf"
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

// ollamaManifest is the subset of an ollama image manifest needed to locate the
// model blob.
type ollamaManifest struct {
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
	} `json:"layers"`
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
	models := envconfig.Models()
	data, err := os.ReadFile(filepath.Join(models, "manifests", n.Filepath()))
	if err != nil {
		return "", false
	}
	var m ollamaManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return "", false
	}
	for _, l := range m.Layers {
		if l.MediaType != "application/vnd.ollama.image.model" {
			continue
		}
		blob := filepath.Join(models, "blobs", strings.ReplaceAll(l.Digest, ":", "-"))
		if isGGUFFile(blob) {
			return blob, true
		}
	}
	return "", false
}
