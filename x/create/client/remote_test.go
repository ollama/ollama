package client

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestPrepareRemoteSourceFilesSelectsJSONAndSafetensors(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"config.json":                  `{"architectures":["TestModel"]}`,
		"model.safetensors":            "tensor-data",
		"model.safetensors.index.json": `{"weight_map":{}}`,
		"tokenizer.model":              "not uploaded",
		"README.md":                    "not uploaded",
		"consolidated.00.safetensors":  "unsupported safetensors",
		"tokenizer_config.json":        `{}`,
		"special_tokens_map.json":      `{}`,
		"generation_config.json":       `{}`,
		"preprocessor_config.json":     `{}`,
		"chat_template.json":           `{}`,
		"processor_config.json":        `{}`,
		"added_tokens.json":            `{}`,
		"tokenizer.json":               `{}`,
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	got, err := prepareRemoteSourceFiles(dir, false)
	if err != nil {
		t.Fatal(err)
	}

	var names []string
	for _, f := range got {
		names = append(names, f.logical)
		if f.digest == "" || f.size == 0 {
			t.Fatalf("file %s digest/size = %q/%d, want populated", f.logical, f.digest, f.size)
		}
		if f.draft {
			t.Fatalf("file %s marked draft", f.logical)
		}
	}
	slices.Sort(names)
	want := []string{
		"added_tokens.json",
		"chat_template.json",
		"config.json",
		"generation_config.json",
		"model.safetensors",
		"model.safetensors.index.json",
		"preprocessor_config.json",
		"processor_config.json",
		"special_tokens_map.json",
		"tokenizer.json",
		"tokenizer_config.json",
	}
	if !slices.Equal(names, want) {
		t.Fatalf("uploaded files = %v, want %v", names, want)
	}
}

func TestNewRemoteCreateRequest(t *testing.T) {
	req := newRemoteCreateRequest(CreateOptions{
		ModelName:     "example",
		Quantize:      "nvfp4",
		DraftQuantize: "mxfp8",
		Modelfile: &ModelfileConfig{
			Template:   "{{ .Prompt }}",
			System:     "system",
			License:    "MIT",
			Parser:     "mf-parser",
			Renderer:   "mf-renderer",
			Requires:   "0.20.0",
			Parameters: map[string]any{"temperature": float32(0.1)},
		},
	}, []remoteSourceFile{
		{logical: "config.json", digest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
		{logical: "model.safetensors", digest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
		{logical: "config.json", digest: "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", draft: true},
		{logical: "model.safetensors", digest: "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", draft: true},
	}, []string{"completion", "thinking"}, "inferred-parser", "inferred-renderer")

	if req.Model != "example" {
		t.Fatalf("request model = %q, want example", req.Model)
	}
	if req.Quantize != "nvfp4" || req.DraftQuantize != "mxfp8" {
		t.Fatalf("request quantize = %q/%q, want nvfp4/mxfp8", req.Quantize, req.DraftQuantize)
	}
	if req.Parser != "mf-parser" || req.Renderer != "mf-renderer" {
		t.Fatalf("parser/renderer = %q/%q, want mf-parser/mf-renderer", req.Parser, req.Renderer)
	}
	if req.Requires != "0.20.0" {
		t.Fatalf("Requires = %q, want 0.20.0", req.Requires)
	}
	if req.Files["model.safetensors"] == "" || req.DraftFiles["model.safetensors"] == "" {
		t.Fatalf("files = %v draft_files = %v, want model entries", req.Files, req.DraftFiles)
	}
	caps, ok := req.Info["capabilities"].([]string)
	if !ok || !slices.Equal(caps, []string{"completion", "thinking"}) {
		t.Fatalf("capabilities = %#v, want completion/thinking", req.Info["capabilities"])
	}
}

func TestRemoteUploadConcurrencyUsesEnvconfigMaxTransferStreams(t *testing.T) {
	t.Setenv("OLLAMA_MAX_TRANSFER_STREAMS", "")
	if got := remoteUploadConcurrency(); got != 4 {
		t.Fatalf("remoteUploadConcurrency() = %d, want default 4", got)
	}

	t.Setenv("OLLAMA_MAX_TRANSFER_STREAMS", "7")
	if got := remoteUploadConcurrency(); got != 7 {
		t.Fatalf("remoteUploadConcurrency() = %d, want 7", got)
	}

	t.Setenv("OLLAMA_MAX_TRANSFER_STREAMS", "0")
	if got := remoteUploadConcurrency(); got != 1 {
		t.Fatalf("remoteUploadConcurrency() = %d, want zero clamped to 1", got)
	}
}
