package client

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/create"
)

func TestModelfileConfig(t *testing.T) {
	// Test that ModelfileConfig struct works as expected
	config := &ModelfileConfig{
		Template: "{{ .Prompt }}",
		System:   "You are a helpful assistant.",
		Licenses: []string{"MIT"},
		Parser:   "qwen3",
		Renderer: "qwen3",
	}

	if config.Template != "{{ .Prompt }}" {
		t.Errorf("Template = %q, want %q", config.Template, "{{ .Prompt }}")
	}
	if config.System != "You are a helpful assistant." {
		t.Errorf("System = %q, want %q", config.System, "You are a helpful assistant.")
	}
	if !slices.Equal(config.Licenses, []string{"MIT"}) {
		t.Errorf("Licenses = %q, want %q", config.Licenses, []string{"MIT"})
	}
	if config.Parser != "qwen3" {
		t.Errorf("Parser = %q, want %q", config.Parser, "qwen3")
	}
	if config.Renderer != "qwen3" {
		t.Errorf("Renderer = %q, want %q", config.Renderer, "qwen3")
	}
}

func TestConfigFromModelfile(t *testing.T) {
	modelfile, err := parser.ParseFile(strings.NewReader(`
FROM ./model
DRAFT ./assistant
TEMPLATE {{ .Prompt }}
REQUIRES 0.20.0
LICENSE MIT
LICENSE Apache-2.0
ADAPTER ./adapter.gguf
MESSAGE user Hello
PARAMETER temperature 0.7
PARAMETER stop USER:
PARAMETER stop ASSISTANT:
`))
	if err != nil {
		t.Fatal(err)
	}

	modelDir, mfConfig, err := ConfigFromModelfile(modelfile)
	if err != nil {
		t.Fatal(err)
	}

	if modelDir != "./model" {
		t.Fatalf("modelDir = %q, want %q", modelDir, "./model")
	}

	if mfConfig.Template != "{{ .Prompt }}" {
		t.Fatalf("Template = %q, want %q", mfConfig.Template, "{{ .Prompt }}")
	}

	if mfConfig.Draft != "./assistant" {
		t.Fatalf("Draft = %q, want %q", mfConfig.Draft, "./assistant")
	}

	if mfConfig.Requires != "0.20.0" {
		t.Fatalf("Requires = %q, want %q", mfConfig.Requires, "0.20.0")
	}
	if !slices.Equal(mfConfig.Licenses, []string{"MIT", "Apache-2.0"}) {
		t.Fatalf("Licenses = %v, want both licenses", mfConfig.Licenses)
	}
	if !slices.Equal(mfConfig.Adapters, []string{"./adapter.gguf"}) {
		t.Fatalf("Adapters = %v, want adapter reference", mfConfig.Adapters)
	}
	if len(mfConfig.Messages) != 1 || mfConfig.Messages[0].Role != "user" || mfConfig.Messages[0].Content != "Hello" {
		t.Fatalf("Messages = %#v, want one user message", mfConfig.Messages)
	}

	if got := mfConfig.Parameters["temperature"]; got != float32(0.7) {
		t.Fatalf("temperature = %#v, want %v", got, float32(0.7))
	}

	if got := mfConfig.Parameters["stop"]; got == nil || len(got.([]string)) != 2 {
		t.Fatalf("unexpected stop params: %#v", got)
	}
}

func TestConfigFromModelfileMatchesCreateRequestMetadata(t *testing.T) {
	modelfile, err := parser.ParseFile(strings.NewReader(`
FROM missing-model
TEMPLATE {{ .Prompt }}
SYSTEM system prompt
LICENSE MIT
LICENSE Apache-2.0
PARSER test-parser
RENDERER test-renderer
REQUIRES 0.20.0
MESSAGE user Hello
PARAMETER temperature 0.7
PARAMETER stop USER:
PARAMETER stop ASSISTANT:
`))
	if err != nil {
		t.Fatal(err)
	}

	modelDir, got, err := ConfigFromModelfile(modelfile)
	if err != nil {
		t.Fatal(err)
	}
	req, err := modelfile.CreateRequest(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	licenses, ok := req.License.([]string)
	if !ok {
		t.Fatalf("CreateRequest license = %#v, want []string", req.License)
	}
	want := &ModelfileConfig{
		Template:   req.Template,
		System:     req.System,
		Licenses:   licenses,
		Parser:     req.Parser,
		Renderer:   req.Renderer,
		Requires:   req.Requires,
		Parameters: req.Parameters,
		Messages:   req.Messages,
	}
	if modelDir != req.From {
		t.Fatalf("model source = %q, want %q", modelDir, req.From)
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("Modelfile metadata mismatch (-want +got):\n%s", diff)
	}
}

func TestConfigFromModelfilePreservesRequiresBelowSafetensorsMinimum(t *testing.T) {
	modelfile, err := parser.ParseFile(strings.NewReader(`
FROM ./model
REQUIRES 0.14.0
`))
	if err != nil {
		t.Fatal(err)
	}

	_, config, err := ConfigFromModelfile(modelfile)
	if err != nil {
		t.Fatal(err)
	}
	if config.Requires != "0.14.0" {
		t.Fatalf("Requires = %q, want 0.14.0", config.Requires)
	}
}

func TestConfigFromModelfile_RequiresInvalidSemver(t *testing.T) {
	modelfile, err := parser.ParseFile(strings.NewReader(`
FROM ./model
REQUIRES not-a-version
`))
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = ConfigFromModelfile(modelfile)
	if err == nil {
		t.Fatal("expected error for invalid semver, got nil")
	}
	if !strings.Contains(err.Error(), "valid semver") {
		t.Fatalf("error = %v, want semver error", err)
	}
}

func TestModelfileConfig_Empty(t *testing.T) {
	config := &ModelfileConfig{}

	if config.Template != "" {
		t.Errorf("Template should be empty, got %q", config.Template)
	}
	if config.System != "" {
		t.Errorf("System should be empty, got %q", config.System)
	}
	if len(config.Licenses) != 0 {
		t.Errorf("Licenses should be empty, got %q", config.Licenses)
	}
	if config.Parser != "" {
		t.Errorf("Parser should be empty, got %q", config.Parser)
	}
	if config.Renderer != "" {
		t.Errorf("Renderer should be empty, got %q", config.Renderer)
	}
}

func TestModelfileConfig_PartialFields(t *testing.T) {
	// Test config with only some fields set
	config := &ModelfileConfig{
		Template: "{{ .Prompt }}",
		// System and License intentionally empty
	}

	if config.Template == "" {
		t.Error("Template should not be empty")
	}
	if config.System != "" {
		t.Error("System should be empty")
	}
	if len(config.Licenses) != 0 {
		t.Error("License should be empty")
	}
	if config.Parser != "" {
		t.Error("Parser should be empty")
	}
	if config.Renderer != "" {
		t.Error("Renderer should be empty")
	}
}

func TestMinOllamaVersion(t *testing.T) {
	// Verify the minimum version constant is set
	if create.SafetensorsMinOllamaVersion == "" {
		t.Error("SafetensorsMinOllamaVersion should not be empty")
	}
	if create.SafetensorsMinOllamaVersion != "0.19.0" {
		t.Errorf("SafetensorsMinOllamaVersion = %q, want %q", create.SafetensorsMinOllamaVersion, "0.19.0")
	}
}

func TestCreateModel_InvalidDir(t *testing.T) {
	// Test that CreateModel returns error for invalid directory
	err := CreateModel(context.Background(), CreateOptions{
		ModelName: "test-model",
		ModelDir:  "/nonexistent/path",
	}, nil)
	if err == nil {
		t.Error("expected error for nonexistent directory, got nil")
	}
}

func TestCreateModel_NotSafetensorsDir(t *testing.T) {
	// Test that CreateModel returns error for directory without safetensors
	dir := t.TempDir()

	err := CreateModel(context.Background(), CreateOptions{
		ModelName: "test-model",
		ModelDir:  dir,
	}, nil)
	if err == nil {
		t.Error("expected error for empty directory, got nil")
	}
}

func TestCreateRejectsSafetensorsAdaptersBeforeReadingSource(t *testing.T) {
	opts := CreateOptions{
		ModelDir: "missing",
		Modelfile: &ModelfileConfig{
			Adapters: []string{"adapter.gguf"},
		},
	}
	if err := CreateModel(t.Context(), opts, nil); !errors.Is(err, errSafetensorsAdapters) {
		t.Fatalf("CreateModel() error = %v, want %v", err, errSafetensorsAdapters)
	}
	if err := CreateModelRemote(t.Context(), nil, opts, nil); !errors.Is(err, errSafetensorsAdapters) {
		t.Fatalf("CreateModelRemote() error = %v, want %v", err, errSafetensorsAdapters)
	}
}

func TestCreateRejectsSameSafetensorsModelAndDraftSource(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	type draftSource struct {
		name string
		path string
	}
	draftSources := []draftSource{{name: "same path", path: dir}}
	alias := filepath.Join(t.TempDir(), "model-alias")
	if err := os.Symlink(dir, alias); err != nil {
		t.Logf("symlink alias unavailable: %v", err)
	} else {
		draftSources = append(draftSources, draftSource{name: "symlink alias", path: alias})
	}
	for _, source := range draftSources {
		t.Run(source.name, func(t *testing.T) {
			opts := CreateOptions{
				ModelDir: dir,
				Modelfile: &ModelfileConfig{
					Draft: source.path,
				},
			}
			for _, tt := range []struct {
				name   string
				create func() error
			}{
				{name: "local", create: func() error { return CreateModel(t.Context(), opts, nil) }},
				{name: "remote", create: func() error { return CreateModelRemote(t.Context(), nil, opts, nil) }},
			} {
				t.Run(tt.name, func(t *testing.T) {
					err := tt.create()
					if err == nil || !strings.Contains(err.Error(), "DRAFT must not reference the same local path as FROM") {
						t.Fatalf("error = %v, want matching source rejection", err)
					}
				})
			}
		})
	}
}

func TestCreateModel_DraftQuantizeRequiresDraft(t *testing.T) {
	err := CreateModel(context.Background(), CreateOptions{
		ModelName:     "test-model",
		ModelDir:      t.TempDir(),
		DraftQuantize: "mxfp8",
	}, nil)
	if err == nil || !strings.Contains(err.Error(), "--draft-quantize requires a DRAFT model") {
		t.Fatalf("error = %v, want draft-quantize requires DRAFT", err)
	}
}

func TestCreateModelCanceledContext(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":["Qwen3ForCausalLM"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), nil, 0o644); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := CreateModel(ctx, CreateOptions{
		ModelName: "test-model",
		ModelDir:  dir,
	}, nil)
	if err != context.Canceled {
		t.Fatalf("CreateModel() error = %v, want context.Canceled", err)
	}
}

func TestCreateOptions(t *testing.T) {
	opts := CreateOptions{
		ModelName:     "my-model",
		ModelDir:      "/path/to/model",
		Quantize:      "fp8",
		DraftQuantize: "mxfp8",
		Modelfile: &ModelfileConfig{
			Template: "test",
			System:   "system",
			Licenses: []string{"MIT"},
			Parser:   "qwen3-thinking",
			Renderer: "qwen3",
			Parameters: map[string]any{
				"temperature": float32(0.7),
			},
		},
	}

	if opts.ModelName != "my-model" {
		t.Errorf("ModelName = %q, want %q", opts.ModelName, "my-model")
	}
	if opts.ModelDir != "/path/to/model" {
		t.Errorf("ModelDir = %q, want %q", opts.ModelDir, "/path/to/model")
	}
	if opts.Quantize != "fp8" {
		t.Errorf("Quantize = %q, want %q", opts.Quantize, "fp8")
	}
	if opts.DraftQuantize != "mxfp8" {
		t.Errorf("DraftQuantize = %q, want %q", opts.DraftQuantize, "mxfp8")
	}
	if opts.Modelfile == nil {
		t.Error("Modelfile should not be nil")
	}
	if opts.Modelfile.Template != "test" {
		t.Errorf("Modelfile.Template = %q, want %q", opts.Modelfile.Template, "test")
	}
	if opts.Modelfile.Parser != "qwen3-thinking" {
		t.Errorf("Modelfile.Parser = %q, want %q", opts.Modelfile.Parser, "qwen3-thinking")
	}
	if opts.Modelfile.Renderer != "qwen3" {
		t.Errorf("Modelfile.Renderer = %q, want %q", opts.Modelfile.Renderer, "qwen3")
	}
	if opts.Modelfile.Parameters["temperature"] != float32(0.7) {
		t.Errorf("Modelfile.Parameters[temperature] = %v, want %v", opts.Modelfile.Parameters["temperature"], float32(0.7))
	}
}

func TestCreateOptions_Defaults(t *testing.T) {
	opts := CreateOptions{
		ModelName: "test",
		ModelDir:  "/tmp",
	}

	// Quantize should default to empty
	if opts.Quantize != "" {
		t.Errorf("Quantize should be empty by default, got %q", opts.Quantize)
	}
	if opts.DraftQuantize != "" {
		t.Errorf("DraftQuantize should be empty by default, got %q", opts.DraftQuantize)
	}

	// Modelfile should default to nil
	if opts.Modelfile != nil {
		t.Error("Modelfile should be nil by default")
	}
}

func TestNewManifestWriter_PopulatesFileTypeFromEffectiveQuantize(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	opts := CreateOptions{
		ModelName: "test-quantized",
		ModelDir:  t.TempDir(),
	}

	writer := newManifestWriter(opts)
	if err := writer(context.Background(), opts.ModelName, create.ManifestInfo{ModelConfig: model.ConfigV2{
		Capabilities: []string{"completion"},
		Parser:       "qwen3",
		Renderer:     "qwen3",
	}, Class: create.Classification{Quantize: "mxfp8"}}); err != nil {
		t.Fatalf("newManifestWriter() error = %v", err)
	}

	name := model.ParseName(opts.ModelName)
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatalf("ParseNamedManifest() error = %v", err)
	}

	configPath, err := manifest.BlobsPath(mf.Config.Digest)
	if err != nil {
		t.Fatalf("BlobsPath() error = %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}

	var cfg model.ConfigV2
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}

	if cfg.FileType != "mxfp8" {
		t.Fatalf("FileType = %q, want %q", cfg.FileType, "mxfp8")
	}
}

func TestNewManifestWriterPreservesMultipleLicenses(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	opts := CreateOptions{
		ModelName: "test-licenses",
		ModelDir:  t.TempDir(),
		Modelfile: &ModelfileConfig{
			Licenses: []string{"MIT", "Apache-2.0"},
		},
	}
	writer := newManifestWriter(opts)
	if err := writer(context.Background(), opts.ModelName, create.ManifestInfo{ModelConfig: model.ConfigV2{Capabilities: []string{"completion"}}}); err != nil {
		t.Fatal(err)
	}

	mf, err := manifest.ParseNamedManifest(model.ParseName(opts.ModelName))
	if err != nil {
		t.Fatal(err)
	}
	var licenses []string
	for _, layer := range mf.Layers {
		if layer.MediaType != "application/vnd.ollama.image.license" {
			continue
		}
		f, err := layer.Open()
		if err != nil {
			t.Fatal(err)
		}
		data, readErr := io.ReadAll(f)
		closeErr := f.Close()
		if readErr != nil {
			t.Fatal(readErr)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		licenses = append(licenses, string(data))
	}
	if !slices.Equal(licenses, []string{"MIT", "Apache-2.0"}) {
		t.Fatalf("licenses = %v, want both licenses", licenses)
	}
}

func TestNewManifestWriter_PopulatesDraftMetadata(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	draftDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(draftDir, "config.json"), []byte(`{"architectures":["DFlashDraftModel"],"model_type":"qwen3"}`), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	opts := CreateOptions{
		ModelName: "test-draft",
		ModelDir:  t.TempDir(),
		Modelfile: &ModelfileConfig{Draft: draftDir},
	}

	writer := newManifestWriter(opts)
	if err := writer(context.Background(), opts.ModelName, create.ManifestInfo{ModelConfig: model.ConfigV2{
		Capabilities: []string{"completion"},
		Parser:       "gemma4",
		Renderer:     "gemma4",
	}}); err != nil {
		t.Fatalf("newManifestWriter() error = %v", err)
	}

	name := model.ParseName(opts.ModelName)
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatalf("ParseNamedManifest() error = %v", err)
	}

	configPath, err := manifest.BlobsPath(mf.Config.Digest)
	if err != nil {
		t.Fatalf("BlobsPath() error = %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}

	var cfg model.ConfigV2
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if cfg.Draft == nil {
		t.Fatal("Draft metadata missing")
	}
	if cfg.Draft.TensorPrefix != "draft." || cfg.Draft.Config != "draft/config.json" {
		t.Fatalf("Draft = %#v, want draft prefix/config", cfg.Draft)
	}
	if cfg.Draft.Architecture != "DFlashDraftModel" {
		t.Fatalf("Draft architecture = %q, want DFlashDraftModel", cfg.Draft.Architecture)
	}
}

func TestCreateModelFromBaseReplacesDraftLayers(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	newLayer := func(mediaType, name, content string) manifest.Layer {
		t.Helper()
		layer, err := manifest.NewLayer(strings.NewReader(content), mediaType)
		if err != nil {
			t.Fatal(err)
		}
		layer.Name = name
		return layer
	}

	baseName := model.ParseName("base-with-draft:latest")
	config := newLayer("application/vnd.docker.container.image.v1+json", "", `{"model_format":"safetensors","capabilities":["completion"]}`)
	baseLayers := []manifest.Layer{
		newLayer(manifest.MediaTypeImageTensor, "model.embed_tokens.weight", "base"),
		newLayer("application/vnd.ollama.image.json", "config.json", `{}`),
		newLayer(manifest.MediaTypeImageTensor, "draft.model.embed_tokens.weight", "old tensor draft"),
		newLayer("application/vnd.ollama.image.json", "draft/config.json", "old config draft"),
		newLayer(manifest.MediaTypeImageDraft, "", "old GGUF draft"),
		newLayer(manifest.MediaTypeImageTensor, "drafting.weight", "not a draft"),
	}
	if err := manifest.WriteManifest(baseName, config, baseLayers); err != nil {
		t.Fatal(err)
	}

	draftDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(draftDir, "config.json"), []byte(`{"architectures":["DFlashDraftModel"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	newDraft := newLayer(manifest.MediaTypeImageTensor, "draft.model.embed_tokens.weight", "new tensor draft")
	draftLayers := []create.LayerInfo{{
		Digest: newDraft.Digest, Size: newDraft.Size, MediaType: newDraft.MediaType, Name: newDraft.Name,
	}}
	opts := CreateOptions{
		ModelName: "base-with-replaced-draft:latest",
		ModelDir:  baseName.String(),
		Modelfile: &ModelfileConfig{Draft: draftDir},
	}
	if err := createModelFromBaseWithDraft(t.Context(), opts, draftLayers, func(string) {}); err != nil {
		t.Fatal(err)
	}

	got, err := manifest.ParseNamedManifest(model.ParseName(opts.ModelName))
	if err != nil {
		t.Fatal(err)
	}
	seen := make(map[string]int)
	for _, layer := range got.Layers {
		seen[layer.Digest]++
	}
	for _, old := range baseLayers[2:5] {
		if seen[old.Digest] != 0 {
			t.Errorf("stale draft layer %q was retained", old.Name)
		}
	}
	if seen[newDraft.Digest] != 1 {
		t.Errorf("new draft layer count = %d, want 1", seen[newDraft.Digest])
	}
	if seen[baseLayers[0].Digest] != 1 || seen[baseLayers[5].Digest] != 1 {
		t.Errorf("non-draft layers were not preserved")
	}
}
