package client

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/create"
)

func sourceMetadataInputsForTest(t *testing.T, dir string) (sourceConfig, string) {
	t.Helper()

	cfg, chatTemplate, err := readSourceMetadataInputs(dir)
	if err != nil {
		t.Fatal(err)
	}
	return cfg, chatTemplate
}

func TestModelfileConfig(t *testing.T) {
	// Test that ModelfileConfig struct works as expected
	config := &ModelfileConfig{
		Template: "{{ .Prompt }}",
		System:   "You are a helpful assistant.",
		License:  "MIT",
		Parser:   "qwen3",
		Renderer: "qwen3",
	}

	if config.Template != "{{ .Prompt }}" {
		t.Errorf("Template = %q, want %q", config.Template, "{{ .Prompt }}")
	}
	if config.System != "You are a helpful assistant." {
		t.Errorf("System = %q, want %q", config.System, "You are a helpful assistant.")
	}
	if config.License != "MIT" {
		t.Errorf("License = %q, want %q", config.License, "MIT")
	}
	if config.Parser != "qwen3" {
		t.Errorf("Parser = %q, want %q", config.Parser, "qwen3")
	}
	if config.Renderer != "qwen3" {
		t.Errorf("Renderer = %q, want %q", config.Renderer, "qwen3")
	}
}

func TestNemotronNanoOmniMetadataInference(t *testing.T) {
	dir := t.TempDir()
	config := `{
		"architectures": ["NemotronH_Nano_Omni_Reasoning_V3"],
		"model_type": "NemotronH_Nano_Omni_Reasoning_V3",
		"vision_config": {"patch_size": 16},
		"sound_config": {"model_type": "parakeet"},
		"llm_config": {"model_type": "nemotron_h"}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	metadata, err := readSourceMetadata(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := metadata.parserName, "nemotron-3-nano"; got != want {
		t.Fatalf("parser = %q, want %q", got, want)
	}
	if got, want := metadata.rendererName, "nemotron-3-nano"; got != want {
		t.Fatalf("renderer = %q, want %q", got, want)
	}
	caps := metadata.capabilities
	if !slices.Equal(caps, []string{"completion", "vision", "audio", "tools", "thinking"}) {
		t.Fatalf("capabilities = %v, want completion/vision/audio/tools/thinking", caps)
	}
}

func TestNemotron35MetadataInference(t *testing.T) {
	dir := t.TempDir()
	config := `{"architectures":["NemotronHForCausalLM"],"model_type":"nemotron_h"}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte("{reasoning effort: efficient}"), 0o644); err != nil {
		t.Fatal(err)
	}
	metadata, err := readSourceMetadata(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := metadata.parserName, "nemotron-3.5-nano"; got != want {
		t.Fatalf("parser = %q, want %q", got, want)
	}
	if got, want := metadata.rendererName, "nemotron-3.5-nano"; got != want {
		t.Fatalf("renderer = %q, want %q", got, want)
	}
}

func TestConfigFromModelfile(t *testing.T) {
	modelfile, err := parser.ParseFile(strings.NewReader(`
FROM ./model
DRAFT ./assistant
TEMPLATE {{ .Prompt }}
REQUIRES 0.20.0
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

	if got := mfConfig.Parameters["temperature"]; got != float32(0.7) {
		t.Fatalf("temperature = %#v, want %v", got, float32(0.7))
	}

	if got := mfConfig.Parameters["stop"]; got == nil || len(got.([]string)) != 2 {
		t.Fatalf("unexpected stop params: %#v", got)
	}
}

func TestConfigFromModelfile_RequiresBelowMinimum(t *testing.T) {
	modelfile, err := parser.ParseFile(strings.NewReader(`
FROM ./model
REQUIRES 0.14.0
`))
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = ConfigFromModelfile(modelfile)
	if err == nil {
		t.Fatal("expected error for REQUIRES below minimum, got nil")
	}
	if !strings.Contains(err.Error(), "minimum supported version") {
		t.Fatalf("error = %v, want error mentioning minimum supported version", err)
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
	if config.License != "" {
		t.Errorf("License should be empty, got %q", config.License)
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
	if config.License != "" {
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

func TestCreateModelMalformedMetadataErrors(t *testing.T) {
	tests := []struct {
		name       string
		files      map[string]string
		wantErrSub string
	}{
		{
			name: "config",
			files: map[string]string{
				"config.json": "{",
			},
			wantErrSub: "parse",
		},
		{
			name: "tokenizer config",
			files: map[string]string{
				"config.json":           `{"architectures":["Qwen3ForCausalLM"]}`,
				"tokenizer_config.json": "{",
			},
			wantErrSub: "parse",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			for name, content := range tt.files {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), nil, 0o644); err != nil {
				t.Fatal(err)
			}

			err := CreateModel(context.Background(), CreateOptions{
				ModelName: "test-model",
				ModelDir:  dir,
			}, nil)
			if err == nil || !strings.Contains(err.Error(), tt.wantErrSub) {
				t.Fatalf("CreateModel() error = %v, want substring %q", err, tt.wantErrSub)
			}
		})
	}
}

func TestReadSourceMetadataMissingTokenizerConfig(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":["Qwen3ForCausalLM"]}`), 0o644); err != nil {
		t.Fatal(err)
	}

	metadata, err := readSourceMetadata(dir, nil)
	if err != nil {
		t.Fatalf("readSourceMetadata() error = %v", err)
	}
	if metadata.parserName != "qwen3" || metadata.rendererName != "qwen3-coder" {
		t.Fatalf("metadata parser/renderer = %q/%q, want qwen3/qwen3-coder", metadata.parserName, metadata.rendererName)
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
			License:  "MIT",
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

func TestResolveParserName(t *testing.T) {
	tests := []struct {
		name     string
		mf       *ModelfileConfig
		inferred string
		want     string
	}{
		{
			name:     "nil modelfile uses inferred",
			mf:       nil,
			inferred: "qwen3",
			want:     "qwen3",
		},
		{
			name: "empty parser uses inferred",
			mf: &ModelfileConfig{
				Parser: "",
			},
			inferred: "qwen3",
			want:     "qwen3",
		},
		{
			name: "explicit parser overrides inferred",
			mf: &ModelfileConfig{
				Parser: "qwen3-thinking",
			},
			inferred: "qwen3",
			want:     "qwen3-thinking",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveParserName(tt.mf, tt.inferred); got != tt.want {
				t.Fatalf("resolveParserName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestResolveRendererName(t *testing.T) {
	tests := []struct {
		name     string
		mf       *ModelfileConfig
		inferred string
		want     string
	}{
		{
			name:     "nil modelfile uses inferred",
			mf:       nil,
			inferred: "qwen3-coder",
			want:     "qwen3-coder",
		},
		{
			name: "empty renderer uses inferred",
			mf: &ModelfileConfig{
				Renderer: "",
			},
			inferred: "qwen3-coder",
			want:     "qwen3-coder",
		},
		{
			name: "explicit renderer overrides inferred",
			mf: &ModelfileConfig{
				Renderer: "qwen3",
			},
			inferred: "qwen3-coder",
			want:     "qwen3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveRendererName(tt.mf, tt.inferred); got != tt.want {
				t.Fatalf("resolveRendererName() = %q, want %q", got, tt.want)
			}
		})
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

func TestInferSafetensorsCapabilities(t *testing.T) {
	tests := []struct {
		name       string
		configJSON string
		want       []string
	}{
		{
			name: "qwen3.5 text model",
			configJSON: `{
				"architectures": ["Qwen3_5ForCausalLM"],
				"model_type": "qwen3"
			}`,
			want: []string{"completion", "thinking"},
		},
		{
			name: "qwen3.5 multimodal model",
			configJSON: `{
				"architectures": ["Qwen3_5ForConditionalGeneration"],
				"model_type": "qwen3",
				"vision_config": {"hidden_size": 1024}
			}`,
			want: []string{"completion", "vision", "thinking"},
		},
		{
			name: "model with audio config",
			configJSON: `{
				"architectures": ["Gemma4ForConditionalGeneration"],
				"model_type": "gemma4",
				"vision_config": {"hidden_size": 1024},
				"audio_config": {"num_mel_bins": 128}
			}`,
			want: []string{"completion", "vision", "audio"},
		},
		{
			name: "model with audio but no vision",
			configJSON: `{
				"architectures": ["SomeAudioModel"],
				"model_type": "other",
				"audio_config": {"num_mel_bins": 128}
			}`,
			want: []string{"completion", "audio"},
		},
		{
			name: "model with sound config",
			configJSON: `{
				"architectures": ["SomeSoundModel"],
				"model_type": "other",
				"sound_config": {"model_type": "parakeet"}
			}`,
			want: []string{"completion", "audio"},
		},
		{
			name: "non-qwen conditional generation model",
			configJSON: `{
				"architectures": ["SomeOtherForConditionalGeneration"],
				"model_type": "other"
			}`,
			want: []string{"completion"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.configJSON), 0o644); err != nil {
				t.Fatal(err)
			}

			cfg, chatTemplate := sourceMetadataInputsForTest(t, dir)
			if got := inferSafetensorsCapabilitiesFromConfig(cfg, chatTemplate, ""); !slices.Equal(got, tt.want) {
				t.Fatalf("inferSafetensorsCapabilitiesFromConfig() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestNewManifestWriter_PopulatesFileTypeFromEffectiveQuantize(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	opts := CreateOptions{
		ModelName: "test-quantized",
		ModelDir:  t.TempDir(),
	}

	writer := newManifestWriter(opts, []string{"completion"}, "qwen3", "qwen3")
	if err := writer(opts.ModelName, create.ManifestInfo{Class: create.Classification{Quantize: "mxfp8"}}); err != nil {
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

	writer := newManifestWriter(opts, []string{"completion"}, "gemma4", "gemma4")
	if err := writer(opts.ModelName, create.ManifestInfo{}); err != nil {
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

func TestDetectCapabilities(t *testing.T) {
	const thinkingTemplate = `{"chat_template": "{%- if '</think>' in content %}{{ content.split('</think>')[-1] }}{%- endif %}<think>\n</think>"}`
	const instructTemplate = `{"chat_template": "{{ '<|im_start|>assistant\n' }}"}`

	tests := []struct {
		name          string
		configJSON    string
		tokenizerJSON string
		want          modelCapabilities
	}{
		{
			name:          "thinking from chat template",
			configJSON:    `{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}`,
			tokenizerJSON: thinkingTemplate,
			want:          modelCapabilities{thinking: true},
		},
		{
			name:          "instruct template has no thinking",
			configJSON:    `{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}`,
			tokenizerJSON: instructTemplate,
			want:          modelCapabilities{thinking: false},
		},
		{
			name:       "plain qwen3 without template has no thinking",
			configJSON: `{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}`,
			want:       modelCapabilities{thinking: false},
		},
		{
			name:          "qwen3.5 moe always thinks without a thinking template",
			configJSON:    `{"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe"}`,
			tokenizerJSON: instructTemplate,
			want:          modelCapabilities{thinking: true},
		},
		{
			name:       "qwen3-next always thinks",
			configJSON: `{"architectures": ["Qwen3NextForCausalLM"]}`,
			want:       modelCapabilities{thinking: true},
		},
		{
			name:       "vision config",
			configJSON: `{"architectures": ["Gemma4ForConditionalGeneration"], "vision_config": {}}`,
			want:       modelCapabilities{vision: true},
		},
		{
			name:       "flat vision flag",
			configJSON: `{"architectures": ["MuseGlimmerForConditionalGeneration"], "model_type": "muse_glimmer", "has_vision": true}`,
			want:       modelCapabilities{vision: true},
		},
		{
			name:       "audio config",
			configJSON: `{"architectures": ["Qwen3OmniForConditionalGeneration"], "audio_config": {}}`,
			want:       modelCapabilities{audio: true},
		},
		{
			name:       "llama has no extra capabilities",
			configJSON: `{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}`,
			want:       modelCapabilities{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if tt.configJSON != "" {
				if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.configJSON), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			if tt.tokenizerJSON != "" {
				if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(tt.tokenizerJSON), 0o644); err != nil {
					t.Fatal(err)
				}
			}

			cfg, chatTemplate := sourceMetadataInputsForTest(t, dir)
			if got := detectCapabilitiesFromConfig(cfg, chatTemplate); got != tt.want {
				t.Errorf("detectCapabilitiesFromConfig() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestInferSafetensorsCapabilitiesFromParser(t *testing.T) {
	tests := []struct {
		name       string
		parserName string
		want       []string
	}{
		{
			name:       "laguna tools and thinking",
			parserName: "laguna",
			want:       []string{"completion", "tools", "thinking"},
		},
		{
			name:       "poolside tools and thinking",
			parserName: "poolside-v1",
			want:       []string{"completion", "tools", "thinking"},
		},
		{
			name:       "functiongemma tools only",
			parserName: "functiongemma",
			want:       []string{"completion", "tools"},
		},
		{
			name:       "glimmer tools and thinking",
			parserName: "glimmer",
			want:       []string{"completion", "tools", "thinking"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o644); err != nil {
				t.Fatal(err)
			}

			cfg, chatTemplate := sourceMetadataInputsForTest(t, dir)
			if got := inferSafetensorsCapabilitiesFromConfig(cfg, chatTemplate, tt.parserName); !slices.Equal(got, tt.want) {
				t.Fatalf("inferSafetensorsCapabilitiesFromConfig() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestInferSafetensorsCapabilitiesGlimmerPreservesVisionMetadata(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{
		"architectures": ["MuseGlimmerForConditionalGeneration"],
		"model_type": "muse_glimmer",
		"has_vision": true
	}`), 0o644); err != nil {
		t.Fatal(err)
	}

	metadata, err := readSourceMetadata(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	got := metadata.capabilities
	want := []string{"completion", "vision", "tools", "thinking"}
	if !slices.Equal(got, want) {
		t.Fatalf("readSourceMetadata() capabilities = %#v, want %#v", got, want)
	}
}

func TestInferSafetensorsCapabilitiesLaguna(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures": ["LagunaForCausalLM"], "model_type": "laguna"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, chatTemplate := sourceMetadataInputsForTest(t, dir)
	got := inferSafetensorsCapabilitiesFromConfig(cfg, chatTemplate, "laguna")
	for _, want := range []string{"completion", "tools", "thinking"} {
		if !slices.Contains(got, want) {
			t.Fatalf("capabilities %v missing %q", got, want)
		}
	}
	if slices.Contains(got, "vision") || slices.Contains(got, "audio") {
		t.Fatalf("unexpected non-text capability in %v", got)
	}
}

func TestGetParserName(t *testing.T) {
	tests := []struct {
		name       string
		configJSON string
		want       string
	}{
		{
			name:       "qwen3 model",
			configJSON: `{"architectures": ["Qwen3ForCausalLM"]}`,
			want:       "qwen3",
		},
		{
			name:       "qwen3.5 model",
			configJSON: `{"architectures": ["Qwen3_5ForConditionalGeneration"]}`,
			want:       "qwen3.5",
		},
		{
			name:       "deepseek model",
			configJSON: `{"architectures": ["DeepseekV3ForCausalLM"]}`,
			want:       "deepseek3",
		},
		{
			name:       "glm4 model",
			configJSON: `{"architectures": ["GLM4ForCausalLM"]}`,
			want:       "glm-4.7",
		},
		{
			name:       "llama model (no parser)",
			configJSON: `{"architectures": ["LlamaForCausalLM"]}`,
			want:       "",
		},
		{
			name:       "qwen3 via model_type",
			configJSON: `{"model_type": "qwen3"}`,
			want:       "qwen3",
		},
		{
			name:       "laguna model",
			configJSON: `{"architectures": ["LagunaForCausalLM"], "model_type": "laguna"}`,
			want:       "laguna",
		},
		{
			name:       "glimmer model",
			configJSON: `{"architectures": ["MuseGlimmerForConditionalGeneration"], "model_type": "muse_glimmer"}`,
			want:       "glimmer",
		},
		{
			name:       "nemotron text architecture",
			configJSON: `{"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h"}`,
			want:       "nemotron-3-nano",
		},
		{
			name:       "nemotron omni architecture",
			configJSON: `{"architectures": ["NemotronH_Nano_Omni_Reasoning_V3"], "model_type": "NemotronH_Nano_Omni_Reasoning_V3"}`,
			want:       "nemotron-3-nano",
		},
		{
			name:       "nemotron nested llm config",
			configJSON: `{"model_type": "nemotron_h_omni", "llm_config": {"model_type": "nemotron_h"}}`,
			want:       "nemotron-3-nano",
		},
		{
			name:       "no config",
			configJSON: `{}`,
			want:       "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.configJSON), 0o644); err != nil {
				t.Fatal(err)
			}

			cfg, chatTemplate := sourceMetadataInputsForTest(t, dir)
			got, err := parserNameForConfig(dir, cfg, chatTemplate)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("parserNameForConfig() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestGetRendererName(t *testing.T) {
	tests := []struct {
		name           string
		configJSON     string
		chatTemplate   string
		standaloneOnly bool
		want           string
	}{
		{
			name:       "qwen3 model",
			configJSON: `{"architectures": ["Qwen3ForCausalLM"]}`,
			want:       "qwen3-coder",
		},
		{
			name:       "qwen3.5 model",
			configJSON: `{"architectures": ["Qwen3_5ForConditionalGeneration"]}`,
			want:       "qwen3.5",
		},
		{
			name:         "qwen3.8 embedded template",
			configJSON:   `{"architectures": ["Qwen3_5ForConditionalGeneration"]}`,
			chatTemplate: `{% set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}{% if preserve_thinking %}{% endif %}`,
			want:         "qwen3.8",
		},
		{
			name:           "qwen3.8 standalone template",
			configJSON:     `{"architectures": ["Qwen3_5ForConditionalGeneration"]}`,
			chatTemplate:   `{% set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}{% if preserve_thinking %}{% endif %}`,
			standaloneOnly: true,
			want:           "qwen3.8",
		},
		{
			name:       "deepseek model",
			configJSON: `{"architectures": ["DeepseekV3ForCausalLM"]}`,
			want:       "deepseek3",
		},
		{
			name:       "glm4 model",
			configJSON: `{"architectures": ["GLM4ForCausalLM"]}`,
			want:       "glm-4.7",
		},
		{
			name:       "llama model (no renderer)",
			configJSON: `{"architectures": ["LlamaForCausalLM"]}`,
			want:       "",
		},
		{
			name:       "laguna model",
			configJSON: `{"architectures": ["LagunaForCausalLM"], "model_type": "laguna"}`,
			want:       "laguna",
		},
		{
			name:       "glimmer model",
			configJSON: `{"architectures": ["MuseGlimmerForConditionalGeneration"], "model_type": "muse_glimmer"}`,
			want:       "glimmer",
		},
		{
			name:       "nemotron text architecture",
			configJSON: `{"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h"}`,
			want:       "nemotron-3-nano",
		},
		{
			name:       "nemotron omni architecture",
			configJSON: `{"architectures": ["NemotronH_Nano_Omni_Reasoning_V3"], "model_type": "NemotronH_Nano_Omni_Reasoning_V3"}`,
			want:       "nemotron-3-nano",
		},
		{
			name:       "nemotron nested llm config",
			configJSON: `{"model_type": "nemotron_h_omni", "llm_config": {"model_type": "nemotron_h"}}`,
			want:       "nemotron-3-nano",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.configJSON), 0o644); err != nil {
				t.Fatal(err)
			}
			if tt.chatTemplate != "" {
				if tt.standaloneOnly {
					if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(tt.chatTemplate), 0o644); err != nil {
						t.Fatal(err)
					}
				} else {
					data, err := json.Marshal(map[string]string{"chat_template": tt.chatTemplate})
					if err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), data, 0o644); err != nil {
						t.Fatal(err)
					}
				}
			}

			cfg, chatTemplate := sourceMetadataInputsForTest(t, dir)
			got, err := rendererNameForConfig(dir, cfg, chatTemplate)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("rendererNameForConfig() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestGetLagunaRendererParserName(t *testing.T) {
	tests := []struct {
		name         string
		chatTemplate string
		want         string
	}{
		{
			name:         "v5",
			chatTemplate: `{#- Iteration on laguna_glm_thinking_v5/chat_template.jinja -#}`,
			want:         "laguna",
		},
		{
			name:         "v8",
			chatTemplate: `{#- Iteration on laguna_glm_thinking_v8/chat_template.jinja -#}`,
			want:         "poolside-v1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":["LagunaForCausalLM"],"model_type":"laguna"}`), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"chat_template":"{% include 'chat_template.jinja' %}"}`), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(tt.chatTemplate), 0o644); err != nil {
				t.Fatal(err)
			}

			cfg, chatTemplate := sourceMetadataInputsForTest(t, dir)
			gotParser, err := parserNameForConfig(dir, cfg, chatTemplate)
			if err != nil {
				t.Fatal(err)
			}
			if gotParser != tt.want {
				t.Errorf("parserNameForConfig() = %q, want %q", gotParser, tt.want)
			}
			gotRenderer, err := rendererNameForConfig(dir, cfg, chatTemplate)
			if err != nil {
				t.Fatal(err)
			}
			if gotRenderer != tt.want {
				t.Errorf("rendererNameForConfig() = %q, want %q", gotRenderer, tt.want)
			}
		})
	}
}
