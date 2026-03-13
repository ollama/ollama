package client

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/parser"
)

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

func TestConfigFromModelfile(t *testing.T) {
	modelfile, err := parser.ParseFile(strings.NewReader(`
FROM ./model
TEMPLATE {{ .Prompt }}
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

	if got := mfConfig.Parameters["temperature"]; got != float32(0.7) {
		t.Fatalf("temperature = %#v, want %v", got, float32(0.7))
	}

	if got := mfConfig.Parameters["stop"]; got == nil || len(got.([]string)) != 2 {
		t.Fatalf("unexpected stop params: %#v", got)
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
	if MinOllamaVersion == "" {
		t.Error("MinOllamaVersion should not be empty")
	}
	if MinOllamaVersion != "0.14.0" {
		t.Errorf("MinOllamaVersion = %q, want %q", MinOllamaVersion, "0.14.0")
	}
}

func TestCreateModel_InvalidDir(t *testing.T) {
	// Test that CreateModel returns error for invalid directory
	err := CreateModel(CreateOptions{
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

	err := CreateModel(CreateOptions{
		ModelName: "test-model",
		ModelDir:  dir,
	}, nil)
	if err == nil {
		t.Error("expected error for empty directory, got nil")
	}
}

func TestCreateOptions(t *testing.T) {
	opts := CreateOptions{
		ModelName: "my-model",
		ModelDir:  "/path/to/model",
		Quantize:  "fp8",
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

	// Modelfile should default to nil
	if opts.Modelfile != nil {
		t.Error("Modelfile should be nil by default")
	}
}

func TestQuantizeSupported(t *testing.T) {
	// This just verifies the function exists and returns a boolean
	// The actual value depends on build tags (mlx vs non-mlx)
	supported := QuantizeSupported()

	// In non-mlx builds, this should be false
	// We can't easily test both cases, so just verify it returns something
	_ = supported
}

func TestCreateModelfileLayersIncludesParameters(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	layers, err := createModelfileLayers(&ModelfileConfig{
		Parameters: map[string]any{
			"temperature": float32(0.7),
			"stop":        []string{"USER:", "ASSISTANT:"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(layers) != 1 {
		t.Fatalf("len(layers) = %d, want 1", len(layers))
	}

	if layers[0].MediaType != "application/vnd.ollama.image.params" {
		t.Fatalf("MediaType = %q, want %q", layers[0].MediaType, "application/vnd.ollama.image.params")
	}

	blobPath, err := manifest.BlobsPath(layers[0].Digest)
	if err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(blobPath)
	if err != nil {
		t.Fatal(err)
	}

	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}

	if got["temperature"] != float64(0.7) {
		t.Fatalf("temperature = %v, want %v", got["temperature"], float64(0.7))
	}
}
