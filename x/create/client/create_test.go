package client

import (
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
	if MinOllamaVersion != "0.19.0" {
		t.Errorf("MinOllamaVersion = %q, want %q", MinOllamaVersion, "0.19.0")
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

			if got := inferSafetensorsCapabilities(dir, ""); !slices.Equal(got, tt.want) {
				t.Fatalf("inferSafetensorsCapabilities() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestParsePerExpertInputs(t *testing.T) {
	makeInput := func(name, quantize string) create.PackedTensorInput {
		return create.PackedTensorInput{Name: name, Quantize: quantize}
	}

	t.Run("uniform quant across projections", func(t *testing.T) {
		inputs := []create.PackedTensorInput{
			makeInput("layer.moe.experts.0.gate_proj.weight", "int4"),
			makeInput("layer.moe.experts.1.gate_proj.weight", "int4"),
			makeInput("layer.moe.experts.0.down_proj.weight", "int4"),
			makeInput("layer.moe.experts.1.down_proj.weight", "int4"),
		}
		groups, projQ := parsePerExpertInputs("layer.moe.experts", inputs)
		if groups == nil {
			t.Fatal("expected non-nil groups")
		}
		if len(groups) != 2 {
			t.Fatalf("expected 2 projection groups, got %d", len(groups))
		}
		if projQ["gate_proj.weight"] != "int4" {
			t.Errorf("gate_proj quant = %q, want int4", projQ["gate_proj.weight"])
		}
		if projQ["down_proj.weight"] != "int4" {
			t.Errorf("down_proj quant = %q, want int4", projQ["down_proj.weight"])
		}
	})

	t.Run("mixed quant across projections", func(t *testing.T) {
		inputs := []create.PackedTensorInput{
			makeInput("layer.moe.experts.0.gate_proj.weight", "int4"),
			makeInput("layer.moe.experts.1.gate_proj.weight", "int4"),
			makeInput("layer.moe.experts.0.down_proj.weight", "int8"),
			makeInput("layer.moe.experts.1.down_proj.weight", "int8"),
		}
		groups, projQ := parsePerExpertInputs("layer.moe.experts", inputs)
		if groups == nil {
			t.Fatal("expected non-nil groups for mixed cross-projection quant")
		}
		if projQ["gate_proj.weight"] != "int4" {
			t.Errorf("gate_proj quant = %q, want int4", projQ["gate_proj.weight"])
		}
		if projQ["down_proj.weight"] != "int8" {
			t.Errorf("down_proj quant = %q, want int8", projQ["down_proj.weight"])
		}
	})

	t.Run("mixed quant within same projection rejected", func(t *testing.T) {
		inputs := []create.PackedTensorInput{
			makeInput("layer.moe.experts.0.down_proj.weight", "int4"),
			makeInput("layer.moe.experts.1.down_proj.weight", "int8"),
		}
		groups, _ := parsePerExpertInputs("layer.moe.experts", inputs)
		if groups != nil {
			t.Fatal("expected nil for mixed quant within same projection")
		}
	})

	t.Run("non-experts group rejected", func(t *testing.T) {
		inputs := []create.PackedTensorInput{
			makeInput("layer.mlp.gate_proj.weight", "int4"),
		}
		groups, _ := parsePerExpertInputs("layer.mlp", inputs)
		if groups != nil {
			t.Fatal("expected nil for non-experts group")
		}
	})
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

func TestNewManifestWriter_PopulatesFileTypeFromQuantize(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	opts := CreateOptions{
		ModelName: "test-quantized",
		ModelDir:  t.TempDir(),
		Quantize:  "MXFP8",
	}

	writer := newManifestWriter(opts, []string{"completion"}, "qwen3", "qwen3")
	if err := writer(opts.ModelName, create.LayerInfo{}, nil); err != nil {
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

func TestSupportsThinking(t *testing.T) {
	tests := []struct {
		name       string
		configJSON string
		want       bool
	}{
		{
			name:       "qwen3 architecture",
			configJSON: `{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}`,
			want:       true,
		},
		{
			name:       "deepseek architecture",
			configJSON: `{"architectures": ["DeepseekV3ForCausalLM"]}`,
			want:       true,
		},
		{
			name:       "glm4moe architecture",
			configJSON: `{"architectures": ["GLM4MoeForCausalLM"]}`,
			want:       true,
		},
		{
			name:       "llama architecture (no thinking)",
			configJSON: `{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}`,
			want:       false,
		},
		{
			name:       "gemma architecture (no thinking)",
			configJSON: `{"architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3"}`,
			want:       false,
		},
		{
			name:       "model_type only",
			configJSON: `{"model_type": "deepseek"}`,
			want:       true,
		},
		{
			name:       "laguna architecture without template",
			configJSON: `{"architectures": ["LagunaForCausalLM"], "model_type": "laguna"}`,
			want:       false,
		},
		{
			name:       "empty config",
			configJSON: `{}`,
			want:       false,
		},
		{
			name:       "invalid json",
			configJSON: `not json`,
			want:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.configJSON), 0o644)

			if got := supportsThinking(dir); got != tt.want {
				t.Errorf("supportsThinking() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSupportsThinking_NoConfig(t *testing.T) {
	if supportsThinking(t.TempDir()) {
		t.Error("supportsThinking should return false for missing config.json")
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
			name:       "functiongemma tools only",
			parserName: "functiongemma",
			want:       []string{"completion", "tools"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o644); err != nil {
				t.Fatal(err)
			}

			if got := inferSafetensorsCapabilities(dir, tt.parserName); !slices.Equal(got, tt.want) {
				t.Fatalf("inferSafetensorsCapabilities() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestInferSafetensorsCapabilitiesLaguna(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures": ["LagunaForCausalLM"], "model_type": "laguna"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	got := inferSafetensorsCapabilities(dir, "laguna")
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
			name:       "no config",
			configJSON: `{}`,
			want:       "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.configJSON), 0o644)

			if got := getParserName(dir); got != tt.want {
				t.Errorf("getParserName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestGetRendererName(t *testing.T) {
	tests := []struct {
		name       string
		configJSON string
		want       string
	}{
		{
			name:       "qwen3 model",
			configJSON: `{"architectures": ["Qwen3ForCausalLM"]}`,
			want:       "qwen3-coder",
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.configJSON), 0o644)

			if got := getRendererName(dir); got != tt.want {
				t.Errorf("getRendererName() = %q, want %q", got, tt.want)
			}
		})
	}
}
