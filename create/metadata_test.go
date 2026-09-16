package create

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestDetectSafetensorsCapabilities(t *testing.T) {
	const thinkingTemplate = `{%- if '</think>' in content %}{{ content.split('</think>')[-1] }}{%- endif %}<think>\n</think>`
	tests := []struct {
		name         string
		configJSON   string
		chatTemplate string
		want         modelCapabilities
	}{
		{name: "thinking template", configJSON: `{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}`, chatTemplate: thinkingTemplate, want: modelCapabilities{thinking: true}},
		{name: "plain qwen3", configJSON: `{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}`, want: modelCapabilities{}},
		{name: "qwen3.5 always thinks", configJSON: `{"architectures":["Qwen3_5MoeForConditionalGeneration"],"model_type":"qwen3_5_moe"}`, want: modelCapabilities{thinking: true}},
		{name: "qwen3-next always thinks", configJSON: `{"architectures":["Qwen3NextForCausalLM"]}`, want: modelCapabilities{thinking: true}},
		{name: "qwen4 always thinks", configJSON: `{"architectures":["Qwen4ExpForConditionalGeneration"],"model_type":"qwen4_exp"}`, want: modelCapabilities{thinking: true}},
		{name: "vision config", configJSON: `{"architectures":["Gemma4ForConditionalGeneration"],"vision_config":{}}`, want: modelCapabilities{vision: true}},
		{name: "flat vision flag", configJSON: `{"architectures":["MuseGlimmerForConditionalGeneration"],"has_vision":true}`, want: modelCapabilities{vision: true}},
		{name: "audio config", configJSON: `{"architectures":["Qwen3OmniForConditionalGeneration"],"audio_config":{}}`, want: modelCapabilities{audio: true}},
		{name: "plain llama", configJSON: `{"architectures":["LlamaForCausalLM"],"model_type":"llama"}`, want: modelCapabilities{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cfg sourceModelConfig
			if err := json.Unmarshal([]byte(tt.configJSON), &cfg); err != nil {
				t.Fatal(err)
			}
			if got := detectCapabilitiesFromConfig(cfg, tt.chatTemplate); got != tt.want {
				t.Fatalf("detectCapabilitiesFromConfig() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestInferSafetensorsConfigAppliesOverrides(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":["Qwen3ForCausalLM"]}`), 0o644); err != nil {
		t.Fatal(err)
	}

	config := inferConfigForTest(t, dir, "functiongemma", "custom-renderer")
	if config.ModelFormat != "safetensors" {
		t.Fatalf("model format = %q, want safetensors", config.ModelFormat)
	}
	if config.Parser != "functiongemma" || config.Renderer != "custom-renderer" {
		t.Fatalf("parser/renderer = %q/%q, want overrides", config.Parser, config.Renderer)
	}
	if want := []string{"completion", "tools"}; !slices.Equal(config.Capabilities, want) {
		t.Fatalf("capabilities = %v, want %v", config.Capabilities, want)
	}
}

func TestInferSafetensorsConfigIncludesGenerationDefaults(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":["Qwen3ForCausalLM"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "generation_config.json"), []byte(`{
		"temperature": 0,
		"top_k": 12,
		"top_p": 0.7,
		"min_p": 0.05,
		"repetition_penalty": 1.2,
		"penalty_last_n": -1
	}`), 0o644); err != nil {
		t.Fatal(err)
	}

	config := inferConfigForTest(t, dir, "", "")
	for key, want := range map[string]any{
		"temperature":    float64(0),
		"top_k":          int64(12),
		"top_p":          float64(0.7),
		"min_p":          float64(0.05),
		"repeat_penalty": float64(1.2),
		"repeat_last_n":  int64(-1),
	} {
		if got := config.GenerationDefaults[key]; got != want {
			t.Errorf("generation default %s = %#v, want %#v", key, got, want)
		}
	}
}

func TestInferSafetensorsConfigFamilies(t *testing.T) {
	tests := []struct {
		name         string
		config       string
		chatTemplate string
		standalone   bool
		wantParser   string
		wantRenderer string
		wantCaps     []string
	}{
		{
			name:         "qwen3",
			config:       `{"architectures":["Qwen3ForCausalLM"]}`,
			wantParser:   "qwen3",
			wantRenderer: "qwen3-coder",
		},
		{
			name:         "qwen3.5 text",
			config:       `{"architectures":["Qwen3_5ForCausalLM"],"model_type":"qwen3"}`,
			wantParser:   "qwen3.5",
			wantRenderer: "qwen3.5",
			wantCaps:     []string{"completion", "tools", "thinking"},
		},
		{
			name:         "qwen3.5 vision",
			config:       `{"architectures":["Qwen3_5ForConditionalGeneration"],"vision_config":{}}`,
			wantParser:   "qwen3.5",
			wantRenderer: "qwen3.5",
			wantCaps:     []string{"completion", "vision", "tools", "thinking"},
		},
		{
			name:         "qwen3.8 embedded template",
			config:       `{"architectures":["Qwen3_5ForConditionalGeneration"]}`,
			chatTemplate: `{% set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}{% if preserve_thinking %}{% endif %}`,
			wantParser:   "qwen3.5",
			wantRenderer: "qwen3.8",
		},
		{
			name:         "qwen3.8 standalone template",
			config:       `{"architectures":["Qwen3_5ForConditionalGeneration"]}`,
			chatTemplate: `{% set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}{% if preserve_thinking %}{% endif %}`,
			standalone:   true,
			wantParser:   "qwen3.5",
			wantRenderer: "qwen3.8",
		},
		{
			name:         "qwen4",
			config:       `{"architectures":["Qwen4ExpForConditionalGeneration"],"model_type":"qwen4_exp"}`,
			wantParser:   "qwen3.5",
			wantRenderer: "qwen3.8",
		},
		{
			name:         "gemma4 vision and audio",
			config:       `{"architectures":["Gemma4ForConditionalGeneration"],"vision_config":{},"audio_config":{}}`,
			wantParser:   "gemma4",
			wantRenderer: "gemma4",
			wantCaps:     []string{"completion", "vision", "audio", "tools", "thinking"},
		},
		{
			name:         "glimmer vision",
			config:       `{"architectures":["MuseGlimmerForConditionalGeneration"],"model_type":"muse_glimmer","has_vision":true}`,
			wantParser:   "glimmer",
			wantRenderer: "glimmer",
			wantCaps:     []string{"completion", "vision", "tools", "thinking"},
		},
		{
			name:         "laguna",
			config:       `{"architectures":["LagunaForCausalLM"],"model_type":"laguna"}`,
			wantParser:   "laguna",
			wantRenderer: "laguna",
			wantCaps:     []string{"completion", "tools", "thinking"},
		},
		{
			name:         "laguna poolside template",
			config:       `{"architectures":["LagunaForCausalLM"],"model_type":"laguna"}`,
			chatTemplate: `{#- Iteration on laguna_glm_thinking_v8/chat_template.jinja -#}`,
			standalone:   true,
			wantParser:   "poolside-v1",
			wantRenderer: "poolside-v1",
		},
		{
			name:         "nemotron text",
			config:       `{"architectures":["NemotronHForCausalLM"],"model_type":"nemotron_h"}`,
			wantParser:   "nemotron-3-nano",
			wantRenderer: "nemotron-3-nano",
		},
		{
			name:         "nemotron 3.5 template",
			config:       `{"architectures":["NemotronHForCausalLM"],"model_type":"nemotron_h"}`,
			chatTemplate: "{reasoning effort: efficient}",
			standalone:   true,
			wantParser:   "nemotron-3.5-nano",
			wantRenderer: "nemotron-3.5-nano",
		},
		{
			name:         "nemotron omni nested config",
			config:       `{"architectures":["NemotronH_Nano_Omni_Reasoning_V3"],"vision_config":{},"sound_config":{},"llm_config":{"model_type":"nemotron_h"}}`,
			wantParser:   "nemotron-3-nano",
			wantRenderer: "nemotron-3-nano",
			wantCaps:     []string{"completion", "vision", "audio", "tools", "thinking"},
		},
		{
			name:         "deepseek",
			config:       `{"architectures":["DeepseekV3ForCausalLM"]}`,
			wantParser:   "deepseek3",
			wantRenderer: "deepseek3",
		},
		{
			name:         "glm4",
			config:       `{"architectures":["GLM4ForCausalLM"]}`,
			wantParser:   "glm-4.7",
			wantRenderer: "glm-4.7",
		},
		{
			name:   "llama",
			config: `{"architectures":["LlamaForCausalLM"]}`,
		},
		{
			name:         "nested llm config",
			config:       `{"model_type":"nemotron_h_omni","llm_config":{"model_type":"nemotron_h"}}`,
			wantParser:   "nemotron-3-nano",
			wantRenderer: "nemotron-3-nano",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.config), 0o600); err != nil {
				t.Fatal(err)
			}
			if tt.chatTemplate != "" {
				name := "tokenizer_config.json"
				data, err := json.Marshal(map[string]string{"chat_template": tt.chatTemplate})
				if tt.standalone {
					name = "chat_template.jinja"
					data = []byte(tt.chatTemplate)
				}
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
					t.Fatal(err)
				}
			}

			config := inferConfigForTest(t, dir, "", "")
			if config.Parser != tt.wantParser || config.Renderer != tt.wantRenderer {
				t.Errorf("parser/renderer = %q/%q, want %q/%q", config.Parser, config.Renderer, tt.wantParser, tt.wantRenderer)
			}
			if tt.wantCaps != nil && !slices.Equal(config.Capabilities, tt.wantCaps) {
				t.Errorf("capabilities = %v, want %v", config.Capabilities, tt.wantCaps)
			}
		})
	}
}

func TestInferSafetensorsCapabilitiesFromParser(t *testing.T) {
	for _, tt := range []struct {
		parser string
		want   []string
	}{
		{parser: "laguna", want: []string{"completion", "tools", "thinking"}},
		{parser: "poolside-v1", want: []string{"completion", "tools", "thinking"}},
		{parser: "functiongemma", want: []string{"completion", "tools"}},
		{parser: "glimmer", want: []string{"completion", "tools", "thinking"}},
	} {
		t.Run(tt.parser, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o600); err != nil {
				t.Fatal(err)
			}
			config := inferConfigForTest(t, dir, tt.parser, "")
			if !slices.Equal(config.Capabilities, tt.want) {
				t.Errorf("capabilities = %v, want %v", config.Capabilities, tt.want)
			}
		})
	}
}

func inferConfigForTest(t *testing.T, modelDir, parser, renderer string) model.ConfigV2 {
	t.Helper()
	cfg, _, err := readSourceModelConfig(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	config, err := inferSafetensorsConfig(modelDir, cfg, parser, renderer)
	if err != nil {
		t.Fatal(err)
	}
	return config
}
