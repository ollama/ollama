package launch

import (
	"strings"
	"testing"
)

func TestLaunchModelDeprecation(t *testing.T) {
	tests := []struct {
		name       string
		deprecated bool
	}{
		{name: "qwen2.5", deprecated: true},
		{name: "qwen2.5:14b", deprecated: true},
		{name: "qwen2.5-coder:32b", deprecated: true},
		{name: "library/qwen2.5-coder:7b", deprecated: true},
		{name: "llama3", deprecated: true},
		{name: "llama3.1:8b", deprecated: true},
		{name: "llama3.2:latest", deprecated: true},
		{name: "llama3.3:70b", deprecated: true},
		{name: "llama3.2:cloud", deprecated: true},
		{name: "codellama", deprecated: true},
		{name: "codellama:13b-code", deprecated: true},
		{name: "library/codellama:7b", deprecated: true},
		{name: "starcoder", deprecated: true},
		{name: "starcoder:15b", deprecated: true},
		{name: "mistral", deprecated: true},
		{name: "mistral:7b", deprecated: true},
		{name: "deepseek-r1", deprecated: true},
		{name: "deepseek-r1:latest", deprecated: true},
		{name: "deepseek-r1:1.5b", deprecated: true},
		{name: "deepseek-r1:7b", deprecated: true},
		{name: "deepseek-r1:8b", deprecated: true},
		{name: "deepseek-r1:14b", deprecated: true},
		{name: "deepseek-r1:32b", deprecated: true},
		{name: "deepseek-r1:32b-cloud", deprecated: true},
		{name: "qwen3.5", deprecated: false},
		{name: "qwen3-coder:30b", deprecated: false},
		{name: "gemma4", deprecated: false},
		{name: "my-qwen2.5-coder", deprecated: false},
		{name: "llama3.2-inspired", deprecated: false},
		{name: "codellama-inspired", deprecated: false},
		{name: "starcoder2:15b", deprecated: false},
		{name: "mixtral:8x7b", deprecated: false},
		{name: "deepseek-r1:70b", deprecated: false},
		{name: "deepseek-r1:671b", deprecated: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isDeprecatedLaunchModel(tt.name); got != tt.deprecated {
				t.Fatalf("isDeprecatedLaunchModel(%q) = %v, want %v", tt.name, got, tt.deprecated)
			}
		})
	}
}

func TestDeprecatedLaunchModelErrorMentionsRecommendedModels(t *testing.T) {
	prompt := deprecatedLaunchModelPrompt("qwen2.5-coder:32b", "Codex", "codex", "recommended-cloud:cloud", "recommended-local")
	if prompt == "" {
		t.Fatal("expected deprecated model prompt")
	}
	for _, want := range []string{"qwen2.5-coder:32b does not work well with Codex", "recommended-cloud:cloud", "recommended-local", "ollama launch codex --model recommended-cloud:cloud", "Launch with qwen2.5-coder:32b anyway?"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt %q does not contain %q", prompt, want)
		}
	}
}
