package launch

import (
	"fmt"
	"slices"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestCodexArgs(t *testing.T) {
	c := &Codex{}

	baseURLArg := fmt.Sprintf("openai_base_url=%q", envconfig.Host().String()+"/v1/")

	tests := []struct {
		name  string
		model string
		args  []string
		want  []string
	}{
		{"with model", "llama3.2", nil, []string{"--oss", "-c", baseURLArg, "-m", "llama3.2"}},
		{"empty model", "", nil, []string{"--oss", "-c", baseURLArg}},
		{"with model and profile", "qwen3.5", []string{"-p", "myprofile"}, []string{"--oss", "-c", baseURLArg, "-m", "qwen3.5", "-p", "myprofile"}},
		{"with sandbox flag", "llama3.2", []string{"--sandbox", "workspace-write"}, []string{"--oss", "-c", baseURLArg, "-m", "llama3.2", "--sandbox", "workspace-write"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := c.args(tt.model, tt.args)
			if !slices.Equal(got, tt.want) {
				t.Errorf("args(%q, %v) = %v, want %v", tt.model, tt.args, got, tt.want)
			}
		})
	}
}
