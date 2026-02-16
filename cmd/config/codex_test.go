package config

import (
	"slices"
	"testing"
)

func TestCodexArgs(t *testing.T) {
	c := &Codex{}

	tests := []struct {
		name  string
		model string
		args  []string
		want  []string
	}{
		{"with model", "llama3.2", nil, []string{"--oss", "-m", "llama3.2"}},
		{"empty model", "", nil, []string{"--oss"}},
		{"with model and profile", "qwen3-coder", []string{"-p", "myprofile"}, []string{"--oss", "-m", "qwen3-coder", "-p", "myprofile"}},
		{"with sandbox flag", "llama3.2", []string{"--sandbox", "workspace-write"}, []string{"--oss", "-m", "llama3.2", "--sandbox", "workspace-write"}},
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
