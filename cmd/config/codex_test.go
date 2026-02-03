package config

import (
	"slices"
	"testing"
)

func TestCodexArgs(t *testing.T) {
	c := &Codex{}

	tests := []struct {
		name      string
		model     string
		args []string
		want      []string
	}{
		{"with model", "llama3.2", nil, []string{"--oss", "-m", "llama3.2"}},
		{"empty model", "", nil, []string{"--oss"}},
		{"with model and extra args", "qwen3-coder", []string{"--yolo"}, []string{"--oss", "-m", "qwen3-coder", "--yolo"}},
		{"empty model with extra args", "", []string{"--help"}, []string{"--oss", "--help"}},
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
