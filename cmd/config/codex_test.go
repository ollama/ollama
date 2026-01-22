package config

import (
	"slices"
	"testing"
)

func TestCodexIntegration(t *testing.T) {
	c := &Codex{}

	t.Run("String", func(t *testing.T) {
		if got := c.String(); got != "Codex" {
			t.Errorf("String() = %q, want %q", got, "Codex")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = c
	})
}

func TestCodexArgs(t *testing.T) {
	c := &Codex{}

	tests := []struct {
		name  string
		model string
		want  []string
	}{
		{"with model", "llama3.2", []string{"--oss", "-m", "llama3.2"}},
		{"empty model", "", []string{"--oss"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := c.args(tt.model)
			if !slices.Equal(got, tt.want) {
				t.Errorf("args(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}
