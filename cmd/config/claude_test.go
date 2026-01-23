package config

import (
	"slices"
	"testing"
)

func TestClaudeIntegration(t *testing.T) {
	c := &Claude{}

	t.Run("String", func(t *testing.T) {
		if got := c.String(); got != "Claude Code" {
			t.Errorf("String() = %q, want %q", got, "Claude Code")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = c
	})
}

func TestClaudeArgs(t *testing.T) {
	c := &Claude{}

	tests := []struct {
		name  string
		model string
		want  []string
	}{
		{"with model", "llama3.2", []string{"--model", "llama3.2"}},
		{"empty model", "", nil},
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
