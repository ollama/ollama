package config

import (
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
