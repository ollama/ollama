package config

import (
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
