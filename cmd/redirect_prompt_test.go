package cmd

import (
	"strings"
	"testing"
)

func TestRequirePromptForRedirectedOutput(t *testing.T) {
	t.Run("redirected without prompt errors with guidance", func(t *testing.T) {
		err := requirePromptForRedirectedOutput("", false, "llama3.1")
		if err == nil {
			t.Fatal("expected an error, got nil")
		}
		msg := err.Error()
		if !strings.Contains(msg, "llama3.1") {
			t.Errorf("expected model name in message, got: %q", msg)
		}
		if !strings.Contains(msg, "your prompt") {
			t.Errorf("expected usage guidance in message, got: %q", msg)
		}
	})

	t.Run("redirected with prompt passes", func(t *testing.T) {
		if err := requirePromptForRedirectedOutput("hello", false, "llama3.1"); err != nil {
			t.Errorf("expected nil, got: %v", err)
		}
	})

	t.Run("terminal without prompt passes", func(t *testing.T) {
		if err := requirePromptForRedirectedOutput("", true, "llama3.1"); err != nil {
			t.Errorf("expected nil, got: %v", err)
		}
	})

	t.Run("terminal with prompt passes", func(t *testing.T) {
		if err := requirePromptForRedirectedOutput("hello", true, "llama3.1"); err != nil {
			t.Errorf("expected nil, got: %v", err)
		}
	})
}
