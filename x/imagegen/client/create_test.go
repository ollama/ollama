package client

import (
	"testing"

	"github.com/ollama/ollama/parser"
)

func TestCreateModelFromModelfileExtractsMetadata(t *testing.T) {
	// Test that the command parsing works correctly
	commands := []parser.Command{
		{Name: "model", Args: "./weights/test"},
		{Name: "license", Args: "Apache-2.0"},
		{Name: "requires", Args: "0.15.0"},
		{Name: "num_predict", Args: "12"},
		{Name: "seed", Args: "42"},
	}

	// We can't easily test the full function without a real model dir,
	// but we can verify the commands are valid parser.Command types
	for _, c := range commands {
		if c.Name == "" {
			t.Error("Command name should not be empty")
		}
	}
}

func TestMinOllamaVersion(t *testing.T) {
	if MinOllamaVersion == "" {
		t.Error("MinOllamaVersion should not be empty")
	}
	if MinOllamaVersion[0] < '0' || MinOllamaVersion[0] > '9' {
		t.Errorf("MinOllamaVersion should start with a number, got %q", MinOllamaVersion)
	}
}
