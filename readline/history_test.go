package readline

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/emirpasic/gods/v2/lists/arraylist"
)

func TestHistoryInitWithOllamaModels(t *testing.T) {
	// Create a temporary directory to use as OLLAMA_MODELS
	tmpDir, err := os.MkdirTemp("", "ollama-models-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Set OLLAMA_MODELS environment variable
	t.Setenv("OLLAMA_MODELS", tmpDir)

	h := &History{
		Buf:      arraylist.New[string](),
		Limit:    100,
		Autosave: true,
		Enabled:  true,
	}

	err = h.Init()
	if err != nil {
		t.Fatalf("History.Init() failed: %v", err)
	}

	// Verify the history path is inside the models directory
	expectedPath := filepath.Join(tmpDir, ".history")
	if h.Filename != expectedPath {
		t.Errorf("expected history path %q, got %q", expectedPath, h.Filename)
	}
}

func TestHistoryInitWithOllamaModelsTrailingSlash(t *testing.T) {
	// Create a temporary directory to use as OLLAMA_MODELS
	tmpDir, err := os.MkdirTemp("", "ollama-models-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Set OLLAMA_MODELS environment variable with trailing slash
	t.Setenv("OLLAMA_MODELS", tmpDir+string(filepath.Separator))

	h := &History{
		Buf:      arraylist.New[string](),
		Limit:    100,
		Autosave: true,
		Enabled:  true,
	}

	err = h.Init()
	if err != nil {
		t.Fatalf("History.Init() failed: %v", err)
	}

	// Verify the history path is inside the models directory (trailing slash should be cleaned)
	expectedPath := filepath.Join(tmpDir, ".history")
	if h.Filename != expectedPath {
		t.Errorf("expected history path %q, got %q", expectedPath, h.Filename)
	}
}

func TestHistoryInitDefaultPath(t *testing.T) {
	// Ensure OLLAMA_MODELS is not set
	t.Setenv("OLLAMA_MODELS", "")

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("failed to get user home dir: %v", err)
	}

	h := &History{
		Buf:      arraylist.New[string](),
		Limit:    100,
		Autosave: true,
		Enabled:  true,
	}

	err = h.Init()
	if err != nil {
		t.Fatalf("History.Init() failed: %v", err)
	}

	// Verify the default history path
	expectedPath := filepath.Join(home, ".ollama", "history")
	if h.Filename != expectedPath {
		t.Errorf("expected history path %q, got %q", expectedPath, h.Filename)
	}
}