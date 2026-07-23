package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestNormalizeMemoryFact(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{name: "empty", input: "   ", want: ""},
		{name: "too short", input: "hello world", want: ""},
		{name: "normal", input: " user prefers concise answers ", want: "user prefers concise answers"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := normalizeMemoryFact(tt.input)
			if got != tt.want {
				t.Fatalf("normalizeMemoryFact(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestLocalSemanticMemoryCompression(t *testing.T) {
	dir := t.TempDir()
	mgr := &localSemanticMemoryManager{
		dir:          dir,
		maxFacts:     2,
		profileFacts: 2,
	}

	facts := []string{
		"user prefers concise technical answers",
		"user likes concise explanations with examples",
		"user is currently working on ollama CLI features",
	}
	for i, fact := range facts {
		path := filepath.Join(dir, "fact-"+time.Now().Add(time.Duration(i)*time.Millisecond).Format("150405.000000000")+".txt")
		if err := os.WriteFile(path, []byte(fact), 0o600); err != nil {
			t.Fatalf("write fact file: %v", err)
		}
		time.Sleep(2 * time.Millisecond)
	}

	if err := mgr.compressIfNeeded(); err != nil {
		t.Fatalf("compressIfNeeded: %v", err)
	}

	files, err := mgr.listFactFiles()
	if err != nil {
		t.Fatalf("listFactFiles: %v", err)
	}
	if len(files) != 2 {
		t.Fatalf("expected 2 files after compression, got %d", len(files))
	}
}

func TestLocalSemanticMemoryInjectSystemMessage(t *testing.T) {
	dir := t.TempDir()
	mgr := &localSemanticMemoryManager{
		dir:          dir,
		maxFacts:     10,
		profileFacts: 2,
	}

	if err := os.WriteFile(filepath.Join(dir, "fact-1.txt"), []byte("user name is cal"), 0o600); err != nil {
		t.Fatalf("write fact-1: %v", err)
	}
	time.Sleep(2 * time.Millisecond)
	if err := os.WriteFile(filepath.Join(dir, "fact-2.txt"), []byte("user prefers concise answers"), 0o600); err != nil {
		t.Fatalf("write fact-2: %v", err)
	}

	msgs := []api.Message{{Role: "user", Content: "hello"}}
	out, err := mgr.injectSystemMessage(msgs)
	if err != nil {
		t.Fatalf("injectSystemMessage: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("expected 2 messages after injection, got %d", len(out))
	}
	if out[0].Role != "system" {
		t.Fatalf("first message role = %q, want system", out[0].Role)
	}
	if !strings.Contains(out[0].Content, localSemanticMemoryHeader) {
		t.Fatalf("system message missing memory header: %q", out[0].Content)
	}
}
