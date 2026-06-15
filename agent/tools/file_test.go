package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/agent"
)

func TestEditReplacesUniqueText(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "note.txt")
	if err := os.WriteFile(path, []byte("hello world\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := NewEdit().Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path":     "note.txt",
		"old_text": "hello",
		"new_text": "hi",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "Updated note.txt") {
		t.Fatalf("result = %q", result.Content)
	}

	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "hi world\n" {
		t.Fatalf("content = %q", content)
	}
}

func TestEditRequiresUniqueMatchByDefault(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "note.txt")
	if err := os.WriteFile(path, []byte("same same\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := NewEdit().Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path":     "note.txt",
		"old_text": "same",
		"new_text": "other",
	})
	if err == nil {
		t.Fatal("expected ambiguous edit to fail")
	}
	if !strings.Contains(err.Error(), "matched 2 times") {
		t.Fatalf("err = %v", err)
	}
}

func TestEditRejectsEscapingPath(t *testing.T) {
	dir := t.TempDir()
	_, err := NewEdit().Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path":     "../outside.txt",
		"old_text": "old",
		"new_text": "new",
	})
	if err == nil {
		t.Fatal("expected escaping path to fail")
	}
	if !strings.Contains(err.Error(), "path escapes working directory") {
		t.Fatalf("err = %v", err)
	}
}

func TestReadRejectsParentOutsideCurrentWorkingDir(t *testing.T) {
	root := t.TempDir()
	subdir := filepath.Join(root, "sub")
	if err := os.Mkdir(subdir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "note.txt"), []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := NewRead().Execute(context.Background(), agent.ToolContext{WorkingDir: subdir}, map[string]any{
		"path": "../note.txt",
	})
	if err == nil {
		t.Fatal("expected parent path to fail")
	}
	if !strings.Contains(err.Error(), "path escapes working directory") {
		t.Fatalf("err = %v", err)
	}
}

func TestReadDefaultsToEntireFile(t *testing.T) {
	dir := t.TempDir()
	content := "one\ntwo\nthree\n"
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := NewRead().Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path": "note.txt",
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != content {
		t.Fatalf("content = %q", result.Content)
	}
}

func TestReadLineRange(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte("one\ntwo\nthree\nfour\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := NewRead().Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path":       "note.txt",
		"line_range": "2-3",
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "two\nthree\n" {
		t.Fatalf("content = %q", result.Content)
	}
}

func TestReadLineCountFromStartLine(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte("one\ntwo\nthree\nfour\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := NewRead().Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path":       "note.txt",
		"start_line": 3,
		"line_count": 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "three\nfour\n" {
		t.Fatalf("content = %q", result.Content)
	}
}

func TestReadRejectsInvalidLineRange(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte("one\ntwo\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := NewRead().Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path":       "note.txt",
		"line_range": "4-2",
	})
	if err == nil {
		t.Fatal("expected invalid range to fail")
	}
	if !strings.Contains(err.Error(), "line_range end") {
		t.Fatalf("err = %v", err)
	}
}
