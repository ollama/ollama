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

	result, err := (&Edit{}).Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
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

	_, err := (&Edit{}).Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
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
	_, err := (&Edit{}).Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
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

func TestEditRejectsSymlinkEscape(t *testing.T) {
	dir := t.TempDir()
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "note.txt"), []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "link")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	_, err := (&Edit{}).Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path":     filepath.Join("link", "note.txt"),
		"old_text": "old",
		"new_text": "new",
	})
	if err == nil {
		t.Fatal("expected symlink escape to fail")
	}
	if !strings.Contains(err.Error(), "path escapes working directory") {
		t.Fatalf("err = %v", err)
	}

	content, err := os.ReadFile(filepath.Join(outside, "note.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "old\n" {
		t.Fatalf("outside content changed to %q", content)
	}
}

func TestEditRejectsFinalSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target.txt")
	if err := os.WriteFile(target, []byte("old\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link.txt")
	if err := os.Symlink("target.txt", link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	_, err := (&Edit{}).Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path":     "link.txt",
		"old_text": "old",
		"new_text": "new",
	})
	if err == nil {
		t.Fatal("expected final symlink edit to fail")
	}
	if !strings.Contains(err.Error(), "is a symlink") {
		t.Fatalf("err = %v", err)
	}
	content, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "old\n" {
		t.Fatalf("target content changed to %q", content)
	}
	info, err := os.Lstat(link)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("link mode = %v, want symlink", info.Mode())
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

	_, err := (&Read{}).Execute(context.Background(), agent.ToolContext{WorkingDir: subdir}, map[string]any{
		"path": "../note.txt",
	})
	if err == nil {
		t.Fatal("expected parent path to fail")
	}
	if !strings.Contains(err.Error(), "path escapes working directory") {
		t.Fatalf("err = %v", err)
	}
}

func TestReadRequiresApproval(t *testing.T) {
	if !agent.ToolRequiresApproval((&Read{}), map[string]any{"path": "note.txt"}) {
		t.Fatal("read should require approval")
	}
}

func TestReadDefaultsToEntireFile(t *testing.T) {
	dir := t.TempDir()
	content := "one\ntwo\nthree\n"
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := (&Read{}).Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path": "note.txt",
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != content {
		t.Fatalf("content = %q", result.Content)
	}
}

func TestReadStartEnd(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte("one\ntwo\nthree\nfour\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := (&Read{}).Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path":  "note.txt",
		"start": 2,
		"end":   3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "two\nthree\n" {
		t.Fatalf("content = %q", result.Content)
	}
}

func TestReadStartOnly(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte("one\ntwo\nthree\nfour\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := (&Read{}).Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path":  "note.txt",
		"start": 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "three\nfour\n" {
		t.Fatalf("content = %q", result.Content)
	}
}

func TestReadEndOnly(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte("one\ntwo\nthree\nfour\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := (&Read{}).Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path": "note.txt",
		"end":  2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "one\ntwo\n" {
		t.Fatalf("content = %q", result.Content)
	}
}

func TestReadSelectionRejectsHugeSingleLine(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte(strings.Repeat("x", maxReadBytes+1)), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := (&Read{}).Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path":  "note.txt",
		"start": 1,
		"end":   1,
	})
	if err == nil {
		t.Fatal("expected huge selected line to fail")
	}
	if !strings.Contains(err.Error(), "selected content is too large") {
		t.Fatalf("err = %v", err)
	}
}

func TestReadRejectsInvalidRange(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte("one\ntwo\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := (&Read{}).Execute(context.Background(), agent.ToolContext{WorkingDir: dir}, map[string]any{
		"path":  "note.txt",
		"start": 4,
		"end":   2,
	})
	if err == nil {
		t.Fatal("expected invalid range to fail")
	}
	if !strings.Contains(err.Error(), "end must") {
		t.Fatalf("err = %v", err)
	}
}
