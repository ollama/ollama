package main

import (
	"path/filepath"
	"testing"
)

func TestPathWithinDir(t *testing.T) {
	tmpDir := t.TempDir()
	applications := filepath.Join(tmpDir, "Applications")
	cases := []struct {
		name string
		path string
		dir  string
		want bool
	}{
		{"app directly in applications", filepath.Join(applications, "Ollama.app"), applications, true},
		{"app in applications subdirectory", filepath.Join(applications, "AI", "Ollama.app"), applications, true},
		{"dir itself", applications, applications, true},
		{"sibling with shared prefix", filepath.Join(tmpDir, "ApplicationsExtra", "Ollama.app"), applications, false},
		{"outside the directory", filepath.Join(tmpDir, "Downloads", "Ollama.app"), applications, false},
		{"parent traversal", filepath.Join(applications, "..", "Downloads", "Ollama.app"), applications, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := pathWithinDir(tc.path, tc.dir); got != tc.want {
				t.Errorf("pathWithinDir(%q, %q) = %v, want %v", tc.path, tc.dir, got, tc.want)
			}
		})
	}
}

// TestPathWithinDirApplicationsLayout covers the exact locations from the
// issue report against the real /Applications prefix, independent of any
// temporary directories.
func TestPathWithinDirApplicationsLayout(t *testing.T) {
	applications := filepath.FromSlash("/Applications")
	cases := []struct {
		path string
		want bool
	}{
		{filepath.FromSlash("/Applications/Ollama.app"), true},
		{filepath.FromSlash("/Applications/Ollama 2.app"), true},
		{filepath.FromSlash("/Applications/AI/Ollama.app"), true},
		{filepath.FromSlash("/Applications/LLM/Local/Ollama.app"), true},
		{filepath.FromSlash("/ApplicationsExtra/Ollama.app"), false},
		{filepath.FromSlash("~/Downloads/Ollama.app"), false},
	}

	for _, tc := range cases {
		t.Run(tc.path, func(t *testing.T) {
			if got := pathWithinDir(tc.path, applications); got != tc.want {
				t.Errorf("pathWithinDir(%q, %q) = %v, want %v", tc.path, applications, got, tc.want)
			}
		})
	}
}
