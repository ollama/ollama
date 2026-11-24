//go:build darwin

package main

import (
	"archive/zip"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInstallClaudeDesktopZip(t *testing.T) {
	archive := writeClaudeDesktopTestZip(t, map[string]claudeDesktopTestZipEntry{
		"Claude.app/":                        {directory: true},
		"Claude.app/Contents/":               {directory: true},
		"Claude.app/Contents/MacOS/":         {directory: true},
		"Claude.app/Contents/MacOS/Claude":   {body: "binary", mode: 0o755},
		"Claude.app/Contents/Resources/":     {directory: true},
		"Claude.app/Contents/Resources/link": {body: "../MacOS/Claude", mode: os.ModeSymlink | 0o777},
	})
	destination := filepath.Join(t.TempDir(), "Applications", "Claude.app")
	var verified string
	installed, err := installClaudeDesktopZip(archive, []string{destination}, func(bundle string) error {
		verified = bundle
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if installed != destination || verified == "" {
		t.Fatalf("installed = %q, verified = %q", installed, verified)
	}
	info, err := os.Stat(filepath.Join(installed, "Contents", "MacOS", "Claude"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode()&0o111 == 0 {
		t.Fatal("installed Claude executable is not executable")
	}
	if target, err := os.Readlink(filepath.Join(installed, "Contents", "Resources", "link")); err != nil || target != "../MacOS/Claude" {
		t.Fatalf("symlink target = %q, err = %v", target, err)
	}
}

func TestInstallClaudeDesktopZipRejectsUnsafeArchives(t *testing.T) {
	for _, test := range []struct {
		name    string
		entries map[string]claudeDesktopTestZipEntry
	}{
		{name: "path traversal", entries: map[string]claudeDesktopTestZipEntry{"../Claude.app/Contents/MacOS/Claude": {body: "binary", mode: 0o755}}},
		{name: "unexpected root", entries: map[string]claudeDesktopTestZipEntry{"README": {body: "nope", mode: 0o644}}},
		{name: "escaping symlink", entries: map[string]claudeDesktopTestZipEntry{
			"Claude.app/Contents/MacOS/Claude": {body: "binary", mode: 0o755},
			"Claude.app/escape":                {body: "../../outside", mode: os.ModeSymlink | 0o777},
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			archive := writeClaudeDesktopTestZip(t, test.entries)
			destination := filepath.Join(t.TempDir(), "Claude.app")
			if _, err := installClaudeDesktopZip(archive, []string{destination}, func(string) error { return nil }); err == nil {
				t.Fatal("installClaudeDesktopZip succeeded")
			}
			if _, err := os.Stat(destination); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("unsafe archive created destination: %v", err)
			}
		})
	}
}

func TestInstallClaudeDesktopZipVerifiesBeforeMove(t *testing.T) {
	archive := writeClaudeDesktopTestZip(t, map[string]claudeDesktopTestZipEntry{
		"Claude.app/Contents/MacOS/Claude": {body: "binary", mode: 0o755},
	})
	destination := filepath.Join(t.TempDir(), "Claude.app")
	wantErr := errors.New("invalid signature")
	if _, err := installClaudeDesktopZip(archive, []string{destination}, func(string) error { return wantErr }); !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
	if _, err := os.Stat(destination); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("invalid bundle created destination: %v", err)
	}
}

func TestInstallClaudeDesktopZipDoesNotOverwrite(t *testing.T) {
	archive := writeClaudeDesktopTestZip(t, map[string]claudeDesktopTestZipEntry{
		"Claude.app/Contents/MacOS/Claude": {body: "binary", mode: 0o755},
	})
	destination := filepath.Join(t.TempDir(), "Claude.app")
	if err := os.MkdirAll(destination, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := installClaudeDesktopZip(archive, []string{destination}, func(string) error { return nil }); !errors.Is(err, errClaudeDesktopDestinationExists) {
		t.Fatalf("error = %v, want destination exists", err)
	}
}

func TestInstallClaudeDesktopZipRealArchive(t *testing.T) {
	archive := os.Getenv("OLLAMA_TEST_CLAUDE_DESKTOP_ZIP")
	if archive == "" {
		t.Skip("set OLLAMA_TEST_CLAUDE_DESKTOP_ZIP to a downloaded Claude Desktop ZIP")
	}
	destination := filepath.Join(t.TempDir(), "Applications", "Claude.app")
	installed, err := installClaudeDesktopZip(
		archive,
		[]string{destination},
		verifyClaudeDesktopBundle,
	)
	if err != nil {
		t.Fatal(err)
	}
	if installed != destination {
		t.Fatalf("installed = %q, want %q", installed, destination)
	}
}

type claudeDesktopTestZipEntry struct {
	body      string
	mode      os.FileMode
	directory bool
}

func writeClaudeDesktopTestZip(t *testing.T, entries map[string]claudeDesktopTestZipEntry) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "Claude.zip")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	writer := zip.NewWriter(file)
	for name, entry := range entries {
		header := &zip.FileHeader{Name: name, Method: zip.Deflate}
		if entry.directory {
			header.SetMode(os.ModeDir | 0o755)
		} else {
			header.SetMode(entry.mode)
		}
		item, err := writer.CreateHeader(header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := item.Write([]byte(entry.body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestSafeClaudeDesktopArchivePath(t *testing.T) {
	for _, name := range []string{"Claude.app", "Claude.app/Contents/MacOS/Claude"} {
		if got, err := safeClaudeDesktopArchivePath(name); err != nil || got != strings.TrimSuffix(name, "/") {
			t.Fatalf("safeClaudeDesktopArchivePath(%q) = %q, %v", name, got, err)
		}
	}
}
