//go:build darwin

package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestInstallCodexDesktopBundle(t *testing.T) {
	bundle := writeCodexDesktopTestBundle(t, "ChatGPT.app", "ChatGPT")
	destination := filepath.Join(t.TempDir(), "Applications", "ChatGPT.app")
	verified := 0
	installed, err := installCodexDesktopBundle(bundle, []string{destination}, func(string) error {
		verified++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if installed != destination {
		t.Fatalf("installed = %q, want %q", installed, destination)
	}
	if verified != 2 {
		t.Fatalf("signature verification count = %d, want 2", verified)
	}
	info, err := os.Stat(filepath.Join(installed, "Contents", "MacOS", "ChatGPT"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode()&0o111 == 0 {
		t.Fatal("installed ChatGPT executable is not executable")
	}
}

func TestInstallCodexDesktopBundleAcceptsCodexNamedSource(t *testing.T) {
	bundle := writeCodexDesktopTestBundle(t, "Codex.app", "Codex")
	destination := filepath.Join(t.TempDir(), "Applications", "ChatGPT.app")
	if _, err := installCodexDesktopBundle(bundle, []string{destination}, func(string) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(destination, "Contents", "MacOS", "Codex")); err != nil {
		t.Fatal(err)
	}
}

func TestInstallCodexDesktopBundleVerifiesBeforeCopy(t *testing.T) {
	bundle := writeCodexDesktopTestBundle(t, "ChatGPT.app", "ChatGPT")
	destination := filepath.Join(t.TempDir(), "ChatGPT.app")
	wantErr := errors.New("invalid signature")
	if _, err := installCodexDesktopBundle(bundle, []string{destination}, func(string) error { return wantErr }); !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
	if _, err := os.Stat(destination); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("invalid bundle created destination: %v", err)
	}
}

func TestInstallCodexDesktopBundleDoesNotOverwrite(t *testing.T) {
	bundle := writeCodexDesktopTestBundle(t, "ChatGPT.app", "ChatGPT")
	destination := filepath.Join(t.TempDir(), "ChatGPT.app")
	if err := os.MkdirAll(destination, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := installCodexDesktopBundle(bundle, []string{destination}, func(string) error { return nil }); !errors.Is(err, errCodexDesktopDestinationExists) {
		t.Fatalf("error = %v, want destination exists", err)
	}
}

func TestInstallCodexDesktopBundleDoesNotOverwriteBrokenSymlink(t *testing.T) {
	bundle := writeCodexDesktopTestBundle(t, "ChatGPT.app", "ChatGPT")
	destination := filepath.Join(t.TempDir(), "ChatGPT.app")
	if err := os.Symlink(filepath.Join(t.TempDir(), "missing"), destination); err != nil {
		t.Fatal(err)
	}
	if _, err := installCodexDesktopBundle(bundle, []string{destination}, func(string) error { return nil }); !errors.Is(err, errCodexDesktopDestinationExists) {
		t.Fatalf("error = %v, want destination exists", err)
	}
}

func TestCodexDesktopBundleOnVolumeRejectsSymlink(t *testing.T) {
	volume := t.TempDir()
	target := writeCodexDesktopTestBundle(t, "ChatGPT.app", "ChatGPT")
	if err := os.Symlink(target, filepath.Join(volume, "ChatGPT.app")); err != nil {
		t.Fatal(err)
	}
	if _, err := codexDesktopBundleOnVolume(volume); err == nil {
		t.Fatal("codexDesktopBundleOnVolume accepted a symlink")
	}
}

func TestInstallCodexDesktopDiskImageRealArchive(t *testing.T) {
	image := os.Getenv("OLLAMA_TEST_CODEX_DESKTOP_DMG")
	if image == "" {
		t.Skip("set OLLAMA_TEST_CODEX_DESKTOP_DMG to the official ChatGPT DMG")
	}
	destination := filepath.Join(t.TempDir(), "Applications", "ChatGPT.app")
	installed, err := installCodexDesktopDiskImage(image, []string{destination}, verifyCodexDesktopBundle)
	if err != nil {
		t.Fatal(err)
	}
	if installed != destination {
		t.Fatalf("installed = %q, want %q", installed, destination)
	}
}

func writeCodexDesktopTestBundle(t *testing.T, appName, executableName string) string {
	t.Helper()
	bundle := filepath.Join(t.TempDir(), appName)
	executable := filepath.Join(bundle, "Contents", "MacOS", executableName)
	if err := os.MkdirAll(filepath.Dir(executable), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(executable, []byte("binary"), 0o755); err != nil {
		t.Fatal(err)
	}
	return bundle
}
