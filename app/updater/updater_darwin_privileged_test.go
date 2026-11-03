package updater

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
)

// runSwapScript runs the privileged script without privileges.
func runSwapScript(app, newApp, owner string) bool {
	return exec.Command("/bin/sh", "-c", swapBundleScript, "sh", app, newApp, owner).Run() == nil
}

func TestSwapBundleScript(t *testing.T) {
	dir := t.TempDir()
	app := filepath.Join(dir, "Ollama.app")
	newApp := filepath.Join(dir, "stage", "Ollama.app")
	for p, c := range map[string]string{
		filepath.Join(app, "Contents", "v"):    "old",
		filepath.Join(newApp, "Contents", "v"): "new",
	} {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(c), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	owner := fmt.Sprintf("%d:%d", os.Getuid(), os.Getgid())

	if !runSwapScript(app, newApp, owner) {
		t.Fatal("swap failed")
	}
	b, _ := os.ReadFile(filepath.Join(app, "Contents", "v"))
	if string(b) != "new" {
		t.Fatalf("content = %q", b)
	}
	if _, err := os.Stat(newApp); !os.IsNotExist(err) {
		t.Fatal("staged app should be gone")
	}
	if m, _ := filepath.Glob(app + ".old.*"); len(m) != 0 {
		t.Fatalf("old bundle left behind: %v", m)
	}

	// Failure (missing staged app) must leave the existing bundle in place.
	if runSwapScript(app, filepath.Join(dir, "missing"), owner) {
		t.Fatal("expected failure")
	}
	b, _ = os.ReadFile(filepath.Join(app, "Contents", "v"))
	if string(b) != "new" {
		t.Fatalf("bundle damaged after failed swap: %q", b)
	}
}

func TestDoUpgradeWithAuthorizationPreservesOwnerAndDoesNotChown(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("permission failure can't be simulated as root")
	}
	tmpDir := t.TempDir()
	parent := filepath.Join(tmpDir, "Applications")
	BundlePath = filepath.Join(parent, "Ollama.app")
	appBackupDir = filepath.Join(tmpDir, "backup")
	UpdateStageDir = filepath.Join(tmpDir, "updates")
	UpgradeMarkerFile = filepath.Join(tmpDir, "upgraded")
	bundle := filepath.Join(UpdateStageDir, "foo", "ollama-darwin.zip")
	if err := os.MkdirAll(filepath.Join(BundlePath, "Contents"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(BundlePath, "Contents", "old"), []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(bundle), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := zipCreationHelper(bundle, []testPayload{
		{Name: "Ollama.app/Contents/MacOS/Ollama", Body: []byte("new binary"), Mode: 0o755},
	}); err != nil {
		t.Fatal(err)
	}

	// Make the rename fail like a root-owned /Applications would.
	if err := os.Chmod(parent, 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(parent, 0o755) })

	orig := swapBundle
	t.Cleanup(func() { swapBundle = orig })

	// Non-interactive must not prompt.
	swapBundle = func(string, string, string) bool { t.Fatal("must not escalate when non-interactive"); return false }
	if err := DoUpgrade(false); err == nil {
		t.Fatal("expected non-interactive failure")
	}

	var gotOwner string
	swapBundle = func(app, newApp, owner string) bool {
		gotOwner = owner
		os.Chmod(parent, 0o755) // act as root
		return runSwapScript(app, newApp, owner)
	}
	if err := DoUpgrade(true); err != nil {
		t.Fatalf("upgrade failed: %v", err)
	}

	fi, _ := os.Stat(BundlePath)
	st := fi.Sys().(*syscall.Stat_t)
	if want := fmt.Sprintf("%d:%d", st.Uid, st.Gid); gotOwner != want {
		t.Fatalf("owner passed = %s, bundle owner = %s", gotOwner, want)
	}
	if b, err := os.ReadFile(filepath.Join(BundlePath, "Contents", "MacOS", "Ollama")); err != nil || string(b) != "new binary" {
		t.Fatalf("new bundle not in place: %q %v", b, err)
	}
	if _, err := os.Stat(filepath.Join(BundlePath, "Contents", "old")); err == nil {
		t.Fatal("old content should be gone")
	}
	if _, err := os.Stat(filepath.Join(appBackupDir, "staged")); !os.IsNotExist(err) {
		t.Fatal("staging dir should be cleaned up")
	}
	if _, err := os.Stat(UpgradeMarkerFile); err != nil {
		t.Fatal("marker file missing")
	}
}

func TestDoUpgradeWithAuthorizationDenied(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("permission failure can't be simulated as root")
	}
	tmpDir := t.TempDir()
	parent := filepath.Join(tmpDir, "Applications")
	BundlePath = filepath.Join(parent, "Ollama.app")
	appBackupDir = filepath.Join(tmpDir, "backup")
	UpdateStageDir = filepath.Join(tmpDir, "updates")
	UpgradeMarkerFile = filepath.Join(tmpDir, "upgraded")
	bundle := filepath.Join(UpdateStageDir, "foo", "ollama-darwin.zip")
	os.MkdirAll(filepath.Join(BundlePath, "Contents"), 0o755)
	os.MkdirAll(filepath.Dir(bundle), 0o755)
	if err := zipCreationHelper(bundle, []testPayload{{Name: "Ollama.app/Contents/MacOS/Ollama", Body: []byte("x")}}); err != nil {
		t.Fatal(err)
	}
	os.Chmod(parent, 0o555)
	t.Cleanup(func() { os.Chmod(parent, 0o755) })
	orig := swapBundle
	t.Cleanup(func() { swapBundle = orig })
	swapBundle = func(string, string, string) bool { return false } // user cancelled

	if err := DoUpgrade(true); err == nil {
		t.Fatal("expected error when authorization denied")
	}
	if _, err := os.Stat(filepath.Join(BundlePath, "Contents")); err != nil {
		t.Fatal("existing bundle must be untouched")
	}
	if _, err := os.Stat(bundle); err != nil {
		t.Fatal("staged update should remain for retry")
	}
}
