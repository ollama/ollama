package updater

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/app/version"
)

func infoPlist(v string) []byte {
	return fmt.Appendf(nil, `<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
	<key>CFBundleShortVersionString</key>
	<string>%s</string>
</dict></plist>`, v)
}

func TestStagedBundleVersion(t *testing.T) {
	dir := t.TempDir()
	z := filepath.Join(dir, "a.zip")
	if err := zipCreationHelper(z, []testPayload{{Name: "Ollama.app/Contents/Info.plist", Body: infoPlist("1.2.3")}}); err != nil {
		t.Fatal(err)
	}
	if got := stagedBundleVersion(z); got != "1.2.3" {
		t.Fatalf("version = %q, want 1.2.3", got)
	}
	z2 := filepath.Join(dir, "b.zip")
	if err := zipCreationHelper(z2, []testPayload{{Name: "Ollama.app/Contents/MacOS/Ollama", Body: []byte("x")}}); err != nil {
		t.Fatal(err)
	}
	if got := stagedBundleVersion(z2); got != "" {
		t.Fatalf("version = %q, want empty", got)
	}
}

func TestGetStagedUpdateDiscardsSameVersion(t *testing.T) {
	old := version.Version
	t.Cleanup(func() { version.Version = old })
	version.Version = "0.40.13"
	UpdateStageDir = filepath.Join(t.TempDir(), "updates")
	z := filepath.Join(UpdateStageDir, "etag", "Ollama-darwin.zip")
	if err := os.MkdirAll(filepath.Dir(z), 0o755); err != nil {
		t.Fatal(err)
	}

	if err := zipCreationHelper(z, []testPayload{{Name: "Ollama.app/Contents/Info.plist", Body: infoPlist("0.40.14")}}); err != nil {
		t.Fatal(err)
	}
	if got := getStagedUpdate(); got != z {
		t.Fatalf("newer update should be kept, got %q", got)
	}
	if !IsUpdatePending() {
		t.Fatal("expected update pending")
	}

	if err := zipCreationHelper(z, []testPayload{{Name: "Ollama.app/Contents/Info.plist", Body: infoPlist("0.40.13")}}); err != nil {
		t.Fatal(err)
	}
	if got := getStagedUpdate(); got != "" {
		t.Fatalf("same-version update should be discarded, got %q", got)
	}
	if IsUpdatePending() {
		t.Fatal("same-version update must not be pending")
	}
	if _, err := os.Stat(z); !os.IsNotExist(err) {
		t.Fatalf("staged zip should be removed: %v", err)
	}
}
