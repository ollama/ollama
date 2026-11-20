package manifest

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/types/model"
)

// A directory junction carries the same reparse tag as an NTFS volume mount
// point (IO_REPARSE_TAG_MOUNT_POINT) and can be created without admin rights.
func createJunction(t *testing.T, link, target string) {
	t.Helper()
	if out, err := exec.Command("cmd", "/c", "mklink", "/J", link, target).CombinedOutput(); err != nil {
		t.Skipf("mklink /J unavailable: %v %s", err, out)
	}
}

func TestModelsThroughJunction(t *testing.T) {
	for _, tt := range []struct {
		name   string
		models func(real, link string) string // OLLAMA_MODELS seen by the server
		target func(real string) string       // what the junction points at
	}{
		// OLLAMA_MODELS=C:\mnt\hdd0\models where C:\mnt\hdd0 is a mount point
		{"ancestor", func(_, link string) string { return filepath.Join(link, "models") }, func(real string) string { return real }},
		// OLLAMA_MODELS is the junction itself
		{"models dir", func(_, link string) string { return link }, func(real string) string { return filepath.Join(real, "models") }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			base := t.TempDir()
			real := filepath.Join(base, "real")
			if err := os.MkdirAll(filepath.Join(real, "models"), 0o755); err != nil {
				t.Fatal(err)
			}

			t.Setenv("OLLAMA_MODELS", filepath.Join(real, "models"))
			name := model.ParseName("example")
			config := Layer{
				MediaType: "application/vnd.docker.container.image.v1+json",
				Digest:    "sha256:" + strings.Repeat("a", 64),
				Size:      12,
			}
			if err := WriteManifest(name, config, nil); err != nil {
				t.Fatal(err)
			}

			link := filepath.Join(base, "mnt")
			createJunction(t, link, tt.target(real))
			t.Setenv("OLLAMA_MODELS", tt.models(real, link))

			ms, err := Manifests(false)
			if err != nil {
				t.Fatalf("Manifests: %v", err)
			}
			if len(ms) != 1 {
				t.Fatalf("Manifests returned %d models, want 1", len(ms))
			}
			if _, err := ReadManifestData(name); err != nil {
				t.Fatalf("ReadManifestData: %v", err)
			}
		})
	}
}
