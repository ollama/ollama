//go:build windows || darwin

package ui

import (
	"io/fs"
	"strings"
	"testing"
)

// TestEmbeddedAssets verifies that the correct UI assets are embedded.
// This test will FAIL THE BUILD if wrong files are embedded.
func TestEmbeddedAssets(t *testing.T) {
	fsys, err := fs.Sub(appFS, "app/dist")
	if err != nil {
		t.Fatal("app/dist not found in embedded filesystem - UI not built")
	}

	data, err := fs.ReadFile(fsys, "index.html")
	if err != nil {
		t.Fatal("index.html not found - run 'go generate' first")
	}

	html := string(data)

	if strings.Contains(html, "/src/main.tsx") {
		t.Fatal("Wrong index.html embedded: has /src/main.tsx (dev paths). The UI was not built. Run 'npm run build' first.")
	}

	if !strings.Contains(html, "/assets/index-") {
		t.Fatal("Wrong index.html embedded: missing /assets/index-* (production paths). The UI was not built correctly.")
	}

	if _, err := fsys.Open("assets"); err != nil {
		t.Fatal("assets/ directory not found - UI build incomplete")
	}

	t.Log("Embedded assets verified - UI built correctly")
}
