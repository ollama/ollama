package updater

import (
	"archive/zip"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBundleLayoutAcceptsReleaseZip(t *testing.T) {
	// Names a `ditto -c -k --norsrc --keepParent Ollama.app` archive contains.
	names := []string{
		"Ollama.app/",
		"Ollama.app/Contents/",
		"Ollama.app/Contents/Info.plist",
		"Ollama.app/Contents/MacOS/",
		"Ollama.app/Contents/MacOS/Ollama",
		"Ollama.app/Contents/Resources/ollama",
	}
	if err := checkBundleNames(t.TempDir(), names); err != nil {
		t.Fatal(err)
	}
}

func TestBundleLayoutRejectsWrongAppName(t *testing.T) {
	// ditto --keepParent on a directory named New.app produces this root.
	// The rehearsal used to zip that name; the updater correctly refuses it.
	dir := t.TempDir()
	err := checkBundleNames(dir, []string{
		"New.app/",
		"New.app/Contents/Info.plist",
	})
	if err == nil || !strings.Contains(err.Error(), "bundle contains invalid path: New.app/") {
		t.Fatalf("err = %v", err)
	}
}

func TestBundleLayoutRejectsTraversal(t *testing.T) {
	for _, name := range []string{
		"Ollama.app/../evil",
		"../Ollama.app/Contents/Info.plist",
		"Ollama.app/Contents/../../etc/passwd",
		"/tmp/Ollama.app/Contents/Info.plist",
		"__MACOSX/Ollama.app/Contents/Info.plist",
		"Ollama.app/Contents/MacOS/../../../../../../etc/passwd",
	} {
		err := checkBundleNames(t.TempDir(), []string{name})
		if err == nil {
			t.Fatalf("accepted %s", name)
		}
		if !strings.Contains(err.Error(), "bundle contains invalid path:") {
			t.Fatalf("%s: %v", name, err)
		}
	}
}

func TestBundleLayoutRelativeScopeStillRejectsParent(t *testing.T) {
	// DoUpgrade strips the Ollama.app/ prefix and then checks the remainder.
	root := t.TempDir()
	if _, err := bundleEntryPath(root, "Contents/MacOS/Ollama", bundleEntryRelative); err != nil {
		t.Fatal(err)
	}
	if _, err := bundleEntryPath(root, "../etc/passwd", bundleEntryRelative); err == nil {
		t.Fatal("relative traversal was accepted")
	}
}

func TestBundleLayoutZipEntries(t *testing.T) {
	good := zipWithNames(t, "Ollama.app/", "Ollama.app/Contents/Info.plist", "Ollama.app/Contents/MacOS/Ollama")
	if err := checkZipEntries(good); err != nil {
		t.Fatal(err)
	}
	bad := zipWithNames(t, "New.app/", "New.app/Contents/Info.plist")
	err := checkZipEntries(bad)
	if err == nil || !strings.Contains(err.Error(), "New.app/") {
		t.Fatalf("err = %v", err)
	}
	escaped := zipWithNames(t, "Ollama.app/../evil")
	if err := checkZipEntries(escaped); err == nil {
		t.Fatal("zip traversal was accepted")
	}
}

func checkBundleNames(root string, names []string) error {
	for _, name := range names {
		if _, err := bundleEntryPath(root, name, bundleEntryWithArchiveRoot); err != nil {
			return err
		}
	}
	return nil
}

// checkZipEntries applies the same per-entry check verifyDownload uses.
func checkZipEntries(path string) error {
	r, err := zip.OpenReader(path)
	if err != nil {
		return err
	}
	defer r.Close()
	root := filepath.Dir(path)
	for _, f := range r.File {
		if _, err := bundleEntryPath(root, f.Name, bundleEntryWithArchiveRoot); err != nil {
			return err
		}
	}
	return nil
}

func zipWithNames(t *testing.T, names ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "bundle.zip")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	w := zip.NewWriter(f)
	for _, name := range names {
		hdr := &zip.FileHeader{Name: name}
		if strings.HasSuffix(name, "/") {
			hdr.SetMode(os.ModeDir | 0o755)
		}
		if _, err := w.CreateHeader(hdr); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}
