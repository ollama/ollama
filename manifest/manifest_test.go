package manifest

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func createManifestAtRoot(t *testing.T, path, root, name string) {
	t.Helper()

	p := filepath.Join(path, root, name)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}

	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := json.NewEncoder(f).Encode(Manifest{}); err != nil {
		t.Fatal(err)
	}
}

func createManifest(t *testing.T, path, name string) {
	t.Helper()
	createManifestAtRoot(t, path, "manifests", name)
}

func TestWriteManifestStoresManifestAsBlob(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	name := model.ParseName("example")
	config := Layer{
		MediaType: "application/vnd.docker.container.image.v1+json",
		Digest:    "sha256:" + strings.Repeat("a", 64),
		Size:      12,
	}

	if err := WriteManifest(name, config, nil); err != nil {
		t.Fatal(err)
	}

	manifestPath, err := V2PathForName(name)
	if err != nil {
		t.Fatal(err)
	}
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}

	sum := sha256.Sum256(manifestData)
	digest := fmt.Sprintf("sha256:%x", sum)
	blobPath, err := BlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	blobData, err := os.ReadFile(blobPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(blobData, manifestData) {
		t.Fatal("manifest path and blob content differ")
	}

	m, err := ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	if got := m.Digest(); got != fmt.Sprintf("%x", sum) {
		t.Fatalf("digest = %q, want %x", got, sum)
	}
	if got := m.BlobDigest(); got != digest {
		t.Fatalf("blob digest = %q, want %q", got, digest)
	}
}

func TestParseNamedManifestLeavesLegacyManifestInPlace(t *testing.T) {
	models := t.TempDir()
	t.Setenv("OLLAMA_MODELS", models)

	name := model.ParseName("example")
	createManifest(t, models, name.Filepath())

	manifestPath, err := PathForName(name)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := ParseNamedManifest(name); err != nil {
		t.Fatal(err)
	}

	fi, err := os.Lstat(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode()&os.ModeSymlink != 0 {
		t.Fatal("legacy manifest was converted to a symlink while reading")
	}

	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	blobPath, err := BlobsPath(fmt.Sprintf("sha256:%x", sum))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(blobPath); !os.IsNotExist(err) {
		t.Fatalf("legacy manifest read created blob: %v", err)
	}
}

func TestMigrateManifestLinks(t *testing.T) {
	models := t.TempDir()
	t.Setenv("OLLAMA_MODELS", models)

	name := model.ParseName("example")
	createManifest(t, models, name.Filepath())

	migrated, err := MigrateManifestLinks()
	if err != nil {
		t.Fatal(err)
	}
	if migrated != 1 {
		t.Fatalf("migrated = %d, want 1", migrated)
	}

	manifestPath, err := V2PathForName(name)
	if err != nil {
		t.Fatal(err)
	}
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(manifestData)
	blobPath, err := BlobsPath(fmt.Sprintf("sha256:%x", sum))
	if err != nil {
		t.Fatal(err)
	}
	blobData, err := os.ReadFile(blobPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(blobData, manifestData) {
		t.Fatal("migrated manifest path and blob content differ")
	}

	legacyPath, err := PathForName(name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(legacyPath); !os.IsNotExist(err) {
		t.Fatalf("legacy manifest still exists: %v", err)
	}

	migrated, err = MigrateManifestLinks()
	if err != nil {
		t.Fatal(err)
	}
	if migrated != 0 {
		t.Fatalf("migrated on second run = %d, want 0", migrated)
	}

	if _, err := MigrateManifestLinks(); err != nil {
		t.Fatal(err)
	}
	manifestDataAfter, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(manifestDataAfter, manifestData) {
		t.Fatal("second migration changed manifest content")
	}
}

func TestRemoveLayersRemovesUnreferencedManifestBlob(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	name := model.ParseName("example")
	if err := WriteManifest(name, Layer{}, nil); err != nil {
		t.Fatal(err)
	}

	m, err := ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	blobPath, err := BlobsPath(m.BlobDigest())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(blobPath); err != nil {
		t.Fatal(err)
	}

	if err := m.Remove(); err != nil {
		t.Fatal(err)
	}
	if err := m.RemoveLayers(); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(blobPath); !os.IsNotExist(err) {
		t.Fatalf("manifest blob still exists: %v", err)
	}
}

func TestParseNamedManifestRejectsUnsafeSymlinks(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	name := model.ParseName("example")
	manifestPath, err := PathForName(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(manifestPath), 0o755); err != nil {
		t.Fatal(err)
	}

	t.Run("non blob basename", func(t *testing.T) {
		target := filepath.Join(t.TempDir(), "not-a-blob")
		if err := os.WriteFile(target, []byte(`{"schemaVersion":2}`), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(manifestPath); err != nil && !os.IsNotExist(err) {
			t.Fatal(err)
		}
		if err := os.Symlink(target, manifestPath); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}

		_, err := ParseNamedManifest(name)
		if err == nil || !strings.Contains(err.Error(), "not a sha256 blob") {
			t.Fatalf("err = %v, want not a sha256 blob", err)
		}
	})

	t.Run("blob basename outside blob store", func(t *testing.T) {
		data := []byte(`{"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json"}`)
		sum := sha256.Sum256(data)
		target := filepath.Join(t.TempDir(), fmt.Sprintf("sha256-%x", sum))
		if err := os.WriteFile(target, data, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(manifestPath); err != nil && !os.IsNotExist(err) {
			t.Fatal(err)
		}
		if err := os.Symlink(target, manifestPath); err != nil {
			t.Skipf("symlink unavailable: %v", err)
		}

		_, err := ParseNamedManifest(name)
		if err == nil || !strings.Contains(err.Error(), "does not match blob") {
			t.Fatalf("err = %v, want does not match blob", err)
		}
	})
}

func TestParseNamedManifestPrefersV2(t *testing.T) {
	models := t.TempDir()
	t.Setenv("OLLAMA_MODELS", models)

	name := model.ParseName("example")

	legacyPath, err := PathForName(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(legacyPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(legacyPath, []byte(`{"schemaVersion":2,"mediaType":"legacy"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := WriteManifestData(name, []byte(`{"schemaVersion":2,"mediaType":"v2"}`)); err != nil {
		t.Fatal(err)
	}

	m, err := ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	if m.MediaType != "v2" {
		t.Fatalf("media type = %q, want %q", m.MediaType, "v2")
	}
}

func TestManifestsV2ShadowsLegacy(t *testing.T) {
	models := t.TempDir()
	t.Setenv("OLLAMA_MODELS", models)

	name := model.ParseName("example")
	createManifest(t, models, name.Filepath())
	if err := WriteManifestData(name, []byte(`{"schemaVersion":2,"mediaType":"v2"}`)); err != nil {
		t.Fatal(err)
	}

	ms, err := Manifests(true)
	if err != nil {
		t.Fatal(err)
	}

	if len(ms) != 1 {
		t.Fatalf("manifest count = %d, want 1", len(ms))
	}

	var m *Manifest
	for gotName, gotManifest := range ms {
		if gotName.EqualFold(model.ParseName("example")) {
			m = gotManifest
			break
		}
	}
	if m == nil {
		t.Fatalf("missing v2 manifest for %s", name)
	}
	if m.MediaType != "v2" {
		t.Fatalf("media type = %q, want %q", m.MediaType, "v2")
	}
}

func TestManifests(t *testing.T) {
	cases := map[string]struct {
		ps               []string
		wantValidCount   int
		wantInvalidCount int
	}{
		"empty": {},
		"single": {
			ps: []string{
				filepath.Join("host", "namespace", "model", "tag"),
			},
			wantValidCount: 1,
		},
		"multiple": {
			ps: []string{
				filepath.Join("registry.ollama.ai", "library", "llama3", "latest"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q4_0"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q4_1"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q8_0"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q5_0"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q5_1"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q2_K"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q3_K_S"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q3_K_M"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q3_K_L"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q4_K_S"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q4_K_M"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q5_K_S"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q5_K_M"),
				filepath.Join("registry.ollama.ai", "library", "llama3", "q6_K"),
			},
			wantValidCount: 15,
		},
		"hidden": {
			ps: []string{
				filepath.Join("host", "namespace", "model", "tag"),
				filepath.Join("host", "namespace", "model", ".hidden"),
			},
			wantValidCount:   1,
			wantInvalidCount: 1,
		},
		"subdir": {
			ps: []string{
				filepath.Join("host", "namespace", "model", "tag", "one"),
				filepath.Join("host", "namespace", "model", "tag", "another", "one"),
			},
			wantInvalidCount: 2,
		},
		"upper tag": {
			ps: []string{
				filepath.Join("host", "namespace", "model", "TAG"),
			},
			wantValidCount: 1,
		},
		"upper model": {
			ps: []string{
				filepath.Join("host", "namespace", "MODEL", "tag"),
			},
			wantValidCount: 1,
		},
		"upper namespace": {
			ps: []string{
				filepath.Join("host", "NAMESPACE", "model", "tag"),
			},
			wantValidCount: 1,
		},
		"upper host": {
			ps: []string{
				filepath.Join("HOST", "namespace", "model", "tag"),
			},
			wantValidCount: 1,
		},
	}

	for n, wants := range cases {
		t.Run(n, func(t *testing.T) {
			d := t.TempDir()
			t.Setenv("OLLAMA_MODELS", d)

			for _, p := range wants.ps {
				createManifest(t, d, p)
			}

			ms, err := Manifests(true)
			if err != nil {
				t.Fatal(err)
			}

			var ns []model.Name
			for k := range ms {
				ns = append(ns, k)
			}

			var gotValidCount, gotInvalidCount int
			for _, p := range wants.ps {
				n := model.ParseNameFromFilepath(p)
				if n.IsValid() {
					gotValidCount++
				} else {
					gotInvalidCount++
				}

				if !n.IsValid() && slices.Contains(ns, n) {
					t.Errorf("unexpected invalid name: %s", p)
				} else if n.IsValid() && !slices.Contains(ns, n) {
					t.Errorf("missing valid name: %s", p)
				}
			}

			if gotValidCount != wants.wantValidCount {
				t.Errorf("got valid count %d, want %d", gotValidCount, wants.wantValidCount)
			}

			if gotInvalidCount != wants.wantInvalidCount {
				t.Errorf("got invalid count %d, want %d", gotInvalidCount, wants.wantInvalidCount)
			}
		})
	}
}
