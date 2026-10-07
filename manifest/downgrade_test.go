package manifest

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func digestOf(data []byte) string {
	sum := sha256.Sum256(data)
	return fmt.Sprintf("%x", sum)
}

// downgradePrune simulates a pre-manifest-list daemon's startup garbage
// collection: every file in blobs that no legacy manifest references through
// its config or layers is deleted.
func downgradePrune(t *testing.T) {
	t.Helper()

	blobs, err := BlobsPath("")
	if err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(blobs)
	if err != nil {
		t.Fatal(err)
	}

	legacyRoot, err := Path()
	if err != nil {
		t.Fatal(err)
	}
	retained := make(map[string]struct{})
	err = filepath.Walk(legacyRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var m Manifest
		if err := json.Unmarshal(data, &m); err != nil {
			return err
		}
		for _, layer := range append(m.Layers, m.Config) {
			if layer.Digest == "" {
				continue
			}
			digest, err := canonicalBlobDigest(layer.Digest)
			if err != nil {
				continue
			}
			retained[digest] = struct{}{}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, entry := range entries {
		digest, ok := DigestReference(entry.Name())
		if !ok {
			continue
		}
		if _, used := retained[digest]; used {
			continue
		}
		if err := os.Remove(filepath.Join(blobs, entry.Name())); err != nil {
			t.Fatal(err)
		}
	}
}

func danglingEntry(t *testing.T, name model.Name) {
	t.Helper()

	v2Path, err := V2PathForName(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(v2Path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(v2Path); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join("..", "..", "blobs", "sha256-"+strings.Repeat("0", 64)), v2Path); err != nil {
		t.Fatal(err)
	}
}

func TestResolvePathForNameFallsBackWhenV2EntryDangling(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	name := model.ParseName("example")

	config, err := NewLayer(strings.NewReader(`{"model_type":"test"}`), MediaTypeImageConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteLegacyManifestData(name, mustMarshal(t, Manifest{SchemaVersion: 2, Config: config})); err != nil {
		t.Fatal(err)
	}
	danglingEntry(t, name)

	path, err := ResolvePathForName(name)
	if err != nil {
		t.Fatalf("resolve with dangling v2 entry: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("resolved path %s: %v", path, err)
	}
	if got, err := V2PathForName(name); err == nil && path == got {
		t.Fatal("resolved dangling v2 entry instead of the legacy manifest")
	}
}

func TestResolvePathForNameDanglingWithoutLegacyIsNotExists(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	name := model.ParseName("example")
	danglingEntry(t, name)

	if _, err := ResolvePathForName(name); err == nil {
		t.Fatal("resolved a dangling v2 entry with no fallback")
	} else if !os.IsNotExist(err) {
		t.Fatalf("resolve error = %v, want not exist", err)
	}
}

func mustMarshal(t *testing.T, m Manifest) []byte {
	t.Helper()

	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// TestDowngradePruneKeepsManifestListBlobs covers the downgrade round trip:
// v2 manifest documents must survive a pre-manifest-list daemon's garbage
// collection when a legacy anchor references them.
func TestDowngradePruneKeepsManifestListBlobs(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	name := model.ParseName("example")

	// Build the model the way a pull does: layers, then a child manifest
	// blob, then the parent list, then the legacy anchor.
	weights, err := NewLayer(strings.NewReader("weights"), MediaTypeImageModel)
	if err != nil {
		t.Fatal(err)
	}
	config, err := NewLayer(strings.NewReader(`{"model_type":"test"}`), MediaTypeImageConfig)
	if err != nil {
		t.Fatal(err)
	}

	child := Manifest{SchemaVersion: 2, Config: config, Layers: []Layer{weights}}
	childDigest, err := WriteManifestBlob(mustMarshal(t, child))
	if err != nil {
		t.Fatal(err)
	}

	childRef, err := NewManifestReference(childDigest, RunnerLlamaCPP, FormatGGUF)
	if err != nil {
		t.Fatal(err)
	}

	parentDigest, err := WriteManifestList(name, []Manifest{childRef})
	if err != nil {
		t.Fatal(err)
	}

	if err := WriteLegacyAnchor(name, &child, parentDigest, childDigest); err != nil {
		t.Fatal(err)
	}

	downgradePrune(t)

	for digest, what := range map[string]string{
		parentDigest:   "parent manifest list blob",
		childDigest:    "child manifest blob",
		weights.Digest: "weights blob",
	} {
		path, err := BlobsPath(digest)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("%s did not survive the downgrade prune: %v", what, err)
		}
	}

	// The v2 entry must still resolve after re-upgrade.
	path, err := ResolvePathForName(name)
	if err != nil {
		t.Fatalf("resolve after downgrade prune: %v", err)
	}
	var parent Manifest
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &parent); err != nil {
		t.Fatal(err)
	}
	if parent.MediaType != MediaTypeManifestList || len(parent.Manifests) != 1 {
		t.Fatalf("resolved manifest = %+v, want the manifest list", parent)
	}
}

// TestDowngradePruneKeepsPlainPullBlobs covers plain pulls: the anchor
// references the model's own manifest document blob.
func TestDowngradePruneKeepsPlainPullBlobs(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	name := model.ParseName("example")

	weights, err := NewLayer(strings.NewReader("weights"), MediaTypeImageModel)
	if err != nil {
		t.Fatal(err)
	}
	config, err := NewLayer(strings.NewReader(`{"model_type":"test"}`), MediaTypeImageConfig)
	if err != nil {
		t.Fatal(err)
	}

	mf := Manifest{SchemaVersion: 2, Config: config, Layers: []Layer{weights}}
	data := mustMarshal(t, mf)
	if err := WriteManifestData(name, data); err != nil {
		t.Fatal(err)
	}

	parentDigest := "sha256:" + digestOf(data)
	if err := WriteLegacyAnchor(name, &mf, parentDigest); err != nil {
		t.Fatal(err)
	}

	downgradePrune(t)

	for digest, what := range map[string]string{
		parentDigest:   "manifest document blob",
		weights.Digest: "weights blob",
	} {
		path, err := BlobsPath(digest)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("%s did not survive the downgrade prune: %v", what, err)
		}
	}
}
