package manifest

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
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

func createManifestForTest(configDigest, layerDigest, runner string) Manifest {
	return Manifest{
		SchemaVersion: 2,
		MediaType:     MediaTypeManifest,
		Runner:        runner,
		Format:        FormatGGUF,
		Config: Layer{
			MediaType: "application/vnd.docker.container.image.v1+json",
			Digest:    configDigest,
			Size:      12,
		},
		Layers: []Layer{
			{
				MediaType: "application/vnd.ollama.image.model",
				Digest:    layerDigest,
				Size:      34,
			},
		},
	}
}

func createManifestListData(t *testing.T, manifests ...Manifest) []byte {
	t.Helper()

	ml := Manifest{
		SchemaVersion: 2,
		MediaType:     MediaTypeManifestList,
		Manifests:     manifests,
	}

	data, err := json.Marshal(ml)
	if err != nil {
		t.Fatal(err)
	}

	return data
}

func writeManifestBlobForTest(t *testing.T, data []byte) string {
	t.Helper()

	digest, err := writeManifestBlob(data)
	if err != nil {
		t.Fatal(err)
	}

	return digest
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

func TestSelectManifestUsesRunnerPreference(t *testing.T) {
	ml := Manifest{
		SchemaVersion: 2,
		MediaType:     MediaTypeManifestList,
		Manifests: []Manifest{
			createManifestForTest("sha256:"+strings.Repeat("a", 64), "sha256:"+strings.Repeat("b", 64), RunnerGGML),
			createManifestForTest("sha256:"+strings.Repeat("c", 64), "sha256:"+strings.Repeat("d", 64), RunnerLlamaCPP),
		},
	}

	child, err := selectManifestWithPreferences(ml.Manifests, []string{RunnerLlamaCPP, RunnerGGML})
	if err != nil {
		t.Fatal(err)
	}
	if child.Runner != RunnerLlamaCPP {
		t.Fatalf("runner = %q, want %q", child.Runner, RunnerLlamaCPP)
	}
}

func TestSelectManifestReferenceDoesNotResolveBlob(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	ref, err := NewManifestReference("sha256:"+strings.Repeat("a", 64), RunnerGGML, FormatGGUF)
	if err != nil {
		t.Fatal(err)
	}

	child, err := selectManifestReferenceWithPreferences([]Manifest{ref}, []string{RunnerGGML})
	if err != nil {
		t.Fatal(err)
	}
	if got := child.BlobDigest(); got != "sha256:"+strings.Repeat("a", 64) {
		t.Fatalf("blob digest = %q, want selected reference digest", got)
	}
}

func TestSelectManifestRejectsOldOllamaRunner(t *testing.T) {
	_, err := selectManifestWithPreferences([]Manifest{
		createManifestForTest("sha256:"+strings.Repeat("a", 64), "sha256:"+strings.Repeat("b", 64), "ollama"),
	}, []string{RunnerGGML})
	if !errors.Is(err, ErrNoCompatibleManifest) {
		t.Fatalf("err = %v, want %v", err, ErrNoCompatibleManifest)
	}
}

func TestParseNamedManifestResolvesManifestList(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	name := model.ParseName("example")

	ggml := createManifestForTest("sha256:"+strings.Repeat("a", 64), "sha256:"+strings.Repeat("b", 64), RunnerGGML)
	ggmlData, err := json.Marshal(ggml)
	if err != nil {
		t.Fatal(err)
	}
	ggmlDigest := writeManifestBlobForTest(t, ggmlData)

	llamacpp := createManifestForTest("sha256:"+strings.Repeat("c", 64), "sha256:"+strings.Repeat("d", 64), RunnerLlamaCPP)
	llamacppData, err := json.Marshal(llamacpp)
	if err != nil {
		t.Fatal(err)
	}
	llamacppDigest := writeManifestBlobForTest(t, llamacppData)

	parentData := createManifestListData(t, llamacpp, ggml)

	if err := WriteManifestData(name, parentData); err != nil {
		t.Fatal(err)
	}

	m, err := ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}

	parentSum := sha256.Sum256(parentData)
	if got := m.Digest(); got != fmt.Sprintf("%x", parentSum) {
		t.Fatalf("digest = %q, want %x", got, parentSum)
	}
	if got := m.BlobDigest(); got != fmt.Sprintf("sha256:%x", parentSum) {
		t.Fatalf("blob digest = %q, want sha256:%x", got, parentSum)
	}
	if got := m.SelectedDigest(); got != strings.TrimPrefix(ggmlDigest, "sha256:") {
		t.Fatalf("selected digest = %q, want %q", got, strings.TrimPrefix(ggmlDigest, "sha256:"))
	}
	if got := m.Runner; got != RunnerGGML {
		t.Fatalf("runner = %q, want %q", got, RunnerGGML)
	}
	if got := m.Format; got != FormatGGUF {
		t.Fatalf("format = %q, want %q", got, FormatGGUF)
	}
	if got := m.Config.Digest; got != "sha256:"+strings.Repeat("a", 64) {
		t.Fatalf("config digest = %q, want selected child config", got)
	}

	m, err = ParseNamedManifestForRunner(name, RunnerLlamaCPP)
	if err != nil {
		t.Fatal(err)
	}
	if got := m.Runner; got != RunnerLlamaCPP {
		t.Fatalf("runner = %q, want %q", got, RunnerLlamaCPP)
	}
	if got := m.SelectedDigest(); got != strings.TrimPrefix(llamacppDigest, "sha256:") {
		t.Fatalf("selected digest = %q, want %q", got, strings.TrimPrefix(llamacppDigest, "sha256:"))
	}
	if got := m.Config.Digest; got != "sha256:"+strings.Repeat("c", 64) {
		t.Fatalf("config digest = %q, want selected child config", got)
	}

	referenced, err := ReferencedBlobDigestsForName(name)
	if err != nil {
		t.Fatal(err)
	}
	for _, digest := range []string{llamacppDigest, ggmlDigest} {
		if !slices.Contains(referenced, digest) {
			t.Fatalf("referenced blob digests missing child manifest %s", digest)
		}
	}

	raw, err := ReadManifestData(name)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, parentData) {
		t.Fatal("ReadManifestData did not return the parent manifest list")
	}

	selected, err := ReadSelectedManifestData(name)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(selected, ggmlData) {
		t.Fatal("ReadSelectedManifestData did not return the selected child manifest")
	}
}

func TestTotalSizeForNameIncludesAllManifestListChildren(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	name := model.ParseName("example")

	sharedLayerData := []byte("shared layer")
	ggufConfigData := []byte("gguf config")
	ggufLayerData := []byte("gguf layer")
	mlxConfigData := []byte("mlx config")
	mlxLayerData := []byte("mlx layer")

	sharedLayerDigest := writeManifestBlobForTest(t, sharedLayerData)
	ggufConfigDigest := writeManifestBlobForTest(t, ggufConfigData)
	ggufLayerDigest := writeManifestBlobForTest(t, ggufLayerData)
	mlxConfigDigest := writeManifestBlobForTest(t, mlxConfigData)
	mlxLayerDigest := writeManifestBlobForTest(t, mlxLayerData)

	gguf := Manifest{
		SchemaVersion: 2,
		MediaType:     MediaTypeManifest,
		Runner:        RunnerGGML,
		Format:        FormatGGUF,
		Config: Layer{
			MediaType: "application/vnd.docker.container.image.v1+json",
			Digest:    ggufConfigDigest,
			Size:      int64(len(ggufConfigData)),
		},
		Layers: []Layer{
			{
				MediaType: "application/vnd.ollama.image.model",
				Digest:    sharedLayerDigest,
				Size:      int64(len(sharedLayerData)),
			},
			{
				MediaType: "application/vnd.ollama.image.model",
				Digest:    ggufLayerDigest,
				Size:      int64(len(ggufLayerData)),
			},
		},
	}
	ggufData, err := json.Marshal(gguf)
	if err != nil {
		t.Fatal(err)
	}
	ggufManifestDigest := writeManifestBlobForTest(t, ggufData)
	ggufRef, err := NewManifestReference(ggufManifestDigest, gguf.Runner, gguf.Format)
	if err != nil {
		t.Fatal(err)
	}

	mlx := Manifest{
		SchemaVersion: 2,
		MediaType:     MediaTypeManifest,
		Runner:        RunnerMLX,
		Format:        FormatSafetensors,
		Config: Layer{
			MediaType: "application/vnd.docker.container.image.v1+json",
			Digest:    mlxConfigDigest,
			Size:      int64(len(mlxConfigData)),
		},
		Layers: []Layer{
			{
				MediaType: MediaTypeImageTensor,
				Digest:    sharedLayerDigest,
				Size:      int64(len(sharedLayerData)),
			},
			{
				MediaType: MediaTypeImageTensor,
				Digest:    mlxLayerDigest,
				Size:      int64(len(mlxLayerData)),
			},
		},
	}
	mlxData, err := json.Marshal(mlx)
	if err != nil {
		t.Fatal(err)
	}
	mlxManifestDigest := writeManifestBlobForTest(t, mlxData)
	mlxRef, err := NewManifestReference(mlxManifestDigest, mlx.Runner, mlx.Format)
	if err != nil {
		t.Fatal(err)
	}

	if err := WriteManifestData(name, createManifestListData(t, ggufRef, mlxRef)); err != nil {
		t.Fatal(err)
	}

	size, err := TotalSizeForName(name)
	if err != nil {
		t.Fatal(err)
	}

	want := int64(len(ggufConfigData) + len(sharedLayerData) + len(ggufLayerData) + len(mlxConfigData) + len(mlxLayerData))
	if size != want {
		t.Fatalf("size = %d, want %d", size, want)
	}
}

func TestPartialManifestListTracksPresentAndMissingChildren(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	name := model.ParseName("example")

	configData := []byte("gguf config")
	layerData := []byte("gguf layer")
	configDigest := writeManifestBlobForTest(t, configData)
	layerDigest := writeManifestBlobForTest(t, layerData)

	child := Manifest{
		SchemaVersion: 2,
		MediaType:     MediaTypeManifest,
		Runner:        RunnerGGML,
		Format:        FormatGGUF,
		Config: Layer{
			MediaType: "application/vnd.docker.container.image.v1+json",
			Digest:    configDigest,
			Size:      int64(len(configData)),
		},
		Layers: []Layer{
			{
				MediaType: "application/vnd.ollama.image.model",
				Digest:    layerDigest,
				Size:      int64(len(layerData)),
			},
		},
	}
	childData, err := json.Marshal(child)
	if err != nil {
		t.Fatal(err)
	}
	childDigest := writeManifestBlobForTest(t, childData)
	childRef, err := NewManifestReference(childDigest, child.Runner, child.Format)
	if err != nil {
		t.Fatal(err)
	}

	missingDigest := "sha256:" + strings.Repeat("e", 64)
	missingRef, err := NewManifestReference(missingDigest, RunnerMLX, FormatSafetensors)
	if err != nil {
		t.Fatal(err)
	}

	parentData := createManifestListData(t, childRef, missingRef)
	if err := WriteManifestData(name, parentData); err != nil {
		t.Fatal(err)
	}
	parentSum := sha256.Sum256(parentData)
	parentDigest := fmt.Sprintf("sha256:%x", parentSum)

	referenced, err := ReferencedBlobDigestsForName(name)
	if err != nil {
		t.Fatal(err)
	}
	for _, digest := range []string{parentDigest, childDigest, missingDigest, configDigest, layerDigest} {
		if !slices.Contains(referenced, digest) {
			t.Fatalf("referenced blob digests missing %s: %#v", digest, referenced)
		}
	}

	size, err := TotalSizeForName(name)
	if err != nil {
		t.Fatal(err)
	}
	want := int64(len(configData) + len(layerData))
	if size != want {
		t.Fatalf("size = %d, want %d", size, want)
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

func TestRemoveNamedRemovesUnreferencedManifestBlob(t *testing.T) {
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

	if err := RemoveNamed(name); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(blobPath); !os.IsNotExist(err) {
		t.Fatalf("manifest blob still exists: %v", err)
	}
}

func TestRemoveNamedTracksManifestListChildBlobs(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	ggmlConfigDigest := writeManifestBlobForTest(t, []byte("ggml config"))
	ggmlLayerDigest := writeManifestBlobForTest(t, []byte("ggml layer"))
	ggml := createManifestForTest(ggmlConfigDigest, ggmlLayerDigest, RunnerGGML)
	ggmlData, err := json.Marshal(ggml)
	if err != nil {
		t.Fatal(err)
	}
	writeManifestBlobForTest(t, ggmlData)

	llamacppConfigDigest := writeManifestBlobForTest(t, []byte("llamacpp config"))
	llamacppLayerDigest := writeManifestBlobForTest(t, []byte("llamacpp layer"))
	llamacpp := createManifestForTest(llamacppConfigDigest, llamacppLayerDigest, RunnerLlamaCPP)
	llamacppData, err := json.Marshal(llamacpp)
	if err != nil {
		t.Fatal(err)
	}
	writeManifestBlobForTest(t, llamacppData)

	parentData := createManifestListData(t, ggml, llamacpp)

	nameA := model.ParseName("example-a")
	nameB := model.ParseName("example-b")
	if err := WriteManifestData(nameA, parentData); err != nil {
		t.Fatal(err)
	}
	if err := WriteManifestData(nameB, parentData); err != nil {
		t.Fatal(err)
	}

	parentSum := sha256.Sum256(parentData)
	parentPath, err := BlobsPath(fmt.Sprintf("sha256:%x", parentSum))
	if err != nil {
		t.Fatal(err)
	}
	referencedBlobs, err := ReferencedBlobDigestsForName(nameA)
	if err != nil {
		t.Fatal(err)
	}

	if err := RemoveNamed(nameA); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(parentPath); err != nil {
		t.Fatalf("parent list blob was removed while another model uses it: %v", err)
	}
	for _, digest := range referencedBlobs {
		blob, err := BlobsPath(digest)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(blob); err != nil {
			t.Fatalf("referenced blob %s was removed while another model uses it: %v", digest, err)
		}
	}

	if err := RemoveNamed(nameB); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(parentPath); !os.IsNotExist(err) {
		t.Fatalf("parent list blob still exists after final remove: %v", err)
	}
	for _, digest := range referencedBlobs {
		blob, err := BlobsPath(digest)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(blob); !os.IsNotExist(err) {
			t.Fatalf("referenced blob %s still exists after final remove: %v", digest, err)
		}
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
