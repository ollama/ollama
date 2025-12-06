package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

func TestDeleteHandler(t *testing.T) {
	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	envconfig.LoadConfig()

	s := &Server{}

	w := createRequest(t, s.DeleteHandler, api.DeleteRequest{Name: "test"})
	if w.Code != http.StatusNotFound {
		t.Errorf("expected status code 404, actual %d", w.Code)
	}

	blobPath := filepath.Join(p, "blobs", "sha256-22222222222222222222222222222222222222222222222222222222222222222")
	if err := os.MkdirAll(filepath.Dir(blobPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(blobPath, []byte("config"), 0o644); err != nil {
		t.Fatal(err)
	}

	n := model.Name{
		Host:      "registry.ollama.ai",
		Namespace: "library",
		Model:     "test",
		Tag:       "latest",
	}

	config := Layer{
		MediaType: "config",
		Digest:    "sha256:2222222222222222222222222222222222222222222222222222222222222222",
		Size:      6,
	}

	manifestPath := filepath.Join(p, "manifests", n.Filepath())
	if err := os.MkdirAll(filepath.Dir(manifestPath), 0o755); err != nil {
		t.Fatal(err)
	}

	m := Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Config:        config,
		Layers:        []Layer{config},
	}

	manifestFile, err := os.Create(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	defer manifestFile.Close()
	if err := json.NewEncoder(manifestFile).Encode(m); err != nil {
		t.Fatal(err)
	}

	// Verify blobs exist
	checkFileExists(t, filepath.Join(p, "blobs", "sha256-*"), []string{blobPath})

	w = createRequest(t, s.DeleteHandler, api.DeleteRequest{Name: "test"})
	if w.Code != http.StatusOK {
		t.Errorf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{})
	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{})
}

func TestDeleteDuplicateLayers(t *testing.T) {
	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	envconfig.LoadConfig()

	s := &Server{}

	n := model.Name{
		Host:      "registry.ollama.ai",
		Namespace: "library",
		Model:     "test",
		Tag:       "latest",
	}

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(&ConfigV2{}); err != nil {
		t.Fatal(err)
	}

	config, err := NewLayer(&b, "application/vnd.docker.container.image.v1+json")
	if err != nil {
		t.Fatal(err)
	}

	// create a manifest with duplicate layers
	if err := WriteManifest(n, config, []Layer{config}); err != nil {
		t.Fatal(err)
	}

	w := createRequest(t, s.DeleteHandler, api.DeleteRequest{Name: "test"})
	if w.Code != http.StatusOK {
		t.Errorf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{})
}
