package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"path/filepath"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestDelete(t *testing.T) {
	gin.SetMode(gin.TestMode)

	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)

	var s Server

	_, digest := createBinFile(t, nil, nil)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:  "test",
		Files: map[string]string{"test.gguf": digest},
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:     "test2",
		Files:    map[string]string{"test.gguf": digest},
		Template: "{{ .System }} {{ .Prompt }}",
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test2", "latest"),
	})

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-136bf7c76bac2ec09d6617885507d37829e04b41acc47687d45e512b544e893a"),
		filepath.Join(p, "blobs", "sha256-6bcdb8859d417753645538d7bbfbd7ca91a3f0c191aef5379c53c05e86b669dd"),
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
		filepath.Join(p, "blobs", "sha256-fe7ac77b725cda2ccad03f88a880ecdfd7a33192d6cae08fce2c0ee1455991ed"),
	})

	w = createRequest(t, s.DeleteHandler, api.DeleteRequest{Name: "test"})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test2", "latest"),
	})

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-136bf7c76bac2ec09d6617885507d37829e04b41acc47687d45e512b544e893a"),
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
		filepath.Join(p, "blobs", "sha256-fe7ac77b725cda2ccad03f88a880ecdfd7a33192d6cae08fce2c0ee1455991ed"),
	})

	w = createRequest(t, s.DeleteHandler, api.DeleteRequest{Name: "test2"})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{})
	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{})
}

func TestDeleteDuplicateLayers(t *testing.T) {
	gin.SetMode(gin.TestMode)

	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	var s Server

	n := model.ParseName("test")

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(&model.ConfigV2{}); err != nil {
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
