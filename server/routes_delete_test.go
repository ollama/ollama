package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
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
		filepath.Join(p, "blobs", "sha256-eb4aa77a05d4846c309cc5ccf00a3eeda12d4d207bde3ec2b597bed8a73c3a9d"),
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
		filepath.Join(p, "blobs", "sha256-eb4aa77a05d4846c309cc5ccf00a3eeda12d4d207bde3ec2b597bed8a73c3a9d"),
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

	config, err := manifest.NewLayer(&b, "application/vnd.docker.container.image.v1+json")
	if err != nil {
		t.Fatal(err)
	}

	// create a manifest with duplicate layers
	if err := manifest.WriteManifest(n, config, []manifest.Layer{config}); err != nil {
		t.Fatal(err)
	}

	w := createRequest(t, s.DeleteHandler, api.DeleteRequest{Name: "test"})
	if w.Code != http.StatusOK {
		t.Errorf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{})
}

func TestDeleteCloudSourceNormalizesToLegacyName(t *testing.T) {
	gin.SetMode(gin.TestMode)

	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)

	var s Server

	_, digest := createBinFile(t, nil, nil)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:  "gpt-oss:20b-cloud",
		Files: map[string]string{"test.gguf": digest},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "gpt-oss", "20b-cloud"),
	})

	w = createRequest(t, s.DeleteHandler, api.DeleteRequest{Name: "gpt-oss:20b:cloud"})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d (%s)", w.Code, w.Body.String())
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{})
}

// TestDeletePrunesOrphanedBlobs reproduces
// https://github.com/ollama/ollama/issues/18595: blobs referenced by no
// manifest must be garbage-collected when a model is deleted, not only by
// the housekeeping pass at server startup.
func TestDeletePrunesOrphanedBlobs(t *testing.T) {
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

	// Plant two orphan blobs referenced by no manifest: one older than the
	// prune grace period and one freshly written. Distinct KV metadata keeps
	// the digests distinct.
	_, oldDigest := createBinFile(t, map[string]any{"general.architecture": "orphan-old"}, nil)
	_, freshDigest := createBinFile(t, map[string]any{"general.architecture": "orphan-fresh"}, nil)

	oldBlob, err := manifest.BlobsPath(oldDigest)
	if err != nil {
		t.Fatal(err)
	}
	freshBlob, err := manifest.BlobsPath(freshDigest)
	if err != nil {
		t.Fatal(err)
	}

	aged := time.Now().Add(-2 * layerPruneGracePeriod)
	if err := os.Chtimes(oldBlob, aged, aged); err != nil {
		t.Fatal(err)
	}

	w = createRequest(t, s.DeleteHandler, api.DeleteRequest{Name: "test"})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{})

	// The model's own blobs and the aged orphan are reclaimed at delete
	// time. The fresh orphan is spared by the grace period: it may belong
	// to an in-flight pull.
	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{freshBlob})
}
