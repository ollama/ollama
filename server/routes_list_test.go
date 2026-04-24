package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"slices"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestList(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Setenv("OLLAMA_MODELS", t.TempDir())

	expectNames := []string{
		"mistral:7b-instruct-q4_0",
		"zephyr:7b-beta-q5_K_M",
		"apple/OpenELM:latest",
		"boreas:2b-code-v1.5-q6_K",
		"notus:7b-v1-IQ2_S",
		// TODO: host:port currently fails on windows (#4107)
		// "localhost:5000/library/eurus:700b-v0.5-iq3_XXS",
		"mynamespace/apeliotes:latest",
		"myhost/mynamespace/lips:code",
	}

	var s Server
	for _, n := range expectNames {
		_, digest := createBinFile(t, nil, nil)

		createRequest(t, s.CreateHandler, api.CreateRequest{
			Name:  n,
			Files: map[string]string{"test.gguf": digest},
		})
	}

	w := createRequest(t, s.ListHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	var resp api.ListResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	if len(resp.Models) != len(expectNames) {
		t.Fatalf("expected %d models, actual %d", len(expectNames), len(resp.Models))
	}

	actualNames := make([]string, len(resp.Models))
	for i, m := range resp.Models {
		actualNames[i] = m.Name
	}

	slices.Sort(actualNames)
	slices.Sort(expectNames)

	if !slices.Equal(actualNames, expectNames) {
		t.Fatalf("expected slices to be equal %v", actualNames)
	}
}

func TestListIncludesAllManifestListChildrenInSize(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Setenv("OLLAMA_MODELS", t.TempDir())

	makeConfig := func(t *testing.T, format string) manifest.Layer {
		t.Helper()

		data, err := json.Marshal(model.ConfigV2{ModelFormat: format})
		if err != nil {
			t.Fatal(err)
		}

		layer, err := manifest.NewLayer(bytes.NewReader(data), "application/vnd.docker.container.image.v1+json")
		if err != nil {
			t.Fatal(err)
		}

		return layer
	}

	ggufConfig := makeConfig(t, manifest.FormatGGUF)
	mlxConfig := makeConfig(t, manifest.FormatSafetensors)

	sharedBlob, err := manifest.NewLayer(bytes.NewReader([]byte("shared-weights")), "application/vnd.ollama.image.model")
	if err != nil {
		t.Fatal(err)
	}
	ggufBlob, err := manifest.NewLayer(bytes.NewReader([]byte("gguf-weights")), "application/vnd.ollama.image.model")
	if err != nil {
		t.Fatal(err)
	}
	mlxBlob, err := manifest.NewLayer(bytes.NewReader([]byte("mlx-weights")), manifest.MediaTypeImageTensor)
	if err != nil {
		t.Fatal(err)
	}

	ggufLayers := []manifest.Layer{
		sharedBlob,
		ggufBlob,
	}
	if err := manifest.WriteManifestWithMetadata(model.ParseName("test-gguf"), ggufConfig, ggufLayers, manifest.RunnerGGML, manifest.FormatGGUF); err != nil {
		t.Fatal(err)
	}

	mlxLayers := []manifest.Layer{
		{
			MediaType: manifest.MediaTypeImageTensor,
			Digest:    sharedBlob.Digest,
			Size:      sharedBlob.Size,
		},
		mlxBlob,
	}
	if err := manifest.WriteManifestWithMetadata(model.ParseName("test-mlx"), mlxConfig, mlxLayers, manifest.RunnerMLX, manifest.FormatSafetensors); err != nil {
		t.Fatal(err)
	}

	ggufManifest, err := manifest.ParseNamedManifest(model.ParseName("test-gguf"))
	if err != nil {
		t.Fatal(err)
	}
	mlxManifest, err := manifest.ParseNamedManifestForRunner(model.ParseName("test-mlx"), manifest.RunnerMLX)
	if err != nil {
		t.Fatal(err)
	}

	ggufRef, err := manifest.NewManifestReference(ggufManifest.BlobDigest(), manifest.RunnerGGML, manifest.FormatGGUF)
	if err != nil {
		t.Fatal(err)
	}
	mlxRef, err := manifest.NewManifestReference(mlxManifest.BlobDigest(), manifest.RunnerMLX, manifest.FormatSafetensors)
	if err != nil {
		t.Fatal(err)
	}

	parentData, err := json.Marshal(manifest.Manifest{
		SchemaVersion: 2,
		MediaType:     manifest.MediaTypeManifestList,
		Manifests:     []manifest.Manifest{ggufRef, mlxRef},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.WriteManifestData(model.ParseName("test-list"), parentData); err != nil {
		t.Fatal(err)
	}

	var s Server
	w := createRequest(t, s.ListHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	var resp api.ListResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	var listed *api.ListModelResponse
	for i := range resp.Models {
		if resp.Models[i].Name == "test-list:latest" {
			listed = &resp.Models[i]
			break
		}
	}
	if listed == nil {
		t.Fatal("test-list:latest not found in list response")
	}

	want := ggufConfig.Size + sharedBlob.Size + ggufBlob.Size + mlxConfig.Size + mlxBlob.Size
	if listed.Size != want {
		t.Fatalf("size = %d, want %d", listed.Size, want)
	}
}
