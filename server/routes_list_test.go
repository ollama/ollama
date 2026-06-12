package server

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/openai"
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

	s := Server{modelCaches: &modelCaches{modelList: newModelListCache()}}
	s.modelCaches.modelList.Start(context.Background())
	if err := s.modelCaches.modelList.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}

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

	for _, m := range resp.Models {
		if !slices.Contains(m.Capabilities, "completion") {
			t.Fatalf("capabilities for %q = %v, want completion", m.Name, m.Capabilities)
		}
	}
}

func TestOpenAIListMatchesTagsModels(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	cache := newModelListCache()
	s := Server{modelCaches: &modelCaches{modelList: cache}}
	cache.Start(context.Background())
	if err := cache.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}

	createModel := func(name string) {
		t.Helper()

		_, digest := createBinFile(t, nil, nil)
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Model:  name,
			Files:  map[string]string{"model.gguf": digest},
			Stream: &stream,
		})
		if w.Code != http.StatusOK {
			t.Fatalf("create %s status = %d, want 200: %s", name, w.Code, w.Body.String())
		}
	}

	createModel("older-model")
	createModel("newer-model")

	setManifestTime := func(name string, modified time.Time) {
		t.Helper()

		parsed := model.ParseName(name)
		path, err := manifest.ResolvePathForName(parsed)
		if err != nil {
			t.Fatalf("manifest path for %s: %v", name, err)
		}
		if err := os.Chtimes(path, modified, modified); err != nil {
			t.Fatalf("set manifest time for %s: %v", name, err)
		}
		if err := cache.RefreshModel(parsed); err != nil {
			t.Fatalf("refresh %s: %v", name, err)
		}
	}

	older := time.Unix(1000, 0).UTC()
	newer := time.Unix(2000, 0).UTC()
	setManifestTime("older-model:latest", older)
	setManifestTime("newer-model:latest", newer)

	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}

	doGet := func(path string, dest any) {
		t.Helper()

		req := httptest.NewRequest(http.MethodGet, path, nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		body, err := io.ReadAll(w.Body)
		if err != nil {
			t.Fatalf("read %s response: %v", path, err)
		}
		if w.Code != http.StatusOK {
			t.Fatalf("GET %s status = %d, want 200: %s", path, w.Code, string(body))
		}
		if err := json.Unmarshal(body, dest); err != nil {
			t.Fatalf("decode %s response: %v", path, err)
		}
	}

	var tags api.ListResponse
	doGet("/api/tags", &tags)

	var models openai.ListCompletion
	doGet("/v1/models", &models)

	if len(tags.Models) != 2 {
		t.Fatalf("/api/tags models = %d, want 2: %+v", len(tags.Models), tags.Models)
	}
	if len(models.Data) != len(tags.Models) {
		t.Fatalf("/v1/models data = %d, want %d", len(models.Data), len(tags.Models))
	}

	for i, tagModel := range tags.Models {
		v1Model := models.Data[i]
		if v1Model.Id != tagModel.Model {
			t.Fatalf("model %d id = %q, want /api/tags model %q", i, v1Model.Id, tagModel.Model)
		}
		if v1Model.Created != tagModel.ModifiedAt.Unix() {
			t.Fatalf("model %d created = %d, want modified_at %d", i, v1Model.Created, tagModel.ModifiedAt.Unix())
		}
	}

	if got, want := models.Data[0].Id, "newer-model:latest"; got != want {
		t.Fatalf("first /v1/models id = %q, want %q", got, want)
	}
}

func TestListIncludesManifestListChildrenAsSeparateRows(t *testing.T) {
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

	s := Server{modelCaches: &modelCaches{modelList: newModelListCache()}}
	s.modelCaches.modelList.Start(context.Background())
	if err := s.modelCaches.modelList.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}

	w := createRequest(t, s.ListHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	var resp api.ListResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	var listed []api.ListModelResponse
	for i := range resp.Models {
		if resp.Models[i].Name == "test-list:latest" {
			listed = append(listed, resp.Models[i])
		}
	}
	if len(listed) != 2 {
		t.Fatalf("test-list:latest rows = %d, want 2: %+v", len(listed), listed)
	}

	wantSizes := map[string]int64{
		ggufManifest.BlobDigest(): ggufConfig.Size + sharedBlob.Size + ggufBlob.Size,
		mlxManifest.BlobDigest():  mlxConfig.Size + sharedBlob.Size + mlxBlob.Size,
	}
	for _, row := range listed {
		want, ok := wantSizes[row.Digest]
		if !ok {
			t.Fatalf("unexpected digest for test-list row: %+v", row)
		}
		if row.Size != want {
			t.Fatalf("size for %s = %d, want %d", row.Digest, row.Size, want)
		}
		delete(wantSizes, row.Digest)
	}
	if len(wantSizes) != 0 {
		t.Fatalf("missing list rows for digests: %+v", wantSizes)
	}
}

func TestCopyManifestListByNameAndChildDigest(t *testing.T) {
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

	ggufBlob, err := manifest.NewLayer(bytes.NewReader([]byte("gguf-weights")), "application/vnd.ollama.image.model")
	if err != nil {
		t.Fatal(err)
	}
	mlxBlob, err := manifest.NewLayer(bytes.NewReader([]byte("mlx-weights")), manifest.MediaTypeImageTensor)
	if err != nil {
		t.Fatal(err)
	}

	if err := manifest.WriteManifestWithMetadata(model.ParseName("copy-gguf"), ggufConfig, []manifest.Layer{ggufBlob}, manifest.RunnerGGML, manifest.FormatGGUF); err != nil {
		t.Fatal(err)
	}
	if err := manifest.WriteManifestWithMetadata(model.ParseName("copy-mlx"), mlxConfig, []manifest.Layer{mlxBlob}, manifest.RunnerMLX, manifest.FormatSafetensors); err != nil {
		t.Fatal(err)
	}

	ggufManifest, err := manifest.ParseNamedManifest(model.ParseName("copy-gguf"))
	if err != nil {
		t.Fatal(err)
	}
	mlxManifest, err := manifest.ParseNamedManifestForRunner(model.ParseName("copy-mlx"), manifest.RunnerMLX)
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
	if err := manifest.WriteManifestData(model.ParseName("copy-list"), parentData); err != nil {
		t.Fatal(err)
	}

	var s Server
	w := createRequest(t, s.CopyHandler, api.CopyRequest{
		Source:      "copy-list",
		Destination: "copy-list-copy",
	})
	if w.Code != http.StatusOK {
		t.Fatalf("copy manifest list status = %d, want 200: %s", w.Code, w.Body.String())
	}
	copiedList, err := manifest.ReadManifestData(model.ParseName("copy-list-copy"))
	if err != nil {
		t.Fatal(err)
	}
	var copiedParent manifest.Manifest
	if err := json.Unmarshal(copiedList, &copiedParent); err != nil {
		t.Fatal(err)
	}
	if copiedParent.MediaType != manifest.MediaTypeManifestList || len(copiedParent.Manifests) != 2 {
		t.Fatalf("copied parent = %+v, want manifest list with 2 children", copiedParent)
	}

	childDigestRef := strings.Replace(ggufManifest.BlobDigest(), ":", "-", 1)
	w = createRequest(t, s.CopyHandler, api.CopyRequest{
		Source:      childDigestRef,
		Destination: "copy-child",
	})
	if w.Code != http.StatusOK {
		t.Fatalf("copy child digest status = %d, want 200: %s", w.Code, w.Body.String())
	}
	copiedChild, err := manifest.ParseNamedManifest(model.ParseName("copy-child"))
	if err != nil {
		t.Fatal(err)
	}
	if copiedChild.MediaType == manifest.MediaTypeManifestList {
		t.Fatal("copying a child digest produced a manifest list")
	}
	if copiedChild.BlobDigest() != ggufManifest.BlobDigest() {
		t.Fatalf("copied child digest = %s, want %s", copiedChild.BlobDigest(), ggufManifest.BlobDigest())
	}
}

func TestCopyRejectsExplicitCloudSource(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	createShowCacheModel(t, "copy-cloud", map[string]any{"test.context_length": uint32(1024)})

	var s Server
	w := createRequest(t, s.CopyHandler, api.CopyRequest{
		Source:      "copy-cloud:cloud",
		Destination: "copy-cloud-local",
	})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("copy cloud source status = %d, want 400: %s", w.Code, w.Body.String())
	}
}
