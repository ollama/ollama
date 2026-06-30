package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
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
		path, err := manifest.PathForName(parsed)
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
