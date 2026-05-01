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

// TestListSafetensorsDetails verifies that /api/tags populates parameter_size and
// quantization_level for safetensors models when those values are stored in the
// manifest config, matching the behaviour of /api/show.
func TestListSafetensorsDetails(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	cfgData, err := json.Marshal(model.ConfigV2{
		ModelFormat:  "safetensors",
		ModelType:    "26B",
		FileType:     "mxfp8",
		Capabilities: []string{"completion"},
	})
	if err != nil {
		t.Fatalf("failed to marshal config: %v", err)
	}

	configLayer, err := manifest.NewLayer(bytes.NewReader(cfgData), "application/vnd.docker.container.image.v1+json")
	if err != nil {
		t.Fatalf("failed to create config layer: %v", err)
	}

	name := model.ParseName("gemma4:26b-mxfp8")
	if err := manifest.WriteManifest(name, configLayer, nil); err != nil {
		t.Fatalf("failed to write manifest: %v", err)
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

	if len(resp.Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(resp.Models))
	}

	got := resp.Models[0].Details
	if got.ParameterSize == "" {
		t.Errorf("ParameterSize is empty, want non-empty for safetensors model")
	}
	if got.QuantizationLevel != "mxfp8" {
		t.Errorf("QuantizationLevel = %q, want %q", got.QuantizationLevel, "mxfp8")
	}
}
