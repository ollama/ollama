package server

import (
	"context"
	"net/http"
	"slices"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func listedModel(t *testing.T, name string) api.ListModelResponse {
	t.Helper()
	models, err := listModels(context.Background())
	if err != nil {
		t.Fatalf("listModels failed: %v", err)
	}
	for _, m := range models {
		if m.Name == name {
			return m
		}
	}
	t.Fatalf("%s not listed; got %v", name, models)
	return api.ListModelResponse{}
}

func TestListModelsDescribesModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-describe", map[string]any{
		"test.context_length":   uint32(4096),
		"test.embedding_length": uint32(384),
	}, "{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}{{ if .suffix }}{{ .suffix }}{{ end }}")

	got := listedModel(t, "list-describe:latest")

	if got.Model != "list-describe:latest" {
		t.Errorf("model = %q", got.Model)
	}
	if got.Digest == "" || got.Size == 0 {
		t.Errorf("digest = %q size = %d, want both set", got.Digest, got.Size)
	}
	if got.Details.Family != "test" || got.Details.Format != "gguf" {
		t.Errorf("details = %+v, want gguf/test", got.Details)
	}
	if got.Details.ContextLength != 4096 {
		t.Errorf("context length = %d, want 4096", got.Details.ContextLength)
	}
	if got.Details.EmbeddingLength != 384 {
		t.Errorf("embedding length = %d, want 384", got.Details.EmbeddingLength)
	}
	// The test GGUF has no general.file_type; the list must not guess F32
	// (FileType 0) from a missing key.
	if !isUnknownQuantization(got.Details.QuantizationLevel) {
		t.Errorf("quantization = %q, want unknown (no general.file_type in GGUF)", got.Details.QuantizationLevel)
	}
	for _, capability := range []model.Capability{model.CapabilityCompletion, model.CapabilityTools, model.CapabilityInsert} {
		if !slices.Contains(got.Capabilities, capability) {
			t.Errorf("capabilities = %v, want %s", got.Capabilities, capability)
		}
	}
}

// Listing reads the manifests every time, so a model created or deleted after
// the last call shows up without anything needing to be told about it.
func TestListModelsFollowsManifestChanges(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-follow-a", map[string]any{"test.context_length": uint32(1024)}, "")

	listedModel(t, "list-follow-a:latest")

	createListCacheModel(t, "list-follow-b", map[string]any{"test.context_length": uint32(2048)}, "")
	listedModel(t, "list-follow-a:latest")
	listedModel(t, "list-follow-b:latest")

	var s Server
	w := createRequest(t, s.DeleteHandler, api.DeleteRequest{Model: "list-follow-a"})
	if w.Code != http.StatusOK {
		t.Fatalf("delete status = %d: %s", w.Code, w.Body.String())
	}

	models, err := listModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(models))
	for _, m := range models {
		names = append(names, m.Name)
	}
	if slices.Contains(names, "list-follow-a:latest") || !slices.Contains(names, "list-follow-b:latest") {
		t.Fatalf("names after delete = %v, want only list-follow-b", names)
	}
}

func TestCapabilitiesSuppressNemotronSafetensorsMedia(t *testing.T) {
	caps := []model.Capability{
		model.CapabilityCompletion,
		model.CapabilityTools,
		model.CapabilityThinking,
		model.CapabilityVision,
		model.CapabilityAudio,
	}
	got := (&Model{Config: model.ConfigV2{
		ModelFormat: "safetensors",
		Renderer:    "nemotron-3-nano",
		Parser:      "nemotron-3-nano",
	}}).filterUnsupportedCapabilities(caps, "")

	for _, capability := range []model.Capability{
		model.CapabilityCompletion,
		model.CapabilityTools,
		model.CapabilityThinking,
	} {
		if !slices.Contains(got, capability) {
			t.Errorf("capabilities = %v, want %s", got, capability)
		}
	}
	for _, capability := range []model.Capability{model.CapabilityVision, model.CapabilityAudio} {
		if slices.Contains(got, capability) {
			t.Errorf("capabilities = %v, did not expect %s", got, capability)
		}
	}
}
