package server

import (
	"context"
	"os"
	"slices"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
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
	createListedModelFromKV(t, "list-describe", map[string]any{
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
	createListedModelFromKV(t, "list-follow-a", map[string]any{"test.context_length": uint32(1024)}, "")

	listedModel(t, "list-follow-a:latest")

	createListedModelFromKV(t, "list-follow-b", map[string]any{"test.context_length": uint32(2048)}, "")
	listedModel(t, "list-follow-a:latest")
	listedModel(t, "list-follow-b:latest")

	deleteModelNamed(t, "list-follow-a")

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

// A model whose layers no longer parse must still be listed: it is the one the
// user most needs to see, in order to remove it.
func TestListModelsKeepsUnloadableModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListedModelFromKV(t, "broken", map[string]any{"test.context_length": uint32(1024)}, "{{ .Prompt }}")

	mf, err := manifest.ParseNamedManifest(model.ParseName("broken"))
	if err != nil {
		t.Fatal(err)
	}
	var corrupted bool
	for _, layer := range mf.Layers {
		if layer.MediaType != "application/vnd.ollama.image.template" {
			continue
		}
		path, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("{{ if }"), 0o644); err != nil {
			t.Fatal(err)
		}
		corrupted = true
	}
	if !corrupted {
		t.Fatal("no template layer to corrupt")
	}
	if _, err := GetModel("broken"); err == nil {
		t.Fatal("model still loads, so the listing is not being asked the question")
	}

	got := listedModel(t, "broken:latest")
	if got.Details.Family != "test" {
		t.Errorf("details = %+v, want the manifest config's family", got.Details)
	}
}

func createListedModelFromKV(t *testing.T, name string, kv map[string]any, tmpl string) {
	t.Helper()
	_, digest := createBinFile(t, kv, nil)
	createModelFromBlob(t, name, digest, tmpl)
}
