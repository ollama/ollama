package create

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestApplyModelfileLayersIncludesParameters(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	layers, err := ApplyModelfileLayers(nil, ModelfileLayerOptions{
		Parameters: map[string]any{
			"temperature": float32(0.7),
			"stop":        []string{"USER:", "ASSISTANT:"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(layers) != 1 {
		t.Fatalf("len(layers) = %d, want 1", len(layers))
	}
	if layers[0].MediaType != "application/vnd.ollama.image.params" {
		t.Fatalf("MediaType = %q, want %q", layers[0].MediaType, "application/vnd.ollama.image.params")
	}

	blobPath, err := manifest.BlobsPath(layers[0].Digest)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(blobPath)
	if err != nil {
		t.Fatal(err)
	}

	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got["temperature"] != float64(0.7) {
		t.Fatalf("temperature = %v, want %v", got["temperature"], float64(0.7))
	}
}

func TestApplyModelfileLayersOverlaysInheritedValues(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	layers, err := ApplyModelfileLayers(nil, ModelfileLayerOptions{
		Template: "{{ .Prompt }}",
		System:   "old system",
		License:  "old license",
		Parameters: map[string]any{
			"temperature": 0.8,
			"top_p":       0.9,
		},
		Messages: []api.Message{{Role: "user", Content: "old message"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	layers, err = appendTextLayer(layers, "application/vnd.ollama.image.prompt", "legacy prompt")
	if err != nil {
		t.Fatal(err)
	}
	layers, err = appendTextLayer(layers, "application/vnd.ollama.image.params", `{"top_p":0.4,"repeat_penalty":1.1}`)
	if err != nil {
		t.Fatal(err)
	}

	overrides := map[string]any{
		"temperature": 0.2,
		"num_ctx":     4096,
	}
	layers, err = ApplyModelfileLayers(layers, ModelfileLayerOptions{
		Template:   "{{ .System }}{{ .Prompt }}",
		System:     "new system",
		License:    "new license",
		Parameters: overrides,
		Messages:   []api.Message{{Role: "assistant", Content: "new message"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(overrides) != 2 {
		t.Fatalf("parameter overrides were mutated: %v", overrides)
	}

	counts := make(map[string]int)
	var gotTemplate, gotSystem string
	var gotParameters map[string]any
	var gotMessages []api.Message
	var gotLicenses []string
	for _, layer := range layers {
		counts[layer.MediaType]++
		switch layer.MediaType {
		case "application/vnd.ollama.image.template":
			gotTemplate = readManifestLayerString(t, layer)
		case "application/vnd.ollama.image.system":
			gotSystem = readManifestLayerString(t, layer)
		case "application/vnd.ollama.image.license":
			gotLicenses = append(gotLicenses, readManifestLayerString(t, layer))
		case "application/vnd.ollama.image.params":
			readManifestLayerJSON(t, layer, &gotParameters)
		case "application/vnd.ollama.image.messages":
			readManifestLayerJSON(t, layer, &gotMessages)
		}
	}

	for _, mediaType := range []string{
		"application/vnd.ollama.image.template",
		"application/vnd.ollama.image.system",
		"application/vnd.ollama.image.params",
		"application/vnd.ollama.image.messages",
	} {
		if counts[mediaType] != 1 {
			t.Errorf("%s layer count = %d, want 1", mediaType, counts[mediaType])
		}
	}
	if counts["application/vnd.ollama.image.prompt"] != 0 {
		t.Errorf("legacy prompt layer count = %d, want 0", counts["application/vnd.ollama.image.prompt"])
	}
	if gotTemplate != "{{ .System }}{{ .Prompt }}" {
		t.Errorf("template = %q, want replacement", gotTemplate)
	}
	if gotSystem != "new system" {
		t.Errorf("system = %q, want replacement", gotSystem)
	}
	if diff := cmp.Diff([]string{"old license", "new license"}, gotLicenses); diff != "" {
		t.Errorf("licenses mismatch (-want +got):\n%s", diff)
	}
	wantParameters := map[string]any{
		"temperature":    0.2,
		"top_p":          0.9,
		"repeat_penalty": 1.1,
		"num_ctx":        float64(4096),
	}
	if diff := cmp.Diff(wantParameters, gotParameters); diff != "" {
		t.Errorf("parameters mismatch (-want +got):\n%s", diff)
	}
	if want := []api.Message{{Role: "assistant", Content: "new message"}}; cmp.Diff(want, gotMessages) != "" {
		t.Fatalf("messages mismatch (-want +got):\n%s", cmp.Diff(want, gotMessages))
	}
}

func TestApplyModelfileLayersRejectsInvalidTemplate(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	if _, err := ApplyModelfileLayers(nil, ModelfileLayerOptions{Template: "{{ if .Prompt }}"}); !errors.Is(err, ErrBadTemplate) {
		t.Fatalf("ApplyModelfileLayers() error = %v, want ErrBadTemplate", err)
	}
}

func TestNewSafetensorsManifestWriterPreservesBaseConfig(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	base := model.ConfigV2{
		ModelFormat:   "safetensors",
		ModelFamily:   "gemma4",
		ModelFamilies: []string{"gemma4", "gemma"},
		ModelType:     "7B",
		FileType:      "nvfp4",
		Renderer:      "gemma4",
		Parser:        "gemma4",
		Requires:      "0.20.0",
		RemoteHost:    "https://example.invalid",
		RemoteModel:   "gemma4:7b",
		Capabilities:  []string{"completion", "vision", "thinking"},
		ContextLen:    131072,
		EmbedLen:      3072,
		BaseName:      "gemma-4",
		Draft: &model.Draft{
			ModelFormat:  "safetensors",
			Architecture: "gemma4_mtp",
			TensorPrefix: "draft.",
			Config:       "draft/config.json",
		},
	}
	writeManifest := NewSafetensorsManifestWriter(SafetensorsManifestOptions{MinVersion: "0.19.0"})
	if err := writeManifest(context.Background(), "test-base-config", ManifestInfo{ModelConfig: base}); err != nil {
		t.Fatal(err)
	}

	got := readSafetensorsManifestConfig(t, "test-base-config")
	if diff := cmp.Diff(base, got); diff != "" {
		t.Fatalf("base config changed (-want +got):\n%s", diff)
	}
}

func TestNewSafetensorsManifestWriterRaisesBaseMinimumVersion(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	base := model.ConfigV2{ModelFormat: "safetensors", Requires: "0.14.0"}
	writeManifest := NewSafetensorsManifestWriter(SafetensorsManifestOptions{MinVersion: "0.19.0"})
	if err := writeManifest(context.Background(), "test-min-version", ManifestInfo{ModelConfig: base}); err != nil {
		t.Fatal(err)
	}

	if got := readSafetensorsManifestConfig(t, "test-min-version").Requires; got != "0.19.0" {
		t.Fatalf("requires = %q, want 0.19.0", got)
	}
}

func TestNewSafetensorsManifestWriterRaisesExplicitMinimumVersion(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	writeManifest := NewSafetensorsManifestWriter(SafetensorsManifestOptions{
		MinVersion: "0.19.0",
	})
	if err := writeManifest(context.Background(), "test-explicit-min-version", ManifestInfo{ModelConfig: model.ConfigV2{Requires: "0.14.0"}}); err != nil {
		t.Fatal(err)
	}

	if got := readSafetensorsManifestConfig(t, "test-explicit-min-version").Requires; got != "0.19.0" {
		t.Fatalf("requires = %q, want 0.19.0", got)
	}
}

func TestNewSafetensorsManifestWriterRejectsInvalidRequires(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	writeManifest := NewSafetensorsManifestWriter(SafetensorsManifestOptions{})
	err := writeManifest(context.Background(), "test-invalid-requires", ManifestInfo{ModelConfig: model.ConfigV2{Requires: "not-semver"}})
	if !errors.Is(err, ErrInvalidRequires) {
		t.Fatalf("writeManifest() error = %v, want ErrInvalidRequires", err)
	}
}

func TestNewSafetensorsManifestWriterDoesNotPublishAfterCancellation(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	ctx, cancel := context.WithCancel(context.Background())
	writeManifest := NewSafetensorsManifestWriter(SafetensorsManifestOptions{
		BeforeWriteManifest: cancel,
	})
	err := writeManifest(ctx, "test-canceled-manifest", ManifestInfo{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("writeManifest() error = %v, want context.Canceled", err)
	}
	if _, err := manifest.ParseNamedManifest(model.ParseName("test-canceled-manifest")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("ParseNamedManifest() error = %v, want os.ErrNotExist", err)
	}
}

func readSafetensorsManifestConfig(t *testing.T, name string) model.ConfigV2 {
	t.Helper()

	m, err := manifest.ParseNamedManifest(model.ParseName(name))
	if err != nil {
		t.Fatal(err)
	}
	configFile, err := m.Config.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer configFile.Close()

	var config model.ConfigV2
	if err := json.NewDecoder(configFile).Decode(&config); err != nil {
		t.Fatal(err)
	}
	return config
}

func readManifestLayerString(t *testing.T, layer manifest.Layer) string {
	t.Helper()
	f, err := layer.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	b, err := io.ReadAll(f)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func readManifestLayerJSON(t *testing.T, layer manifest.Layer, dst any) {
	t.Helper()
	f, err := layer.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := json.NewDecoder(f).Decode(dst); err != nil {
		t.Fatal(err)
	}
}
