package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
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

func TestCapabilitiesExposeNemotronSafetensorsVision(t *testing.T) {
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
		model.CapabilityVision,
	} {
		if !slices.Contains(got, capability) {
			t.Errorf("capabilities = %v, want %s", got, capability)
		}
	}
	if slices.Contains(got, model.CapabilityAudio) {
		t.Errorf("capabilities = %v, did not expect audio", got)
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

// A migrated GGUF model is a manifest list with the original ggml child and
// the converted llamacpp child, plus a rollback tag named llamacpp:<digest>.
// Listing should show that model once and keep distinct models and runners.
func TestListHidesMigratedModelDuplicateAndShadow(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	migrated := writeRunnerList(t, "gemma-embed",
		manifestListFixtureChild{name: "mig-ggml", runner: manifest.RunnerGGML, format: manifest.FormatGGUF},
		manifestListFixtureChild{name: "mig-llamacpp", runner: manifest.RunnerLlamaCPP, format: manifest.FormatGGUF},
	)
	migratedHex := writeConvertedShadow(t, childByRunner(t, migrated, manifest.RunnerLlamaCPP))

	writeRunnerList(t, "published-both",
		manifestListFixtureChild{name: "pub-ggml", runner: manifest.RunnerGGML, format: manifest.FormatGGUF},
		manifestListFixtureChild{name: "pub-llamacpp", runner: manifest.RunnerLlamaCPP, format: manifest.FormatGGUF},
	)

	hybrid := writeRunnerList(t, "hybrid",
		manifestListFixtureChild{name: "hyb-ggml", runner: manifest.RunnerGGML, format: manifest.FormatGGUF},
		manifestListFixtureChild{name: "hyb-mlx", runner: manifest.RunnerMLX, format: manifest.FormatSafetensors},
		manifestListFixtureChild{name: "hyb-llamacpp", runner: manifest.RunnerLlamaCPP, format: manifest.FormatGGUF},
	)
	writeConvertedShadow(t, childByRunner(t, hybrid, manifest.RunnerLlamaCPP))

	writeListedRunnerManifest(t, "llamacpp:latest", manifest.RunnerLlamaCPP, manifest.FormatGGUF)
	decoyTag := strings.Repeat("ab", 32)
	writeListedRunnerManifest(t, "llamacpp:"+decoyTag, manifest.RunnerLlamaCPP, manifest.FormatGGUF)
	writeListedRunnerManifest(t, "other-model", manifest.RunnerGGML, manifest.FormatGGUF)
	orphanHex := writeOrphanConversionShadow(t)

	models, err := listModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	migratedRows := rowsNamed(models, "gemma-embed:latest")
	if len(migratedRows) != 1 {
		t.Fatalf("gemma-embed:latest rows = %d, want 1: %+v", len(migratedRows), migratedRows)
	}
	if migratedRows[0].Digest != migratedHex {
		t.Fatalf("gemma-embed digest = %s, want converted child %s", migratedRows[0].Digest, migratedHex)
	}
	if migratedRows[0].Details.Runner != manifest.RunnerLlamaCPP {
		t.Fatalf("gemma-embed runner = %q, want %q", migratedRows[0].Details.Runner, manifest.RunnerLlamaCPP)
	}
	if rows := rowsNamed(models, "llamacpp:"+migratedHex); len(rows) != 0 {
		t.Fatalf("rollback tag listed: %+v", rows)
	}

	published := rowsNamed(models, "published-both:latest")
	if len(published) != 2 {
		t.Fatalf("published-both rows = %d, want 2: %+v", len(published), published)
	}
	if !rowRunners(published, manifest.RunnerGGML, manifest.RunnerLlamaCPP) {
		t.Fatalf("published-both runners = %+v, want ggml and llamacpp", published)
	}

	hybridRows := rowsNamed(models, "hybrid:latest")
	if len(hybridRows) != 2 {
		t.Fatalf("hybrid rows = %d, want 2: %+v", len(hybridRows), hybridRows)
	}
	if !rowRunners(hybridRows, manifest.RunnerMLX, manifest.RunnerLlamaCPP) {
		t.Fatalf("hybrid runners = %+v, want mlx and llamacpp", hybridRows)
	}

	for _, name := range []string{
		"llamacpp:latest",
		"llamacpp:" + decoyTag,
		"llamacpp:" + orphanHex,
		"other-model:latest",
	} {
		if len(rowsNamed(models, name)) != 1 {
			t.Fatalf("name %q rows = %+v, want exactly one", name, rowsNamed(models, name))
		}
	}
}

func rowsNamed(models []api.ListModelResponse, name string) []api.ListModelResponse {
	var rows []api.ListModelResponse
	for _, row := range models {
		if row.Name == name {
			rows = append(rows, row)
		}
	}
	return rows
}

func rowRunners(rows []api.ListModelResponse, want ...string) bool {
	if len(rows) != len(want) {
		return false
	}
	got := make([]string, len(rows))
	for i, row := range rows {
		got[i] = row.Details.Runner
	}
	slices.Sort(got)
	want = append([]string(nil), want...)
	slices.Sort(want)
	return slices.Equal(got, want)
}

func writeRunnerList(t *testing.T, name string, children ...manifestListFixtureChild) []*manifest.Manifest {
	t.Helper()

	for _, child := range children {
		format := child.format
		if format == "" {
			format = manifest.FormatGGUF
		}
		writeListedRunnerManifest(t, child.name, child.runner, format)
	}
	written := writeManifestListFixture(t, name, children...)
	for _, child := range children {
		if _, err := manifest.RemoveNamed(model.ParseName(child.name)); err != nil {
			t.Fatalf("remove intermediate manifest %s: %v", child.name, err)
		}
	}
	return written
}

func writeListedRunnerManifest(t *testing.T, name, runner, format string) {
	t.Helper()

	cfg := makeManifestListConfig(t, format)
	mediaType := "application/vnd.ollama.image.model"
	if format == manifest.FormatSafetensors {
		mediaType = manifest.MediaTypeImageTensor
	}
	layer, err := manifest.NewLayer(bytes.NewReader([]byte(name)), mediaType)
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.WriteManifestWithMetadata(model.ParseName(name), cfg, []manifest.Layer{layer}, runner, format); err != nil {
		t.Fatal(err)
	}
}

func writeConvertedShadow(t *testing.T, mf *manifest.Manifest) string {
	t.Helper()

	digest := strings.TrimPrefix(mf.BlobDigest(), "sha256:")
	path, err := manifest.BlobsPath(mf.BlobDigest())
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.WriteLegacyManifestData(model.ParseName(manifest.RunnerLlamaCPP+":"+digest), data); err != nil {
		t.Fatal(err)
	}
	return digest
}

func writeOrphanConversionShadow(t *testing.T) string {
	t.Helper()

	cfg := makeManifestListConfig(t, manifest.FormatGGUF)
	layer, err := manifest.NewLayer(bytes.NewReader([]byte("orphan-weights")), "application/vnd.ollama.image.model")
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(manifest.Manifest{
		SchemaVersion: 2,
		MediaType:     manifest.MediaTypeManifest,
		Config:        cfg,
		Layers:        []manifest.Layer{layer},
		Runner:        manifest.RunnerLlamaCPP,
		Format:        manifest.FormatGGUF,
	})
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	hex := fmt.Sprintf("%x", sum)
	if err := manifest.WriteLegacyManifestData(model.ParseName("llamacpp:"+hex), data); err != nil {
		t.Fatal(err)
	}
	return hex
}

func childByRunner(t *testing.T, children []*manifest.Manifest, runner string) *manifest.Manifest {
	t.Helper()
	for _, child := range children {
		if child.Runner == runner {
			return child
		}
	}
	t.Fatalf("no child with runner %s", runner)
	return nil
}

func createListedModelFromKV(t *testing.T, name string, kv map[string]any, tmpl string) {
	t.Helper()
	_, digest := createBinFile(t, kv, nil)
	createModelFromBlob(t, name, digest, tmpl)
}
