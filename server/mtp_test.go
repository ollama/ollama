package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestMetadataHasMTPLayers(t *testing.T) {
	tests := []struct {
		name string
		kv   map[string]any
		want bool
	}{
		{
			name: "architecture key",
			kv:   map[string]any{"glm4moe.nextn_predict_layers": json.Number("1")},
			want: true,
		},
		{
			name: "json number from show model_info",
			kv:   map[string]any{"qwen35.nextn_predict_layers": float64(3)},
			want: true,
		},
		{
			name: "unsigned metadata",
			kv:   map[string]any{"glm4moe.nextn_predict_layers": uint32(1)},
			want: true,
		},
		{
			name: "unprefixed key",
			kv:   map[string]any{"nextn_predict_layers": json.Number("2")},
			want: true,
		},
		{
			name: "zero is not MTP",
			kv:   map[string]any{"glm4moe.nextn_predict_layers": uint32(0)},
			want: false,
		},
		{
			name: "missing key",
			kv:   map[string]any{"glm4moe.block_count": uint32(40), "general.architecture": "glm4moe"},
			want: false,
		},
		{
			name: "unrelated suffix",
			kv:   map[string]any{"glm4moe.nextn_predict_layers_extra": uint32(1)},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := metadataHasMTPLayers(tt.kv); got != tt.want {
				t.Fatalf("metadataHasMTPLayers() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestExistingDraftNumPredictReadsAsEnabled(t *testing.T) {
	// The shape stored for `PARAMETER draft_num_predict 3` alongside num_ctx.
	raw := []byte(`{"num_ctx":1048576,"draft_num_predict":3}`)
	params := decodeParams(t, raw)

	got, ok := draftNumPredictValue(params)
	if !ok || got != 3 {
		t.Fatalf("draftNumPredictValue() = (%d, %v), want (3, true)", got, ok)
	}

	if _, present := params["num_ctx"]; !present {
		t.Fatal("num_ctx was dropped while reading draft_num_predict")
	}
}

func TestApplyDraftNumPredictPreservesOtherParameters(t *testing.T) {
	original := []byte(`{"num_ctx":1048576,"repeat_penalty":1.1,"stop":["<|end|>"],"draft_num_predict":3}`)

	t.Run("existing value reads as on and is left untouched", func(t *testing.T) {
		updated, changed, err := applyDraftNumPredict(original, 3)
		if err != nil {
			t.Fatal(err)
		}
		if changed {
			t.Fatalf("applyDraftNumPredict rewrote parameters:\n got %s\nwant %s", updated, original)
		}
	})

	t.Run("off removes only draft_num_predict", func(t *testing.T) {
		updated, changed, err := applyDraftNumPredict(original, 0)
		if err != nil {
			t.Fatal(err)
		}
		if !changed {
			t.Fatal("expected draft_num_predict to be removed")
		}
		assertParams(t, updated, map[string]any{
			"num_ctx":        json.Number("1048576"),
			"repeat_penalty": json.Number("1.1"),
			"stop":           []any{"<|end|>"},
		})
		if bytes.Contains(updated, []byte("draft_num_predict")) {
			t.Fatalf("draft_num_predict still present: %s", updated)
		}
		if !bytes.Contains(updated, []byte(`"num_ctx":1048576`)) {
			t.Fatalf("num_ctx literal changed: %s", updated)
		}
		if !bytes.Contains(updated, []byte(`"repeat_penalty":1.1`)) {
			t.Fatalf("repeat_penalty literal changed: %s", updated)
		}
		if !bytes.Contains(updated, []byte(`"<|end|>"`)) {
			t.Fatalf("stop token was escaped or dropped: %s", updated)
		}
	})

	t.Run("on adds draft_num_predict without rewriting other literals", func(t *testing.T) {
		// Scientific notation is what /api/show prints for large floats. The
		// stored JSON keeps whatever literal was written; toggling must not
		// expand it.
		source := []byte(`{"num_ctx":1.048576e+06,"stop":["<|end|>"]}`)
		updated, changed, err := applyDraftNumPredict(source, 3)
		if err != nil {
			t.Fatal(err)
		}
		if !changed {
			t.Fatal("expected draft_num_predict to be added")
		}
		assertParams(t, updated, map[string]any{
			"num_ctx":           json.Number("1.048576e+06"),
			"stop":              []any{"<|end|>"},
			"draft_num_predict": json.Number("3"),
		})
		if !bytes.Contains(updated, []byte(`"num_ctx":1.048576e+06`)) {
			t.Fatalf("num_ctx literal changed: %s", updated)
		}
	})

	t.Run("out of range does not write", func(t *testing.T) {
		_, _, err := applyDraftNumPredict(original, 8)
		if !errors.Is(err, errDraftNumPredictRange) {
			t.Fatalf("error = %v, want range error", err)
		}
	})
}

func TestSetModelDraftNumPredictPreservesManifest(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	gin.SetMode(gin.TestMode)

	const params = `{"num_ctx":1048576,"stop":["<|end|>"],"draft_num_predict":3}`
	name := writeMTPManifest(t, "library/glm-mtp:latest", params, manifest.RunnerLlamaCPP, manifest.FormatGGUF)
	before := readManifestBytes(t, name)
	modelDigest := layerDigest(t, name, manifest.MediaTypeImageModel)
	templateDigest := layerDigest(t, name, manifest.MediaTypeImageTemplate)
	configBefore := configDigest(t, name)

	if err := setModelDraftNumPredict(name, 3); err != nil {
		t.Fatal(err)
	}
	if after := readManifestBytes(t, name); !bytes.Equal(before, after) {
		t.Fatalf("setting the existing draft depth rewrote the manifest:\n got %s\nwant %s", after, before)
	}

	if err := setModelDraftNumPredict(name, 0); err != nil {
		t.Fatal(err)
	}
	assertStoredParams(t, name, map[string]any{
		"num_ctx": json.Number("1048576"),
		"stop":    []any{"<|end|>"},
	})
	if layerDigest(t, name, manifest.MediaTypeImageModel) != modelDigest {
		t.Fatal("model blob changed while removing draft_num_predict")
	}
	if layerDigest(t, name, manifest.MediaTypeImageTemplate) != templateDigest {
		t.Fatal("template changed while removing draft_num_predict")
	}
	if configDigest(t, name) != configBefore {
		t.Fatal("config changed while removing draft_num_predict")
	}
	if got := layerFrom(t, name, manifest.MediaTypeImageModel); got != "parent-weights" {
		t.Fatalf("model from = %q, want parent-weights", got)
	}

	if err := setModelDraftNumPredict(name, 3); err != nil {
		t.Fatal(err)
	}
	assertStoredParams(t, name, map[string]any{
		"num_ctx":           json.Number("1048576"),
		"stop":              []any{"<|end|>"},
		"draft_num_predict": json.Number("3"),
	})
	if layerDigest(t, name, manifest.MediaTypeImageModel) != modelDigest {
		t.Fatal("model blob changed while setting draft_num_predict")
	}
}

func TestSetModelDraftNumPredictManifestList(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	const params = `{"num_ctx":1048576,"draft_num_predict":3}`
	ggml := writeMTPManifest(t, "library/mtp-ggml:latest", params, manifest.RunnerGGML, manifest.FormatGGUF)
	mlx := writeMTPManifest(t, "library/mtp-mlx:latest", `{"num_ctx":4096}`, manifest.RunnerMLX, manifest.FormatSafetensors)
	ggmlModel := layerDigest(t, ggml, manifest.MediaTypeImageModel)
	mlxModel := layerDigest(t, mlx, manifest.MediaTypeImageModel)

	ggmlRef := manifestReference(t, ggml)
	mlxRef := manifestReference(t, mlx)
	absent, err := manifest.NewManifestReference(fmt.Sprintf("sha256:%064d", 1), manifest.RunnerLlamaCPP, manifest.FormatGGUF)
	if err != nil {
		t.Fatal(err)
	}

	list := model.ParseName("library/mtp-list:latest")
	if _, err := manifest.WriteManifestList(list, []manifest.Manifest{ggmlRef, mlxRef, absent}); err != nil {
		t.Fatal(err)
	}

	if err := setModelDraftNumPredict(list, 0); err != nil {
		t.Fatal(err)
	}
	// The named children are independent manifests. The list stores its own
	// copies, and those are what the toggle updates.
	assertListChildParams(t, list, manifest.RunnerGGML, map[string]any{
		"num_ctx": json.Number("1048576"),
	})
	assertListChildParams(t, list, manifest.RunnerMLX, map[string]any{
		"num_ctx": json.Number("4096"),
	})
	if listChildLayerDigest(t, list, manifest.RunnerGGML, manifest.MediaTypeImageModel) != ggmlModel ||
		listChildLayerDigest(t, list, manifest.RunnerMLX, manifest.MediaTypeImageModel) != mlxModel {
		t.Fatal("child weight blobs changed")
	}

	raw := readManifestBytes(t, list)
	if !bytes.Contains(raw, []byte(absent.BlobDigest())) {
		t.Fatalf("missing child reference was dropped: %s", raw)
	}

	if err := setModelDraftNumPredict(list, 4); err != nil {
		t.Fatal(err)
	}
	assertListChildParams(t, list, manifest.RunnerGGML, map[string]any{
		"num_ctx":           json.Number("1048576"),
		"draft_num_predict": json.Number("4"),
	})
	assertListChildParams(t, list, manifest.RunnerMLX, map[string]any{
		"num_ctx":           json.Number("4096"),
		"draft_num_predict": json.Number("4"),
	})
}

func TestModelMTPHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	s := &Server{}
	paramsLayer := mustLayer(t, `{"num_ctx":1048576,"stop":["<|end|>"],"draft_num_predict":3}`, manifest.MediaTypeImageParams)
	templateLayer := mustLayer(t, "{{ .Prompt }}", manifest.MediaTypeImageTemplate)
	writeShowManifestVariant(t, "glm-mtp", manifest.RunnerLlamaCPP, manifest.FormatGGUF, model.ConfigV2{
		ModelFormat: manifest.FormatGGUF,
		ModelFamily: "glm4moe",
	}, map[string]any{
		"general.architecture": "glm4moe",
		"nextn_predict_layers": uint32(1),
	}, templateLayer, paramsLayer)

	templateBefore := layerDigest(t, model.ParseName("glm-mtp"), manifest.MediaTypeImageTemplate)

	t.Run("off removes draft and keeps other parameters", func(t *testing.T) {
		w := createRequest(t, s.ModelMTPHandler, map[string]any{
			"model":             "glm-mtp",
			"draft_num_predict": 0,
		})
		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, body = %s", w.Code, w.Body)
		}
		assertStoredParams(t, model.ParseName("glm-mtp"), map[string]any{
			"num_ctx": json.Number("1048576"),
			"stop":    []any{"<|end|>"},
		})
		if layerDigest(t, model.ParseName("glm-mtp"), manifest.MediaTypeImageTemplate) != templateBefore {
			t.Fatal("template changed")
		}
	})

	t.Run("on sets the default depth", func(t *testing.T) {
		w := createRequest(t, s.ModelMTPHandler, map[string]any{
			"model":             "glm-mtp",
			"draft_num_predict": 3,
		})
		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, body = %s", w.Code, w.Body)
		}
		got, ok := draftNumPredictValue(storedParams(t, model.ParseName("glm-mtp")))
		if !ok || got != 3 {
			t.Fatalf("draft_num_predict = (%d, %v), want (3, true)", got, ok)
		}
		if _, present := storedParams(t, model.ParseName("glm-mtp"))["num_ctx"]; !present {
			t.Fatal("num_ctx was dropped")
		}
	})

	t.Run("model without nextn layers is rejected", func(t *testing.T) {
		plain := mustLayer(t, `{"num_ctx":8192}`, manifest.MediaTypeImageParams)
		writeShowManifestVariant(t, "plain", manifest.RunnerLlamaCPP, manifest.FormatGGUF, model.ConfigV2{
			ModelFormat: manifest.FormatGGUF,
			ModelFamily: "llama",
		}, map[string]any{"general.architecture": "llama"}, plain)
		before := readManifestBytes(t, model.ParseName("plain"))

		w := createRequest(t, s.ModelMTPHandler, map[string]any{
			"model":             "plain",
			"draft_num_predict": 3,
		})
		if w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400, body = %s", w.Code, w.Body)
		}
		if !strings.Contains(w.Body.String(), "no multi-token prediction layers") {
			t.Fatalf("body = %s", w.Body)
		}
		if after := readManifestBytes(t, model.ParseName("plain")); !bytes.Equal(before, after) {
			t.Fatalf("rejected request changed the manifest:\n got %s\nwant %s", after, before)
		}
	})

	t.Run("cloud models are rejected", func(t *testing.T) {
		w := createRequest(t, s.ModelMTPHandler, map[string]any{
			"model":             "gpt-oss:20b:cloud",
			"draft_num_predict": 3,
		})
		if w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400, body = %s", w.Code, w.Body)
		}
	})

	t.Run("missing draft value is rejected", func(t *testing.T) {
		w := createRequest(t, s.ModelMTPHandler, map[string]any{"model": "glm-mtp"})
		if w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400, body = %s", w.Code, w.Body)
		}
		got, ok := draftNumPredictValue(storedParams(t, model.ParseName("glm-mtp")))
		if !ok || got != 3 {
			t.Fatalf("draft_num_predict = (%d, %v), want it to stay 3", got, ok)
		}
	})

	t.Run("depth above 4 is rejected", func(t *testing.T) {
		w := createRequest(t, s.ModelMTPHandler, map[string]any{
			"model":             "glm-mtp",
			"draft_num_predict": 8,
		})
		if w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400, body = %s", w.Code, w.Body)
		}
	})
}

func writeMTPManifest(t *testing.T, name, params, runner, format string) model.Name {
	t.Helper()

	n := model.ParseName(name)
	config := mustLayer(t, `{"model_format":"`+format+`"}`, manifest.MediaTypeImageConfig)
	weights := mustLayer(t, "weights-"+name, manifest.MediaTypeImageModel)
	weights.From = "parent-weights"
	template := mustLayer(t, "{{ .Prompt }}", manifest.MediaTypeImageTemplate)
	layers := []manifest.Layer{weights, template}
	if params != "" {
		layers = append(layers, mustLayer(t, params, manifest.MediaTypeImageParams))
	}
	if err := manifest.WriteManifestWithMetadata(n, config, layers, runner, format); err != nil {
		t.Fatal(err)
	}
	return n
}

func mustLayer(t *testing.T, contents, mediaType string) manifest.Layer {
	t.Helper()
	layer, err := manifest.NewLayer(strings.NewReader(contents), mediaType)
	if err != nil {
		t.Fatal(err)
	}
	return layer
}

func manifestReference(t *testing.T, name model.Name) manifest.Manifest {
	t.Helper()
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := manifest.NewManifestReference(mf.BlobDigest(), mf.Runner, mf.Format)
	if err != nil {
		t.Fatal(err)
	}
	return ref
}

func readManifestBytes(t *testing.T, name model.Name) []byte {
	t.Helper()
	data, err := manifest.ReadManifestData(name)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func configDigest(t *testing.T, name model.Name) string {
	t.Helper()
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	return mf.Config.Digest
}

func layerDigest(t *testing.T, name model.Name, mediaType string) string {
	t.Helper()
	for _, layer := range parsedLayers(t, name) {
		if layer.MediaType == mediaType {
			return layer.Digest
		}
	}
	t.Fatalf("no %s layer on %s", mediaType, name)
	return ""
}

func layerFrom(t *testing.T, name model.Name, mediaType string) string {
	t.Helper()
	for _, layer := range parsedLayers(t, name) {
		if layer.MediaType == mediaType {
			return layer.From
		}
	}
	t.Fatalf("no %s layer on %s", mediaType, name)
	return ""
}

func parsedLayers(t *testing.T, name model.Name) []manifest.Layer {
	t.Helper()
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	return mf.Layers
}

func listDocument(t *testing.T, name model.Name) manifest.Manifest {
	t.Helper()
	var doc manifest.Manifest
	if err := json.Unmarshal(readManifestBytes(t, name), &doc); err != nil {
		t.Fatal(err)
	}
	return doc
}

func listChildManifest(t *testing.T, name model.Name, runner string) manifest.Manifest {
	t.Helper()
	doc := listDocument(t, name)
	for _, child := range doc.Manifests {
		if child.Runner != runner {
			continue
		}
		if child.BlobDigest() == "" {
			return child
		}
		loaded, err := readManifestAtDigest(child.BlobDigest())
		if err != nil {
			t.Fatal(err)
		}
		return loaded
	}
	t.Fatalf("no %s child on %s", runner, name)
	return manifest.Manifest{}
}

func assertListChildParams(t *testing.T, name model.Name, runner string, want map[string]any) {
	t.Helper()
	child := listChildManifest(t, name, runner)
	for _, layer := range child.Layers {
		if layer.MediaType != manifest.MediaTypeImageParams {
			continue
		}
		path, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		assertParams(t, data, want)
		return
	}
	t.Fatalf("no params layer on %s child of %s", runner, name)
}

func listChildLayerDigest(t *testing.T, name model.Name, runner, mediaType string) string {
	t.Helper()
	child := listChildManifest(t, name, runner)
	for _, layer := range child.Layers {
		if layer.MediaType == mediaType {
			return layer.Digest
		}
	}
	t.Fatalf("no %s layer on %s child of %s", mediaType, runner, name)
	return ""
}

func storedParams(t *testing.T, name model.Name) map[string]any {
	t.Helper()
	for _, layer := range parsedLayers(t, name) {
		if layer.MediaType != manifest.MediaTypeImageParams {
			continue
		}
		path, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return decodeParams(t, data)
	}
	t.Fatalf("no params layer on %s", name)
	return nil
}

func assertStoredParams(t *testing.T, name model.Name, want map[string]any) {
	t.Helper()
	assertParams(t, mustParamsJSON(t, name), want)
}

func mustParamsJSON(t *testing.T, name model.Name) []byte {
	t.Helper()
	for _, layer := range parsedLayers(t, name) {
		if layer.MediaType != manifest.MediaTypeImageParams {
			continue
		}
		path, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	t.Fatalf("no params layer on %s", name)
	return nil
}

func decodeParams(t *testing.T, raw []byte) map[string]any {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var params map[string]any
	if err := dec.Decode(&params); err != nil {
		t.Fatal(err)
	}
	return params
}

func assertParams(t *testing.T, raw []byte, want map[string]any) {
	t.Helper()
	got := decodeParams(t, raw)
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("parameters mismatch (-want +got):\n%s\nraw: %s", diff, raw)
	}
}
