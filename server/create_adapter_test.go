package server

import (
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestCreateMLXAdapter(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	raw := []byte(`{"architectures":["Qwen3ForCausalLM"],"hidden_size":4,"num_attention_heads":1,"num_hidden_layers":1,"intermediate_size":4,"vocab_size":8}`)
	baseLayers := []manifest.Layer{
		{Name: "config.json", MediaType: manifest.MediaTypeImageJSON, Digest: createTestBlob(t, raw), Size: int64(len(raw))},
		{Name: "model.norm.weight", MediaType: manifest.MediaTypeImageTensor, Digest: createTestSafetensorsBlob(t, []*safetensors.TensorData{
			safetensors.NewTensorDataFromBytes("model.norm.weight", "F32", []int32{4}, make([]byte, 16)),
		})},
	}
	createSafetensorsTestModel(t, "qwen-base", model.ConfigV2{ModelFormat: manifest.FormatSafetensors, Capabilities: []string{"completion"}}, baseLayers)
	const proj = "base_model.model.model.layers.0.self_attn.v_proj."
	weights := createTestSafetensorsBlob(t, []*safetensors.TensorData{
		safetensors.NewTensorDataFromBytes(proj+"lora_A.weight", "BF16", []int32{2, 4}, make([]byte, 16)),
		safetensors.NewTensorDataFromBytes(proj+"lora_B.weight", "BF16", []int32{4, 2}, make([]byte, 16)),
	})
	cfg := createTestBlob(t, []byte(`{"peft_type":"LORA","r":2,"lora_alpha":4,"target_modules":["v_proj"]}`))
	request := api.CreateRequest{Model: "qwen-adapted", From: "qwen-base", Adapters: map[string]string{manifest.AdapterConfigName: cfg, manifest.AdapterWeightsName: weights}, Stream: &stream}
	var s Server
	w := createRequest(t, s.CreateHandler, request)
	if w.Code != http.StatusOK {
		t.Fatalf("create: %d %s", w.Code, w.Body.String())
	}
	base, err := GetModel(request.From)
	if err != nil {
		t.Fatal(err)
	}
	adapted, err := GetModel(request.Model)
	if err != nil {
		t.Fatal(err)
	}
	if adapted.ManifestDigest == base.ManifestDigest || schedulerModelKey(base) != schedulerModelKey(adapted) {
		t.Fatal("expected distinct tags to share the base runner key")
	}
	if adapted.Adapter != (manifest.AdapterRef{Config: cfg, Weights: weights}) || base.Adapter != (manifest.AdapterRef{}) {
		t.Fatalf("base=%+v adapted=%+v", base.Adapter, adapted.Adapter)
	}
	mf, err := manifest.ParseNamedManifest(model.ParseName(request.Model))
	if err != nil {
		t.Fatal(err)
	}
	original, err := manifest.ParseNamedManifest(model.ParseName(request.From))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.EqualFunc(mf.TensorLayers(), original.TensorLayers(), func(a, b manifest.Layer) bool { return a.Name == b.Name && a.Digest == b.Digest }) {
		t.Fatalf("unexpected base weights: %v", mf.TensorLayers())
	}

	request.Adapters[manifest.AdapterConfigName] = createTestBlob(t, []byte(`{"peft_type":"LORA","r":3,"lora_alpha":4,"target_modules":["v_proj"]}`))
	w = createRequest(t, s.CreateHandler, request)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "shape") {
		t.Fatalf("invalid replacement: %d %s", w.Code, w.Body.String())
	}
	after, err := GetModel(request.Model)
	if err != nil {
		t.Fatal(err)
	}
	if after.ManifestDigest != adapted.ManifestDigest {
		t.Fatal("failed create replaced the previous tag")
	}
}
