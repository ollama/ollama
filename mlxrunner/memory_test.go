package mlxrunner

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/manifest"
)

func TestResidencyExcludesConfiguredEngramAndDraft(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	config, err := manifest.NewLayer(strings.NewReader(`{
		"architectures": ["HYV4FlashForConditionalGeneration"],
		"text_config": {"num_hidden_layers": 4, "engram_layer_ids": [2]}
	}`), manifest.MediaTypeImageJSON)
	if err != nil {
		t.Fatal(err)
	}
	config.Name = "config.json"
	tensor := func(name string, bytes int64) manifest.Layer {
		return manifest.Layer{Name: name, MediaType: manifest.MediaTypeImageTensor, Size: bytes}
	}
	blob := func(names ...string) manifest.Layer {
		t.Helper()
		tensors := make([]*safetensors.TensorData, len(names))
		for i, name := range names {
			tensors[i] = safetensors.NewTensorDataFromBytes(name, "F32", []int32{1}, make([]byte, 4))
		}
		layer, err := manifest.NewLayer(safetensors.BuildPackedSafetensorsReader(tensors), manifest.MediaTypeImageTensor)
		if err != nil {
			t.Fatal(err)
		}
		layer.Name = names[0]
		return layer
	}
	mixed := blob("model.dfly_layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.weight")
	m := &manifest.Manifest{Layers: []manifest.Layer{
		config,
		blob("model.layers.1.engram_layer.multi_head_embedding.embedding.weight"),
		tensor("model.layers.2.engram_layer.multi_head_embedding.embedding.weight", 32<<30),
		blob("model.dfly_layers.0.norm.weight"),
		mixed,
		tensor("model.layers.1.engram_layer.value_proj.weight", 1<<20),
		tensor("model.layers.1.mlp.experts.down_proj", 4<<30),
	}}
	got, err := modelGPUBytes(m)
	if want := uint64((36 << 30) + (1 << 20) + mixed.Size); err != nil || got != want {
		t.Fatalf("estimated resident bytes = %d, want %d, err = %v", got, want, err)
	}
}
