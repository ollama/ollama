package lora

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/manifest"
)

func TestInspectPEFT(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	const module = "model.layers.1.v_proj"
	const prefix = "base_model.model." + module
	layout := Layout{module: {Input: 4, Output: 3}}
	a := safetensors.NewTensorDataFromBytes(prefix+".lora_A.weight", "F16", []int32{2, 4}, make([]byte, 16))
	b := safetensors.NewTensorDataFromBytes(prefix+".lora_B.weight", "F16", []int32{3, 2}, make([]byte, 12))
	for _, tc := range []struct {
		name    string
		option  string
		value   any
		tensors []*safetensors.TensorData
		want    string
	}{
		{name: "valid suffix target", tensors: []*safetensors.TensorData{a, b}},
		{name: "all linear", option: "target_modules", value: "all-linear", tensors: []*safetensors.TensorData{a, b}},
		{name: "missing B", tensors: []*safetensors.TensorData{a}, want: "missing LoRA factor"},
		{name: "wrong rank", option: "r", value: 3, tensors: []*safetensors.TensorData{a, b}, want: "shape"},
		{name: "unsupported target", option: "target_modules", value: []string{"v_proj", "router"}, tensors: []*safetensors.TensorData{a, b}, want: "router"},
		{name: "unexpected tensor", tensors: []*safetensors.TensorData{a, b, a.WithName("router.weight")}, want: "router.weight"},
		{name: "alias collision", tensors: []*safetensors.TensorData{a, b, safetensors.NewTensorDataFromBytes(module+".lora_A.weight", "F16", []int32{2, 4}, make([]byte, 16))}, want: "duplicate A"},
		{name: "mixed dtype", tensors: []*safetensors.TensorData{a, safetensors.NewTensorDataFromBytes(prefix+".lora_B.weight", "BF16", []int32{3, 2}, make([]byte, 12))}, want: "mixed factor dtypes"},
		{name: "DoRA", option: "use_dora", value: true, tensors: []*safetensors.TensorData{a, b}, want: "use_dora"},
		{name: "rank overrides", option: "rank_pattern", value: map[string]int{"v_proj": 2}, tensors: []*safetensors.TensorData{a, b}, want: "rank_pattern"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := map[string]any{"peft_type": "LORA", "r": 2, "lora_alpha": 3, "target_modules": []string{"v_proj"}}
			if tc.option != "" {
				cfg[tc.option] = tc.value
			}
			data, err := json.Marshal(cfg)
			if err != nil {
				t.Fatal(err)
			}
			config, err := manifest.NewLayer(strings.NewReader(string(data)), manifest.MediaTypeImageJSON)
			if err != nil {
				t.Fatal(err)
			}
			weights, err := manifest.NewLayer(safetensors.BuildPackedSafetensorsReader(tc.tensors), manifest.MediaTypePEFTAdapter)
			if err != nil {
				t.Fatal(err)
			}
			got, err := Inspect(manifest.AdapterRef{Config: config.Digest, Weights: weights.Digest}, layout)
			if tc.want != "" {
				if err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("got %v, want error containing %q", err, tc.want)
				}
			} else if err != nil || got.scale != 1.5 || got.pairs[module].b != b.Name {
				t.Fatalf("checkpoint=%+v, error=%v", got, err)
			}
		})
	}
}
