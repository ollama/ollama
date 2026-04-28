package create

import (
	"io"
	"os"
	"path/filepath"
	"testing"

	st "github.com/ollama/ollama/x/safetensors"
)

func TestCreateSafetensorsModel_LagunaHFFP8RespectsSourceTensorPrecision(t *testing.T) {
	tests := []struct {
		name          string
		requested     string
		wantFP8Gate   string
		wantFP8Up     string
		wantFP8Down   string
		wantBF16QProj string
	}{
		{
			name:          "default mxfp8 import keeps source bf16 tensors",
			requested:     "",
			wantFP8Gate:   "mxfp8",
			wantFP8Up:     "mxfp8",
			wantFP8Down:   "mxfp8",
			wantBF16QProj: "",
		},
		{
			name:          "nvfp4 import keeps source bf16 tensors and preserves down_proj at mxfp8",
			requested:     "nvfp4",
			wantFP8Gate:   "nvfp4",
			wantFP8Up:     "nvfp4",
			wantFP8Down:   "mxfp8",
			wantBF16QProj: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			configJSON := `{
				"model_type": "laguna",
				"architectures": ["LagunaForCausalLM"],
				"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}
			}`
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
				t.Fatalf("failed to write config.json: %v", err)
			}

			createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.0.gate_proj.weight", "F8_E4M3", []int32{128, 128}, make([]byte, 128*128)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv", "BF16", []int32{1, 1}, make([]byte, 2)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.0.up_proj.weight", "F8_E4M3", []int32{128, 128}, make([]byte, 128*128)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.0.up_proj.weight_scale_inv", "BF16", []int32{1, 1}, make([]byte, 2)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.0.down_proj.weight", "F8_E4M3", []int32{128, 128}, make([]byte, 128*128)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.0.down_proj.weight_scale_inv", "BF16", []int32{1, 1}, make([]byte, 2)),
				st.NewTensorDataFromBytes("model.layers.0.self_attn.q_proj.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
				st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
				st.NewTensorDataFromBytes("lm_head.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.gate.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
			})

			quantizeByName := make(map[string]string)

			createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
				if _, err := io.ReadAll(r); err != nil {
					return LayerInfo{}, err
				}
				return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
			}
			createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
				if _, err := io.ReadAll(r); err != nil {
					return nil, err
				}
				quantizeByName[name] = quantize
				return []LayerInfo{{Name: name, Digest: "sha256:tensor_" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
			}
			writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error { return nil }

			if err := CreateSafetensorsModel("test-model", dir, tt.requested, createLayer, createTensorLayer, writeManifest, func(string) {}); err != nil {
				t.Fatalf("CreateSafetensorsModel failed: %v", err)
			}

			if got := quantizeByName["model.layers.0.mlp.experts.0.gate_proj.weight"]; got != tt.wantFP8Gate {
				t.Fatalf("gate_proj quantization = %q, want %q", got, tt.wantFP8Gate)
			}
			if got := quantizeByName["model.layers.0.mlp.experts.0.up_proj.weight"]; got != tt.wantFP8Up {
				t.Fatalf("up_proj quantization = %q, want %q", got, tt.wantFP8Up)
			}
			if got := quantizeByName["model.layers.0.mlp.experts.0.down_proj.weight"]; got != tt.wantFP8Down {
				t.Fatalf("down_proj quantization = %q, want %q", got, tt.wantFP8Down)
			}
			for _, name := range []string{
				"model.layers.0.self_attn.q_proj.weight",
				"model.embed_tokens.weight",
				"lm_head.weight",
				"model.layers.0.mlp.gate.weight",
			} {
				if got := quantizeByName[name]; got != tt.wantBF16QProj {
					t.Fatalf("%s quantization = %q, want %q", name, got, tt.wantBF16QProj)
				}
			}
		})
	}
}

func TestCreateSafetensorsModel_LagunaBF16QuantizesOnlyRoutedExperts(t *testing.T) {
	tests := []struct {
		name      string
		requested string
		want      map[string]string
	}{
		{
			name:      "int8 quantizes only routed experts",
			requested: "int8",
			want: map[string]string{
				"model.layers.0.mlp.experts.0.gate_proj.weight":      "int8",
				"model.layers.0.mlp.experts.0.up_proj.weight":        "int8",
				"model.layers.0.mlp.experts.0.down_proj.weight":      "int8",
				"model.layers.0.mlp.shared_experts.gate_proj.weight": "",
				"model.layers.0.mlp.shared_experts.down_proj.weight": "",
				"model.layers.0.self_attn.q_proj.weight":             "",
				"model.layers.0.mlp.down_proj.weight":                "",
				"model.embed_tokens.weight":                          "",
				"lm_head.weight":                                     "",
				"model.layers.0.mlp.gate.weight":                     "",
			},
		},
		{
			name:      "int4 keeps routed down_proj at int8 and leaves others bf16",
			requested: "int4",
			want: map[string]string{
				"model.layers.0.mlp.experts.0.gate_proj.weight":      "int4",
				"model.layers.0.mlp.experts.0.up_proj.weight":        "int4",
				"model.layers.0.mlp.experts.0.down_proj.weight":      "int8",
				"model.layers.0.mlp.shared_experts.gate_proj.weight": "",
				"model.layers.0.mlp.shared_experts.down_proj.weight": "",
				"model.layers.0.self_attn.q_proj.weight":             "",
				"model.layers.0.mlp.down_proj.weight":                "",
				"model.embed_tokens.weight":                          "",
				"lm_head.weight":                                     "",
				"model.layers.0.mlp.gate.weight":                     "",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			configJSON := `{
				"model_type": "laguna",
				"architectures": ["LagunaForCausalLM"]
			}`
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
				t.Fatalf("failed to write config.json: %v", err)
			}

			createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.0.gate_proj.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.0.up_proj.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.experts.0.down_proj.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.shared_experts.gate_proj.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.shared_experts.down_proj.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
				st.NewTensorDataFromBytes("model.layers.0.self_attn.q_proj.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.down_proj.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
				st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
				st.NewTensorDataFromBytes("lm_head.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
				st.NewTensorDataFromBytes("model.layers.0.mlp.gate.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
			})

			quantizeByName := make(map[string]string)

			createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
				if _, err := io.ReadAll(r); err != nil {
					return LayerInfo{}, err
				}
				return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
			}
			createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
				if _, err := io.ReadAll(r); err != nil {
					return nil, err
				}
				quantizeByName[name] = quantize
				return []LayerInfo{{Name: name, Digest: "sha256:tensor_" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
			}
			writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error { return nil }

			if err := CreateSafetensorsModel("test-model", dir, tt.requested, createLayer, createTensorLayer, writeManifest, func(string) {}); err != nil {
				t.Fatalf("CreateSafetensorsModel failed: %v", err)
			}

			for name, want := range tt.want {
				if got := quantizeByName[name]; got != want {
					t.Fatalf("%s quantization = %q, want %q", name, got, want)
				}
			}
		})
	}
}
