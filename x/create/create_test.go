package create

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	st "github.com/ollama/ollama/x/safetensors"
)

func TestIsTensorModelDir(t *testing.T) {
	tests := []struct {
		name     string
		setup    func(dir string) error
		expected bool
	}{
		{
			name: "valid diffusers model with model_index.json",
			setup: func(dir string) error {
				return os.WriteFile(filepath.Join(dir, "model_index.json"), []byte(`{"_class_name": "FluxPipeline"}`), 0o644)
			},
			expected: true,
		},
		{
			name: "empty directory",
			setup: func(dir string) error {
				return nil
			},
			expected: false,
		},
		{
			name: "directory with other files but no model_index.json",
			setup: func(dir string) error {
				return os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o644)
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := tt.setup(dir); err != nil {
				t.Fatalf("setup failed: %v", err)
			}

			got := IsTensorModelDir(dir)
			if got != tt.expected {
				t.Errorf("IsTensorModelDir() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestValidateScalarFloat32TensorData(t *testing.T) {
	td := st.NewTensorDataFromBytes("linear.weight_scale_2", "F32", []int32{}, encodeFloat32s(2))

	got, err := validateScalarFloat32TensorData(td, "linear.weight.global_scale")
	if err != nil {
		t.Fatalf("validateScalarFloat32TensorData returned error: %v", err)
	}

	if got.Name != "linear.weight.global_scale" {
		t.Fatalf("name = %q, want %q", got.Name, "linear.weight.global_scale")
	}
	if got.Dtype != "F32" {
		t.Fatalf("dtype = %q, want F32", got.Dtype)
	}
	if len(got.Shape) != 0 {
		t.Fatalf("shape = %v, want scalar", got.Shape)
	}
}

func TestValidateScalarFloat32TensorDataRejectsNonScalar(t *testing.T) {
	td := st.NewTensorDataFromBytes("linear.weight_scale_2", "F32", []int32{2}, encodeFloat32s(2, 4))

	_, err := validateScalarFloat32TensorData(td, "linear.weight.global_scale")
	if err == nil || !strings.Contains(err.Error(), "expected scalar F32 tensor") {
		t.Fatalf("validateScalarFloat32TensorData error = %v, want scalar-shape failure", err)
	}
}

func TestInvertScalarFloat32TensorDataRejectsNonF32(t *testing.T) {
	td := st.NewTensorDataFromBytes("linear.weight_global_scale", "BF16", []int32{}, []byte{0, 0})

	_, err := invertScalarFloat32TensorData(td, "linear.weight.global_scale")
	if err == nil || !strings.Contains(err.Error(), "expected F32 tensor") {
		t.Fatalf("invertScalarFloat32TensorData error = %v, want dtype failure", err)
	}
}

func TestIsSafetensorsModelDir(t *testing.T) {
	tests := []struct {
		name     string
		setup    func(dir string) error
		expected bool
	}{
		{
			name: "valid safetensors model with config.json and .safetensors file",
			setup: func(dir string) error {
				if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type": "gemma3"}`), 0o644); err != nil {
					return err
				}
				return os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("dummy"), 0o644)
			},
			expected: true,
		},
		{
			name: "config.json only, no safetensors files",
			setup: func(dir string) error {
				return os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o644)
			},
			expected: false,
		},
		{
			name: "safetensors file only, no config.json",
			setup: func(dir string) error {
				return os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("dummy"), 0o644)
			},
			expected: false,
		},
		{
			name: "empty directory",
			setup: func(dir string) error {
				return nil
			},
			expected: false,
		},
		{
			name: "multiple safetensors files with config.json",
			setup: func(dir string) error {
				if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o644); err != nil {
					return err
				}
				if err := os.WriteFile(filepath.Join(dir, "model-00001-of-00002.safetensors"), []byte("dummy"), 0o644); err != nil {
					return err
				}
				return os.WriteFile(filepath.Join(dir, "model-00002-of-00002.safetensors"), []byte("dummy"), 0o644)
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := tt.setup(dir); err != nil {
				t.Fatalf("setup failed: %v", err)
			}

			got := IsSafetensorsModelDir(dir)
			if got != tt.expected {
				t.Errorf("IsSafetensorsModelDir() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestIsSafetensorsModelDir_NonexistentDir(t *testing.T) {
	got := IsSafetensorsModelDir("/nonexistent/path/that/does/not/exist")
	if got != false {
		t.Errorf("IsSafetensorsModelDir() = %v for nonexistent dir, want false", got)
	}
}

func createTestSafetensors(t *testing.T, path string, tensors []*st.TensorData) {
	t.Helper()

	data, err := io.ReadAll(st.BuildPackedSafetensorsReader(tensors))
	if err != nil {
		t.Fatalf("failed to build packed safetensors: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("failed to write safetensors: %v", err)
	}
}

func encodeFloat32s(vals ...float32) []byte {
	raw := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(raw[i*4:(i+1)*4], math.Float32bits(v))
	}
	return raw
}

func readSafetensorsHeaderNames(t *testing.T, data []byte) []string {
	t.Helper()

	var headerSize uint64
	if err := binary.Read(bytes.NewReader(data[:8]), binary.LittleEndian, &headerSize); err != nil {
		t.Fatalf("failed to read header size: %v", err)
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(data[8:8+headerSize], &header); err != nil {
		t.Fatalf("failed to parse header: %v", err)
	}

	names := make([]string, 0, len(header))
	for name := range header {
		if name == "__metadata__" {
			continue
		}
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

func readPackedTensorRaw(t *testing.T, data []byte, tensorName string) []byte {
	t.Helper()

	var headerSize uint64
	if err := binary.Read(bytes.NewReader(data[:8]), binary.LittleEndian, &headerSize); err != nil {
		t.Fatalf("failed to read header size: %v", err)
	}

	var header map[string]struct {
		Dtype       string  `json:"dtype"`
		Shape       []int32 `json:"shape"`
		DataOffsets [2]int  `json:"data_offsets"`
	}
	if err := json.Unmarshal(data[8:8+headerSize], &header); err != nil {
		t.Fatalf("failed to parse header: %v", err)
	}

	info, ok := header[tensorName]
	if !ok {
		t.Fatalf("tensor %q not found in header", tensorName)
	}

	start := 8 + int(headerSize) + info.DataOffsets[0]
	end := 8 + int(headerSize) + info.DataOffsets[1]
	return data[start:end]
}

func TestResolveManifestPath(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		wantParts []string // Parts that should appear in the path
	}{
		{
			name:      "simple model name",
			modelName: "llama2",
			wantParts: []string{"registry.ollama.ai", "library", "llama2", "latest"},
		},
		{
			name:      "model name with tag",
			modelName: "llama2:7b",
			wantParts: []string{"registry.ollama.ai", "library", "llama2", "7b"},
		},
		{
			name:      "model name with namespace",
			modelName: "myuser/mymodel",
			wantParts: []string{"registry.ollama.ai", "myuser", "mymodel", "latest"},
		},
		{
			name:      "model name with namespace and tag",
			modelName: "myuser/mymodel:v1",
			wantParts: []string{"registry.ollama.ai", "myuser", "mymodel", "v1"},
		},
		{
			name:      "fully qualified model name",
			modelName: "registry.example.com/namespace/model:tag",
			wantParts: []string{"registry.example.com", "namespace", "model", "tag"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := resolveManifestPath(tt.modelName)

			for _, part := range tt.wantParts {
				if !strings.Contains(got, part) {
					t.Errorf("resolveManifestPath(%q) = %q, missing part %q", tt.modelName, got, part)
				}
			}
		})
	}
}

func TestLayerInfo(t *testing.T) {
	layer := LayerInfo{
		Digest:    "sha256:abc123",
		Size:      1024,
		MediaType: "application/vnd.ollama.image.tensor",
		Name:      "model.weight",
	}

	if layer.Digest != "sha256:abc123" {
		t.Errorf("Digest = %q, want %q", layer.Digest, "sha256:abc123")
	}
	if layer.Size != 1024 {
		t.Errorf("Size = %d, want %d", layer.Size, 1024)
	}
	if layer.MediaType != "application/vnd.ollama.image.tensor" {
		t.Errorf("MediaType = %q, want %q", layer.MediaType, "application/vnd.ollama.image.tensor")
	}
	if layer.Name != "model.weight" {
		t.Errorf("Name = %q, want %q", layer.Name, "model.weight")
	}
}

func TestModelConfig(t *testing.T) {
	config := ModelConfig{
		ModelFormat:  "safetensors",
		Capabilities: []string{"completion", "chat"},
	}

	if config.ModelFormat != "safetensors" {
		t.Errorf("ModelFormat = %q, want %q", config.ModelFormat, "safetensors")
	}
	if len(config.Capabilities) != 2 {
		t.Errorf("Capabilities length = %d, want %d", len(config.Capabilities), 2)
	}
}

func TestManifest(t *testing.T) {
	manifest := Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.oci.image.manifest.v1+json",
		Config: ManifestLayer{
			MediaType: "application/vnd.docker.container.image.v1+json",
			Digest:    "sha256:config",
			Size:      100,
		},
		Layers: []ManifestLayer{
			{
				MediaType: "application/vnd.ollama.image.tensor",
				Digest:    "sha256:layer1",
				Size:      1000,
				Name:      "weight.bin",
			},
		},
	}

	if manifest.SchemaVersion != 2 {
		t.Errorf("SchemaVersion = %d, want %d", manifest.SchemaVersion, 2)
	}
	if manifest.Config.Digest != "sha256:config" {
		t.Errorf("Config.Digest = %q, want %q", manifest.Config.Digest, "sha256:config")
	}
	if len(manifest.Layers) != 1 {
		t.Errorf("Layers length = %d, want %d", len(manifest.Layers), 1)
	}
	if manifest.Layers[0].Name != "weight.bin" {
		t.Errorf("Layers[0].Name = %q, want %q", manifest.Layers[0].Name, "weight.bin")
	}
}

func TestShouldQuantize(t *testing.T) {
	tests := []struct {
		name      string
		tensor    string
		component string
		want      bool
	}{
		// VAE component should never be quantized
		{"vae weight", "decoder.weight", "vae", false},
		{"vae bias", "decoder.bias", "vae", false},

		// Embeddings should not be quantized
		{"embedding weight", "embed_tokens.weight", "", false},
		{"embedding in name", "token_embedding.weight", "", false},

		// Norms should not be quantized
		{"layer norm", "layer_norm.weight", "", false},
		{"rms norm", "rms_norm.weight", "", false},
		{"ln prefix", "ln_1.weight", "", false},
		{"layernorm in name", "input_layernorm.weight", "", false},

		// Audio encoder tensors should not be quantized
		{"audio tower weight", "model.audio_tower.layers.0.weight", "", false},
		{"audio tower norm", "model.audio_tower.norm.weight", "", false},
		{"embed audio weight", "embed_audio.weight", "", false},

		// Biases should not be quantized
		{"bias tensor", "attention.bias", "", false},
		{"proj bias", "o_proj.bias", "", false},

		// Linear weights should be quantized
		{"linear weight", "q_proj.weight", "", true},
		{"attention weight", "self_attn.weight", "", true},
		{"mlp weight", "mlp.gate_proj.weight", "", true},

		// Transformer component weights should be quantized
		{"transformer weight", "layers.0.weight", "transformer", true},
		{"text_encoder weight", "encoder.weight", "text_encoder", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ShouldQuantize(tt.tensor, tt.component)
			if got != tt.want {
				t.Errorf("ShouldQuantize(%q, %q) = %v, want %v", tt.tensor, tt.component, got, tt.want)
			}
		})
	}
}

func TestExpertGroupPrefix(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		// Expert tensors should return the group prefix
		{"model.layers.1.mlp.experts.0.down_proj.weight", "model.layers.1.mlp.experts"},
		{"model.layers.1.mlp.experts.63.gate_proj.weight", "model.layers.1.mlp.experts"},
		{"model.layers.0.mlp.experts.0.up_proj.weight", "model.layers.0.mlp.experts"},

		// MoE expert tensors (Gemma-style .moe.experts.)
		{"model.layers.0.moe.experts.0.gate_proj.weight", "model.layers.0.moe.experts"},
		{"model.layers.1.moe.experts.42.down_proj.weight", "model.layers.1.moe.experts"},
		{"language_model.model.layers.2.moe.experts.127.up_proj.weight", "language_model.model.layers.2.moe.experts"},

		// Expert tensors with language_model prefix should also match
		{"language_model.model.layers.0.mlp.experts.0.gate_proj.weight", "language_model.model.layers.0.mlp.experts"},
		{"language_model.model.layers.1.mlp.experts.255.down_proj.weight", "language_model.model.layers.1.mlp.experts"},

		// Shared expert tensors should return their own group prefix
		{"model.layers.1.mlp.shared_experts.down_proj.weight", "model.layers.1.mlp.shared_experts"},
		{"model.layers.2.mlp.shared_experts.gate_proj.weight", "model.layers.2.mlp.shared_experts"},

		// Rewritten Qwen switch_mlp tensors should also be packed per-layer.
		{"model.layers.1.mlp.switch_mlp.down_proj.weight", "model.layers.1.mlp.switch_mlp"},
		{"language_model.layers.2.mlp.switch_mlp.gate_proj.weight", "language_model.layers.2.mlp.switch_mlp"},
		{"language_model.model.layers.3.mlp.switch_mlp.up_proj.weight", "language_model.model.layers.3.mlp.switch_mlp"},
		{"model.language_model.layers.4.mlp.switch_mlp.gate_proj.weight", "model.language_model.layers.4.mlp.switch_mlp"},

		// Non-expert tensors should return empty string
		{"model.layers.0.mlp.down_proj.weight", ""},    // dense layer, no experts
		{"model.layers.1.mlp.gate.weight", ""},         // routing gate, not an expert
		{"model.embed_tokens.weight", ""},              // embedding
		{"model.layers.0.self_attn.q_proj.weight", ""}, // attention
		{"model.norm.weight", ""},                      // norm
		{"lm_head.weight", ""},                         // output head
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExpertGroupPrefix(tt.name)
			if got != tt.want {
				t.Errorf("ExpertGroupPrefix(%q) = %q, want %q", tt.name, got, tt.want)
			}
		})
	}
}

func TestGetTensorQuantization_StackedExpert3D(t *testing.T) {
	gateUp := GetTensorQuantization(
		"model.layers.1.mlp.switch_mlp.gate_up_proj.weight",
		[]int32{64, 22016, 4096},
		"int4",
	)
	if gateUp != "int4" {
		t.Fatalf("gate_up_proj quantization = %q, want %q", gateUp, "int4")
	}

	down := GetTensorQuantization(
		"model.layers.1.mlp.experts.down_proj.weight",
		[]int32{64, 4096, 14336},
		"int4",
	)
	if down != "int8" {
		t.Fatalf("down_proj quantization = %q, want %q", down, "int8")
	}

	combinedGateUp := GetTensorQuantization(
		"model.language_model.layers.0.mlp.experts.gate_up_proj",
		[]int32{256, 1024, 2048},
		"int8",
	)
	if combinedGateUp != "int8" {
		t.Fatalf("combined gate_up_proj quantization = %q, want %q", combinedGateUp, "int8")
	}

	combinedDown := GetTensorQuantization(
		"model.language_model.layers.0.mlp.experts.down_proj",
		[]int32{256, 2048, 512},
		"int4",
	)
	if combinedDown != "int8" {
		t.Fatalf("combined down_proj quantization = %q, want %q", combinedDown, "int8")
	}

	nvfp4GateUp := GetTensorQuantization(
		"language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
		[]int32{64, 11008, 4096},
		"nvfp4",
	)
	if nvfp4GateUp != "nvfp4" {
		t.Fatalf("nvfp4 gate_proj quantization = %q, want %q", nvfp4GateUp, "nvfp4")
	}

	nvfp4Down := GetTensorQuantization(
		"language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
		[]int32{64, 4096, 11008},
		"nvfp4",
	)
	if nvfp4Down != "nvfp4" {
		t.Fatalf("nvfp4 down_proj quantization = %q, want %q", nvfp4Down, "nvfp4")
	}

	mxfp4GateUp := GetTensorQuantization(
		"language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
		[]int32{64, 11008, 4096},
		"mxfp4",
	)
	if mxfp4GateUp != "mxfp4" {
		t.Fatalf("mxfp4 gate_proj quantization = %q, want %q", mxfp4GateUp, "mxfp4")
	}

	mxfp4Down := GetTensorQuantization(
		"language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
		[]int32{64, 4096, 11008},
		"mxfp4",
	)
	if mxfp4Down != "mxfp4" {
		t.Fatalf("mxfp4 down_proj quantization = %q, want %q", mxfp4Down, "mxfp4")
	}
}

func TestIsAligned(t *testing.T) {
	tests := []struct {
		name      string
		shape     []int32
		quantType string
		want      bool
	}{
		// int4/int8: group_size=64
		{"int4 aligned", []int32{1024, 4096}, "int4", true},
		{"int4 unaligned", []int32{1024, 48}, "int4", false},
		{"int8 aligned", []int32{1024, 128}, "int8", true},
		{"int8 unaligned", []int32{1024, 32}, "int8", false},

		// nvfp4: group_size=16
		{"nvfp4 aligned", []int32{1024, 48}, "nvfp4", true},
		{"nvfp4 unaligned", []int32{1024, 24}, "nvfp4", false},
		{"nvfp4 aligned 16", []int32{1024, 16}, "nvfp4", true},

		// mxfp4/mxfp8: group_size=32
		{"mxfp4 aligned", []int32{1024, 64}, "mxfp4", true},
		{"mxfp4 unaligned", []int32{1024, 48}, "mxfp4", false},
		{"mxfp8 aligned", []int32{1024, 32}, "mxfp8", true},
		{"mxfp8 unaligned", []int32{1024, 24}, "mxfp8", false},

		// Edge cases
		{"empty shape", []int32{}, "int4", false},
		{"1D tensor", []int32{4096}, "int4", true},
		{"3D stacked expert", []int32{128, 4096, 2816}, "int4", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isAligned(tt.shape, tt.quantType)
			if got != tt.want {
				t.Errorf("isAligned(%v, %q) = %v, want %v", tt.shape, tt.quantType, got, tt.want)
			}
		})
	}
}

func TestGetTensorQuantization_MixedPrecisionPromotion(t *testing.T) {
	aligned := []int32{4096, 4096} // divisible by 64

	tests := []struct {
		name     string
		tensor   string
		shape    []int32
		quantize string
		want     string
	}{
		// int4 → int8 promotion for sensitive tensors
		{"v_proj int4 promoted", "model.layers.0.self_attn.v_proj.weight", aligned, "int4", "int8"},
		{"k_proj int4 promoted", "model.layers.0.self_attn.k_proj.weight", aligned, "int4", "int8"},
		{"down_proj int4 promoted", "model.layers.0.mlp.down_proj.weight", aligned, "int4", "int8"},

		// Non-sensitive int4 tensors stay int4
		{"q_proj int4 stays", "model.layers.0.self_attn.q_proj.weight", aligned, "int4", "int4"},
		{"o_proj int4 stays", "model.layers.0.self_attn.o_proj.weight", aligned, "int4", "int4"},
		{"gate_proj int4 stays", "model.layers.0.mlp.gate_proj.weight", aligned, "int4", "int4"},
		{"up_proj int4 stays", "model.layers.0.mlp.up_proj.weight", aligned, "int4", "int4"},

		// nvfp4/mxfp4 → mxfp8 promotion for sensitive tensors; mxfp8 stays uniform
		{"v_proj nvfp4 promoted", "model.layers.0.self_attn.v_proj.weight", aligned, "nvfp4", "mxfp8"},
		{"down_proj mxfp4 promoted", "model.layers.0.mlp.down_proj.weight", aligned, "mxfp4", "mxfp8"},
		{"q_proj nvfp4 stays", "model.layers.0.self_attn.q_proj.weight", aligned, "nvfp4", "nvfp4"},
		{"v_proj mxfp8 uniform", "model.layers.0.self_attn.v_proj.weight", aligned, "mxfp8", "mxfp8"},

		// int8: already 8-bit, no promotion
		{"v_proj int8 stays", "model.layers.0.self_attn.v_proj.weight", aligned, "int8", "int8"},

		// lm_head stays at source precision for fp modes, quantizes for affine
		{"lm_head nvfp4 kept", "lm_head.weight", aligned, "nvfp4", ""},
		{"lm_head mxfp8 kept", "lm_head.weight", aligned, "mxfp8", ""},
		{"lm_head int4 stays", "lm_head.weight", aligned, "int4", "int4"},

		// Expert tensors: down_proj also promoted for int4
		{"expert down_proj int4", "model.layers.0.mlp.experts.down_proj.weight", []int32{128, 4096, 2816}, "int4", "int8"},
		{"moe expert down_proj int4", "model.layers.0.moe.experts.down_proj.weight", []int32{128, 4096, 2816}, "int4", "int8"},

		// Unaligned: falls back to bf16 (empty string)
		{"v_proj int4 unaligned", "model.layers.0.self_attn.v_proj.weight", []int32{1024, 48}, "int4", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetTensorQuantization(tt.tensor, tt.shape, tt.quantize)
			if got != tt.want {
				t.Errorf("GetTensorQuantization(%q, %v, %q) = %q, want %q",
					tt.tensor, tt.shape, tt.quantize, got, tt.want)
			}
		})
	}
}
