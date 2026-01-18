package create

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
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

// createMinimalSafetensors creates a minimal valid safetensors file with one tensor
func createMinimalSafetensors(t *testing.T, path string) {
	t.Helper()

	// Create a minimal safetensors file with a single float32 tensor
	header := map[string]interface{}{
		"test_tensor": map[string]interface{}{
			"dtype":        "F32",
			"shape":        []int{2, 2},
			"data_offsets": []int{0, 16}, // 4 float32 values = 16 bytes
		},
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("failed to marshal header: %v", err)
	}

	// Pad header to 8-byte alignment
	padding := (8 - len(headerJSON)%8) % 8
	headerJSON = append(headerJSON, bytes.Repeat([]byte(" "), padding)...)

	// Write file
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	defer f.Close()

	// Write header size (8 bytes, little endian)
	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerJSON))); err != nil {
		t.Fatalf("failed to write header size: %v", err)
	}

	// Write header
	if _, err := f.Write(headerJSON); err != nil {
		t.Fatalf("failed to write header: %v", err)
	}

	// Write tensor data (16 bytes of zeros for 4 float32 values)
	if _, err := f.Write(make([]byte, 16)); err != nil {
		t.Fatalf("failed to write tensor data: %v", err)
	}
}

func TestCreateSafetensorsModel(t *testing.T) {
	dir := t.TempDir()

	// Create config.json
	configJSON := `{"model_type": "test", "architectures": ["TestModel"]}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	// Create a minimal safetensors file
	createMinimalSafetensors(t, filepath.Join(dir, "model.safetensors"))

	// Track what was created
	var createdLayers []LayerInfo
	var manifestWritten bool
	var manifestModelName string
	var manifestConfigLayer LayerInfo
	var manifestLayers []LayerInfo
	var statusMessages []string

	// Mock callbacks
	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		data, err := io.ReadAll(r)
		if err != nil {
			return LayerInfo{}, err
		}
		layer := LayerInfo{
			Digest:    "sha256:test",
			Size:      int64(len(data)),
			MediaType: mediaType,
			Name:      name,
		}
		createdLayers = append(createdLayers, layer)
		return layer, nil
	}

	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		data, err := io.ReadAll(r)
		if err != nil {
			return nil, err
		}
		layer := LayerInfo{
			Digest:    "sha256:tensor_" + name,
			Size:      int64(len(data)),
			MediaType: "application/vnd.ollama.image.tensor",
			Name:      name,
		}
		createdLayers = append(createdLayers, layer)
		return []LayerInfo{layer}, nil
	}

	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error {
		manifestWritten = true
		manifestModelName = modelName
		manifestConfigLayer = config
		manifestLayers = layers
		return nil
	}

	progressFn := func(status string) {
		statusMessages = append(statusMessages, status)
	}

	// Run CreateSafetensorsModel
	err := CreateSafetensorsModel("test-model", dir, "", createLayer, createTensorLayer, writeManifest, progressFn)
	if err != nil {
		t.Fatalf("CreateSafetensorsModel failed: %v", err)
	}

	// Verify manifest was written
	if !manifestWritten {
		t.Error("manifest was not written")
	}

	if manifestModelName != "test-model" {
		t.Errorf("manifest model name = %q, want %q", manifestModelName, "test-model")
	}

	// Verify config layer was set
	if manifestConfigLayer.Name != "config.json" {
		t.Errorf("config layer name = %q, want %q", manifestConfigLayer.Name, "config.json")
	}

	// Verify we have at least one tensor and one config layer
	hasTensor := false
	hasConfig := false
	for _, layer := range manifestLayers {
		if layer.Name == "test_tensor" {
			hasTensor = true
		}
		if layer.Name == "config.json" {
			hasConfig = true
		}
	}

	if !hasTensor {
		t.Error("no tensor layer found in manifest")
	}
	if !hasConfig {
		t.Error("no config layer found in manifest")
	}

	// Verify status messages were sent
	if len(statusMessages) == 0 {
		t.Error("no status messages received")
	}
}

func TestCreateSafetensorsModel_NoConfigJson(t *testing.T) {
	dir := t.TempDir()

	// Create only a safetensors file, no config.json
	createMinimalSafetensors(t, filepath.Join(dir, "model.safetensors"))

	// Mock callbacks (minimal)
	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		io.ReadAll(r)
		return LayerInfo{Name: name}, nil
	}
	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		io.ReadAll(r)
		return []LayerInfo{{Name: name}}, nil
	}
	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error {
		return nil
	}
	progressFn := func(status string) {}

	err := CreateSafetensorsModel("test-model", dir, "", createLayer, createTensorLayer, writeManifest, progressFn)
	if err == nil {
		t.Error("expected error for missing config.json, got nil")
	}
}

func TestCreateSafetensorsModel_EmptyDir(t *testing.T) {
	dir := t.TempDir()

	// Mock callbacks
	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		return LayerInfo{}, nil
	}
	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		return []LayerInfo{{}}, nil
	}
	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error {
		return nil
	}
	progressFn := func(status string) {}

	err := CreateSafetensorsModel("test-model", dir, "", createLayer, createTensorLayer, writeManifest, progressFn)
	if err == nil {
		t.Error("expected error for empty directory, got nil")
	}
}

func TestCreateSafetensorsModel_SkipsIndexJson(t *testing.T) {
	dir := t.TempDir()

	// Create config.json
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	// Create model.safetensors.index.json (should be skipped)
	indexJSON := `{"metadata": {"total_size": 100}, "weight_map": {}}`
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(indexJSON), 0o644); err != nil {
		t.Fatalf("failed to write index.json: %v", err)
	}

	// Create a minimal safetensors file
	createMinimalSafetensors(t, filepath.Join(dir, "model.safetensors"))

	var configNames []string

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		io.ReadAll(r)
		configNames = append(configNames, name)
		return LayerInfo{Name: name, Digest: "sha256:test"}, nil
	}
	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		io.ReadAll(r)
		return []LayerInfo{{Name: name}}, nil
	}
	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error {
		return nil
	}
	progressFn := func(status string) {}

	err := CreateSafetensorsModel("test-model", dir, "", createLayer, createTensorLayer, writeManifest, progressFn)
	if err != nil {
		t.Fatalf("CreateSafetensorsModel failed: %v", err)
	}

	// Verify model.safetensors.index.json was not included
	for _, name := range configNames {
		if name == "model.safetensors.index.json" {
			t.Error("model.safetensors.index.json should have been skipped")
		}
	}
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

func TestShouldQuantizeTensor(t *testing.T) {
	tests := []struct {
		name   string
		tensor string
		shape  []int32
		want   bool
	}{
		// 2D tensors with sufficient size should be quantized
		{"large 2D weight", "q_proj.weight", []int32{4096, 4096}, true},
		{"medium 2D weight", "small_proj.weight", []int32{128, 128}, true},

		// Small tensors should not be quantized (< 1024 elements)
		{"tiny 2D weight", "tiny.weight", []int32{16, 16}, false},
		{"small 2D weight", "small.weight", []int32{31, 31}, false},

		// 1D tensors should not be quantized
		{"1D tensor", "layer_norm.weight", []int32{4096}, false},

		// 3D+ tensors should not be quantized
		{"3D tensor", "conv.weight", []int32{64, 64, 3}, false},
		{"4D tensor", "conv2d.weight", []int32{64, 64, 3, 3}, false},

		// Embeddings should not be quantized regardless of shape
		{"embedding 2D", "embed_tokens.weight", []int32{32000, 4096}, false},

		// Norms should not be quantized regardless of shape
		{"norm 2D", "layer_norm.weight", []int32{4096, 1}, false},

		// Biases should not be quantized
		{"bias 2D", "proj.bias", []int32{4096, 1}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ShouldQuantizeTensor(tt.tensor, tt.shape)
			if got != tt.want {
				t.Errorf("ShouldQuantizeTensor(%q, %v) = %v, want %v", tt.tensor, tt.shape, got, tt.want)
			}
		})
	}
}

func TestCreateSafetensorsModel_WithQuantize(t *testing.T) {
	dir := t.TempDir()

	// Create config.json
	configJSON := `{"model_type": "test", "architectures": ["TestModel"]}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	// Create a minimal safetensors file
	createMinimalSafetensors(t, filepath.Join(dir, "model.safetensors"))

	var quantizeRequested []string

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		io.ReadAll(r)
		return LayerInfo{Name: name, Digest: "sha256:test"}, nil
	}

	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		io.ReadAll(r)
		quantizeRequested = append(quantizeRequested, quantize)
		return []LayerInfo{{Name: name}}, nil
	}

	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error {
		return nil
	}

	progressFn := func(status string) {}

	// Run with quantize enabled
	err := CreateSafetensorsModel("test-model", dir, "fp8", createLayer, createTensorLayer, writeManifest, progressFn)
	if err != nil {
		t.Fatalf("CreateSafetensorsModel failed: %v", err)
	}

	// Verify quantize was passed to callback (will be false for small test tensor)
	if len(quantizeRequested) == 0 {
		t.Error("no tensors processed")
	}
}

// createMinimalImageGenModel creates a minimal diffusers-style model directory
func createMinimalImageGenModel(t *testing.T, dir string) {
	t.Helper()

	// Create model_index.json
	modelIndex := `{"_class_name": "FluxPipeline", "_diffusers_version": "0.30.0"}`
	if err := os.WriteFile(filepath.Join(dir, "model_index.json"), []byte(modelIndex), 0o644); err != nil {
		t.Fatalf("failed to write model_index.json: %v", err)
	}

	// Create transformer directory with a safetensors file
	transformerDir := filepath.Join(dir, "transformer")
	if err := os.MkdirAll(transformerDir, 0o755); err != nil {
		t.Fatalf("failed to create transformer dir: %v", err)
	}
	createMinimalSafetensors(t, filepath.Join(transformerDir, "model.safetensors"))

	// Create transformer config
	transformerConfig := `{"hidden_size": 3072}`
	if err := os.WriteFile(filepath.Join(transformerDir, "config.json"), []byte(transformerConfig), 0o644); err != nil {
		t.Fatalf("failed to write transformer config: %v", err)
	}
}

func TestCreateImageGenModel(t *testing.T) {
	dir := t.TempDir()
	createMinimalImageGenModel(t, dir)

	var manifestWritten bool
	var manifestModelName string
	var statusMessages []string

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		io.ReadAll(r)
		return LayerInfo{Name: name, Digest: "sha256:test"}, nil
	}

	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		io.ReadAll(r)
		return []LayerInfo{{Name: name, Digest: "sha256:tensor"}}, nil
	}

	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error {
		manifestWritten = true
		manifestModelName = modelName
		return nil
	}

	progressFn := func(status string) {
		statusMessages = append(statusMessages, status)
	}

	err := CreateImageGenModel("test-imagegen", dir, "", createLayer, createTensorLayer, writeManifest, progressFn)
	if err != nil {
		t.Fatalf("CreateImageGenModel failed: %v", err)
	}

	if !manifestWritten {
		t.Error("manifest was not written")
	}

	if manifestModelName != "test-imagegen" {
		t.Errorf("manifest model name = %q, want %q", manifestModelName, "test-imagegen")
	}

	if len(statusMessages) == 0 {
		t.Error("no status messages received")
	}
}

func TestCreateImageGenModel_NoModelIndex(t *testing.T) {
	dir := t.TempDir()

	// Create only transformer without model_index.json
	transformerDir := filepath.Join(dir, "transformer")
	if err := os.MkdirAll(transformerDir, 0o755); err != nil {
		t.Fatalf("failed to create transformer dir: %v", err)
	}
	createMinimalSafetensors(t, filepath.Join(transformerDir, "model.safetensors"))

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		io.ReadAll(r)
		return LayerInfo{Name: name}, nil
	}
	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		io.ReadAll(r)
		return []LayerInfo{{Name: name}}, nil
	}
	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error {
		return nil
	}
	progressFn := func(status string) {}

	err := CreateImageGenModel("test-imagegen", dir, "", createLayer, createTensorLayer, writeManifest, progressFn)
	if err == nil {
		t.Error("expected error for missing model_index.json, got nil")
	}
}

func TestCreateImageGenModel_WithQuantize(t *testing.T) {
	dir := t.TempDir()
	createMinimalImageGenModel(t, dir)

	var quantizeRequested []string

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		io.ReadAll(r)
		return LayerInfo{Name: name, Digest: "sha256:test"}, nil
	}

	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		io.ReadAll(r)
		quantizeRequested = append(quantizeRequested, quantize)
		return []LayerInfo{{Name: name}}, nil
	}

	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error {
		return nil
	}

	progressFn := func(status string) {}

	err := CreateImageGenModel("test-imagegen", dir, "fp8", createLayer, createTensorLayer, writeManifest, progressFn)
	if err != nil {
		t.Fatalf("CreateImageGenModel failed: %v", err)
	}

	if len(quantizeRequested) == 0 {
		t.Error("no tensors processed")
	}
}
