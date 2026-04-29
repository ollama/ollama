package create

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/d4l3k/go-bfloat16"
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

func readSingleTensorHeader(t *testing.T, data []byte) (string, []int32) {
	t.Helper()

	var headerSize uint64
	if err := binary.Read(bytes.NewReader(data[:8]), binary.LittleEndian, &headerSize); err != nil {
		t.Fatalf("failed to read header size: %v", err)
	}

	var header map[string]struct {
		Dtype string  `json:"dtype"`
		Shape []int32 `json:"shape"`
	}
	if err := json.Unmarshal(data[8:8+headerSize], &header); err != nil {
		t.Fatalf("failed to parse header: %v", err)
	}

	for name, info := range header {
		if name == "__metadata__" {
			continue
		}
		return info.Dtype, info.Shape
	}

	t.Fatal("no tensor entry found in header")
	return "", nil
}

func readSingleTensorRaw(t *testing.T, data []byte) []byte {
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

	for name, info := range header {
		if name == "__metadata__" {
			continue
		}
		start := 8 + int(headerSize) + info.DataOffsets[0]
		end := 8 + int(headerSize) + info.DataOffsets[1]
		return data[start:end]
	}

	t.Fatal("no tensor entry found in header")
	return nil
}

func encodeFloat32s(vals ...float32) []byte {
	raw := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(raw[i*4:(i+1)*4], math.Float32bits(v))
	}
	return raw
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

func TestCreateSafetensorsModel_PacksPrequantizedTensorTriplets(t *testing.T) {
	dir := t.TempDir()

	configJSON := `{
		"model_type": "test",
		"architectures": ["TestModel"],
		"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("linear.weight", "U32", []int32{4, 4}, make([]byte, 16)),
		st.NewTensorDataFromBytes("linear.scales", "BF16", []int32{4, 1}, make([]byte, 8)),
		st.NewTensorDataFromBytes("linear.biases", "BF16", []int32{4, 1}, make([]byte, 8)),
		st.NewTensorDataFromBytes("plain.weight", "F32", []int32{2, 2}, make([]byte, 16)),
	})

	var packedHeader map[string]json.RawMessage
	var tensorLayerNames []string
	var createTensorLayerNames []string

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		data, err := io.ReadAll(r)
		if err != nil {
			return LayerInfo{}, err
		}
		if mediaType == "application/vnd.ollama.image.tensor" && name == "linear.weight" {
			var headerSize uint64
			if err := binary.Read(bytes.NewReader(data[:8]), binary.LittleEndian, &headerSize); err != nil {
				return LayerInfo{}, err
			}
			if err := json.Unmarshal(data[8:8+headerSize], &packedHeader); err != nil {
				return LayerInfo{}, err
			}
		}
		tensorLayerNames = append(tensorLayerNames, name)
		return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType, Size: int64(len(data))}, nil
	}

	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		if _, err := io.ReadAll(r); err != nil {
			return nil, err
		}
		createTensorLayerNames = append(createTensorLayerNames, name)
		return []LayerInfo{{Name: name, Digest: "sha256:tensor_" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
	}

	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error {
		return nil
	}

	progressFn := func(status string) {}

	if err := CreateSafetensorsModel("test-model", dir, "", createLayer, createTensorLayer, writeManifest, progressFn); err != nil {
		t.Fatalf("CreateSafetensorsModel failed: %v", err)
	}

	if packedHeader == nil {
		t.Fatal("expected packed quantized header for linear.weight")
	}
	if _, ok := packedHeader["linear.weight"]; !ok {
		t.Fatalf("packed header missing linear.weight: %v", packedHeader)
	}
	if _, ok := packedHeader["linear.weight.scale"]; !ok {
		t.Fatalf("packed header missing linear.weight.scale: %v", packedHeader)
	}
	if _, ok := packedHeader["linear.weight.bias"]; !ok {
		t.Fatalf("packed header missing linear.weight.bias: %v", packedHeader)
	}

	var metadata map[string]string
	if metaRaw, ok := packedHeader["__metadata__"]; ok {
		if err := json.Unmarshal(metaRaw, &metadata); err != nil {
			t.Fatalf("failed to parse packed metadata: %v", err)
		}
	}
	if metadata["quant_type"] != "int4" {
		t.Fatalf("quant_type = %q, want %q", metadata["quant_type"], "int4")
	}
	if metadata["group_size"] != "64" {
		t.Fatalf("group_size = %q, want %q", metadata["group_size"], "64")
	}

	if slices.Contains(createTensorLayerNames, "linear.weight") {
		t.Fatalf("linear.weight unexpectedly handled by createTensorLayer: %v", createTensorLayerNames)
	}
	if slices.Contains(createTensorLayerNames, "linear.scales") || slices.Contains(createTensorLayerNames, "linear.biases") {
		t.Fatalf("quantized companions unexpectedly handled separately: %v", createTensorLayerNames)
	}
	if !slices.Contains(createTensorLayerNames, "plain.weight") {
		t.Fatalf("plain.weight missing from createTensorLayer calls: %v", createTensorLayerNames)
	}
	if slices.Contains(tensorLayerNames, "linear.scales") || slices.Contains(tensorLayerNames, "linear.biases") {
		t.Fatalf("quantized companions unexpectedly emitted as layers: %v", tensorLayerNames)
	}
}

func TestCreateSafetensorsModel_HFFP8AutoConvertsToMXFP8(t *testing.T) {
	dir := t.TempDir()

	configJSON := `{
		"model_type": "test",
		"architectures": ["TestModel"],
		"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("linear.weight", "F8_E4M3", []int32{2, 2}, []byte{1, 2, 3, 4}),
		st.NewTensorDataFromBytes("linear.weight_scale_inv", "BF16", []int32{1, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("dense.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
		st.NewTensorDataFromBytes("norm.weight", "BF16", []int32{2}, make([]byte, 4)),
	})

	quantizeByName := make(map[string]string)
	headerNamesByName := make(map[string][]string)

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		_, err := io.ReadAll(r)
		if err != nil {
			return LayerInfo{}, err
		}
		return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
	}

	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		data, err := io.ReadAll(r)
		if err != nil {
			return nil, err
		}
		quantizeByName[name] = quantize
		headerNamesByName[name] = readSafetensorsHeaderNames(t, data)
		return []LayerInfo{{Name: name, Digest: "sha256:tensor_" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
	}

	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error { return nil }

	var statusMessages []string
	progressFn := func(status string) {
		statusMessages = append(statusMessages, status)
	}

	if err := CreateSafetensorsModel("test-model", dir, "", createLayer, createTensorLayer, writeManifest, progressFn); err != nil {
		t.Fatalf("CreateSafetensorsModel failed: %v", err)
	}

	if len(statusMessages) == 0 {
		t.Fatal("no status messages received")
	}
	if got, want := statusMessages[0], "importing model.safetensors (4 tensors, converting source E4M3 block-FP8 to MLX mxfp8)"; got != want {
		t.Fatalf("status = %q, want %q", got, want)
	}

	if got := quantizeByName["linear.weight"]; got != "mxfp8" {
		t.Fatalf("linear.weight quantization = %q, want %q", got, "mxfp8")
	}

	if got := quantizeByName["norm.weight"]; got != "" {
		t.Fatalf("norm.weight quantization = %q, want empty", got)
	}
	if got := quantizeByName["dense.weight"]; got != "" {
		t.Fatalf("dense.weight quantization = %q, want empty", got)
	}

	if _, ok := quantizeByName["linear.weight_scale_inv"]; ok {
		t.Fatal("linear.weight_scale_inv should not be imported as a standalone tensor")
	}

	if got := headerNamesByName["linear.weight"]; !slices.Equal(got, []string{"linear.weight", "linear.weight.scale_inv"}) {
		t.Fatalf("linear.weight blob tensors = %v, want %v", got, []string{"linear.weight", "linear.weight.scale_inv"})
	}

	if got := headerNamesByName["norm.weight"]; !slices.Equal(got, []string{"norm.weight"}) {
		t.Fatalf("norm.weight blob tensors = %v, want %v", got, []string{"norm.weight"})
	}
	if got := headerNamesByName["dense.weight"]; !slices.Equal(got, []string{"dense.weight"}) {
		t.Fatalf("dense.weight blob tensors = %v, want %v", got, []string{"dense.weight"})
	}
}

func TestCreateSafetensorsModel_CompressedTensorsFP8WeightScale(t *testing.T) {
	dir := t.TempDir()

	configJSON := `{
		"model_type": "test",
		"architectures": ["TestModel"],
		"compression_config": {
			"quant_method": "compressed-tensors",
			"format": "float-quantized",
			"config_groups": {
				"group_0": {
					"format": "float-quantized",
					"weights": {
						"type": "float",
						"num_bits": 8,
						"block_structure": [128, 128]
					}
				}
			}
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("linear.weight", "F8_E4M3", []int32{2, 2}, []byte{1, 2, 3, 4}),
		st.NewTensorDataFromBytes("linear.weight_scale", "BF16", []int32{1, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("norm.weight", "BF16", []int32{2}, make([]byte, 4)),
	})

	quantizeByName := make(map[string]string)
	headerNamesByName := make(map[string][]string)

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		if _, err := io.ReadAll(r); err != nil {
			return LayerInfo{}, err
		}
		return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
	}
	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		data, err := io.ReadAll(r)
		if err != nil {
			return nil, err
		}
		quantizeByName[name] = quantize
		headerNamesByName[name] = readSafetensorsHeaderNames(t, data)
		return []LayerInfo{{Name: name, Digest: "sha256:tensor_" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
	}
	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error { return nil }

	var statusMessages []string
	progressFn := func(status string) {
		statusMessages = append(statusMessages, status)
	}

	if err := CreateSafetensorsModel("test-model", dir, "", createLayer, createTensorLayer, writeManifest, progressFn); err != nil {
		t.Fatalf("CreateSafetensorsModel failed: %v", err)
	}
	if len(statusMessages) == 0 {
		t.Fatal("no status messages received")
	}
	if got, want := statusMessages[0], "importing model.safetensors (3 tensors, converting source E4M3 block-FP8 to MLX mxfp8)"; got != want {
		t.Fatalf("status = %q, want %q", got, want)
	}
	if got := quantizeByName["linear.weight"]; got != "mxfp8" {
		t.Fatalf("linear.weight quantization = %q, want mxfp8", got)
	}
	if _, ok := quantizeByName["linear.weight_scale"]; ok {
		t.Fatal("linear.weight_scale should not be imported as a standalone tensor")
	}
	if got := headerNamesByName["linear.weight"]; !slices.Equal(got, []string{"linear.weight", "linear.weight.scale"}) {
		t.Fatalf("linear.weight blob tensors = %v, want %v", got, []string{"linear.weight", "linear.weight.scale"})
	}
}

func TestCreateSafetensorsModel_HFFP8SourceCanConvertToNVFP4(t *testing.T) {
	dir := t.TempDir()

	configJSON := `{
		"model_type": "test",
		"architectures": ["TestModel"],
		"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("linear.weight", "F8_E4M3", []int32{128, 128}, make([]byte, 128*128)),
		st.NewTensorDataFromBytes("linear.weight_scale_inv", "BF16", []int32{1, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.0.down_proj.weight", "F8_E4M3", []int32{128, 128}, make([]byte, 128*128)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.experts.0.down_proj.weight_scale_inv", "BF16", []int32{1, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("model.layers.0.self_attn.q_proj.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
		st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
		st.NewTensorDataFromBytes("lm_head.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
		st.NewTensorDataFromBytes("model.layers.0.mlp.gate.weight", "BF16", []int32{128, 128}, make([]byte, 128*128*2)),
		st.NewTensorDataFromBytes("norm.weight", "BF16", []int32{128}, make([]byte, 256)),
	})

	quantizeByName := make(map[string]string)
	headerNamesByName := make(map[string][]string)

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		if _, err := io.ReadAll(r); err != nil {
			return LayerInfo{}, err
		}
		return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
	}
	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		data, err := io.ReadAll(r)
		if err != nil {
			return nil, err
		}
		quantizeByName[name] = quantize
		headerNamesByName[name] = readSafetensorsHeaderNames(t, data)
		return []LayerInfo{{Name: name, Digest: "sha256:tensor_" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
	}
	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error { return nil }

	var statusMessages []string
	progressFn := func(status string) {
		statusMessages = append(statusMessages, status)
	}

	if err := CreateSafetensorsModel("test-model", dir, "nvfp4", createLayer, createTensorLayer, writeManifest, progressFn); err != nil {
		t.Fatalf("CreateSafetensorsModel failed: %v", err)
	}
	if len(statusMessages) == 0 {
		t.Fatal("no status messages received")
	}
	if got, want := statusMessages[0], "importing model.safetensors (9 tensors, converting source E4M3 block-FP8 to MLX nvfp4)"; got != want {
		t.Fatalf("status = %q, want %q", got, want)
	}
	if got := quantizeByName["linear.weight"]; got != "nvfp4" {
		t.Fatalf("linear.weight quantization = %q, want nvfp4", got)
	}
	if got := quantizeByName["model.layers.0.mlp.experts.0.down_proj.weight"]; got != "mxfp8" {
		t.Fatalf("source fp8 down_proj quantization = %q, want mxfp8", got)
	}
	for _, name := range []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.embed_tokens.weight",
		"lm_head.weight",
	} {
		if got := quantizeByName[name]; got != "mxfp8" {
			t.Fatalf("%s quantization = %q, want mxfp8", name, got)
		}
	}
	if got := quantizeByName["model.layers.0.mlp.gate.weight"]; got != "" {
		t.Fatalf("router gate quantization = %q, want empty", got)
	}
	if got := quantizeByName["norm.weight"]; got != "" {
		t.Fatalf("norm.weight quantization = %q, want empty", got)
	}
	if got := headerNamesByName["linear.weight"]; !slices.Equal(got, []string{"linear.weight", "linear.weight.scale_inv"}) {
		t.Fatalf("linear.weight blob tensors = %v, want %v", got, []string{"linear.weight", "linear.weight.scale_inv"})
	}
}

func TestCreateSafetensorsModel_RejectsRequantizingQuantizedSources(t *testing.T) {
	tests := []struct {
		name       string
		configJSON string
		tensors    []*st.TensorData
		wantErr    string
	}{
		{
			name:       "prequantized affine",
			configJSON: `{"model_type": "test", "architectures": ["TestModel"]}`,
			tensors: []*st.TensorData{
				st.NewTensorDataFromBytes("linear.weight", "U32", []int32{4, 4}, make([]byte, 16)),
				st.NewTensorDataFromBytes("linear.scales", "BF16", []int32{4, 1}, make([]byte, 8)),
			},
			wantErr: `cannot requantize already-quantized source model with --quantize "int4"`,
		},
		{
			name: "hf fp8 source",
			configJSON: `{
				"model_type": "test",
				"architectures": ["TestModel"],
				"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}
			}`,
			tensors: []*st.TensorData{
				st.NewTensorDataFromBytes("linear.weight", "F8_E4M3", []int32{2, 2}, []byte{1, 2, 3, 4}),
				st.NewTensorDataFromBytes("linear.weight_scale_inv", "BF16", []int32{1, 1}, make([]byte, 2)),
			},
			wantErr: `cannot convert already-quantized fp8 source model with --quantize "int4"`,
		},
		{
			name: "packed nvfp4 source",
			configJSON: `{
				"model_type": "test",
				"architectures": ["TestModel"],
				"compression_config": {"format": "nvfp4-pack-quantized"}
			}`,
			tensors: []*st.TensorData{
				st.NewTensorDataFromBytes("linear.weight_packed", "U8", []int32{16, 8}, make([]byte, 128)),
				st.NewTensorDataFromBytes("linear.weight_scale", "F8_E4M3", []int32{16, 1}, make([]byte, 16)),
			},
			wantErr: `cannot requantize already-quantized source model with --quantize "int4"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.configJSON), 0o644); err != nil {
				t.Fatalf("failed to write config.json: %v", err)
			}
			createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), tt.tensors)

			createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
				return LayerInfo{}, nil
			}
			createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
				return nil, nil
			}
			writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error { return nil }

			err := CreateSafetensorsModel("test-model", dir, "int4", createLayer, createTensorLayer, writeManifest, func(string) {})
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %q, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestCreateSafetensorsModel_PackedNVFP4PreservesSourceLayout(t *testing.T) {
	dir := t.TempDir()

	configJSON := `{
		"model_type": "test",
		"architectures": ["TestModel"],
		"compression_config": {"format": "nvfp4-pack-quantized"}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("linear.weight_packed", "U8", []int32{16, 8}, make([]byte, 128)),
		st.NewTensorDataFromBytes("linear.weight_scale", "F8_E4M3", []int32{16, 1}, make([]byte, 16)),
		st.NewTensorDataFromBytes("linear.weight_global_scale", "F32", []int32{}, encodeFloat32s(4)),
		st.NewTensorDataFromBytes("linear.input_global_scale", "F32", []int32{}, encodeFloat32s(8)),
		st.NewTensorDataFromBytes("norm.weight", "BF16", []int32{16}, make([]byte, 32)),
	})

	var statusMessages []string
	layerHeaders := make(map[string]map[string]json.RawMessage)
	layerData := make(map[string][]byte)
	var tensorLayerNames []string

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		data, err := io.ReadAll(r)
		if err != nil {
			return LayerInfo{}, err
		}
		if mediaType == "application/vnd.ollama.image.tensor" {
			if len(data) < 8 {
				return LayerInfo{}, io.ErrUnexpectedEOF
			}
			var headerSize uint64
			if err := binary.Read(bytes.NewReader(data[:8]), binary.LittleEndian, &headerSize); err != nil {
				return LayerInfo{}, err
			}
			var header map[string]json.RawMessage
			if err := json.Unmarshal(data[8:8+headerSize], &header); err != nil {
				return LayerInfo{}, err
			}
			layerHeaders[name] = header
			layerData[name] = data
		}
		return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
	}
	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		if _, err := io.ReadAll(r); err != nil {
			return nil, err
		}
		tensorLayerNames = append(tensorLayerNames, name)
		return []LayerInfo{{Name: name, Digest: "sha256:tensor_" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
	}
	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error { return nil }
	progressFn := func(status string) { statusMessages = append(statusMessages, status) }

	if err := CreateSafetensorsModel("test-model", dir, "", createLayer, createTensorLayer, writeManifest, progressFn); err != nil {
		t.Fatalf("CreateSafetensorsModel failed: %v", err)
	}

	if len(statusMessages) == 0 {
		t.Fatal("no status messages received")
	}
	if got, want := statusMessages[0], "importing model.safetensors (5 tensors, preserving source quantization)"; got != want {
		t.Fatalf("status = %q, want %q", got, want)
	}

	if slices.Contains(tensorLayerNames, "linear.weight_scale") || slices.Contains(tensorLayerNames, "linear.weight_global_scale") || slices.Contains(tensorLayerNames, "linear.input_global_scale") {
		t.Fatalf("packed nvfp4 companions unexpectedly emitted as standalone tensor layers: %v", tensorLayerNames)
	}

	packedHeader := layerHeaders["linear.weight"]
	if packedHeader == nil {
		t.Fatalf("missing packed layer header for linear.weight")
	}
	for _, key := range []string{
		"linear.weight",
		"linear.weight.scale",
		"linear.weight.global_scale",
	} {
		if _, ok := packedHeader[key]; !ok {
			t.Fatalf("packed header missing %s: %v", key, packedHeader)
		}
	}
	if _, ok := packedHeader["linear.weight.input_global_scale"]; ok {
		t.Fatalf("packed header unexpectedly includes input_global_scale: %v", packedHeader)
	}
	globalRaw := readPackedTensorRaw(t, layerData["linear.weight"], "linear.weight.global_scale")
	if got := math.Float32frombits(binary.LittleEndian.Uint32(globalRaw)); got != 0.25 {
		t.Fatalf("linear.weight.global_scale = %v, want 0.25", got)
	}

	var metadata map[string]string
	if metaRaw, ok := packedHeader["__metadata__"]; ok {
		if err := json.Unmarshal(metaRaw, &metadata); err != nil {
			t.Fatalf("failed to parse metadata: %v", err)
		}
	}
	if metadata["quant_type"] != "nvfp4" {
		t.Fatalf("quant_type = %q, want %q", metadata["quant_type"], "nvfp4")
	}
	if metadata["group_size"] != "16" {
		t.Fatalf("group_size = %q, want %q", metadata["group_size"], "16")
	}
}

func TestCreateSafetensorsModel_PackedNVFP4CrossShardCompanions(t *testing.T) {
	dir := t.TempDir()

	configJSON := `{
		"model_type": "test",
		"architectures": ["TestModel"],
		"compression_config": {"format": "nvfp4-pack-quantized"}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	createTestSafetensors(t, filepath.Join(dir, "model-00001-of-00002.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("linear.weight_packed", "U8", []int32{16, 8}, make([]byte, 128)),
		st.NewTensorDataFromBytes("norm.weight", "BF16", []int32{16}, make([]byte, 32)),
	})
	createTestSafetensors(t, filepath.Join(dir, "model-00002-of-00002.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("linear.weight_scale", "F8_E4M3", []int32{16, 1}, make([]byte, 16)),
		st.NewTensorDataFromBytes("linear.weight_global_scale", "F32", []int32{}, encodeFloat32s(2)),
		st.NewTensorDataFromBytes("linear.input_global_scale", "F32", []int32{}, encodeFloat32s(8)),
	})
	indexJSON := `{
		"metadata": {"total_size": 152},
		"weight_map": {
			"linear.weight_packed": "model-00001-of-00002.safetensors",
			"norm.weight": "model-00001-of-00002.safetensors",
			"linear.weight_scale": "model-00002-of-00002.safetensors",
			"linear.weight_global_scale": "model-00002-of-00002.safetensors",
			"linear.input_global_scale": "model-00002-of-00002.safetensors"
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(indexJSON), 0o644); err != nil {
		t.Fatalf("failed to write index: %v", err)
	}

	layerHeaders := make(map[string]map[string]json.RawMessage)
	var tensorLayerNames []string

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		data, err := io.ReadAll(r)
		if err != nil {
			return LayerInfo{}, err
		}
		if mediaType == "application/vnd.ollama.image.tensor" {
			var headerSize uint64
			if err := binary.Read(bytes.NewReader(data[:8]), binary.LittleEndian, &headerSize); err != nil {
				return LayerInfo{}, err
			}
			var header map[string]json.RawMessage
			if err := json.Unmarshal(data[8:8+headerSize], &header); err != nil {
				return LayerInfo{}, err
			}
			layerHeaders[name] = header
		}
		return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
	}
	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		if _, err := io.ReadAll(r); err != nil {
			return nil, err
		}
		tensorLayerNames = append(tensorLayerNames, name)
		return []LayerInfo{{Name: name, Digest: "sha256:tensor_" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
	}
	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error { return nil }

	packedCreator := func(groupName string, tensors []PackedTensorInput) (LayerInfo, error) {
		return LayerInfo{}, fmt.Errorf("unexpected packedCreator call for %s", groupName)
	}
	if err := CreateSafetensorsModel("test-model", dir, "", createLayer, createTensorLayer, writeManifest, func(string) {}, packedCreator); err != nil {
		t.Fatalf("CreateSafetensorsModel failed: %v", err)
	}

	if slices.Contains(tensorLayerNames, "linear.weight_packed") || slices.Contains(tensorLayerNames, "linear.weight_scale") || slices.Contains(tensorLayerNames, "linear.weight_global_scale") || slices.Contains(tensorLayerNames, "linear.input_global_scale") {
		t.Fatalf("packed nvfp4 tensors unexpectedly emitted as standalone tensor layers: %v", tensorLayerNames)
	}

	packedHeader := layerHeaders["linear.weight"]
	if packedHeader == nil {
		t.Fatalf("missing packed layer header for linear.weight")
	}
	for _, key := range []string{
		"linear.weight",
		"linear.weight.scale",
		"linear.weight.global_scale",
	} {
		if _, ok := packedHeader[key]; !ok {
			t.Fatalf("packed header missing %s: %v", key, packedHeader)
		}
	}
	if _, ok := packedHeader["linear.weight.input_global_scale"]; ok {
		t.Fatalf("packed header unexpectedly includes input_global_scale: %v", packedHeader)
	}
}

func TestCreateSafetensorsModel_PackedNVFP4StacksExperts(t *testing.T) {
	dir := t.TempDir()

	configJSON := `{
		"model_type": "test",
		"architectures": ["TestModel"],
		"compression_config": {"format": "nvfp4-pack-quantized"}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.layers.1.mlp.experts.0.gate_proj.weight_packed", "U8", []int32{2, 8}, make([]byte, 16)),
		st.NewTensorDataFromBytes("model.layers.1.mlp.experts.0.gate_proj.weight_scale", "F8_E4M3", []int32{2, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("model.layers.1.mlp.experts.0.gate_proj.weight_global_scale", "F32", []int32{1}, encodeFloat32s(2)),
		st.NewTensorDataFromBytes("model.layers.1.mlp.experts.0.gate_proj.input_global_scale", "F32", []int32{1}, encodeFloat32s(32)),
		st.NewTensorDataFromBytes("model.layers.1.mlp.experts.1.gate_proj.weight_packed", "U8", []int32{2, 8}, make([]byte, 16)),
		st.NewTensorDataFromBytes("model.layers.1.mlp.experts.1.gate_proj.weight_scale", "F8_E4M3", []int32{2, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("model.layers.1.mlp.experts.1.gate_proj.weight_global_scale", "F32", []int32{1}, encodeFloat32s(4)),
		st.NewTensorDataFromBytes("model.layers.1.mlp.experts.1.gate_proj.input_global_scale", "F32", []int32{1}, encodeFloat32s(64)),
		st.NewTensorDataFromBytes("norm.weight", "BF16", []int32{2}, make([]byte, 4)),
	})

	layerHeaders := make(map[string]map[string]json.RawMessage)
	layerData := make(map[string][]byte)
	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		data, err := io.ReadAll(r)
		if err != nil {
			return LayerInfo{}, err
		}
		if mediaType == "application/vnd.ollama.image.tensor" {
			var headerSize uint64
			if err := binary.Read(bytes.NewReader(data[:8]), binary.LittleEndian, &headerSize); err != nil {
				return LayerInfo{}, err
			}
			var header map[string]json.RawMessage
			if err := json.Unmarshal(data[8:8+headerSize], &header); err != nil {
				return LayerInfo{}, err
			}
			layerHeaders[name] = header
			layerData[name] = data
		}
		return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
	}
	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		if _, err := io.ReadAll(r); err != nil {
			return nil, err
		}
		return []LayerInfo{{Name: name, Digest: "sha256:tensor_" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
	}
	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error { return nil }
	packedCreator := func(groupName string, tensors []PackedTensorInput) (LayerInfo, error) {
		return LayerInfo{}, fmt.Errorf("unexpected packedCreator call for %s", groupName)
	}

	if err := CreateSafetensorsModel("test-model", dir, "", createLayer, createTensorLayer, writeManifest, func(string) {}, packedCreator); err != nil {
		t.Fatalf("CreateSafetensorsModel failed: %v", err)
	}

	header := layerHeaders["model.layers.1.mlp.experts"]
	if header == nil {
		t.Fatalf("missing packed expert layer header")
	}
	for _, key := range []string{
		"model.layers.1.mlp.switch_mlp.gate_proj.weight",
		"model.layers.1.mlp.switch_mlp.gate_proj.weight.scale",
		"model.layers.1.mlp.switch_mlp.gate_proj.weight.global_scale",
	} {
		if _, ok := header[key]; !ok {
			t.Fatalf("stacked header missing %s: %v", key, header)
		}
	}
	if _, ok := header["model.layers.1.mlp.switch_mlp.gate_proj.weight.input_global_scale"]; ok {
		t.Fatalf("stacked header unexpectedly includes input_global_scale: %v", header)
	}
	if _, ok := header["model.layers.1.mlp.experts.0.gate_proj.weight"]; ok {
		t.Fatalf("unexpected per-expert tensor left in packed header: %v", header)
	}

	var weightInfo struct {
		Dtype string  `json:"dtype"`
		Shape []int32 `json:"shape"`
	}
	if err := json.Unmarshal(header["model.layers.1.mlp.switch_mlp.gate_proj.weight"], &weightInfo); err != nil {
		t.Fatalf("failed to unmarshal stacked weight info: %v", err)
	}
	if weightInfo.Dtype != "U32" || !slices.Equal(weightInfo.Shape, []int32{2, 2, 2}) {
		t.Fatalf("stacked weight = dtype %s shape %v, want U32 [2 2 2]", weightInfo.Dtype, weightInfo.Shape)
	}

	var globalInfo struct {
		Dtype string  `json:"dtype"`
		Shape []int32 `json:"shape"`
	}
	if err := json.Unmarshal(header["model.layers.1.mlp.switch_mlp.gate_proj.weight.global_scale"], &globalInfo); err != nil {
		t.Fatalf("failed to unmarshal stacked global scale info: %v", err)
	}
	if globalInfo.Dtype != "F32" || !slices.Equal(globalInfo.Shape, []int32{2, 1, 1}) {
		t.Fatalf("stacked global scale = dtype %s shape %v, want F32 [2 1 1]", globalInfo.Dtype, globalInfo.Shape)
	}
	globalRaw := readPackedTensorRaw(t, layerData["model.layers.1.mlp.experts"], "model.layers.1.mlp.switch_mlp.gate_proj.weight.global_scale")
	if got0 := math.Float32frombits(binary.LittleEndian.Uint32(globalRaw[0:4])); got0 != 0.5 {
		t.Fatalf("stacked global scale[0] = %v, want 0.5", got0)
	}
	if got1 := math.Float32frombits(binary.LittleEndian.Uint32(globalRaw[4:8])); got1 != 0.25 {
		t.Fatalf("stacked global scale[1] = %v, want 0.25", got1)
	}
}

func TestCreateSafetensorsModel_HFFP8PacksExperts(t *testing.T) {
	dir := t.TempDir()

	configJSON := `{
		"model_type": "test",
		"architectures": ["Qwen3_5MoeForConditionalGeneration"],
		"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	// Create 2 experts so stacking produces a [2, 128, 128] tensor
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.0.gate_proj.weight", "F8_E4M3", []int32{128, 128}, make([]byte, 128*128)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv", "BF16", []int32{1, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.0.up_proj.weight", "F8_E4M3", []int32{128, 128}, make([]byte, 128*128)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.0.up_proj.weight_scale_inv", "BF16", []int32{1, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.0.down_proj.weight", "F8_E4M3", []int32{128, 128}, make([]byte, 128*128)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.0.down_proj.weight_scale_inv", "BF16", []int32{1, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.1.gate_proj.weight", "F8_E4M3", []int32{128, 128}, make([]byte, 128*128)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv", "BF16", []int32{1, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.1.up_proj.weight", "F8_E4M3", []int32{128, 128}, make([]byte, 128*128)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.1.up_proj.weight_scale_inv", "BF16", []int32{1, 1}, make([]byte, 2)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.1.down_proj.weight", "F8_E4M3", []int32{128, 128}, make([]byte, 128*128)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.1.down_proj.weight_scale_inv", "BF16", []int32{1, 1}, make([]byte, 2)),
	})

	var packedLayerNames []string
	var packedLayerTensors [][]PackedTensorInput

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
		return []LayerInfo{{Name: name, Digest: "sha256:tensor_" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
	}

	createPackedLayer := func(groupName string, tensors []PackedTensorInput) (LayerInfo, error) {
		packedLayerNames = append(packedLayerNames, groupName)
		packedLayerTensors = append(packedLayerTensors, tensors)
		return LayerInfo{Name: groupName, Digest: "sha256:packed_" + groupName, MediaType: "application/vnd.ollama.image.tensor"}, nil
	}

	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error { return nil }

	if err := CreateSafetensorsModel("test-model", dir, "", createLayer, createTensorLayer, writeManifest, func(string) {}, createPackedLayer); err != nil {
		t.Fatalf("CreateSafetensorsModel failed: %v", err)
	}

	if len(packedLayerNames) != 1 {
		t.Fatalf("expected 1 packed layer, got %d: %v", len(packedLayerNames), packedLayerNames)
	}
	if packedLayerNames[0] != "language_model.model.layers.0.mlp.experts" {
		t.Fatalf("unexpected packed layer name: %s", packedLayerNames[0])
	}

	// Verify all 6 expert tensors (2 experts × 3 proj types) were accumulated
	tensors := packedLayerTensors[0]
	if len(tensors) != 6 {
		t.Fatalf("expected 6 tensors in packed group, got %d", len(tensors))
	}

	// All should be marked for mxfp8 quantization
	for _, tensor := range tensors {
		if tensor.Quantize != "mxfp8" {
			t.Fatalf("expected mxfp8 quantize for %s, got %q", tensor.Name, tensor.Quantize)
		}
	}

	packedLayerNames = nil
	packedLayerTensors = nil
	if err := CreateSafetensorsModel("test-model", dir, "nvfp4", createLayer, createTensorLayer, writeManifest, func(string) {}, createPackedLayer); err != nil {
		t.Fatalf("CreateSafetensorsModel nvfp4 failed: %v", err)
	}

	if len(packedLayerNames) != 1 {
		t.Fatalf("expected 1 packed layer for nvfp4, got %d: %v", len(packedLayerNames), packedLayerNames)
	}

	for _, tensor := range packedLayerTensors[0] {
		want := "nvfp4"
		if strings.Contains(tensor.Name, "down_proj") {
			want = "mxfp8"
		}
		if tensor.Quantize != want {
			t.Fatalf("nvfp4 packed tensor %s quantize = %q, want %q", tensor.Name, tensor.Quantize, want)
		}
	}
}

func TestCreateSafetensorsModel_Qwen35Transforms(t *testing.T) {
	dir := t.TempDir()

	configJSON := `{
		"model_type": "test",
		"architectures": ["Qwen3_5MoeForConditionalGeneration"],
		"text_config": {"dtype": "bfloat16"}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	gateUpValues := make([]float32, 2*128*64)
	for expert := range 2 {
		base := expert * 128 * 64
		for i := range 64 * 64 {
			gateUpValues[base+i] = 1
			gateUpValues[base+64*64+i] = 2
		}
	}

	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.language_model.embed_tokens.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.input_layernorm.weight", "F32", []int32{64}, make([]byte, 64*4)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.linear_attn.A_log", "F32", []int32{32}, make([]byte, 32*4)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.linear_attn.conv1d.weight", "BF16", []int32{64, 1, 4}, make([]byte, 64*1*4*2)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.gate.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.gate_up_proj", "BF16", []int32{2, 128, 64}, bfloat16.EncodeFloat32(gateUpValues)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.down_proj", "BF16", []int32{2, 64, 64}, bfloat16.EncodeFloat32(make([]float32, 2*64*64))),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.shared_expert.down_proj.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
		st.NewTensorDataFromBytes("model.visual.blocks.0.attn.proj.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
		st.NewTensorDataFromBytes("mtp.layers.0.foo.weight", "F32", []int32{64, 64}, make([]byte, 64*64*4)),
	})

	type tensorCall struct {
		dtype    string
		shape    []int32
		quantize string
		raw      []byte
	}
	calls := make(map[string]tensorCall)

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		_, _ = io.ReadAll(r)
		return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
	}

	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		data, err := io.ReadAll(r)
		if err != nil {
			return nil, err
		}
		headerDType, headerShape := readSingleTensorHeader(t, data)
		calls[name] = tensorCall{
			dtype:    headerDType,
			shape:    headerShape,
			quantize: quantize,
			raw:      readSingleTensorRaw(t, data),
		}
		return []LayerInfo{{Name: name, Digest: "sha256:" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
	}

	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error {
		return nil
	}

	if err := CreateSafetensorsModel("test-model", dir, "int4", createLayer, createTensorLayer, writeManifest, func(string) {}); err != nil {
		t.Fatalf("CreateSafetensorsModel failed: %v", err)
	}

	if _, ok := calls["mtp.layers.0.foo.weight"]; ok {
		t.Fatal("mtp tensor should have been dropped")
	}

	layerNorm := calls["language_model.model.layers.0.input_layernorm.weight"]
	if layerNorm.dtype != "BF16" {
		t.Fatalf("input_layernorm dtype = %q, want %q", layerNorm.dtype, "BF16")
	}
	if layerNorm.quantize != "" {
		t.Fatalf("input_layernorm quantize = %q, want empty", layerNorm.quantize)
	}
	layerNormValues := bfloat16.DecodeFloat32(layerNorm.raw)
	if len(layerNormValues) == 0 || layerNormValues[0] != 1.0 {
		t.Fatalf("input_layernorm first value = %v, want 1.0 after +1 shift", layerNormValues[0])
	}

	alog := calls["language_model.model.layers.0.linear_attn.A_log"]
	if alog.dtype != "F32" {
		t.Fatalf("A_log dtype = %q, want %q", alog.dtype, "F32")
	}

	conv := calls["language_model.model.layers.0.linear_attn.conv1d.weight"]
	if !slices.Equal(conv.shape, []int32{64, 4, 1}) {
		t.Fatalf("conv1d shape = %v, want %v", conv.shape, []int32{64, 4, 1})
	}

	if got := calls["language_model.model.embed_tokens.weight"].quantize; got != "int4" {
		t.Fatalf("embed_tokens quantize = %q, want %q", got, "int4")
	}
	if got := calls["language_model.model.layers.0.mlp.gate.weight"].quantize; got != "int4" {
		t.Fatalf("mlp.gate quantize = %q, want %q", got, "int4")
	}
	if got := calls["language_model.model.layers.0.mlp.shared_expert.down_proj.weight"].quantize; got != "int4" {
		t.Fatalf("down_proj quantize = %q, want %q", got, "int4")
	}

	if _, ok := calls["language_model.model.layers.0.mlp.experts.gate_up_proj"]; ok {
		t.Fatal("combined gate_up_proj tensor should have been rewritten")
	}
	if _, ok := calls["language_model.model.layers.0.mlp.experts.down_proj"]; ok {
		t.Fatal("combined down_proj tensor should have been rewritten")
	}

	gateProj := calls["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]
	if !slices.Equal(gateProj.shape, []int32{2, 64, 64}) {
		t.Fatalf("gate_proj shape = %v, want %v", gateProj.shape, []int32{2, 64, 64})
	}
	gateProjValues := bfloat16.DecodeFloat32(gateProj.raw)
	if len(gateProjValues) == 0 || gateProjValues[0] != 1.0 {
		t.Fatalf("gate_proj first value = %v, want 1.0", gateProjValues[0])
	}

	upProj := calls["language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"]
	if !slices.Equal(upProj.shape, []int32{2, 64, 64}) {
		t.Fatalf("up_proj shape = %v, want %v", upProj.shape, []int32{2, 64, 64})
	}
	upProjValues := bfloat16.DecodeFloat32(upProj.raw)
	if len(upProjValues) == 0 || upProjValues[0] != 2.0 {
		t.Fatalf("up_proj first value = %v, want 2.0", upProjValues[0])
	}

	if got := calls["language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"].quantize; got != "int4" {
		t.Fatalf("switch_mlp down_proj quantize = %q, want %q", got, "int4")
	}

	vision := calls["vision_tower.blocks.0.attn.proj.weight"]
	if vision.dtype != "BF16" {
		t.Fatalf("vision weight dtype = %q, want %q", vision.dtype, "BF16")
	}
	if vision.quantize != "" {
		t.Fatalf("vision weight quantize = %q, want empty", vision.quantize)
	}
	if _, ok := calls["language_model.model.visual.blocks.0.attn.proj.weight"]; ok {
		t.Fatal("vision tensor should have been rewritten to vision_tower.*")
	}
}

func TestCreateSafetensorsModel_Qwen35DirectNonAffineKeepsSensitiveWeightsBF16(t *testing.T) {
	for _, quantize := range []string{"nvfp4", "mxfp8", "mxfp4"} {
		t.Run(quantize, func(t *testing.T) {
			dir := t.TempDir()

			configJSON := `{
		"model_type": "test",
		"architectures": ["Qwen3_5MoeForConditionalGeneration"],
		"text_config": {"dtype": "bfloat16"}
	}`
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
				t.Fatalf("failed to write config.json: %v", err)
			}

			gateUpValues := make([]float32, 2*128*64)
			for expert := range 2 {
				base := expert * 128 * 64
				for i := range 64 * 64 {
					gateUpValues[base+i] = 1
					gateUpValues[base+64*64+i] = 2
				}
			}

			createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
				st.NewTensorDataFromBytes("model.language_model.embed_tokens.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
				st.NewTensorDataFromBytes("lm_head.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
				st.NewTensorDataFromBytes("model.language_model.layers.0.linear_attn.in_proj_a.weight", "BF16", []int32{32, 64}, make([]byte, 32*64*2)),
				st.NewTensorDataFromBytes("model.language_model.layers.0.linear_attn.in_proj_b.weight", "BF16", []int32{32, 64}, make([]byte, 32*64*2)),
				st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.gate.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
				st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.shared_expert_gate.weight", "BF16", []int32{1, 64}, make([]byte, 64*2)),
				st.NewTensorDataFromBytes("model.language_model.layers.0.self_attn.q_proj.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
				st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.gate_up_proj", "BF16", []int32{2, 128, 64}, bfloat16.EncodeFloat32(gateUpValues)),
				st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.down_proj", "BF16", []int32{2, 64, 64}, bfloat16.EncodeFloat32(make([]float32, 2*64*64))),
			})

			type tensorCall struct {
				quantize string
			}
			type packedTensorCall struct {
				Name     string
				Quantize string
			}

			tensorCalls := make(map[string]tensorCall)
			packedCalls := make(map[string][]packedTensorCall)

			createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
				_, _ = io.ReadAll(r)
				return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
			}

			createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantizeType string) ([]LayerInfo, error) {
				_, _ = io.ReadAll(r)
				tensorCalls[name] = tensorCall{quantize: quantizeType}
				return []LayerInfo{{Name: name, Digest: "sha256:" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
			}

			createPackedLayer := func(groupName string, tensors []PackedTensorInput) (LayerInfo, error) {
				group := make([]packedTensorCall, 0, len(tensors))
				for _, tensor := range tensors {
					group = append(group, packedTensorCall{
						Name:     tensor.Name,
						Quantize: tensor.Quantize,
					})
				}
				packedCalls[groupName] = group
				return LayerInfo{Name: groupName, Digest: "sha256:" + groupName, MediaType: "application/vnd.ollama.image.tensor"}, nil
			}

			writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error {
				return nil
			}

			if err := CreateSafetensorsModel("test-model", dir, quantize, createLayer, createTensorLayer, writeManifest, func(string) {}, createPackedLayer); err != nil {
				t.Fatalf("CreateSafetensorsModel failed: %v", err)
			}

			for _, name := range []string{
				"language_model.model.embed_tokens.weight",
				"language_model.lm_head.weight",
				"language_model.model.layers.0.linear_attn.in_proj_a.weight",
				"language_model.model.layers.0.linear_attn.in_proj_b.weight",
				"language_model.model.layers.0.mlp.gate.weight",
				"language_model.model.layers.0.mlp.shared_expert_gate.weight",
			} {
				if got := tensorCalls[name].quantize; got != "" {
					t.Fatalf("%s quantize = %q, want empty", name, got)
				}
			}

			if got := tensorCalls["language_model.model.layers.0.self_attn.q_proj.weight"].quantize; got != quantize {
				t.Fatalf("q_proj quantize = %q, want %q", got, quantize)
			}

			group := packedCalls["language_model.model.layers.0.mlp.switch_mlp"]
			if len(group) != 3 {
				t.Fatalf("packed switch_mlp tensor count = %d, want 3", len(group))
			}
			for _, tensor := range group {
				if tensor.Quantize != quantize {
					t.Fatalf("packed tensor %q quantize = %q, want %q", tensor.Name, tensor.Quantize, quantize)
				}
			}
		})
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

func TestShouldQuantizeTensor(t *testing.T) {
	tests := []struct {
		name     string
		tensor   string
		shape    []int32
		quantize string
		want     bool
	}{
		// 2D tensors with sufficient size should be quantized
		{"large 2D weight fp8", "q_proj.weight", []int32{4096, 4096}, "fp8", true},
		{"medium 2D weight fp8", "small_proj.weight", []int32{128, 128}, "fp8", true},
		{"large 2D weight nvfp4", "q_proj.weight", []int32{4096, 4096}, "nvfp4", true},
		{"large 2D weight mxfp4", "q_proj.weight", []int32{4096, 4096}, "mxfp4", true},

		// Small tensors should not be quantized (< 1024 elements)
		{"tiny 2D weight", "tiny.weight", []int32{16, 16}, "fp8", false},
		{"small 2D weight", "small.weight", []int32{31, 31}, "fp8", false},

		// 1D tensors should not be quantized
		{"1D tensor", "layer_norm.weight", []int32{4096}, "fp8", false},

		// 3D+ tensors should not be quantized
		{"3D tensor", "conv.weight", []int32{64, 64, 3}, "fp8", false},
		{"4D tensor", "conv2d.weight", []int32{64, 64, 3, 3}, "fp8", false},
		{"stacked expert switch_mlp gate_up 3D int8", "model.layers.1.mlp.switch_mlp.gate_up_proj.weight", []int32{64, 22016, 4096}, "int8", true},
		{"stacked expert experts down_proj 3D int8", "model.layers.1.mlp.experts.down_proj.weight", []int32{64, 4096, 14336}, "int8", true},
		{"stacked expert combined gate_up 3D int8", "model.language_model.layers.0.mlp.experts.gate_up_proj", []int32{256, 1024, 2048}, "int8", true},
		{"stacked expert combined down_proj 3D int8", "model.language_model.layers.0.mlp.experts.down_proj", []int32{256, 2048, 512}, "int8", true},

		// Embeddings should not be quantized regardless of shape
		{"embedding 2D", "embed_tokens.weight", []int32{32000, 4096}, "fp8", false},

		// Norms should not be quantized regardless of shape
		{"norm 2D", "layer_norm.weight", []int32{4096, 1}, "fp8", false},

		// Biases should not be quantized
		{"bias 2D", "proj.bias", []int32{4096, 1}, "fp8", false},

		// Group size divisibility tests
		// FP8/FP4/MXFP4 require divisible by 32
		{"not divisible by 32 fp8", "proj.weight", []int32{128, 48}, "fp8", false},
		{"divisible by 32 fp8", "proj.weight", []int32{128, 64}, "fp8", true},
		{"not divisible by 32 mxfp4", "proj.weight", []int32{128, 48}, "mxfp4", false},
		{"divisible by 32 mxfp4", "proj.weight", []int32{128, 64}, "mxfp4", true},
		// NVFP4 requires divisible by 16
		{"not divisible by 16 nvfp4", "proj.weight", []int32{128, 24}, "nvfp4", false},
		{"divisible by 16 nvfp4", "proj.weight", []int32{128, 48}, "nvfp4", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ShouldQuantizeTensor(tt.tensor, tt.shape, tt.quantize)
			if got != tt.want {
				t.Errorf("ShouldQuantizeTensor(%q, %v, %q) = %v, want %v", tt.tensor, tt.shape, tt.quantize, got, tt.want)
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

		// nvfp4/mxfp4/mxfp8: no promotion (uniform quantization)
		{"v_proj nvfp4 uniform", "model.layers.0.self_attn.v_proj.weight", aligned, "nvfp4", "nvfp4"},
		{"down_proj mxfp4 uniform", "model.layers.0.mlp.down_proj.weight", aligned, "mxfp4", "mxfp4"},
		{"v_proj mxfp8 uniform", "model.layers.0.self_attn.v_proj.weight", aligned, "mxfp8", "mxfp8"},

		// int8: already 8-bit, no promotion
		{"v_proj int8 stays", "model.layers.0.self_attn.v_proj.weight", aligned, "int8", "int8"},

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

func TestCreateSafetensorsModel_Qwen35NVFP4PacksSwitchMLPExperts(t *testing.T) {
	dir := t.TempDir()

	configJSON := `{
		"model_type": "test",
		"architectures": ["Qwen3_5MoeForConditionalGeneration"],
		"text_config": {"dtype": "bfloat16"}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatalf("failed to write config.json: %v", err)
	}

	gateUpValues := make([]float32, 2*128*64)
	for expert := range 2 {
		base := expert * 128 * 64
		for i := range 64 * 64 {
			gateUpValues[base+i] = 1
			gateUpValues[base+64*64+i] = 2
		}
	}

	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.language_model.embed_tokens.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.gate.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.gate_up_proj", "BF16", []int32{2, 128, 64}, bfloat16.EncodeFloat32(gateUpValues)),
		st.NewTensorDataFromBytes("model.language_model.layers.0.mlp.experts.down_proj", "BF16", []int32{2, 64, 64}, bfloat16.EncodeFloat32(make([]float32, 2*64*64))),
	})

	type tensorCall struct {
		quantize string
	}
	type packedTensorCall struct {
		Name     string
		Dtype    string
		Shape    []int32
		Quantize string
	}

	tensorCalls := make(map[string]tensorCall)
	packedCalls := make(map[string][]packedTensorCall)

	createLayer := func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		_, _ = io.ReadAll(r)
		return LayerInfo{Name: name, Digest: "sha256:" + name, MediaType: mediaType}, nil
	}

	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error) {
		_, _ = io.ReadAll(r)
		tensorCalls[name] = tensorCall{quantize: quantize}
		return []LayerInfo{{Name: name, Digest: "sha256:" + name, MediaType: "application/vnd.ollama.image.tensor"}}, nil
	}

	createPackedLayer := func(groupName string, tensors []PackedTensorInput) (LayerInfo, error) {
		group := make([]packedTensorCall, 0, len(tensors))
		for _, tensor := range tensors {
			group = append(group, packedTensorCall{
				Name:     tensor.Name,
				Dtype:    tensor.Dtype,
				Shape:    append([]int32(nil), tensor.Shape...),
				Quantize: tensor.Quantize,
			})
		}
		packedCalls[groupName] = group
		return LayerInfo{Name: groupName, Digest: "sha256:" + groupName, MediaType: "application/vnd.ollama.image.tensor"}, nil
	}

	writeManifest := func(modelName string, config LayerInfo, layers []LayerInfo) error {
		return nil
	}

	if err := CreateSafetensorsModel("test-model", dir, "nvfp4", createLayer, createTensorLayer, writeManifest, func(string) {}, createPackedLayer); err != nil {
		t.Fatalf("CreateSafetensorsModel failed: %v", err)
	}

	groupName := "language_model.model.layers.0.mlp.switch_mlp"
	group, ok := packedCalls[groupName]
	if !ok {
		t.Fatalf("missing packed group %q: %v", groupName, packedCalls)
	}

	if len(group) != 3 {
		t.Fatalf("packed group %q has %d tensors, want 3", groupName, len(group))
	}

	gotNames := make([]string, 0, len(group))
	for _, tensor := range group {
		gotNames = append(gotNames, tensor.Name)
		if tensor.Quantize != "nvfp4" {
			t.Fatalf("packed tensor %q quantize = %q, want %q", tensor.Name, tensor.Quantize, "nvfp4")
		}
		if tensor.Dtype != "BF16" {
			t.Fatalf("packed tensor %q dtype = %q, want %q", tensor.Name, tensor.Dtype, "BF16")
		}
	}
	slices.Sort(gotNames)

	wantNames := []string{
		"language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
		"language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
		"language_model.model.layers.0.mlp.switch_mlp.up_proj.weight",
	}
	if !slices.Equal(gotNames, wantNames) {
		t.Fatalf("packed tensor names = %v, want %v", gotNames, wantNames)
	}

	for _, name := range wantNames {
		if _, ok := tensorCalls[name]; ok {
			t.Fatalf("packed expert tensor %q unexpectedly handled by createTensorLayer", name)
		}
	}

	if got := tensorCalls["language_model.model.embed_tokens.weight"].quantize; got != "" {
		t.Fatalf("embed_tokens quantize = %q, want empty", got)
	}
	if got := tensorCalls["language_model.model.layers.0.mlp.gate.weight"].quantize; got != "" {
		t.Fatalf("mlp.gate quantize = %q, want empty", got)
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

	err := CreateImageGenModel("test-imagegen", dir, "int8", createLayer, createTensorLayer, writeManifest, progressFn)
	if err != nil {
		t.Fatalf("CreateImageGenModel failed: %v", err)
	}

	if len(quantizeRequested) == 0 {
		t.Error("no tensors processed")
	}
}
