package server

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/x/imagegen"
)

func TestBuildModelInfo(t *testing.T) {
	tests := []struct {
		name             string
		config           modelConfig
		totalTensorBytes int64
		tensorCount      int64
		wantArch         string
		wantContextLen   int
		wantEmbedLen     int
		wantBlockCount   int
		wantParamCount   int64
	}{
		{
			name: "gemma3 model with model_type",
			config: modelConfig{
				ModelType:             "gemma3",
				HiddenSize:            2560,
				NumHiddenLayers:       34,
				MaxPositionEmbeddings: 131072,
				IntermediateSize:      10240,
				NumAttentionHeads:     8,
				NumKeyValueHeads:      4,
				VocabSize:             262144,
				TorchDtype:            "bfloat16",
			},
			totalTensorBytes: 8_600_000_088, // ~4.3B params * 2 bytes + 88 bytes header
			tensorCount:      1,
			wantArch:         "gemma3",
			wantContextLen:   131072,
			wantEmbedLen:     2560,
			wantBlockCount:   34,
			wantParamCount:   4_300_000_000,
		},
		{
			name: "llama model with architectures array",
			config: modelConfig{
				Architectures:         []string{"LlamaForCausalLM"},
				HiddenSize:            4096,
				NumHiddenLayers:       32,
				MaxPositionEmbeddings: 4096,
				IntermediateSize:      11008,
				NumAttentionHeads:     32,
				NumKeyValueHeads:      32,
				VocabSize:             32000,
				TorchDtype:            "float16",
			},
			totalTensorBytes: 14_000_000_088, // ~7B params * 2 bytes + 88 bytes header
			tensorCount:      1,
			wantArch:         "llama",
			wantContextLen:   4096,
			wantEmbedLen:     4096,
			wantBlockCount:   32,
			wantParamCount:   7_000_000_000,
		},
		{
			name: "multimodal model with text_config",
			config: modelConfig{
				Architectures: []string{"Gemma3ForConditionalGeneration"},
				HiddenSize:    1152, // vision hidden size
				TextConfig: &struct {
					HiddenSize            int `json:"hidden_size"`
					MaxPositionEmbeddings int `json:"max_position_embeddings"`
					NumHiddenLayers       int `json:"num_hidden_layers"`
				}{
					HiddenSize:            2560,
					MaxPositionEmbeddings: 131072,
					NumHiddenLayers:       34,
				},
				NumAttentionHeads: 8,
				NumKeyValueHeads:  4,
				VocabSize:         262144,
				TorchDtype:        "bfloat16",
			},
			totalTensorBytes: 8_600_000_088,
			tensorCount:      1,
			wantArch:         "gemma3",
			wantContextLen:   131072,
			wantEmbedLen:     2560,
			wantBlockCount:   34,
			wantParamCount:   4_300_000_000,
		},
		{
			name: "float32 model",
			config: modelConfig{
				ModelType:             "test",
				HiddenSize:            512,
				NumHiddenLayers:       6,
				MaxPositionEmbeddings: 2048,
				TorchDtype:            "float32",
			},
			totalTensorBytes: 400_000_088, // 100M params * 4 bytes + 88 bytes header
			tensorCount:      1,
			wantArch:         "test",
			wantContextLen:   2048,
			wantEmbedLen:     512,
			wantBlockCount:   6,
			wantParamCount:   100_000_000,
		},
		{
			name: "multiple tensors with header overhead",
			config: modelConfig{
				ModelType:             "test",
				HiddenSize:            256,
				NumHiddenLayers:       4,
				MaxPositionEmbeddings: 1024,
				TorchDtype:            "bfloat16",
			},
			totalTensorBytes: 2_000_880, // 1M params * 2 bytes + 10 tensors * 88 bytes
			tensorCount:      10,
			wantArch:         "test",
			wantContextLen:   1024,
			wantEmbedLen:     256,
			wantBlockCount:   4,
			wantParamCount:   1_000_000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := buildModelInfo(tt.config, tt.totalTensorBytes, tt.tensorCount)

			// Check architecture
			if arch, ok := info["general.architecture"].(string); !ok || arch != tt.wantArch {
				t.Errorf("architecture = %v, want %v", info["general.architecture"], tt.wantArch)
			}

			// Check context length
			contextKey := tt.wantArch + ".context_length"
			if contextLen, ok := info[contextKey].(int); !ok || contextLen != tt.wantContextLen {
				t.Errorf("context_length = %v, want %v", info[contextKey], tt.wantContextLen)
			}

			// Check embedding length
			embedKey := tt.wantArch + ".embedding_length"
			if embedLen, ok := info[embedKey].(int); !ok || embedLen != tt.wantEmbedLen {
				t.Errorf("embedding_length = %v, want %v", info[embedKey], tt.wantEmbedLen)
			}

			// Check block count
			blockKey := tt.wantArch + ".block_count"
			if blockCount, ok := info[blockKey].(int); !ok || blockCount != tt.wantBlockCount {
				t.Errorf("block_count = %v, want %v", info[blockKey], tt.wantBlockCount)
			}

			// Check parameter count
			if paramCount, ok := info["general.parameter_count"].(int64); !ok || paramCount != tt.wantParamCount {
				t.Errorf("parameter_count = %v, want %v", info["general.parameter_count"], tt.wantParamCount)
			}
		})
	}
}

func TestBuildModelInfo_ArchitectureConversion(t *testing.T) {
	tests := []struct {
		name          string
		architectures []string
		modelType     string
		wantArch      string
	}{
		{
			name:          "LlamaForCausalLM",
			architectures: []string{"LlamaForCausalLM"},
			wantArch:      "llama",
		},
		{
			name:          "Gemma3ForCausalLM",
			architectures: []string{"Gemma3ForCausalLM"},
			wantArch:      "gemma3",
		},
		{
			name:          "Gemma3ForConditionalGeneration",
			architectures: []string{"Gemma3ForConditionalGeneration"},
			wantArch:      "gemma3",
		},
		{
			name:          "Qwen2ForCausalLM",
			architectures: []string{"Qwen2ForCausalLM"},
			wantArch:      "qwen2",
		},
		{
			name:          "model_type takes precedence",
			architectures: []string{"LlamaForCausalLM"},
			modelType:     "custom",
			wantArch:      "custom",
		},
		{
			name:          "empty architectures with model_type",
			architectures: nil,
			modelType:     "mymodel",
			wantArch:      "mymodel",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := modelConfig{
				Architectures: tt.architectures,
				ModelType:     tt.modelType,
			}
			info := buildModelInfo(config, 0, 0)

			if arch, ok := info["general.architecture"].(string); !ok || arch != tt.wantArch {
				t.Errorf("architecture = %v, want %v", info["general.architecture"], tt.wantArch)
			}
		})
	}
}

func TestBuildModelInfo_BytesPerParam(t *testing.T) {
	tests := []struct {
		name           string
		dtype          string
		totalBytes     int64
		tensorCount    int64
		wantParamCount int64
	}{
		{
			name:           "bfloat16",
			dtype:          "bfloat16",
			totalBytes:     2_000_088, // 1M * 2 + 88
			tensorCount:    1,
			wantParamCount: 1_000_000,
		},
		{
			name:           "float16",
			dtype:          "float16",
			totalBytes:     2_000_088,
			tensorCount:    1,
			wantParamCount: 1_000_000,
		},
		{
			name:           "float32",
			dtype:          "float32",
			totalBytes:     4_000_088, // 1M * 4 + 88
			tensorCount:    1,
			wantParamCount: 1_000_000,
		},
		{
			name:           "int8",
			dtype:          "int8",
			totalBytes:     1_000_088, // 1M * 1 + 88
			tensorCount:    1,
			wantParamCount: 1_000_000,
		},
		{
			name:           "unknown dtype defaults to 2 bytes",
			dtype:          "unknown",
			totalBytes:     2_000_088,
			tensorCount:    1,
			wantParamCount: 1_000_000,
		},
		{
			name:           "empty dtype defaults to 2 bytes",
			dtype:          "",
			totalBytes:     2_000_088,
			tensorCount:    1,
			wantParamCount: 1_000_000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := modelConfig{
				ModelType:  "test",
				TorchDtype: tt.dtype,
			}
			info := buildModelInfo(config, tt.totalBytes, tt.tensorCount)

			if paramCount, ok := info["general.parameter_count"].(int64); !ok || paramCount != tt.wantParamCount {
				t.Errorf("parameter_count = %v, want %v", info["general.parameter_count"], tt.wantParamCount)
			}
		})
	}
}

func TestParseSafetensorsHeader(t *testing.T) {
	tests := []struct {
		name      string
		header    map[string]any
		wantDtype string
		wantShape []int64
		wantErr   bool
	}{
		{
			name: "simple tensor",
			header: map[string]any{
				"weight": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{2560, 262144},
					"data_offsets": []int64{0, 1342177280},
				},
			},
			wantDtype: "BF16",
			wantShape: []int64{2560, 262144},
		},
		{
			name: "with metadata",
			header: map[string]any{
				"__metadata__": map[string]any{
					"format": "pt",
				},
				"bias": map[string]any{
					"dtype":        "F32",
					"shape":        []int64{1024},
					"data_offsets": []int64{0, 4096},
				},
			},
			wantDtype: "F32",
			wantShape: []int64{1024},
		},
		{
			name: "float16 tensor",
			header: map[string]any{
				"layer.weight": map[string]any{
					"dtype":        "F16",
					"shape":        []int64{512, 512, 3, 3},
					"data_offsets": []int64{0, 4718592},
				},
			},
			wantDtype: "F16",
			wantShape: []int64{512, 512, 3, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create safetensors format: 8-byte size + JSON header
			headerJSON, err := json.Marshal(tt.header)
			if err != nil {
				t.Fatalf("failed to marshal header: %v", err)
			}

			var buf bytes.Buffer
			if err := binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON))); err != nil {
				t.Fatalf("failed to write header size: %v", err)
			}
			buf.Write(headerJSON)

			info, err := parseSafetensorsHeader(&buf)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseSafetensorsHeader() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			if info.Dtype != tt.wantDtype {
				t.Errorf("Dtype = %v, want %v", info.Dtype, tt.wantDtype)
			}

			if len(info.Shape) != len(tt.wantShape) {
				t.Errorf("Shape length = %v, want %v", len(info.Shape), len(tt.wantShape))
			} else {
				for i, s := range info.Shape {
					if s != tt.wantShape[i] {
						t.Errorf("Shape[%d] = %v, want %v", i, s, tt.wantShape[i])
					}
				}
			}
		})
	}
}

func TestParseSafetensorsHeader_Errors(t *testing.T) {
	tests := []struct {
		name    string
		data    []byte
		wantErr string
	}{
		{
			name:    "empty data",
			data:    []byte{},
			wantErr: "failed to read header size",
		},
		{
			name:    "truncated header size",
			data:    []byte{0x01, 0x02, 0x03},
			wantErr: "failed to read header size",
		},
		{
			name: "header size too large",
			data: func() []byte {
				var buf bytes.Buffer
				binary.Write(&buf, binary.LittleEndian, uint64(2*1024*1024)) // 2MB
				return buf.Bytes()
			}(),
			wantErr: "header size too large",
		},
		{
			name: "truncated header",
			data: func() []byte {
				var buf bytes.Buffer
				binary.Write(&buf, binary.LittleEndian, uint64(100))
				buf.Write([]byte("short"))
				return buf.Bytes()
			}(),
			wantErr: "failed to read header",
		},
		{
			name: "invalid JSON",
			data: func() []byte {
				var buf bytes.Buffer
				binary.Write(&buf, binary.LittleEndian, uint64(10))
				buf.Write([]byte("not json!!"))
				return buf.Bytes()
			}(),
			wantErr: "failed to parse header",
		},
		{
			name: "no tensors in header",
			data: func() []byte {
				header := map[string]any{
					"__metadata__": map[string]any{"format": "pt"},
				}
				headerJSON, _ := json.Marshal(header)
				var buf bytes.Buffer
				binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON)))
				buf.Write(headerJSON)
				return buf.Bytes()
			}(),
			wantErr: "no tensor found in header",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseSafetensorsHeader(bytes.NewReader(tt.data))
			if err == nil {
				t.Error("expected error, got nil")
				return
			}
			if !bytes.Contains([]byte(err.Error()), []byte(tt.wantErr)) {
				t.Errorf("error = %v, want error containing %v", err, tt.wantErr)
			}
		})
	}
}

func TestGetTensorInfoFromManifest(t *testing.T) {
	// Create a temp directory for blobs
	tempDir := t.TempDir()

	// Create test tensor blobs
	tensors := []struct {
		name   string
		digest string
		dtype  string
		shape  []int64
	}{
		{
			name:   "model.embed_tokens.weight",
			digest: "sha256:abc123",
			dtype:  "BF16",
			shape:  []int64{262144, 2560},
		},
		{
			name:   "model.layers.0.self_attn.q_proj.weight",
			digest: "sha256:def456",
			dtype:  "BF16",
			shape:  []int64{2560, 2560},
		},
		{
			name:   "model.norm.weight",
			digest: "sha256:ghi789",
			dtype:  "F32",
			shape:  []int64{2560},
		},
	}

	// Create blob files
	var layers []imagegen.ManifestLayer
	for _, tensor := range tensors {
		// Create safetensors blob
		header := map[string]any{
			tensor.name: map[string]any{
				"dtype":        tensor.dtype,
				"shape":        tensor.shape,
				"data_offsets": []int64{0, 1000},
			},
		}
		headerJSON, _ := json.Marshal(header)

		var buf bytes.Buffer
		binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON)))
		buf.Write(headerJSON)

		// Write blob file
		blobName := "sha256-" + tensor.digest[7:]
		blobPath := filepath.Join(tempDir, blobName)
		if err := os.WriteFile(blobPath, buf.Bytes(), 0o644); err != nil {
			t.Fatalf("failed to write blob: %v", err)
		}

		layers = append(layers, imagegen.ManifestLayer{
			MediaType: "application/vnd.ollama.image.tensor",
			Digest:    tensor.digest,
			Size:      int64(buf.Len() + 1000), // header + fake data
			Name:      tensor.name,
		})
	}

	// Add a non-tensor layer (should be skipped)
	layers = append(layers, imagegen.ManifestLayer{
		MediaType: "application/vnd.ollama.image.json",
		Digest:    "sha256:config",
		Size:      100,
		Name:      "config.json",
	})

	manifest := &imagegen.ModelManifest{
		Manifest: &imagegen.Manifest{
			Layers: layers,
		},
		BlobDir: tempDir,
	}

	result, err := getTensorInfoFromManifest(manifest)
	if err != nil {
		t.Fatalf("getTensorInfoFromManifest() error = %v", err)
	}

	if len(result) != 3 {
		t.Errorf("got %d tensors, want 3", len(result))
	}

	// Verify each tensor
	for i, tensor := range tensors {
		if i >= len(result) {
			break
		}
		if result[i].Name != tensor.name {
			t.Errorf("tensor[%d].Name = %v, want %v", i, result[i].Name, tensor.name)
		}
		if result[i].Type != tensor.dtype {
			t.Errorf("tensor[%d].Type = %v, want %v", i, result[i].Type, tensor.dtype)
		}
		if len(result[i].Shape) != len(tensor.shape) {
			t.Errorf("tensor[%d].Shape length = %v, want %v", i, len(result[i].Shape), len(tensor.shape))
		}
	}
}

func TestReadSafetensorsHeader(t *testing.T) {
	// Create a temp file with a valid safetensors header
	tempDir := t.TempDir()

	header := map[string]any{
		"test_tensor": map[string]any{
			"dtype":        "BF16",
			"shape":        []int64{1024, 768},
			"data_offsets": []int64{0, 1572864},
		},
	}
	headerJSON, _ := json.Marshal(header)

	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON)))
	buf.Write(headerJSON)

	filePath := filepath.Join(tempDir, "test.safetensors")
	if err := os.WriteFile(filePath, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	info, err := readSafetensorsHeader(filePath)
	if err != nil {
		t.Fatalf("readSafetensorsHeader() error = %v", err)
	}

	if info.Dtype != "BF16" {
		t.Errorf("Dtype = %v, want BF16", info.Dtype)
	}
	if len(info.Shape) != 2 || info.Shape[0] != 1024 || info.Shape[1] != 768 {
		t.Errorf("Shape = %v, want [1024, 768]", info.Shape)
	}
}

func TestReadSafetensorsHeader_FileNotFound(t *testing.T) {
	_, err := readSafetensorsHeader("/nonexistent/path/file.safetensors")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}
