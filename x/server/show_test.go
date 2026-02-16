package server

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/manifest"
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
			totalTensorBytes: 8_600_000_150, // ~4.3B params * 2 bytes + 150 bytes header
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
			totalTensorBytes: 14_000_000_150, // ~7B params * 2 bytes + 150 bytes header
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
			totalTensorBytes: 8_600_000_150,
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
			totalTensorBytes: 400_000_150, // 100M params * 4 bytes + 150 bytes header
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
			totalTensorBytes: 2_001_500, // 1M params * 2 bytes + 10 tensors * 150 bytes
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
			totalBytes:     2_000_150, // 1M * 2 + 150
			tensorCount:    1,
			wantParamCount: 1_000_000,
		},
		{
			name:           "float16",
			dtype:          "float16",
			totalBytes:     2_000_150,
			tensorCount:    1,
			wantParamCount: 1_000_000,
		},
		{
			name:           "float32",
			dtype:          "float32",
			totalBytes:     4_000_150, // 1M * 4 + 150
			tensorCount:    1,
			wantParamCount: 1_000_000,
		},
		{
			name:           "int8",
			dtype:          "int8",
			totalBytes:     1_000_150, // 1M * 1 + 150
			tensorCount:    1,
			wantParamCount: 1_000_000,
		},
		{
			name:           "unknown dtype defaults to 2 bytes",
			dtype:          "unknown",
			totalBytes:     2_000_150,
			tensorCount:    1,
			wantParamCount: 1_000_000,
		},
		{
			name:           "empty dtype defaults to 2 bytes",
			dtype:          "",
			totalBytes:     2_000_150,
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
		name          string
		header        map[string]any
		wantDtype     string
		wantShape     []int64
		wantQuantType string
		wantGroupSize string
		wantErr       bool
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
			name: "tensor keyed by name",
			header: map[string]any{
				"model.layers.0.weight": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{2560, 2560},
					"data_offsets": []int64{0, 13107200},
				},
			},
			wantDtype: "BF16",
			wantShape: []int64{2560, 2560},
		},
		{
			name: "with int4 quant metadata",
			header: map[string]any{
				"__metadata__": map[string]any{
					"quant_type": "int4",
					"group_size": "32",
				},
				"model.layers.0.mlp.up_proj.weight": map[string]any{
					"dtype":        "U32",
					"shape":        []int64{2560, 320},
					"data_offsets": []int64{0, 3276800},
				},
				"model.layers.0.mlp.up_proj.weight.scale": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{2560, 80},
					"data_offsets": []int64{3276800, 3686400},
				},
				"model.layers.0.mlp.up_proj.weight.bias": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{2560, 80},
					"data_offsets": []int64{3686400, 4096000},
				},
			},
			wantDtype:     "U32",
			wantShape:     []int64{2560, 320},
			wantQuantType: "int4",
			wantGroupSize: "32",
		},
		{
			name: "int8 quant metadata",
			header: map[string]any{
				"__metadata__": map[string]any{
					"quant_type": "int8",
					"group_size": "64",
				},
				"model.layers.0.mlp.down_proj.weight": map[string]any{
					"dtype":        "U32",
					"shape":        []int64{2560, 640},
					"data_offsets": []int64{0, 6553600},
				},
				"model.layers.0.mlp.down_proj.weight.scale": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{2560, 40},
					"data_offsets": []int64{6553600, 6963200},
				},
			},
			wantDtype:     "U32",
			wantShape:     []int64{2560, 640},
			wantQuantType: "int8",
			wantGroupSize: "64",
		},
		{
			name: "with old-style format metadata",
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

			if info.QuantType != tt.wantQuantType {
				t.Errorf("QuantType = %v, want %v", info.QuantType, tt.wantQuantType)
			}
			if info.GroupSize != tt.wantGroupSize {
				t.Errorf("GroupSize = %v, want %v", info.GroupSize, tt.wantGroupSize)
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
	// Create a temp directory for blobs and set OLLAMA_MODELS
	tempDir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", tempDir)

	blobDir := filepath.Join(tempDir, "blobs")
	if err := os.MkdirAll(blobDir, 0o755); err != nil {
		t.Fatalf("failed to create blobs dir: %v", err)
	}

	// Create test tensor blobs with __metadata__
	tensors := []struct {
		name   string
		digest string
		dtype  string
		shape  []int64
	}{
		{
			name:   "model.embed_tokens.weight",
			digest: "sha256:abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abc0",
			dtype:  "BF16",
			shape:  []int64{262144, 2560},
		},
		{
			name:   "model.layers.0.self_attn.q_proj.weight",
			digest: "sha256:def456def456def456def456def456def456def456def456def456def456def0",
			dtype:  "BF16",
			shape:  []int64{2560, 2560},
		},
		{
			name:   "model.norm.weight",
			digest: "sha256:789789789789789789789789789789789789789789789789789789789789abc0",
			dtype:  "F32",
			shape:  []int64{2560},
		},
	}

	// Create blob files with tensor keyed by name
	var layers []manifest.Layer
	for _, tensor := range tensors {
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

		// Write blob file using the digest format expected by GetBlobsPath
		blobPath, err := manifest.BlobsPath(tensor.digest)
		if err != nil {
			t.Fatalf("failed to get blob path: %v", err)
		}
		if err := os.WriteFile(blobPath, buf.Bytes(), 0o644); err != nil {
			t.Fatalf("failed to write blob: %v", err)
		}

		layers = append(layers, manifest.Layer{
			MediaType: manifest.MediaTypeImageTensor,
			Digest:    tensor.digest,
			Size:      int64(buf.Len() + 1000), // header + fake data
			Name:      tensor.name,
		})
	}

	// Add a non-tensor layer (should be skipped)
	layers = append(layers, manifest.Layer{
		MediaType: "application/vnd.ollama.image.json",
		Digest:    "sha256:0000000000000000000000000000000000000000000000000000000000000000",
		Size:      100,
		Name:      "config.json",
	})

	mf := &manifest.Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Layers:        layers,
	}

	result, err := getTensorInfoFromManifest(mf)
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

func TestGetTensorInfoFromManifest_Quantized(t *testing.T) {
	// Create a temp directory for blobs and set OLLAMA_MODELS
	tempDir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", tempDir)

	blobDir := filepath.Join(tempDir, "blobs")
	if err := os.MkdirAll(blobDir, 0o755); err != nil {
		t.Fatalf("failed to create blobs dir: %v", err)
	}

	// Create a combined quantized blob with __metadata__
	header := map[string]any{
		"__metadata__": map[string]string{
			"quant_type": "int4",
			"group_size": "32",
		},
		"model.layers.0.mlp.up_proj.weight": map[string]any{
			"dtype":        "U32",
			"shape":        []int64{2560, 320}, // packed: 2560 / 8 = 320
			"data_offsets": []int64{0, 3276800},
		},
		"model.layers.0.mlp.up_proj.weight.scale": map[string]any{
			"dtype":        "BF16",
			"shape":        []int64{2560, 80}, // 2560 / 32 = 80
			"data_offsets": []int64{3276800, 3686400},
		},
		"model.layers.0.mlp.up_proj.weight.bias": map[string]any{
			"dtype":        "BF16",
			"shape":        []int64{2560, 80},
			"data_offsets": []int64{3686400, 4096000},
		},
	}
	headerJSON, _ := json.Marshal(header)

	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON)))
	buf.Write(headerJSON)

	digest := "sha256:aabb11aabb11aabb11aabb11aabb11aabb11aabb11aabb11aabb11aabb11aabb"
	blobPath, err := manifest.BlobsPath(digest)
	if err != nil {
		t.Fatalf("failed to get blob path: %v", err)
	}
	if err := os.WriteFile(blobPath, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("failed to write blob: %v", err)
	}

	mf := &manifest.Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Layers: []manifest.Layer{
			{
				MediaType: manifest.MediaTypeImageTensor,
				Digest:    digest,
				Size:      int64(buf.Len() + 4096000),
				Name:      "model.layers.0.mlp.up_proj.weight",
			},
		},
	}

	result, err := getTensorInfoFromManifest(mf)
	if err != nil {
		t.Fatalf("getTensorInfoFromManifest() error = %v", err)
	}

	if len(result) != 1 {
		t.Fatalf("got %d tensors, want 1", len(result))
	}

	tensor := result[0]
	if tensor.Name != "model.layers.0.mlp.up_proj.weight" {
		t.Errorf("Name = %v, want model.layers.0.mlp.up_proj.weight", tensor.Name)
	}
	if tensor.Type != "INT4" {
		t.Errorf("Type = %v, want INT4", tensor.Type)
	}
	// Shape should be unpacked: 320 * 8 = 2560
	if len(tensor.Shape) != 2 || tensor.Shape[0] != 2560 || tensor.Shape[1] != 2560 {
		t.Errorf("Shape = %v, want [2560, 2560]", tensor.Shape)
	}
}

func TestParseSafetensorsAllHeaders(t *testing.T) {
	tests := []struct {
		name       string
		header     map[string]any
		wantCount  int
		wantNames  []string
		wantDtypes []string
		wantQuants []string
		wantErr    bool
	}{
		{
			name: "single tensor blob",
			header: map[string]any{
				"model.layers.0.weight": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{2560, 2560},
					"data_offsets": []int64{0, 13107200},
				},
			},
			wantCount:  1,
			wantNames:  []string{"model.layers.0.weight"},
			wantDtypes: []string{"BF16"},
			wantQuants: []string{""},
		},
		{
			name: "packed unquantized blob",
			header: map[string]any{
				"model.layers.0.mlp.experts.0.down_proj.weight": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{2560, 10240},
					"data_offsets": []int64{0, 52428800},
				},
				"model.layers.0.mlp.experts.0.gate_proj.weight": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{10240, 2560},
					"data_offsets": []int64{52428800, 104857600},
				},
				"model.layers.0.mlp.experts.0.up_proj.weight": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{10240, 2560},
					"data_offsets": []int64{104857600, 157286400},
				},
			},
			wantCount: 3,
			wantNames: []string{
				"model.layers.0.mlp.experts.0.down_proj.weight",
				"model.layers.0.mlp.experts.0.gate_proj.weight",
				"model.layers.0.mlp.experts.0.up_proj.weight",
			},
			wantDtypes: []string{"BF16", "BF16", "BF16"},
			wantQuants: []string{"", "", ""},
		},
		{
			name: "packed quantized blob with global metadata",
			header: map[string]any{
				"__metadata__": map[string]any{
					"quant_type": "int4",
					"group_size": "32",
				},
				"model.layers.0.mlp.experts.0.gate_proj.weight": map[string]any{
					"dtype":        "U32",
					"shape":        []int64{10240, 320},
					"data_offsets": []int64{0, 13107200},
				},
				"model.layers.0.mlp.experts.0.gate_proj.weight.scale": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{10240, 80},
					"data_offsets": []int64{13107200, 14745600},
				},
				"model.layers.0.mlp.experts.0.gate_proj.weight.bias": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{10240, 80},
					"data_offsets": []int64{14745600, 16384000},
				},
				"model.layers.0.mlp.experts.0.up_proj.weight": map[string]any{
					"dtype":        "U32",
					"shape":        []int64{10240, 320},
					"data_offsets": []int64{16384000, 29491200},
				},
				"model.layers.0.mlp.experts.0.up_proj.weight.scale": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{10240, 80},
					"data_offsets": []int64{29491200, 31129600},
				},
				"model.layers.0.mlp.experts.0.up_proj.weight.bias": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{10240, 80},
					"data_offsets": []int64{31129600, 32768000},
				},
			},
			wantCount: 2,
			wantNames: []string{
				"model.layers.0.mlp.experts.0.gate_proj.weight",
				"model.layers.0.mlp.experts.0.up_proj.weight",
			},
			wantDtypes: []string{"U32", "U32"},
			wantQuants: []string{"int4", "int4"},
		},
		{
			name: "packed mixed-precision blob (no global metadata)",
			header: map[string]any{
				"model.layers.0.mlp.experts.0.gate_proj.weight": map[string]any{
					"dtype":        "U32",
					"shape":        []int64{10240, 320},
					"data_offsets": []int64{0, 13107200},
				},
				"model.layers.0.mlp.experts.0.gate_proj.weight.scale": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{10240, 80},
					"data_offsets": []int64{13107200, 14745600},
				},
				"model.layers.0.mlp.experts.0.gate_proj.weight.bias": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{10240, 80},
					"data_offsets": []int64{14745600, 16384000},
				},
				"model.layers.0.mlp.experts.0.down_proj.weight": map[string]any{
					"dtype":        "U32",
					"shape":        []int64{2560, 2560},
					"data_offsets": []int64{16384000, 42598400},
				},
				"model.layers.0.mlp.experts.0.down_proj.weight.scale": map[string]any{
					"dtype":        "BF16",
					"shape":        []int64{2560, 160},
					"data_offsets": []int64{42598400, 43417600},
				},
			},
			wantCount: 2,
			wantNames: []string{
				"model.layers.0.mlp.experts.0.down_proj.weight",
				"model.layers.0.mlp.experts.0.gate_proj.weight",
			},
			wantDtypes: []string{"U32", "U32"},
			wantQuants: []string{"int8", "int4"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			headerJSON, err := json.Marshal(tt.header)
			if err != nil {
				t.Fatalf("failed to marshal header: %v", err)
			}

			var buf bytes.Buffer
			if err := binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON))); err != nil {
				t.Fatalf("failed to write header size: %v", err)
			}
			buf.Write(headerJSON)

			results, err := parseSafetensorsAllHeaders(&buf)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseSafetensorsAllHeaders() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			if len(results) != tt.wantCount {
				t.Fatalf("got %d tensors, want %d", len(results), tt.wantCount)
			}

			for i, info := range results {
				if info.Name != tt.wantNames[i] {
					t.Errorf("tensor[%d].Name = %v, want %v", i, info.Name, tt.wantNames[i])
				}
				if info.Dtype != tt.wantDtypes[i] {
					t.Errorf("tensor[%d].Dtype = %v, want %v", i, info.Dtype, tt.wantDtypes[i])
				}
				if info.QuantType != tt.wantQuants[i] {
					t.Errorf("tensor[%d].QuantType = %v, want %v", i, info.QuantType, tt.wantQuants[i])
				}
			}
		})
	}
}

func TestGetTensorInfoFromManifest_Packed(t *testing.T) {
	// Create a temp directory for blobs and set OLLAMA_MODELS
	tempDir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", tempDir)

	blobDir := filepath.Join(tempDir, "blobs")
	if err := os.MkdirAll(blobDir, 0o755); err != nil {
		t.Fatalf("failed to create blobs dir: %v", err)
	}

	// Create a packed blob with multiple expert tensors (mixed quantization)
	header := map[string]any{
		"model.layers.0.mlp.experts.0.gate_proj.weight": map[string]any{
			"dtype":        "U32",
			"shape":        []int64{10240, 320},
			"data_offsets": []int64{0, 13107200},
		},
		"model.layers.0.mlp.experts.0.gate_proj.weight.scale": map[string]any{
			"dtype":        "BF16",
			"shape":        []int64{10240, 80},
			"data_offsets": []int64{13107200, 14745600},
		},
		"model.layers.0.mlp.experts.0.gate_proj.weight.bias": map[string]any{
			"dtype":        "BF16",
			"shape":        []int64{10240, 80},
			"data_offsets": []int64{14745600, 16384000},
		},
		"model.layers.0.mlp.experts.0.down_proj.weight": map[string]any{
			"dtype":        "U32",
			"shape":        []int64{2560, 2560},
			"data_offsets": []int64{16384000, 42598400},
		},
		"model.layers.0.mlp.experts.0.down_proj.weight.scale": map[string]any{
			"dtype":        "BF16",
			"shape":        []int64{2560, 160},
			"data_offsets": []int64{42598400, 43417600},
		},
	}
	headerJSON, _ := json.Marshal(header)

	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON)))
	buf.Write(headerJSON)

	packedDigest := "sha256:aaaa000000000000000000000000000000000000000000000000000000000001"
	blobPath, err := manifest.BlobsPath(packedDigest)
	if err != nil {
		t.Fatalf("failed to get blob path: %v", err)
	}
	if err := os.WriteFile(blobPath, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("failed to write packed blob: %v", err)
	}

	// Also create a regular (single-tensor) blob
	singleHeader := map[string]any{
		"model.embed_tokens.weight": map[string]any{
			"dtype":        "BF16",
			"shape":        []int64{262144, 2560},
			"data_offsets": []int64{0, 1342177280},
		},
	}
	singleHeaderJSON, _ := json.Marshal(singleHeader)
	var singleBuf bytes.Buffer
	binary.Write(&singleBuf, binary.LittleEndian, uint64(len(singleHeaderJSON)))
	singleBuf.Write(singleHeaderJSON)

	singleDigest := "sha256:bbbb000000000000000000000000000000000000000000000000000000000002"
	singleBlobPath, err := manifest.BlobsPath(singleDigest)
	if err != nil {
		t.Fatalf("failed to get blob path: %v", err)
	}
	if err := os.WriteFile(singleBlobPath, singleBuf.Bytes(), 0o644); err != nil {
		t.Fatalf("failed to write single blob: %v", err)
	}

	mf := &manifest.Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Layers: []manifest.Layer{
			{
				MediaType: manifest.MediaTypeImageTensor,
				Digest:    singleDigest,
				Size:      int64(singleBuf.Len()),
				Name:      "model.embed_tokens.weight",
			},
			{
				MediaType: manifest.MediaTypeImageTensor,
				Digest:    packedDigest,
				Size:      int64(buf.Len()),
				Name:      "model.layers.0.mlp.experts", // group prefix
			},
		},
	}

	result, err := getTensorInfoFromManifest(mf)
	if err != nil {
		t.Fatalf("getTensorInfoFromManifest() error = %v", err)
	}

	// Should have 3 tensors: 1 single + 2 packed main tensors
	if len(result) != 3 {
		t.Fatalf("got %d tensors, want 3. Tensors: %v", len(result), result)
	}

	// First tensor should be the single blob
	if result[0].Name != "model.embed_tokens.weight" {
		t.Errorf("tensor[0].Name = %v, want model.embed_tokens.weight", result[0].Name)
	}
	if result[0].Type != "BF16" {
		t.Errorf("tensor[0].Type = %v, want BF16", result[0].Type)
	}

	// Packed tensors should have their actual names (sorted)
	packedNames := make(map[string]bool)
	for _, r := range result[1:] {
		packedNames[r.Name] = true
	}
	if !packedNames["model.layers.0.mlp.experts.0.down_proj.weight"] {
		t.Error("missing packed tensor: model.layers.0.mlp.experts.0.down_proj.weight")
	}
	if !packedNames["model.layers.0.mlp.experts.0.gate_proj.weight"] {
		t.Error("missing packed tensor: model.layers.0.mlp.experts.0.gate_proj.weight")
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
