package server

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

// GGUF type constants (matching gguf package)
const (
	typeUint8   = uint32(0)
	typeInt8    = uint32(1)
	typeUint16  = uint32(2)
	typeInt16   = uint32(3)
	typeUint32  = uint32(4)
	typeInt32   = uint32(5)
	typeFloat32 = uint32(6)
	typeBool    = uint32(7)
	typeString  = uint32(8)
	typeArray   = uint32(9)
	typeUint64  = uint32(10)
	typeInt64   = uint32(11)
	typeFloat64 = uint32(12)
)

type testTensorInfo struct {
	Name  string
	Shape []uint64
	Type  uint32
}

// Helper function to create test GGUF files (matching gguf package approach)
func createTestGGUFFile(path string, keyValues map[string]any, tensors []testTensorInfo) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write GGUF magic
	if _, err := file.Write([]byte("GGUF")); err != nil {
		return err
	}

	// Write version
	if err := binary.Write(file, binary.LittleEndian, uint32(3)); err != nil {
		return err
	}

	// Write tensor count
	if err := binary.Write(file, binary.LittleEndian, uint64(len(tensors))); err != nil {
		return err
	}

	// Write metadata count
	if err := binary.Write(file, binary.LittleEndian, uint64(len(keyValues))); err != nil {
		return err
	}

	// Write metadata
	for key, value := range keyValues {
		if err := writeKeyValue(file, key, value); err != nil {
			return err
		}
	}

	// Write tensor info
	for _, tensor := range tensors {
		if err := writeTensorInfo(file, tensor); err != nil {
			return err
		}
	}

	// Write some dummy tensor data
	dummyData := make([]byte, 1024)
	file.Write(dummyData)

	return nil
}

func writeKeyValue(file *os.File, key string, value any) error {
	// Write key length and key
	if err := binary.Write(file, binary.LittleEndian, uint64(len(key))); err != nil {
		return err
	}
	if _, err := file.Write([]byte(key)); err != nil {
		return err
	}

	// Write value based on type
	switch v := value.(type) {
	case string:
		if err := binary.Write(file, binary.LittleEndian, uint32(typeString)); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		_, err := file.Write([]byte(v))
		return err
	case int64:
		if err := binary.Write(file, binary.LittleEndian, typeInt64); err != nil {
			return err
		}
		return binary.Write(file, binary.LittleEndian, v)
	case uint32:
		if err := binary.Write(file, binary.LittleEndian, typeUint32); err != nil {
			return err
		}
		return binary.Write(file, binary.LittleEndian, v)
	case bool:
		if err := binary.Write(file, binary.LittleEndian, typeBool); err != nil {
			return err
		}
		return binary.Write(file, binary.LittleEndian, v)
	case float64:
		if err := binary.Write(file, binary.LittleEndian, uint32(typeFloat64)); err != nil {
			return err
		}
		return binary.Write(file, binary.LittleEndian, v)
	case []string:
		if err := binary.Write(file, binary.LittleEndian, uint32(typeArray)); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, typeString); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, s := range v {
			if err := binary.Write(file, binary.LittleEndian, uint64(len(s))); err != nil {
				return err
			}
			if _, err := file.Write([]byte(s)); err != nil {
				return err
			}
		}
		return nil
	case []int64:
		if err := binary.Write(file, binary.LittleEndian, uint32(typeArray)); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, typeInt64); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, i := range v {
			if err := binary.Write(file, binary.LittleEndian, i); err != nil {
				return err
			}
		}
		return nil
	case []float64:
		if err := binary.Write(file, binary.LittleEndian, typeArray); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, typeFloat64); err != nil {
			return err
		}
		if err := binary.Write(file, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, f := range v {
			if err := binary.Write(file, binary.LittleEndian, f); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unsupported value type: %T", value)
	}
}

func writeTensorInfo(file *os.File, tensor testTensorInfo) error {
	// Write tensor name
	if err := binary.Write(file, binary.LittleEndian, uint64(len(tensor.Name))); err != nil {
		return err
	}
	if _, err := file.Write([]byte(tensor.Name)); err != nil {
		return err
	}

	// Write dimensions
	if err := binary.Write(file, binary.LittleEndian, uint32(len(tensor.Shape))); err != nil {
		return err
	}
	for _, dim := range tensor.Shape {
		if err := binary.Write(file, binary.LittleEndian, dim); err != nil {
			return err
		}
	}

	// Write type
	if err := binary.Write(file, binary.LittleEndian, tensor.Type); err != nil {
		return err
	}

	// Write offset (dummy value)
	return binary.Write(file, binary.LittleEndian, uint64(0))
}

func TestModelCapabilities(t *testing.T) {
	// Create a temporary directory for test files
	tempDir := t.TempDir()

	// Create different types of mock model files
	completionModelPath := filepath.Join(tempDir, "model.bin")
	visionModelPath := filepath.Join(tempDir, "vision_model.bin")
	embeddingModelPath := filepath.Join(tempDir, "embedding_model.bin")
	// Create a simple model file for tests that don't depend on GGUF content
	simpleModelPath := filepath.Join(tempDir, "simple_model.bin")

	// Create completion model (llama architecture without vision)
	if err := createTestGGUFFile(completionModelPath, map[string]any{
		"general.architecture": "llama",
	}, []testTensorInfo{
		{Name: "token_embd.weight", Shape: []uint64{1000, 512}, Type: 1}, // F16
	}); err != nil {
		t.Fatalf("Failed to create completion model file: %v", err)
	}

	// Create vision model (llama architecture with vision block count)
	if err := createTestGGUFFile(visionModelPath, map[string]any{
		"general.architecture":     "llama",
		"llama.vision.block_count": uint32(1),
	}, []testTensorInfo{
		{Name: "token_embd.weight", Shape: []uint64{1000, 512}, Type: 1}, // F16
	}); err != nil {
		t.Fatalf("Failed to create vision model file: %v", err)
	}

	// Create embedding model (bert architecture with pooling type)
	if err := createTestGGUFFile(embeddingModelPath, map[string]any{
		"general.architecture": "bert",
		"bert.pooling_type":    uint32(1),
	}, []testTensorInfo{
		{Name: "token_embd.weight", Shape: []uint64{1000, 512}, Type: 1}, // F16
	}); err != nil {
		t.Fatalf("Failed to create embedding model file: %v", err)
	}

	// Create simple model file for tests that don't depend on GGUF content
	if err := os.WriteFile(simpleModelPath, []byte("dummy model data"), 0o644); err != nil {
		t.Fatalf("Failed to create simple model file: %v", err)
	}

	toolsInsertTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}{{ if .suffix }}{{ .suffix }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}
	chatTemplate, err := template.Parse("{{ .prompt }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}
	toolsTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	testModels := []struct {
		name         string
		model        Model
		expectedCaps []model.Capability
	}{
		{
			name: "model with completion capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion},
		},

		{
			name: "model with completion, tools, and insert capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsInsertTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model with tools and insert capability",
			model: Model{
				ModelPath: simpleModelPath,
				Template:  toolsInsertTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model with tools capability",
			model: Model{
				ModelPath: simpleModelPath,
				Template:  toolsTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityTools},
		},
		{
			name: "model with vision capability",
			model: Model{
				ModelPath: visionModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision},
		},
		{
			name: "model with vision, tools, and insert capability",
			model: Model{
				ModelPath: visionModelPath,
				Template:  toolsInsertTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision, model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model with embedding capability",
			model: Model{
				ModelPath: embeddingModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityEmbedding},
		},
	}

	// compare two slices of model.Capability regardless of order
	compareCapabilities := func(a, b []model.Capability) bool {
		if len(a) != len(b) {
			return false
		}

		aCount := make(map[model.Capability]int)
		for _, cap := range a {
			aCount[cap]++
		}

		bCount := make(map[model.Capability]int)
		for _, cap := range b {
			bCount[cap]++
		}

		for cap, count := range aCount {
			if bCount[cap] != count {
				return false
			}
		}

		return true
	}

	for _, tt := range testModels {
		t.Run(tt.name, func(t *testing.T) {
			// Test Capabilities method
			caps := tt.model.Capabilities()
			if !compareCapabilities(caps, tt.expectedCaps) {
				t.Errorf("Expected capabilities %v, got %v", tt.expectedCaps, caps)
			}
		})
	}
}

func TestModelCheckCapabilities(t *testing.T) {
	// Create a temporary directory for test files
	tempDir := t.TempDir()

	visionModelPath := filepath.Join(tempDir, "vision_model.bin")
	simpleModelPath := filepath.Join(tempDir, "model.bin")
	embeddingModelPath := filepath.Join(tempDir, "embedding_model.bin")

	// Create vision model (llama architecture with vision block count)
	if err := createTestGGUFFile(visionModelPath, map[string]any{
		"general.architecture":     "llama",
		"llama.vision.block_count": uint32(1),
	}, []testTensorInfo{
		{Name: "token_embd.weight", Shape: []uint64{1000, 512}, Type: 1}, // F16
	}); err != nil {
		t.Fatalf("Failed to create vision model file: %v", err)
	}

	// Create embedding model (bert architecture with pooling type)
	if err := createTestGGUFFile(embeddingModelPath, map[string]any{
		"general.architecture": "bert",
		"bert.pooling_type":    uint32(1),
	}, []testTensorInfo{
		{Name: "token_embd.weight", Shape: []uint64{1000, 512}, Type: 1}, // F16
	}); err != nil {
		t.Fatalf("Failed to create embedding model file: %v", err)
	}

	// Create simple model file for tests that don't depend on GGUF content
	if err := os.WriteFile(simpleModelPath, []byte("dummy model data"), 0o644); err != nil {
		t.Fatalf("Failed to create simple model file: %v", err)
	}

	toolsInsertTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}{{ if .suffix }}{{ .suffix }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}
	chatTemplate, err := template.Parse("{{ .prompt }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}
	toolsTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	tests := []struct {
		name           string
		model          Model
		checkCaps      []model.Capability
		expectedErrMsg string
	}{
		{
			name: "completion model without tools capability",
			model: Model{
				ModelPath: simpleModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityTools},
			expectedErrMsg: "does not support tools",
		},
		{
			name: "model with all needed capabilities",
			model: Model{
				ModelPath: simpleModelPath,
				Template:  toolsInsertTemplate,
			},
			checkCaps: []model.Capability{model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model missing insert capability",
			model: Model{
				ModelPath: simpleModelPath,
				Template:  toolsTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityInsert},
			expectedErrMsg: "does not support insert",
		},
		{
			name: "model missing vision capability",
			model: Model{
				ModelPath: simpleModelPath,
				Template:  toolsTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityVision},
			expectedErrMsg: "does not support vision",
		},
		{
			name: "model with vision capability",
			model: Model{
				ModelPath: visionModelPath,
				Template:  chatTemplate,
			},
			checkCaps: []model.Capability{model.CapabilityVision},
		},
		{
			name: "model with embedding capability",
			model: Model{
				ModelPath: embeddingModelPath,
				Template:  chatTemplate,
			},
			checkCaps: []model.Capability{model.CapabilityEmbedding},
		},
		{
			name: "unknown capability",
			model: Model{
				ModelPath: simpleModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{"unknown"},
			expectedErrMsg: "unknown capability",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test CheckCapabilities method
			err := tt.model.CheckCapabilities(tt.checkCaps...)
			if tt.expectedErrMsg == "" {
				if err != nil {
					t.Errorf("Expected no error, got: %v", err)
				}
			} else {
				if err == nil {
					t.Errorf("Expected error containing %q, got nil", tt.expectedErrMsg)
				} else if !strings.Contains(err.Error(), tt.expectedErrMsg) {
					t.Errorf("Expected error containing %q, got: %v", tt.expectedErrMsg, err)
				}
			}
		})
	}
}
