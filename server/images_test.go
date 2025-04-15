package server

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

// Constants for GGUF magic bytes and version
var (
	ggufMagic = []byte{0x47, 0x47, 0x55, 0x46} // "GGUF"
	ggufVer   = uint32(3)                      // Version 3
)

// Helper function to create mock GGUF data
func createMockGGUFData(architecture string, vision bool) []byte {
	var buf bytes.Buffer

	// Write GGUF header
	buf.Write(ggufMagic)
	binary.Write(&buf, binary.LittleEndian, ggufVer)

	// Write tensor count (0 for our test)
	var numTensors uint64 = 0
	binary.Write(&buf, binary.LittleEndian, numTensors)

	// Calculate number of metadata entries
	numMetaEntries := uint64(1) // architecture entry
	if vision {
		numMetaEntries++
	}
	// Add embedding entry if architecture is "bert"
	if architecture == "bert" {
		numMetaEntries++
	}
	binary.Write(&buf, binary.LittleEndian, numMetaEntries)

	// Write architecture metadata
	archKey := "general.architecture"
	keyLen := uint64(len(archKey))
	binary.Write(&buf, binary.LittleEndian, keyLen)
	buf.WriteString(archKey)

	// String type (8)
	var strType uint32 = 8
	binary.Write(&buf, binary.LittleEndian, strType)

	// String length
	strLen := uint64(len(architecture))
	binary.Write(&buf, binary.LittleEndian, strLen)
	buf.WriteString(architecture)

	if vision {
		visionKey := architecture + ".vision.block_count"
		keyLen = uint64(len(visionKey))
		binary.Write(&buf, binary.LittleEndian, keyLen)
		buf.WriteString(visionKey)

		// uint32 type (4)
		var uint32Type uint32 = 4
		binary.Write(&buf, binary.LittleEndian, uint32Type)

		// uint32 value (1)
		var countVal uint32 = 1
		binary.Write(&buf, binary.LittleEndian, countVal)
	}
	// Write embedding metadata if architecture is "bert"
	if architecture == "bert" {
		poolKey := architecture + ".pooling_type"
		keyLen = uint64(len(poolKey))
		binary.Write(&buf, binary.LittleEndian, keyLen)
		buf.WriteString(poolKey)

		// uint32 type (4)
		var uint32Type uint32 = 4
		binary.Write(&buf, binary.LittleEndian, uint32Type)

		// uint32 value (1)
		var poolingVal uint32 = 1
		binary.Write(&buf, binary.LittleEndian, poolingVal)
	}

	return buf.Bytes()
}

func TestModelCapabilities(t *testing.T) {
	// Create a temporary directory for test files
	tempDir, err := os.MkdirTemp("", "model_capabilities_test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Create different types of mock model files
	completionModelPath := filepath.Join(tempDir, "model.bin")
	visionModelPath := filepath.Join(tempDir, "vision_model.bin")
	embeddingModelPath := filepath.Join(tempDir, "embedding_model.bin")
	// Create a simple model file for tests that don't depend on GGUF content
	simpleModelPath := filepath.Join(tempDir, "simple_model.bin")

	err = os.WriteFile(completionModelPath, createMockGGUFData("llama", false), 0o644)
	if err != nil {
		t.Fatalf("Failed to create completion model file: %v", err)
	}
	err = os.WriteFile(visionModelPath, createMockGGUFData("llama", true), 0o644)
	if err != nil {
		t.Fatalf("Failed to create completion model file: %v", err)
	}
	err = os.WriteFile(embeddingModelPath, createMockGGUFData("bert", false), 0o644)
	if err != nil {
		t.Fatalf("Failed to create embedding model file: %v", err)
	}
	err = os.WriteFile(simpleModelPath, []byte("dummy model data"), 0o644)
	if err != nil {
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
	tempDir, err := os.MkdirTemp("", "model_check_capabilities_test")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	visionModelPath := filepath.Join(tempDir, "vision_model.bin")
	simpleModelPath := filepath.Join(tempDir, "model.bin")
	embeddingModelPath := filepath.Join(tempDir, "embedding_model.bin")

	err = os.WriteFile(simpleModelPath, []byte("dummy model data"), 0o644)
	if err != nil {
		t.Fatalf("Failed to create simple model file: %v", err)
	}
	err = os.WriteFile(visionModelPath, createMockGGUFData("llama", true), 0o644)
	if err != nil {
		t.Fatalf("Failed to create vision model file: %v", err)
	}
	err = os.WriteFile(embeddingModelPath, createMockGGUFData("bert", false), 0o644)
	if err != nil {
		t.Fatalf("Failed to create embedding model file: %v", err)
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
