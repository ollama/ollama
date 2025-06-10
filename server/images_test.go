package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"slices"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

func TestModelCapabilities(t *testing.T) {
	// Create different types of model files using real GGUF format
	completionModelPath, _ := createBinFile(t, map[string]any{
		"general.architecture": "llama",
	}, []*ggml.Tensor{
		{
			Name:     "dummy.weight",
			Kind:     0, // F32
			Shape:    []uint64{100},
			WriterTo: bytes.NewReader(make([]byte, 100*4)),
		},
	})

	visionModelPath, _ := createBinFile(t, map[string]any{
		"general.architecture":     "llama",
		"llama.vision.block_count": uint32(1),
	}, []*ggml.Tensor{
		{
			Name:     "dummy.weight",
			Kind:     0, // F32
			Shape:    []uint64{100},
			WriterTo: bytes.NewReader(make([]byte, 100*4)),
		},
	})

	embeddingModelPath, _ := createBinFile(t, map[string]any{
		"general.architecture": "bert",
		"bert.pooling_type":    uint32(1),
	}, []*ggml.Tensor{
		{
			Name:     "dummy.weight",
			Kind:     0, // F32
			Shape:    []uint64{100},
			WriterTo: bytes.NewReader(make([]byte, 100*4)),
		},
	})

	// Create simple non-GGUF model file for template-only tests
	simpleModelFile, err := os.CreateTemp(t.TempDir(), "simple.*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer simpleModelFile.Close()
	if _, err := simpleModelFile.Write([]byte("dummy model data")); err != nil {
		t.Fatal(err)
	}
	simpleModelPath := simpleModelFile.Name()

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
			caps := Capabilities(&tt.model)
			if !compareCapabilities(caps, tt.expectedCaps) {
				t.Errorf("Expected capabilities %v, got %v", tt.expectedCaps, caps)
			}
		})
	}
}

func TestModelCheckCapabilities(t *testing.T) {
	tests := []struct {
		name           string
		model          Model
		checkCaps      []model.Capability
		expectedErrMsg string
	}{
		{
			name: "completion model without tools capability",
			model: Model{
				Capabilities: []model.Capability{
					model.CapabilityCompletion,
				},
			},
			checkCaps:      []model.Capability{model.CapabilityTools},
			expectedErrMsg: "does not support tools",
		},
		{
			name: "model with all needed capabilities",
			model: Model{
				Capabilities: []model.Capability{
					model.CapabilityTools,
					model.CapabilityInsert,
				},
			},
			checkCaps: []model.Capability{model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model missing insert capability",
			model: Model{
				Capabilities: []model.Capability{
					model.CapabilityCompletion,
				},
			},
			checkCaps:      []model.Capability{model.CapabilityInsert},
			expectedErrMsg: "does not support insert",
		},
		{
			name: "model missing vision capability",
			model: Model{
				Capabilities: []model.Capability{
					model.CapabilityCompletion,
				},
			},
			checkCaps:      []model.Capability{model.CapabilityVision},
			expectedErrMsg: "does not support vision",
		},
		{
			name: "model with vision capability",
			model: Model{
				Capabilities: []model.Capability{
					model.CapabilityVision,
				},
			},
			checkCaps: []model.Capability{model.CapabilityVision},
		},
		{
			name: "model with embedding capability",
			model: Model{
				Capabilities: []model.Capability{
					model.CapabilityEmbedding,
				},
			},
			checkCaps: []model.Capability{model.CapabilityEmbedding},
		},
		{
			name: "unknown capability",
			model: Model{
				Capabilities: []model.Capability{
					model.CapabilityCompletion,
				},
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

// Create a blob and return its digest and size
func createBlob(t *testing.T, modelsDir string, content []byte) (string, int64) {
	t.Helper()

	hasher := sha256.New()
	hasher.Write(content)
	hash := hex.EncodeToString(hasher.Sum(nil))
	digest := "sha256-" + hash
	size := int64(len(content))

	// Create blob directory and file
	blobPath := filepath.Join(modelsDir, "blobs", digest)
	if err := os.MkdirAll(filepath.Dir(blobPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(blobPath, content, 0o644); err != nil {
		t.Fatal(err)
	}

	return digest, size
}

// Create a complete model with manifest and blobs
func createTestModel(t *testing.T, modelsDir, modelName string, layers []Layer) {
	t.Helper()

	// Create manifest
	manifestPath := filepath.Join(modelsDir, "manifests", "registry.ollama.ai", "library", modelName, "latest")
	if err := os.MkdirAll(filepath.Dir(manifestPath), 0o755); err != nil {
		t.Fatal(err)
	}

	manifest := Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Layers:        layers,
	}

	f, err := os.Create(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := json.NewEncoder(f).Encode(manifest); err != nil {
		t.Fatal(err)
	}
}

func TestGetModel(t *testing.T) {
	tests := []struct {
		name          string
		modelName     string
		setupFunc     func(t *testing.T, modelsDir string)
		expectedError string
		validateFunc  func(t *testing.T, model *Model)
	}{
		{
			name:      "model not found",
			modelName: "nonexistent:latest",
			setupFunc: func(t *testing.T, modelsDir string) {
				// No setup - model should not exist
			},
			expectedError: "no such file or directory",
		},
		{
			name:      "model with basic config",
			modelName: "test-basic:latest",
			setupFunc: func(t *testing.T, modelsDir string) {
				modelPath, _ := createBinFile(t, map[string]any{
					"general.architecture": "llama",
				}, []*ggml.Tensor{
					{
						Name:     "dummy.weight",
						Kind:     0,
						Shape:    []uint64{100},
						WriterTo: bytes.NewReader(make([]byte, 100*4)),
					},
				})

				content, err := os.ReadFile(modelPath)
				if err != nil {
					t.Fatal(err)
				}
				digest, size := createBlob(t, modelsDir, content)

				createTestModel(t, modelsDir, "test-basic", []Layer{{
					MediaType: "application/vnd.ollama.image.model",
					Digest:    digest,
					Size:      size,
				}})
			},
			validateFunc: func(t *testing.T, model *Model) {
				if !strings.Contains(model.Name, "test-basic") {
					t.Errorf("Expected model name to contain 'test-basic', got %q", model.Name)
				}
				if model.Template == nil {
					t.Error("Expected model to have a template")
				}
				if model.ModelPath == "" {
					t.Error("Expected model to have a model path")
				}
			},
		},
		{
			name:      "model with vision capabilities",
			modelName: "test-vision:latest",
			setupFunc: func(t *testing.T, modelsDir string) {
				visionPath, _ := createBinFile(t, map[string]any{
					"general.architecture":     "llama",
					"llama.vision.block_count": uint32(1),
				}, []*ggml.Tensor{
					{
						Name:     "dummy.weight",
						Kind:     0,
						Shape:    []uint64{100},
						WriterTo: bytes.NewReader(make([]byte, 100*4)),
					},
				})

				content, err := os.ReadFile(visionPath)
				if err != nil {
					t.Fatal(err)
				}
				digest, size := createBlob(t, modelsDir, content)

				createTestModel(t, modelsDir, "test-vision", []Layer{{
					MediaType: "application/vnd.ollama.image.model",
					Digest:    digest,
					Size:      size,
				}})
			},
			validateFunc: func(t *testing.T, m *Model) {
				if !slices.Contains(m.Capabilities, model.CapabilityVision) {
					t.Error("Expected model to have vision capability")
				}
			},
		},
		{
			name:      "model with custom template",
			modelName: "test-template:latest",
			setupFunc: func(t *testing.T, modelsDir string) {
				modelDigest, modelSize := createBlob(t, modelsDir, []byte("dummy model data"))
				templateDigest, templateSize := createBlob(t, modelsDir, []byte("{{ .System }}{{ .Prompt }}{{ if .Tools }}{{ .Tools }}{{ end }}"))

				createTestModel(t, modelsDir, "test-template", []Layer{
					{MediaType: "application/vnd.ollama.image.model", Digest: modelDigest, Size: modelSize},
					{MediaType: "application/vnd.ollama.image.template", Digest: templateDigest, Size: templateSize},
				})
			},
			validateFunc: func(t *testing.T, m *Model) {
				if m.Template == nil {
					t.Error("Expected model to have a template")
				}
				if !slices.Contains(m.Capabilities, model.CapabilityTools) {
					t.Error("Expected model to have tools capability based on template")
				}
			},
		},
		{
			name:      "model with system prompt",
			modelName: "test-system:latest",
			setupFunc: func(t *testing.T, modelsDir string) {
				modelDigest, modelSize := createBlob(t, modelsDir, []byte("dummy model data"))
				systemDigest, systemSize := createBlob(t, modelsDir, []byte("You are a helpful assistant."))

				createTestModel(t, modelsDir, "test-system", []Layer{
					{MediaType: "application/vnd.ollama.image.model", Digest: modelDigest, Size: modelSize},
					{MediaType: "application/vnd.ollama.image.system", Digest: systemDigest, Size: systemSize},
				})
			},
			validateFunc: func(t *testing.T, model *Model) {
				if model.System != "You are a helpful assistant." {
					t.Errorf("Expected system prompt 'You are a helpful assistant.', got %q", model.System)
				}
			},
		},
		{
			name:      "cached model retrieval",
			modelName: "test-cached:latest",
			setupFunc: func(t *testing.T, modelsDir string) {
				modelDigest, modelSize := createBlob(t, modelsDir, []byte("dummy cached model data"))

				createTestModel(t, modelsDir, "test-cached", []Layer{{
					MediaType: "application/vnd.ollama.image.model",
					Digest:    modelDigest,
					Size:      modelSize,
				}})

				// pre-populate cache
				cachedModel := &Model{
					Name:      "test-cached:latest",
					ShortName: "test-cached",
					Template:  template.DefaultTemplate,
				}
				modelCache.set("test-cached:latest", cachedModel)
			},
			validateFunc: func(t *testing.T, model *Model) {
				if model.Name != "test-cached:latest" {
					t.Errorf("Expected cached model name 'test-cached:latest', got %q", model.Name)
				}
				if model.ShortName != "test-cached" {
					t.Errorf("Expected cached model short name 'test-cached', got %q", model.ShortName)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup
			modelsDir := t.TempDir()
			t.Setenv("OLLAMA_MODELS", modelsDir)
			// Clear cache before each test (except for cache test)
			if tt.name != "cached model retrieval" {
				modelCache = &ModelCache{cache: make(map[string]*CachedModel)} // Reset cache
			}
			if tt.setupFunc != nil {
				tt.setupFunc(t, modelsDir)
			}

			// Test
			model, err := GetModel(tt.modelName)

			// Validate
			if tt.expectedError != "" {
				if err == nil {
					t.Errorf("Expected error containing %q, got nil", tt.expectedError)
				} else if !strings.Contains(err.Error(), tt.expectedError) {
					t.Errorf("Expected error containing %q, got: %v", tt.expectedError, err)
				}
				return
			}
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}
			if tt.validateFunc != nil {
				tt.validateFunc(t, model)
			}
		})
	}
}
