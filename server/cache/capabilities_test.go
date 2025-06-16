package cache

import (
	"bytes"
	"maps"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

// testGGUF creates a temporary GGUF model file for testing with custom key-value pairs
func testGGUF(tb testing.TB, customKV ggml.KV) string {
	tb.Helper()
	f, err := os.CreateTemp(tb.TempDir(), "test*.gguf")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	kv := ggml.KV{}
	maps.Copy(kv, customKV)

	tensors := []*ggml.Tensor{
		{
			Name:     "token_embd.weight",
			Kind:     0,
			Shape:    []uint64{1, 1},
			WriterTo: bytes.NewBuffer(make([]byte, 4)),
		},
	}

	if err := ggml.WriteGGUF(f, kv, tensors); err != nil {
		tb.Fatal(err)
	}

	return f.Name()
}

func TestCapabilities(t *testing.T) {
	ggufCapabilities.Range(func(key, value any) bool {
		ggufCapabilities.Delete(key)
		return true
	})

	// Create test model paths
	completionModelPath := testGGUF(t, ggml.KV{
		"general.architecture": "llama",
	})

	visionModelPath := testGGUF(t, ggml.KV{
		"general.architecture":     "llama",
		"llama.vision.block_count": uint32(1),
	})

	embeddingModelPath := testGGUF(t, ggml.KV{
		"general.architecture": "bert",
		"bert.pooling_type":    uint32(1),
	})

	// Create templates
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

	testCases := []struct {
		name         string
		model        ModelInfo
		expectedCaps []model.Capability
	}{
		{
			name: "model with completion capability",
			model: ModelInfo{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion},
		},
		{
			name: "model with completion, tools, and insert capability",
			model: ModelInfo{
				ModelPath: completionModelPath,
				Template:  toolsInsertTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model with tools capability",
			model: ModelInfo{
				ModelPath: completionModelPath,
				Template:  toolsTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools},
		},
		{
			name: "model with vision capability from gguf",
			model: ModelInfo{
				ModelPath: visionModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision},
		},
		{
			name: "model with vision capability from projector",
			model: ModelInfo{
				ModelPath:      completionModelPath,
				ProjectorPaths: []string{"/path/to/projector"},
				Template:       chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision},
		},
		{
			name: "model with vision, tools, and insert capability",
			model: ModelInfo{
				ModelPath: visionModelPath,
				Template:  toolsInsertTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision, model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model with embedding capability",
			model: ModelInfo{
				ModelPath: embeddingModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityEmbedding},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// First call - should read from file
			caps := Capabilities(tc.model)
			slices.Sort(caps)
			slices.Sort(tc.expectedCaps)
			if !slices.Equal(caps, tc.expectedCaps) {
				t.Errorf("Expected capabilities %v, got %v", tc.expectedCaps, caps)
			}

			// Verify caching for models that read from GGUF
			if tc.model.ModelPath != "" {
				// Check that entry is cached
				_, ok := ggufCapabilities.Load(tc.model.ModelPath)
				if !ok {
					t.Error("Expected capabilities to be cached")
				}

				// Second call - should use cache
				caps2 := Capabilities(tc.model)
				slices.Sort(caps2)
				if !slices.Equal(caps, caps2) {
					t.Errorf("Cached capabilities don't match original: expected %v, got %v", caps, caps2)
				}
			}
		})
	}

	// Test cache invalidation on file modification
	t.Run("cache invalidation", func(t *testing.T) {
		// Use completion model for this test
		info := ModelInfo{
			ModelPath: completionModelPath,
			Template:  chatTemplate,
		}

		// Get initial cached entry
		cached, ok := ggufCapabilities.Load(completionModelPath)
		if !ok {
			t.Fatal("Expected model to be cached from previous tests")
		}
		entry := cached.(cacheEntry)

		// Modify the file's timestamp to the future
		future := time.Now().Add(time.Hour)
		err := os.Chtimes(completionModelPath, future, future)
		if err != nil {
			t.Fatalf("Failed to update file timestamp: %v", err)
		}

		// Call should re-read from file due to changed modtime
		caps := Capabilities(info)
		if len(caps) != 1 || caps[0] != model.CapabilityCompletion {
			t.Errorf("Expected [CapabilityCompletion], got %v", caps)
		}

		// Check that cache was updated with new modtime
		cached2, ok := ggufCapabilities.Load(completionModelPath)
		if !ok {
			t.Error("Expected capabilities to be cached after re-read")
		}
		entry2 := cached2.(cacheEntry)
		if entry2.modTime.Equal(entry.modTime) {
			t.Error("Expected cache entry to have updated modTime")
		}
	})
}
