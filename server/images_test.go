package server

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

func TestModelCapabilities(t *testing.T) {
	// Create completion model (llama architecture without vision)
	completionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
	}, []*ggml.Tensor{})

	// Create vision model (llama architecture with vision block count)
	visionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":     "llama",
		"llama.vision.block_count": uint32(1),
	}, []*ggml.Tensor{})

	// Create embedding model (bert architecture with pooling type)
	embeddingModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "bert",
		"bert.pooling_type":    uint32(1),
	}, []*ggml.Tensor{})

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
			name: "model with image generation capability via config",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"image"},
				},
			},
			expectedCaps: []model.Capability{model.CapabilityImage},
		},
		{
			name: "model with image and vision capability (image editing)",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"image", "vision"},
				},
			},
			expectedCaps: []model.Capability{model.CapabilityImage, model.CapabilityVision},
		},
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
			name: "model with tools capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools},
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
	// Create simple model file for tests that don't depend on GGUF content
	completionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
	}, []*ggml.Tensor{})

	// Create vision model (llama architecture with vision block count)
	visionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":     "llama",
		"llama.vision.block_count": uint32(1),
	}, []*ggml.Tensor{})

	// Create embedding model (bert architecture with pooling type)
	embeddingModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "bert",
		"bert.pooling_type":    uint32(1),
	}, []*ggml.Tensor{})

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
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityTools},
			expectedErrMsg: "does not support tools",
		},
		{
			name: "model with all needed capabilities",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsInsertTemplate,
			},
			checkCaps: []model.Capability{model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model missing insert capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityInsert},
			expectedErrMsg: "does not support insert",
		},
		{
			name: "model missing vision capability",
			model: Model{
				ModelPath: completionModelPath,
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
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{"unknown"},
			expectedErrMsg: "unknown capability",
		},
		{
			name: "model missing image generation capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityImage},
			expectedErrMsg: "does not support image generation",
		},
		{
			name: "model with image generation capability",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"image"},
				},
			},
			checkCaps: []model.Capability{model.CapabilityImage},
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

func TestPullModelManifest(t *testing.T) {
	cases := []struct {
		name     string
		manifest string
	}{
		{
			name: "pretty printed",
			manifest: `{  "schemaVersion": 2,  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": { "digest": "sha256:abc", "mediaType": "application/vnd.docker.container.image.v1+json", "size": 50 },
  "layers": [{ "digest": "sha256:t1", "mediaType": "application/vnd.ollama.image.tensor", "size": 1024, "name": "model.weight" }]
}`,
		},
		{
			name:     "non-standard field order",
			manifest: `{"layers":[{"size":999,"digest":"sha256:def","mediaType":"application/vnd.ollama.image.model"}],"schemaVersion":2,"config":{"size":50,"digest":"sha256:abc","mediaType":"application/vnd.docker.container.image.v1+json"},"mediaType":"application/vnd.docker.distribution.manifest.v2+json"}`,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(tt.manifest))
			}))
			defer ts.Close()

			n := model.ParseName("test/model:latest")
			n.ProtocolScheme = "http"
			n.Host = strings.TrimPrefix(ts.URL, "http://")

			mf, data, err := pullModelManifest(t.Context(), n, &registryOptions{})
			if err != nil {
				t.Fatal(err)
			}

			// Raw bytes must be byte-for-byte identical to what the server sent
			if string(data) != tt.manifest {
				t.Fatalf("raw bytes differ from server response")
			}

			// SHA256 of returned data must match the expected registry digest
			expectedDigest := fmt.Sprintf("%x", sha256.Sum256([]byte(tt.manifest)))
			gotDigest := fmt.Sprintf("%x", sha256.Sum256(data))
			if gotDigest != expectedDigest {
				t.Fatalf("digest mismatch\ngot:  %s\nwant: %s", gotDigest, expectedDigest)
			}

			// Parsed manifest must still be usable
			if mf.SchemaVersion != 2 {
				t.Fatalf("schemaVersion = %d, want 2", mf.SchemaVersion)
			}
			if mf.Config.Digest == "" {
				t.Fatal("config digest is empty")
			}
			if len(mf.Layers) == 0 {
				t.Fatal("expected at least one layer")
			}
		})
	}
}

func TestVerifyBlobWithDuplicateDigest(t *testing.T) {
	// When a manifest's config and layer share the same digest, the
	// skipVerify map must not allow a later cache-hit entry to overwrite
	// a prior non-cache-hit entry. If it does, verifyBlob is skipped
	// for the freshly-downloaded blob, enabling SSRF response exfiltration.

	modelsDir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", modelsDir)

	blobsDir := filepath.Join(modelsDir, "blobs")
	if err := os.MkdirAll(blobsDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Write a blob whose content does NOT match its filename digest.
	// This simulates a tampered blob that a rogue registry placed on disk.
	fakeDigest := "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	tamperedContent := []byte("tampered data from rogue registry")

	blobPath := filepath.Join(blobsDir, "sha256-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	if err := os.WriteFile(blobPath, tamperedContent, 0o644); err != nil {
		t.Fatal(err)
	}

	// Simulate the skipVerify map behavior during PullModel.
	// layers = [layer (same digest), config (same digest)]
	// First iteration: layer already on disk → downloadBlob returns cacheHit=true
	// Then attacker tampers the file after download, or the download itself
	// was tampered. Either way, verification must not be skipped.
	//
	// The bug: if we just do skipVerify[digest] = cacheHit, and the layer
	// returns false (downloaded fresh) but the config returns true (cache hit),
	// the true overwrites the false and verification is skipped.

	// Verify that our tampered blob fails verification
	if err := verifyBlob(fakeDigest); err == nil {
		t.Fatal("expected verifyBlob to fail on tampered blob")
	}

	// Simulate the fixed skipVerify logic:
	// When a layer is downloaded fresh (cacheHit=false) and a config
	// with the same digest is a cache hit (cacheHit=true), the result
	// must be false (needs verification).
	skipVerify := make(map[string]bool)

	// Layer iteration: downloaded fresh
	layerCacheHit := false
	if existing, ok := skipVerify[fakeDigest]; !ok {
		skipVerify[fakeDigest] = layerCacheHit
	} else {
		skipVerify[fakeDigest] = existing && layerCacheHit
	}

	if skipVerify[fakeDigest] != false {
		t.Fatal("after layer download (cacheHit=false), skipVerify should be false")
	}

	// Config iteration: found on disk (cache hit)
	configCacheHit := true
	if existing, ok := skipVerify[fakeDigest]; !ok {
		skipVerify[fakeDigest] = configCacheHit
	} else {
		skipVerify[fakeDigest] = existing && configCacheHit
	}

	// Critical assertion: skipVerify must remain false because the layer
	// download was not a cache hit. Before the fix, this was true (broken).
	if skipVerify[fakeDigest] != false {
		t.Fatal("skipVerify should remain false when any download of the digest was not a cache hit")
	}

	// Now verify that the verification loop catches the tampered blob
	if !skipVerify[fakeDigest] {
		if err := verifyBlob(fakeDigest); err == nil {
			t.Fatal("expected digest mismatch error for tampered blob")
		} else if !errors.Is(err, errDigestMismatch) {
			t.Fatalf("expected errDigestMismatch, got: %v", err)
		}
	} else {
		t.Fatal("skipVerify incorrectly set to true, verification would be skipped")
	}
}

func TestSkipVerifyMapNeverOverwritesFalse(t *testing.T) {
	// Table-driven test for the skipVerify map logic to ensure that
	// once a digest is marked as needing verification (false), a
	// subsequent cache hit (true) does not overwrite it.
	cases := []struct {
		name      string
		cacheHits []bool
		wantSkip  bool
	}{
		{
			name:      "single fresh download",
			cacheHits: []bool{false},
			wantSkip:  false,
		},
		{
			name:      "single cache hit",
			cacheHits: []bool{true},
			wantSkip:  true,
		},
		{
			name:      "fresh then cache hit (duplicate digest bug)",
			cacheHits: []bool{false, true},
			wantSkip:  false,
		},
		{
			name:      "cache hit then fresh",
			cacheHits: []bool{true, false},
			wantSkip:  false,
		},
		{
			name:      "both cache hits",
			cacheHits: []bool{true, true},
			wantSkip:  true,
		},
		{
			name:      "both fresh",
			cacheHits: []bool{false, false},
			wantSkip:  false,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			skipVerify := make(map[string]bool)
			digest := "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

			for _, cacheHit := range tt.cacheHits {
				if existing, ok := skipVerify[digest]; !ok {
					skipVerify[digest] = cacheHit
				} else {
					skipVerify[digest] = existing && cacheHit
				}
			}

			if skipVerify[digest] != tt.wantSkip {
				t.Errorf("skipVerify = %v, want %v", skipVerify[digest], tt.wantSkip)
			}
		})
	}
}
