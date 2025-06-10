package server

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestModelCacheGet(t *testing.T) {
	testModel := &Model{
		Name:      "test-model:latest",
		ShortName: "test-model",
	}

	tests := []struct {
		name           string
		modelName      string
		setupFunc      func(t *testing.T, modelsDir string, cache *ModelCache) string // returns manifest path
		expectedModel  *Model
		expectedExists bool
	}{
		{
			name:      "cache hit - valid cached model",
			modelName: "test-model:latest",
			setupFunc: func(t *testing.T, modelsDir string, cache *ModelCache) string {
				createTestModel(t, modelsDir, "test-model", []Layer{
					{MediaType: "application/vnd.ollama.image.model", Digest: "sha256-abc123", Size: 1000},
				})

				manifestPath := filepath.Join(modelsDir, "manifests", "registry.ollama.ai", "library", "test-model", "latest")
				info, err := os.Stat(manifestPath)
				if err != nil {
					t.Fatal(err)
				}

				cache.cache["test-model:latest"] = &CachedModel{
					model:    testModel,
					modTime:  info.ModTime(),
					fileSize: info.Size(),
				}
				return manifestPath
			},
			expectedModel:  testModel,
			expectedExists: true,
		},
		{
			name:      "cache miss - no cached entry",
			modelName: "missing-model:latest",
			setupFunc: func(t *testing.T, modelsDir string, cache *ModelCache) string {
				createTestModel(t, modelsDir, "missing-model", []Layer{
					{MediaType: "application/vnd.ollama.image.model", Digest: "sha256-def456", Size: 2000},
				})
				return filepath.Join(modelsDir, "manifests", "registry.ollama.ai", "library", "missing-model", "latest")
			},
			expectedModel:  nil,
			expectedExists: false,
		},
		{
			name:      "cache stale - modification time changed",
			modelName: "stale-model:latest",
			setupFunc: func(t *testing.T, modelsDir string, cache *ModelCache) string {
				createTestModel(t, modelsDir, "stale-model", []Layer{
					{MediaType: "application/vnd.ollama.image.model", Digest: "sha256-ghi789", Size: 3000},
				})

				manifestPath := filepath.Join(modelsDir, "manifests", "registry.ollama.ai", "library", "stale-model", "latest")
				info, err := os.Stat(manifestPath)
				if err != nil {
					t.Fatal(err)
				}

				cache.cache["stale-model:latest"] = &CachedModel{
					model:    testModel,
					modTime:  info.ModTime().Add(-time.Hour), // Stale time
					fileSize: info.Size(),
				}
				return manifestPath
			},
			expectedModel:  nil,
			expectedExists: false,
		},
		{
			name:      "cache stale - file size changed",
			modelName: "stale-size-model:latest",
			setupFunc: func(t *testing.T, modelsDir string, cache *ModelCache) string {
				createTestModel(t, modelsDir, "stale-size-model", []Layer{
					{MediaType: "application/vnd.ollama.image.model", Digest: "sha256-jkl012", Size: 4000},
				})

				manifestPath := filepath.Join(modelsDir, "manifests", "registry.ollama.ai", "library", "stale-size-model", "latest")
				info, err := os.Stat(manifestPath)
				if err != nil {
					t.Fatal(err)
				}

				cache.cache["stale-size-model:latest"] = &CachedModel{
					model:    testModel,
					modTime:  info.ModTime(),
					fileSize: info.Size() + 100, // Different size
				}
				return manifestPath
			},
			expectedModel:  nil,
			expectedExists: false,
		},
		{
			name:      "manifest file does not exist",
			modelName: "nonexistent-model:latest",
			setupFunc: func(t *testing.T, modelsDir string, cache *ModelCache) string {
				// Add to cache but don't create manifest file
				cache.cache["nonexistent-model:latest"] = &CachedModel{
					model:    testModel,
					modTime:  time.Now(),
					fileSize: 100,
				}
				return "" // No manifest created
			},
			expectedModel:  nil,
			expectedExists: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			modelsDir := t.TempDir()
			t.Setenv("OLLAMA_MODELS", modelsDir)

			// Create fresh cache instance for each test
			cache := &ModelCache{
				cache: make(map[string]*CachedModel),
			}

			if tt.setupFunc != nil {
				tt.setupFunc(t, modelsDir, cache)
			}

			model, exists := cache.get(tt.modelName)

			if exists != tt.expectedExists {
				t.Errorf("get() exists = %v, want %v", exists, tt.expectedExists)
			}

			if tt.expectedModel != nil && model != tt.expectedModel {
				t.Errorf("get() model = %v, want %v", model, tt.expectedModel)
			}

			if tt.expectedModel == nil && model != nil {
				t.Errorf("get() model = %v, want nil", model)
			}
		})
	}
}

func TestModelCacheSet(t *testing.T) {
	testModel := &Model{
		Name:      "test-model:latest",
		ShortName: "test-model",
	}

	tests := []struct {
		name         string
		modelName    string
		model        *Model
		setupFunc    func(t *testing.T, modelsDir string) string // returns manifest path
		expectCached bool
	}{
		{
			name:      "successful cache set",
			modelName: "test-model:latest",
			model:     testModel,
			setupFunc: func(t *testing.T, modelsDir string) string {
				createTestModel(t, modelsDir, "test-model", []Layer{
					{MediaType: "application/vnd.ollama.image.model", Digest: "sha256-abc123", Size: 1000},
				})
				return filepath.Join(modelsDir, "manifests", "registry.ollama.ai", "library", "test-model", "latest")
			},
			expectCached: true,
		},
		{
			name:      "manifest file does not exist",
			modelName: "nonexistent-model:latest",
			model:     testModel,
			setupFunc: func(t *testing.T, modelsDir string) string {
				// Don't create manifest file
				return ""
			},
			expectCached: false,
		},
		{
			name:      "overwrite existing cache entry",
			modelName: "existing-model:latest",
			model:     testModel,
			setupFunc: func(t *testing.T, modelsDir string) string {
				createTestModel(t, modelsDir, "existing-model", []Layer{
					{MediaType: "application/vnd.ollama.image.model", Digest: "sha256-def456", Size: 2000},
				})
				return filepath.Join(modelsDir, "manifests", "registry.ollama.ai", "library", "existing-model", "latest")
			},
			expectCached: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			modelsDir := t.TempDir()
			t.Setenv("OLLAMA_MODELS", modelsDir)

			// Create fresh cache instance for each test
			cache := &ModelCache{
				cache: make(map[string]*CachedModel),
			}

			if tt.name == "overwrite existing cache entry" {
				cache.cache[tt.modelName] = &CachedModel{
					model:    &Model{Name: "old-model"},
					modTime:  time.Now().Add(-time.Hour),
					fileSize: 50,
				}
			}

			if tt.setupFunc != nil {
				tt.setupFunc(t, modelsDir)
			}

			cache.set(tt.modelName, tt.model)

			cached, exists := cache.cache[tt.modelName]

			if tt.expectCached {
				if !exists {
					t.Errorf("set() expected model to be cached, but it wasn't")
					return
				}

				if cached.model != tt.model {
					t.Errorf("set() cached model = %v, want %v", cached.model, tt.model)
				}

				// Verify file info is captured correctly if manifest exists
				expectedManifestPath := filepath.Join(modelsDir, "manifests", "registry.ollama.ai", "library", tt.modelName[:len(tt.modelName)-7], "latest")
				if info, err := os.Stat(expectedManifestPath); err == nil {
					if !cached.modTime.Equal(info.ModTime()) {
						t.Errorf("set() cached modTime = %v, want %v", cached.modTime, info.ModTime())
					}
					if cached.fileSize != info.Size() {
						t.Errorf("set() cached fileSize = %v, want %v", cached.fileSize, info.Size())
					}
				}
			} else {
				if exists {
					t.Errorf("set() expected model not to be cached, but it was")
				}
			}
		})
	}
}
