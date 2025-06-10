package server

import (
	"log/slog"
	"os"
	"sync"
	"time"
)

type ModelCache struct {
	mu    sync.RWMutex
	cache map[string]*CachedModel
}

type CachedModel struct {
	model    *Model
	modTime  time.Time
	fileSize int64
}

var modelCache = &ModelCache{
	cache: make(map[string]*CachedModel),
}

func init() {
	modelCache.fill()
}

func (c *ModelCache) fill() {
	manifests, err := Manifests(true) // continues on error
	if err != nil {
		slog.Warn("Failed to get manifests during cache fill", "error", err)
		return
	}

	for modelName := range manifests {
		nameStr := modelName.String()

		// Load the model (this will populate the cache via GetModel -> set)
		_, err := GetModel(nameStr)
		if err != nil {
			slog.Debug("Failed to load model during cache fill", "name", nameStr, "error", err)
			continue
		}
	}

	slog.Debug("Model cache filled")
}

func (c *ModelCache) get(name string) (*Model, bool) {
	mp := ParseModelPath(name)
	manifestPath, err := mp.GetManifestPath()
	if err != nil {
		return nil, false
	}

	// Check manifest file modification time
	info, err := os.Stat(manifestPath)
	if err != nil {
		return nil, false
	}

	cached, exists := c.cache[name]
	if exists && cached.modTime.Equal(info.ModTime()) && cached.fileSize == info.Size() {
		// Cache hit - return cached model
		return cached.model, true
	}

	// Cache miss or stale
	return nil, false
}

func (c *ModelCache) set(name string, model *Model) {
	mp := ParseModelPath(name)
	manifestPath, err := mp.GetManifestPath()
	if err != nil {
		slog.Debug("Failed to get manifest path for model", "name", name, "error", err)
		return
	}

	info, err := os.Stat(manifestPath)
	if err != nil {
		slog.Debug("Failed to stat manifest file", "path", manifestPath, "error", err)
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache[name] = &CachedModel{
		model:    model,
		modTime:  info.ModTime(),
		fileSize: info.Size(),
	}
}
