package server

import (
	"fmt"
	"sync"

	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/types/model"
)

type loadedModel struct {
	model     llama.Model
	modelPath string
}

type modelLoader struct {
	cache sync.Map
}

// modelCache stores loaded models keyed by their full path and params hash
var modelCache sync.Map // map[string]*loadedModel

func (ml *modelLoader) LoadModel(name string, params llama.ModelParams) (*loadedModel, error) {
	modelName := model.ParseName(name)
	if !modelName.IsValid() {
		return nil, fmt.Errorf("invalid model name: %s", modelName)
	}

	modelPath, err := GetModel(modelName.String())
	if err != nil {
		return nil, fmt.Errorf("model not found: %s", modelName)
	}

	// Create cache key from model path and params hash
	cacheKey := fmt.Sprintf("%s-%+v", modelPath.ModelPath, params)
	if cached, ok := modelCache.Load(cacheKey); ok {
		return cached.(*loadedModel), nil
	}

	// Evict existing model if any
	ml.evictExistingModel()

	model, err := llama.LoadModelFromFile(modelPath.ModelPath, params)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %v", err)
	}

	loaded := &loadedModel{
		model:     *model,
		modelPath: modelPath.ModelPath,
	}
	modelCache.Store(cacheKey, loaded)

	return loaded, nil
}

// evictExistingModel removes any currently loaded model from the cache
// Currently only supports a single model in cache at a time
// TODO: Add proper cache eviction policy (LRU/size/TTL based)
func (ml *modelLoader) evictExistingModel() {
	ml.cache.Range(func(key, value any) bool {
		if cached, ok := ml.cache.LoadAndDelete(key); ok {
			llama.FreeModel(&cached.(*loadedModel).model)
		}
		return true
	})
}
