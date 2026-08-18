package server

import (
	"maps"
	"slices"
	"strconv"
	"sync"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
	"golang.org/x/sync/singleflight"
)

// inferenceModelCache stores fully resolved model metadata and capabilities.
// Model blobs are content-addressed, and the manifest digest is the freshness
// boundary, so cached entries remain valid until the model is recreated or
// pulled with different content.
type inferenceModelCache struct {
	mu      sync.RWMutex
	entries map[inferenceModelCacheKey]inferenceModelCacheEntry
	loads   singleflight.Group

	loadModel func(string) (*Model, error)
}

type inferenceModelCacheKey struct {
	name          string
	goTemplate    bool
	goTemplateSet bool
}

type inferenceModelCacheEntry struct {
	digest string
	model  *Model
}

func newInferenceModelCache() *inferenceModelCache {
	return &inferenceModelCache{
		entries:   make(map[inferenceModelCacheKey]inferenceModelCacheEntry),
		loadModel: GetModel,
	}
}

func (c *inferenceModelCache) Get(name string) (*Model, error) {
	n := model.ParseName(name)
	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		return nil, err
	}

	key := inferenceModelCacheKey{
		name:          n.String(),
		goTemplate:    envconfig.GoTemplate(true),
		goTemplateSet: goTemplateEnvSet(),
	}
	digest := mf.Digest()

	c.mu.RLock()
	entry, ok := c.entries[key]
	c.mu.RUnlock()
	if ok && entry.digest == digest {
		return cloneInferenceModel(entry.model), nil
	}

	loadKey := key.name + "\x00" + digest + "\x00" + strconv.FormatBool(key.goTemplate) + "\x00" + strconv.FormatBool(key.goTemplateSet)
	v, err, _ := c.loads.Do(loadKey, func() (any, error) {
		c.mu.RLock()
		entry, ok := c.entries[key]
		c.mu.RUnlock()
		if ok && entry.digest == digest {
			return entry.model, nil
		}

		m, err := c.loadModel(name)
		if err != nil {
			return nil, err
		}
		m.capabilities = m.Capabilities()
		m.capabilitiesCached = true

		c.mu.Lock()
		c.entries[key] = inferenceModelCacheEntry{digest: m.Digest, model: m}
		c.mu.Unlock()
		return m, nil
	})
	if err != nil {
		return nil, err
	}

	return cloneInferenceModel(v.(*Model)), nil
}

func cloneInferenceModel(src *Model) *Model {
	if src == nil {
		return nil
	}

	dst := *src
	dst.Config.ModelFamilies = slices.Clone(src.Config.ModelFamilies)
	dst.Config.Capabilities = slices.Clone(src.Config.Capabilities)
	if src.Config.Draft != nil {
		draft := *src.Config.Draft
		dst.Config.Draft = &draft
	}
	dst.AdapterPaths = slices.Clone(src.AdapterPaths)
	dst.ProjectorPaths = slices.Clone(src.ProjectorPaths)
	dst.License = slices.Clone(src.License)
	dst.Options = maps.Clone(src.Options)
	dst.Messages = slices.Clone(src.Messages)
	dst.capabilities = slices.Clone(src.capabilities)

	return &dst
}

func (s *Server) getModel(name string) (*Model, error) {
	if s != nil && s.modelCaches != nil && s.modelCaches.inference != nil {
		return s.modelCaches.inference.Get(name)
	}
	return GetModel(name)
}
