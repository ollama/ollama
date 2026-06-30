package server

import (
	"cmp"
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"slices"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/model/parsers"
	ollamatemplate "github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/model"
)

type modelListSummary struct {
	Model        string
	Name         string
	RemoteModel  string
	RemoteHost   string
	Size         int64
	Digest       string
	ModifiedAt   time.Time
	Details      api.ModelDetails
	Capabilities []model.Capability
}

type modelListCacheEntry struct {
	Digest  string
	Summary modelListSummary
}

type modelListCache struct {
	mu sync.RWMutex

	entries map[string]modelListCacheEntry

	once       sync.Once
	readyOnce  sync.Once
	ready      chan struct{}
	hydrateErr error
	build      func(model.Name, *manifest.Manifest) (modelListSummary, error)
}

func newModelListCache() *modelListCache {
	return &modelListCache{
		entries: make(map[string]modelListCacheEntry),
		ready:   make(chan struct{}),
		build:   buildModelListSummary,
	}
}

func (c *modelListCache) Start(ctx context.Context) {
	if c == nil {
		return
	}

	c.once.Do(func() {
		slog.Debug("starting model list cache")
		go func() {
			err := c.hydrate(ctx)
			c.markReady(err)
			if err != nil {
				if ctx != nil && ctx.Err() != nil {
					return
				}
				slog.Warn("model list cache hydration failed", "error", err)
			}
		}()
	})
}

func (c *modelListCache) hydrate(ctx context.Context) error {
	start := time.Now()

	manifests, err := manifest.Manifests(true)
	if err != nil {
		return err
	}

	var hydrated, failed int
	for name, mf := range manifests {
		if ctx != nil {
			if err := ctx.Err(); err != nil {
				return err
			}
		}

		summary, err := c.build(name, mf)
		if err != nil {
			failed++
			slog.Warn("failed to hydrate model list cache", "model", name.String(), "error", err)
			continue
		}

		c.set(name, mf.Digest(), summary)
		hydrated++
	}

	slog.Info("model list cache hydration complete", "models", hydrated, "failures", failed, "elapsed", time.Since(start))
	return nil
}

func (c *modelListCache) markReady(err error) {
	c.mu.Lock()
	c.hydrateErr = err
	c.mu.Unlock()

	c.readyOnce.Do(func() {
		close(c.ready)
	})
}

func (c *modelListCache) Wait(ctx context.Context) error {
	if c == nil {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}

	select {
	case <-c.ready:
		c.mu.RLock()
		err := c.hydrateErr
		c.mu.RUnlock()
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (c *modelListCache) List(ctx context.Context) ([]api.ListModelResponse, error) {
	if err := c.Wait(ctx); err != nil {
		return nil, err
	}
	if err := c.syncManifests(ctx); err != nil {
		return nil, err
	}

	c.mu.RLock()
	models := make([]api.ListModelResponse, 0, len(c.entries))
	for _, entry := range c.entries {
		models = append(models, entry.Summary.ListModelResponse())
	}
	c.mu.RUnlock()

	sortListModelResponses(models)
	return models, nil
}

func (c *modelListCache) syncManifests(ctx context.Context) error {
	manifests, err := manifest.Manifests(true)
	if err != nil {
		return err
	}

	c.mu.RLock()
	current := make(map[string]string, len(c.entries))
	for name, entry := range c.entries {
		current[name] = entry.Digest
	}
	c.mu.RUnlock()

	type update struct {
		name    model.Name
		digest  string
		summary modelListSummary
	}

	seen := make(map[string]struct{}, len(manifests))
	stale := make(map[string]struct{})
	var updates []update
	for name, mf := range manifests {
		if ctx != nil {
			if err := ctx.Err(); err != nil {
				return err
			}
		}

		key := name.String()
		digest := mf.Digest()
		seen[key] = struct{}{}
		if current[key] == digest {
			continue
		}

		summary, err := c.build(name, mf)
		if err != nil {
			slog.Warn("failed to refresh model list cache", "model", key, "error", err)
			if _, ok := current[key]; ok {
				stale[key] = struct{}{}
			}
			continue
		}
		updates = append(updates, update{name: name, digest: digest, summary: summary})
	}

	c.mu.Lock()
	for name := range c.entries {
		if _, ok := seen[name]; !ok {
			delete(c.entries, name)
			continue
		}
		if _, ok := stale[name]; ok {
			delete(c.entries, name)
		}
	}
	for _, update := range updates {
		c.entries[update.name.String()] = modelListCacheEntry{
			Digest:  update.digest,
			Summary: cloneModelListSummary(update.summary),
		}
	}
	c.mu.Unlock()

	return nil
}

func (c *modelListCache) RefreshModel(name model.Name) error {
	if c == nil {
		return nil
	}

	if !name.IsFullyQualified() {
		var err error
		name, err = getExistingName(name)
		if err != nil {
			return err
		}
	}

	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		c.DeleteModel(name)
		return err
	}

	summary, err := c.build(name, mf)
	if err != nil {
		c.DeleteModel(name)
		return err
	}

	c.set(name, mf.Digest(), summary)
	return nil
}

func (c *modelListCache) DeleteModel(name model.Name) {
	if c == nil {
		return
	}

	c.mu.Lock()
	delete(c.entries, name.String())
	c.mu.Unlock()
}

func (c *modelListCache) Get(name model.Name) (modelListSummary, bool) {
	if c == nil {
		return modelListSummary{}, false
	}

	if !name.IsFullyQualified() {
		if existing, err := getExistingName(name); err == nil {
			name = existing
		}
	}

	c.mu.RLock()
	entry, ok := c.entries[name.String()]
	c.mu.RUnlock()
	if !ok {
		return modelListSummary{}, false
	}

	return cloneModelListSummary(entry.Summary), true
}

func (c *modelListCache) Len() int {
	if c == nil {
		return 0
	}

	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries)
}

func (c *modelListCache) set(name model.Name, digest string, summary modelListSummary) {
	c.mu.Lock()
	c.entries[name.String()] = modelListCacheEntry{
		Digest:  digest,
		Summary: cloneModelListSummary(summary),
	}
	c.mu.Unlock()
}

func buildModelListSummary(name model.Name, mf *manifest.Manifest) (modelListSummary, error) {
	cfg, err := readModelListConfig(mf)
	if err != nil {
		return modelListSummary{}, err
	}

	var modified time.Time
	if fi := mf.FileInfo(); fi != nil {
		modified = fi.ModTime()
	}

	summary := modelListSummary{
		Model:       name.DisplayShortest(),
		Name:        name.DisplayShortest(),
		RemoteModel: cfg.RemoteModel,
		RemoteHost:  cfg.RemoteHost,
		Size:        mf.Size(),
		Digest:      mf.Digest(),
		ModifiedAt:  modified,
		Details: api.ModelDetails{
			Format:            cfg.ModelFormat,
			Family:            cfg.ModelFamily,
			Families:          append([]string(nil), cfg.ModelFamilies...),
			ParameterSize:     cfg.ModelType,
			QuantizationLevel: cfg.FileType,
			ContextLength:     cfg.ContextLen,
			EmbeddingLength:   cfg.EmbedLen,
		},
	}

	modelPath, projectorCount, tmpl, err := readModelListLayers(mf, &summary)
	if err != nil {
		return modelListSummary{}, err
	}

	if cfg.RemoteHost == "" && cfg.RemoteModel == "" && modelPath != "" {
		info, err := readModelListGGUF(modelPath)
		if err != nil {
			slog.Debug("failed to read gguf model metadata", "model", name.String(), "error", err)
		} else {
			summary.Capabilities = appendModelListCapabilities(summary.Capabilities, info.Capabilities...)
			if summary.Details.ContextLength == 0 {
				summary.Details.ContextLength = info.ContextLength
			}
			if summary.Details.EmbeddingLength == 0 {
				summary.Details.EmbeddingLength = info.EmbeddingLength
			}
			if isUnknownQuantization(summary.Details.QuantizationLevel) && !isUnknownQuantization(info.FileType) {
				summary.Details.QuantizationLevel = info.FileType
			}
		}
	}

	for _, c := range cfg.Capabilities {
		summary.Capabilities = appendModelListCapability(summary.Capabilities, model.Capability(c))
	}

	builtinParser := parsers.ParserForName(cfg.Parser)
	if tmpl != nil {
		vars, err := tmpl.Vars()
		if err != nil {
			slog.Warn("model template contains errors", "model", name.String(), "error", err)
		}
		if slices.Contains(vars, "tools") || (builtinParser != nil && builtinParser.HasToolSupport()) {
			summary.Capabilities = appendModelListCapability(summary.Capabilities, model.CapabilityTools)
		}
		if slices.Contains(vars, "suffix") {
			summary.Capabilities = appendModelListCapability(summary.Capabilities, model.CapabilityInsert)
		}

		openingTag, closingTag := thinking.InferTags(tmpl.Template)
		hasTags := openingTag != "" && closingTag != ""
		isGptoss := slices.Contains([]string{"gptoss", "gpt-oss"}, cfg.ModelFamily)
		if !slices.Contains(summary.Capabilities, model.CapabilityThinking) &&
			(hasTags || isGptoss || (builtinParser != nil && builtinParser.HasThinkingSupport())) {
			summary.Capabilities = appendModelListCapability(summary.Capabilities, model.CapabilityThinking)
		}
	}

	if projectorCount > 0 {
		summary.Capabilities = appendModelListCapability(summary.Capabilities, model.CapabilityVision)
	}

	if cfg.ModelFormat == "safetensors" && isGemma4Renderer(cfg.Renderer) {
		summary.Capabilities = slices.DeleteFunc(summary.Capabilities, func(c model.Capability) bool {
			return c == model.CapabilityVision || c == model.CapabilityAudio
		})
	}

	return summary, nil
}

func readModelListConfig(mf *manifest.Manifest) (model.ConfigV2, error) {
	var cfg model.ConfigV2
	if mf == nil || mf.Config.Digest == "" {
		return cfg, nil
	}

	f, err := mf.Config.Open()
	if err != nil {
		return cfg, err
	}
	defer f.Close()

	if err := json.NewDecoder(f).Decode(&cfg); err != nil {
		return cfg, err
	}

	return cfg, nil
}

func readModelListLayers(mf *manifest.Manifest, summary *modelListSummary) (string, int, *ollamatemplate.Template, error) {
	var modelPath string
	var projectorCount int
	tmpl := ollamatemplate.DefaultTemplate

	for _, layer := range mf.Layers {
		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			filename, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return "", 0, nil, err
			}
			modelPath = filename
			summary.Details.ParentModel = layer.From
		case "application/vnd.ollama.image.projector":
			projectorCount++
		case "application/vnd.ollama.image.prompt",
			"application/vnd.ollama.image.template":
			filename, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return "", 0, nil, err
			}
			bts, err := os.ReadFile(filename)
			if err != nil {
				return "", 0, nil, err
			}

			tmpl, err = ollamatemplate.Parse(string(bts))
			if err != nil {
				return "", 0, nil, err
			}
		}
	}

	return modelPath, projectorCount, tmpl, nil
}

type modelListGGUF struct {
	Capabilities    []model.Capability
	ContextLength   int
	EmbeddingLength int
	FileType        string
}

func readModelListGGUF(path string) (modelListGGUF, error) {
	keyValues, err := gguf.ScanKeyValues(path, keepModelListGGUFKey)
	if err != nil {
		return modelListGGUF{}, err
	}

	var info modelListGGUF
	var architecture string
	var metadata ggufArchitectureMetadata
	byArchitecture := make(map[string]ggufArchitectureMetadata)
	for _, kv := range keyValues {
		switch kv.Key {
		case ggufKeyGeneralArchitecture:
			architecture = kv.String()
			metadata = byArchitecture[architecture]
		case ggufKeyGeneralFileType:
			info.FileType = ggml.FileType(ggufMetadataInt(kv.Value)).String()
		case ggufKeyTokenizerChatTemplate:
			info.Capabilities = chatTemplateCapabilities(info.Capabilities, kv.String())
		default:
			arch, suffix, _ := cutGGUFArchitectureKey(kv.Key)
			value := byArchitecture[arch]
			updateGGUFArchitectureMetadata(&value, suffix, kv.Value)
			byArchitecture[arch] = value
			if arch == architecture {
				metadata = value
			}
		}
	}

	info.Capabilities = appendGGUFMetadataCapabilities(info.Capabilities, metadata)
	info.ContextLength = metadata.ContextLength
	info.EmbeddingLength = metadata.EmbeddingLength

	return info, nil
}

func keepModelListGGUFKey(key string) bool {
	switch key {
	case ggufKeyGeneralArchitecture, ggufKeyGeneralFileType, ggufKeyTokenizerChatTemplate:
		return true
	}

	return isGGUFArchitectureMetadataKey(key)
}

func appendModelListCapabilities(capabilities []model.Capability, values ...model.Capability) []model.Capability {
	for _, capability := range values {
		capabilities = appendCapability(capabilities, capability)
	}
	return capabilities
}

func appendModelListCapability(capabilities []model.Capability, capability model.Capability) []model.Capability {
	return appendCapability(capabilities, capability)
}

func isUnknownQuantization(quantization string) bool {
	return quantization == "" || quantization == "unknown"
}

func cloneModelListSummary(summary modelListSummary) modelListSummary {
	summary.Details.Families = append([]string(nil), summary.Details.Families...)
	summary.Capabilities = append([]model.Capability(nil), summary.Capabilities...)
	return summary
}

func (s modelListSummary) ListModelResponse() api.ListModelResponse {
	resp := api.ListModelResponse{
		Model:       s.Model,
		Name:        s.Name,
		RemoteModel: s.RemoteModel,
		RemoteHost:  s.RemoteHost,
		Size:        s.Size,
		Digest:      s.Digest,
		ModifiedAt:  s.ModifiedAt,
		Details: api.ModelDetails{
			ParentModel:       s.Details.ParentModel,
			Format:            s.Details.Format,
			Family:            s.Details.Family,
			Families:          append([]string(nil), s.Details.Families...),
			ParameterSize:     s.Details.ParameterSize,
			QuantizationLevel: s.Details.QuantizationLevel,
			ContextLength:     s.Details.ContextLength,
			EmbeddingLength:   s.Details.EmbeddingLength,
		},
	}

	resp.Capabilities = append([]model.Capability(nil), s.Capabilities...)

	return resp
}

func sortListModelResponses(models []api.ListModelResponse) {
	slices.SortStableFunc(models, func(i, j api.ListModelResponse) int {
		// Preserve the existing /api/tags order: most recently modified first.
		return cmp.Compare(j.ModifiedAt.Unix(), i.ModifiedAt.Unix())
	})
}

func (s *Server) refreshModelListCache(name model.Name) {
	if s == nil || s.modelCaches == nil || s.modelCaches.modelList == nil {
		return
	}

	if err := s.modelCaches.modelList.RefreshModel(name); err != nil {
		slog.Warn("failed to refresh model list cache", "model", name.String(), "error", err)
	}
}

func (s *Server) deleteModelListCache(name model.Name) {
	if s == nil || s.modelCaches == nil || s.modelCaches.modelList == nil {
		return
	}

	s.modelCaches.modelList.DeleteModel(name)
}
