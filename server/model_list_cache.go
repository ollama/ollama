package server

import (
	"bufio"
	"cmp"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
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
}

const (
	modelListGGUFMagicLE = 0x46554747
	modelListGGUFMagicBE = 0x47475546
)

const (
	modelListGGUFTypeUint8 uint32 = iota
	modelListGGUFTypeInt8
	modelListGGUFTypeUint16
	modelListGGUFTypeInt16
	modelListGGUFTypeUint32
	modelListGGUFTypeInt32
	modelListGGUFTypeFloat32
	modelListGGUFTypeBool
	modelListGGUFTypeString
	modelListGGUFTypeArray
	modelListGGUFTypeUint64
	modelListGGUFTypeInt64
	modelListGGUFTypeFloat64
)

// readModelListGGUF scans only the small GGUF header values launch needs
// and stops before tokenizer arrays. Using gguf.File.KeyValue for missing keys
// can otherwise advance through large arrays just to discover absence.
func readModelListGGUF(path string) (modelListGGUF, error) {
	f, err := os.Open(path)
	if err != nil {
		return modelListGGUF{}, err
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, 32<<10)
	var magic uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return modelListGGUF{}, err
	}

	var byteOrder binary.ByteOrder = binary.LittleEndian
	switch magic {
	case modelListGGUFMagicLE:
	case modelListGGUFMagicBE:
		byteOrder = binary.BigEndian
	default:
		return modelListGGUF{}, fmt.Errorf("invalid file magic")
	}

	var version uint32
	if err := binary.Read(r, byteOrder, &version); err != nil {
		return modelListGGUF{}, err
	}

	var numKV uint64
	switch version {
	case 1:
		var header struct {
			NumTensor uint32
			NumKV     uint32
		}
		if err := binary.Read(r, byteOrder, &header); err != nil {
			return modelListGGUF{}, err
		}
		numKV = uint64(header.NumKV)
	default:
		var header struct {
			NumTensor uint64
			NumKV     uint64
		}
		if err := binary.Read(r, byteOrder, &header); err != nil {
			return modelListGGUF{}, err
		}
		numKV = header.NumKV
	}

	info := modelListGGUF{}
	var architecture string
	var hasPoolingType bool

	for range numKV {
		key, err := readModelListGGUFString(r, byteOrder, version)
		if err != nil {
			return modelListGGUF{}, err
		}

		var valueType uint32
		if err := binary.Read(r, byteOrder, &valueType); err != nil {
			return modelListGGUF{}, err
		}

		if key == "general.architecture" {
			value, err := readModelListGGUFStringValue(r, byteOrder, version, valueType)
			if err != nil {
				return modelListGGUF{}, err
			}
			architecture = value
			continue
		}

		if architecture != "" && strings.HasPrefix(key, "tokenizer.") {
			break
		}

		if architecture != "" && strings.HasPrefix(key, architecture+".") {
			switch strings.TrimPrefix(key, architecture+".") {
			case "pooling_type":
				hasPoolingType = true
			case "vision.block_count":
				info.Capabilities = appendModelListCapability(info.Capabilities, model.CapabilityVision)
			case "audio.block_count":
				info.Capabilities = appendModelListCapability(info.Capabilities, model.CapabilityAudio)
			case "context_length":
				value, err := readModelListGGUFIntValue(r, byteOrder, version, valueType)
				if err != nil {
					return modelListGGUF{}, err
				}
				info.ContextLength = value
				continue
			case "embedding_length":
				value, err := readModelListGGUFIntValue(r, byteOrder, version, valueType)
				if err != nil {
					return modelListGGUF{}, err
				}
				info.EmbeddingLength = value
				continue
			}
		}

		if err := skipModelListGGUFValue(r, byteOrder, version, valueType); err != nil {
			return modelListGGUF{}, err
		}
	}

	if hasPoolingType {
		info.Capabilities = appendModelListCapability(info.Capabilities, model.CapabilityEmbedding)
	} else {
		info.Capabilities = appendModelListCapability(info.Capabilities, model.CapabilityCompletion)
	}

	return info, nil
}

func readModelListGGUFStringValue(r io.Reader, byteOrder binary.ByteOrder, version uint32, valueType uint32) (string, error) {
	if valueType != modelListGGUFTypeString {
		if err := skipModelListGGUFValue(r, byteOrder, version, valueType); err != nil {
			return "", err
		}
		return "", fmt.Errorf("unexpected gguf string type %d", valueType)
	}
	return readModelListGGUFString(r, byteOrder, version)
}

func readModelListGGUFIntValue(r io.Reader, byteOrder binary.ByteOrder, version uint32, valueType uint32) (int, error) {
	switch valueType {
	case modelListGGUFTypeUint8:
		var value uint8
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case modelListGGUFTypeInt8:
		var value int8
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case modelListGGUFTypeUint16:
		var value uint16
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case modelListGGUFTypeInt16:
		var value int16
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case modelListGGUFTypeUint32:
		var value uint32
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case modelListGGUFTypeInt32:
		var value int32
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case modelListGGUFTypeUint64:
		var value uint64
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case modelListGGUFTypeInt64:
		var value int64
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	default:
		if err := skipModelListGGUFValue(r, byteOrder, version, valueType); err != nil {
			return 0, err
		}
		return 0, fmt.Errorf("unexpected gguf integer type %d", valueType)
	}
}

func skipModelListGGUFValue(r io.Reader, byteOrder binary.ByteOrder, version uint32, valueType uint32) error {
	switch valueType {
	case modelListGGUFTypeUint8, modelListGGUFTypeInt8, modelListGGUFTypeBool:
		return discardModelListGGUFBytes(r, 1)
	case modelListGGUFTypeUint16, modelListGGUFTypeInt16:
		return discardModelListGGUFBytes(r, 2)
	case modelListGGUFTypeUint32, modelListGGUFTypeInt32, modelListGGUFTypeFloat32:
		return discardModelListGGUFBytes(r, 4)
	case modelListGGUFTypeUint64, modelListGGUFTypeInt64, modelListGGUFTypeFloat64:
		return discardModelListGGUFBytes(r, 8)
	case modelListGGUFTypeString:
		return skipModelListGGUFString(r, byteOrder, version)
	case modelListGGUFTypeArray:
		var arrayType uint32
		if err := binary.Read(r, byteOrder, &arrayType); err != nil {
			return err
		}
		var count uint64
		if err := binary.Read(r, byteOrder, &count); err != nil {
			return err
		}
		return skipModelListGGUFArray(r, byteOrder, version, arrayType, count)
	default:
		return fmt.Errorf("unsupported gguf value type %d", valueType)
	}
}

func skipModelListGGUFArray(r io.Reader, byteOrder binary.ByteOrder, version uint32, arrayType uint32, count uint64) error {
	var size uint64
	switch arrayType {
	case modelListGGUFTypeUint8, modelListGGUFTypeInt8, modelListGGUFTypeBool:
		size = 1
	case modelListGGUFTypeUint16, modelListGGUFTypeInt16:
		size = 2
	case modelListGGUFTypeUint32, modelListGGUFTypeInt32, modelListGGUFTypeFloat32:
		size = 4
	case modelListGGUFTypeUint64, modelListGGUFTypeInt64, modelListGGUFTypeFloat64:
		size = 8
	case modelListGGUFTypeString:
		for range count {
			if err := skipModelListGGUFString(r, byteOrder, version); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unsupported gguf array type %d", arrayType)
	}
	return discardModelListGGUFBytes(r, int64(count*size))
}

func readModelListGGUFString(r io.Reader, byteOrder binary.ByteOrder, version uint32) (string, error) {
	var length uint64
	if err := binary.Read(r, byteOrder, &length); err != nil {
		return "", err
	}

	if length == 0 {
		return "", nil
	}

	bts := make([]byte, length)
	if _, err := io.ReadFull(r, bts); err != nil {
		return "", err
	}
	if version == 1 && bts[len(bts)-1] == 0 {
		bts = bts[:len(bts)-1]
	}
	return string(bts), nil
}

func skipModelListGGUFString(r io.Reader, byteOrder binary.ByteOrder, version uint32) error {
	var length uint64
	if err := binary.Read(r, byteOrder, &length); err != nil {
		return err
	}
	return discardModelListGGUFBytes(r, int64(length))
}

func discardModelListGGUFBytes(r io.Reader, n int64) error {
	if n <= 0 {
		return nil
	}
	_, err := io.CopyN(io.Discard, r, n)
	return err
}

func appendModelListCapabilities(capabilities []model.Capability, values ...model.Capability) []model.Capability {
	for _, capability := range values {
		capabilities = appendModelListCapability(capabilities, capability)
	}
	return capabilities
}

func appendModelListCapability(capabilities []model.Capability, capability model.Capability) []model.Capability {
	if capability == "" || slices.Contains(capabilities, capability) {
		return capabilities
	}
	return append(capabilities, capability)
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
