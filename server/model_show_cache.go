package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

/*
The /api/show cache stores full api.ShowResponse values because callers use
more than capabilities: launch flows also need context length, embeddings
metadata, quantization details, remote metadata, and model-specific fields.

Local model entries are content-addressed by canonical model name, manifest
digest, and the verbose flag. The manifest digest is the freshness boundary:
if the model content changes, the digest changes, so the previous entry is not
used. Requests with System or Options overlays bypass the cache because those
overlays mutate the effective show response.

Cloud model entries are keyed by normalized cloud base model name and verbose.
They use stale-while-revalidate behavior: a warm read returns the cached
response immediately and starts a throttled background refresh for that model.
Cold cloud reads preserve existing proxy behavior. Local and cloud entries live
in separate maps, so a local "qwen3.5" and an explicit "qwen3.5:cloud" cannot
collide. The cloud suffix is request routing intent; api.ShowResponse does not
carry a model-name field to reconstruct on the way out.

Snapshots live under ~/.ollama/cache/show/. Startup loads snapshots
synchronously so warm entries are immediately available, then hydrates local
and cloud caches in the background without delaying server readiness.
*/

const (
	modelShowCloudFetchTimeout          = 3 * time.Second
	modelShowCloudReadRefreshCooldown   = 5 * time.Second
	modelShowCloudHydrationConcurrency  = 4
	modelShowLocalSnapshotFilename      = "local.json"
	modelShowCloudSnapshotFilename      = "cloud.json"
	modelShowSnapshotDirectoryComponent = "show"
)

var errModelShowNoCloud = errors.New("cloud disabled")

// modelShowCache owns local and cloud show response caches plus their snapshot
// persistence. All cached responses are cloned at read/write boundaries so
// handler-specific mutations, such as user-agent compatibility tweaks, cannot
// leak back into the cache.
type modelShowCache struct {
	mu sync.RWMutex

	local map[modelShowLocalKey]modelShowLocalEntry
	cloud map[modelShowCloudKey]modelShowCloudEntry

	cloudRefreshing           map[modelShowCloudKey]bool
	cloudNextReadRefreshAfter map[modelShowCloudKey]time.Time

	once         sync.Once
	client       *http.Client
	getModelInfo func(api.ShowRequest) (*api.ShowResponse, error)
}

// modelShowLocalKey describes the local cache freshness boundary. The
// canonical model name handles aliases/casing, Digest tracks the manifest
// content address, and Verbose separates compact metadata from tensor-heavy
// responses.
type modelShowLocalKey struct {
	Model   string
	Digest  string
	Verbose bool
}

// modelShowCloudKey intentionally excludes any local digest because cloud
// models are refreshed through SWR and normalized by cloud base model name.
type modelShowCloudKey struct {
	Model   string
	Verbose bool
}

type modelShowLocalEntry struct {
	Model    string            `json:"model"`
	Digest   string            `json:"digest"`
	Verbose  bool              `json:"verbose"`
	CachedAt time.Time         `json:"cached_at"`
	Response *api.ShowResponse `json:"response"`
}

type modelShowCloudEntry struct {
	Model    string            `json:"model"`
	Verbose  bool              `json:"verbose"`
	CachedAt time.Time         `json:"cached_at"`
	Response *api.ShowResponse `json:"response"`
}

type modelShowLocalSnapshot struct {
	Entries []modelShowLocalEntry `json:"entries"`
}

type modelShowCloudSnapshot struct {
	Entries []modelShowCloudEntry `json:"entries"`
}

func newModelShowCache() *modelShowCache {
	return &modelShowCache{
		local:                     make(map[modelShowLocalKey]modelShowLocalEntry),
		cloud:                     make(map[modelShowCloudKey]modelShowCloudEntry),
		cloudRefreshing:           make(map[modelShowCloudKey]bool),
		cloudNextReadRefreshAfter: make(map[modelShowCloudKey]time.Time),
		client:                    http.DefaultClient,
		getModelInfo:              GetModelInfo,
	}
}

// modelShowCacheable returns whether a request can use the shared show cache.
// System and Options overlays are request-specific response variants, so v1
// bypasses caching for those rather than expanding the key space.
func modelShowCacheable(req api.ShowRequest) bool {
	return req.System == "" && len(req.Options) == 0
}

// Start loads persisted snapshots immediately and kicks off non-blocking
// startup hydration. Snapshot loading is intentionally synchronous so cached
// cloud and local entries are available to the first request after Serve starts.
func (c *modelShowCache) Start(ctx context.Context) {
	c.once.Do(func() {
		slog.Debug("starting model show cache")
		c.loadSnapshots()
		go c.runStartup(ctx)
	})
}

// runStartup hydrates local and cloud caches concurrently. It is only called in
// a goroutine from Start, so manifest scans and cloud requests cannot delay the
// listener from accepting traffic.
func (c *modelShowCache) runStartup(ctx context.Context) {
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		if err := c.hydrateLocal(ctx); err != nil && !errors.Is(err, context.Canceled) {
			slog.Warn("model show local cache hydration failed", "error", err)
		}
	}()

	go func() {
		defer wg.Done()
		if err := c.hydrateCloud(ctx); err != nil {
			switch {
			case errors.Is(err, context.Canceled):
			case errors.Is(err, errModelShowNoCloud):
				slog.Debug("skipping model show cloud cache hydration because cloud is disabled")
			default:
				slog.Warn("model show cloud cache hydration failed", "error", err)
			}
		}
	}()

	wg.Wait()
}

// GetLocal returns a cached local show response when the current manifest
// digest matches. On a miss, it falls back to GetModelInfo, stores non-remote
// local responses, persists the snapshot, and returns a clone to the caller.
func (c *modelShowCache) GetLocal(req api.ShowRequest) (*api.ShowResponse, error) {
	key, err := modelShowLocalKeyForRequest(req)
	if err != nil {
		return nil, err
	}

	if resp, ok := c.getLocal(key); ok {
		return resp, nil
	}

	req.Model = key.Model
	resp, err := c.getModelInfo(req)
	if err != nil {
		return nil, err
	}

	if resp.RemoteHost == "" {
		c.setLocal(key, resp)
		if err := c.persistLocalSnapshot(); err != nil {
			slog.Warn("failed to persist local model show snapshot", "error", err)
		}
	}

	return cloneShowResponse(resp), nil
}

// GetCloudSWR returns a cached cloud show response and triggers a throttled
// background refresh. The boolean is false on a cold miss so callers can
// preserve existing synchronous proxy behavior.
func (c *modelShowCache) GetCloudSWR(ctx context.Context, req api.ShowRequest) (*api.ShowResponse, bool) {
	key := modelShowCloudKeyForModel(req.Model, req.Verbose)
	resp, ok := c.getCloud(key)
	if !ok {
		return nil, false
	}

	c.triggerCloudRefreshOnRead(ctx, key)
	return resp, true
}

func (c *modelShowCache) getLocal(key modelShowLocalKey) (*api.ShowResponse, bool) {
	c.mu.RLock()
	entry, ok := c.local[key]
	c.mu.RUnlock()
	if !ok || entry.Response == nil {
		return nil, false
	}
	return cloneShowResponse(entry.Response), true
}

func (c *modelShowCache) setLocal(key modelShowLocalKey, resp *api.ShowResponse) {
	c.mu.Lock()
	c.local[key] = modelShowLocalEntry{
		Model:    key.Model,
		Digest:   key.Digest,
		Verbose:  key.Verbose,
		CachedAt: time.Now(),
		Response: cloneShowResponse(resp),
	}
	c.mu.Unlock()
}

func (c *modelShowCache) hasLocal(key modelShowLocalKey) bool {
	c.mu.RLock()
	entry, ok := c.local[key]
	c.mu.RUnlock()
	return ok && entry.Response != nil
}

func (c *modelShowCache) hasLocalVerboseVariant(modelName string, verbose bool) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for key, entry := range c.local {
		if key.Model == modelName && key.Verbose == verbose && entry.Response != nil {
			return true
		}
	}
	return false
}

func (c *modelShowCache) getCloud(key modelShowCloudKey) (*api.ShowResponse, bool) {
	c.mu.RLock()
	entry, ok := c.cloud[key]
	c.mu.RUnlock()
	if !ok || entry.Response == nil {
		return nil, false
	}
	return cloneShowResponse(entry.Response), true
}

func (c *modelShowCache) setCloud(key modelShowCloudKey, resp *api.ShowResponse) {
	c.mu.Lock()
	c.cloud[key] = modelShowCloudEntry{
		Model:    key.Model,
		Verbose:  key.Verbose,
		CachedAt: time.Now(),
		Response: cloneShowResponse(resp),
	}
	c.mu.Unlock()
}

func (c *modelShowCache) beginCloudReadRefresh(key modelShowCloudKey) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	if c.cloudRefreshing[key] || now.Before(c.cloudNextReadRefreshAfter[key]) {
		return false
	}

	c.cloudRefreshing[key] = true
	return true
}

func (c *modelShowCache) endCloudReadRefresh(key modelShowCloudKey) {
	c.mu.Lock()
	c.cloudRefreshing[key] = false
	c.cloudNextReadRefreshAfter[key] = time.Now().Add(modelShowCloudReadRefreshCooldown)
	c.mu.Unlock()
}

// triggerCloudRefreshOnRead starts the revalidation side of SWR. The refresh
// uses context.WithoutCancel so a completed client request does not cancel the
// cache update it initiated.
func (c *modelShowCache) triggerCloudRefreshOnRead(ctx context.Context, key modelShowCloudKey) {
	if !c.beginCloudReadRefresh(key) {
		return
	}
	if ctx == nil {
		ctx = context.Background()
	}
	ctx = context.WithoutCancel(ctx)

	slog.Debug("triggering model show cloud refresh on read", "model", key.Model, "verbose", key.Verbose)
	go func() {
		defer c.endCloudReadRefresh(key)

		if err := c.refreshCloud(ctx, key); err != nil {
			switch {
			case errors.Is(err, errModelShowNoCloud):
				slog.Debug("skipping model show cloud read refresh because cloud is disabled", "model", key.Model)
			default:
				slog.Warn("model show cloud read refresh failed", "model", key.Model, "error", err)
			}
		}
	}()
}

// refreshCloud fetches and stores one cloud show response. Refresh failures are
// returned without touching the existing cached entry, which preserves stale
// data for future reads.
func (c *modelShowCache) refreshCloud(ctx context.Context, key modelShowCloudKey) error {
	if disabled, _ := internalcloud.Status(); disabled {
		return errModelShowNoCloud
	}

	resp, err := c.fetchCloudShow(ctx, key.Model, key.Verbose)
	if err != nil {
		return err
	}

	c.setCloud(key, resp)
	if err := c.persistCloudSnapshot(); err != nil {
		slog.Warn("failed to persist cloud model show snapshot", "error", err)
	}
	return nil
}

// hydrateLocal scans manifests at startup and refreshes only entries missing
// for the current digest. Non-verbose responses are hydrated for every local
// model; verbose responses are hydrated only when a verbose snapshot already
// exists, avoiding an expensive tensor walk for users who have never asked for
// verbose show data.
func (c *modelShowCache) hydrateLocal(ctx context.Context) error {
	manifests, err := manifest.Manifests(true)
	if err != nil {
		return err
	}

	changed := false
	for name, mf := range manifests {
		if err := ctx.Err(); err != nil {
			return err
		}

		if modelShowManifestIsRemote(mf) {
			continue
		}

		modelName := name.String()
		variants := []bool{false}
		if c.hasLocalVerboseVariant(modelName, true) {
			variants = append(variants, true)
		}

		for _, verbose := range variants {
			key := modelShowLocalKey{
				Model:   modelName,
				Digest:  mf.Digest(),
				Verbose: verbose,
			}
			if c.hasLocal(key) {
				continue
			}

			resp, err := c.getModelInfo(api.ShowRequest{Model: modelName, Verbose: verbose})
			if err != nil {
				slog.Warn("failed to hydrate local model show cache", "model", modelName, "verbose", verbose, "error", err)
				continue
			}
			if resp.RemoteHost != "" {
				continue
			}

			c.setLocal(key, resp)
			changed = true
		}
	}

	if changed {
		if err := c.persistLocalSnapshot(); err != nil {
			slog.Warn("failed to persist local model show snapshot after hydration", "error", err)
		}
	}
	return nil
}

// hydrateCloud refreshes cloud show entries by listing cloud tags and fetching
// /api/show for each returned model with bounded concurrency. Per-model show
// failures are logged and skipped so one bad cloud entry does not prevent the
// rest of the cache from warming.
func (c *modelShowCache) hydrateCloud(ctx context.Context) error {
	if disabled, _ := internalcloud.Status(); disabled {
		return errModelShowNoCloud
	}

	models, err := c.fetchCloudTags(ctx)
	if err != nil {
		return err
	}

	jobs := make(chan string)
	var wg sync.WaitGroup
	var changed bool
	var changedMu sync.Mutex

	worker := func() {
		defer wg.Done()
		for modelName := range jobs {
			if ctx.Err() != nil {
				continue
			}

			key := modelShowCloudKeyForModel(modelName, false)
			resp, err := c.fetchCloudShow(ctx, key.Model, key.Verbose)
			if err != nil {
				slog.Warn("failed to hydrate cloud model show cache", "model", key.Model, "error", err)
				continue
			}

			c.setCloud(key, resp)
			changedMu.Lock()
			changed = true
			changedMu.Unlock()
		}
	}

	workers := min(modelShowCloudHydrationConcurrency, max(1, len(models)))
	for range workers {
		wg.Add(1)
		go worker()
	}

sendLoop:
	for _, modelName := range models {
		select {
		case <-ctx.Done():
			break sendLoop
		case jobs <- modelName:
		}
	}
	close(jobs)
	wg.Wait()

	if err := ctx.Err(); err != nil {
		return err
	}

	changedMu.Lock()
	shouldPersist := changed
	changedMu.Unlock()
	if shouldPersist {
		if err := c.persistCloudSnapshot(); err != nil {
			slog.Warn("failed to persist cloud model show snapshot after hydration", "error", err)
		}
	}

	return nil
}

// fetchCloudTags returns de-duplicated cloud model names normalized to their
// show-cache key form. It accepts either ListModelResponse.Model or the legacy
// Name field because /api/tags responses may contain both.
func (c *modelShowCache) fetchCloudTags(ctx context.Context) ([]string, error) {
	var payload api.ListResponse
	if err := c.doCloudJSON(ctx, http.MethodGet, "/api/tags", nil, &payload); err != nil {
		return nil, err
	}

	seen := make(map[string]struct{}, len(payload.Models))
	models := make([]string, 0, len(payload.Models))
	for _, item := range payload.Models {
		name := strings.TrimSpace(item.Model)
		if name == "" {
			name = strings.TrimSpace(item.Name)
		}
		name = modelShowNormalizeCloudModel(name)
		if name == "" {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		models = append(models, name)
	}

	return models, nil
}

func (c *modelShowCache) fetchCloudShow(ctx context.Context, modelName string, verbose bool) (*api.ShowResponse, error) {
	payload := api.ShowRequest{
		Model:   modelShowNormalizeCloudModel(modelName),
		Verbose: verbose,
	}

	var resp api.ShowResponse
	if err := c.doCloudJSON(ctx, http.MethodPost, "/api/show", payload, &resp); err != nil {
		return nil, err
	}

	if resp.ModelInfo == nil {
		resp.ModelInfo = map[string]any{}
	}
	return &resp, nil
}

// doCloudJSON is the cache's direct cloud client. It mirrors the cloud proxy's
// signing and client-version behavior but uses an internal timeout because
// hydration and refreshes must not hang indefinitely.
func (c *modelShowCache) doCloudJSON(ctx context.Context, method, path string, payload any, out any) error {
	reqCtx, cancel := context.WithTimeout(ctx, modelShowCloudFetchTimeout)
	defer cancel()

	baseURL, err := url.Parse(cloudProxyBaseURL)
	if err != nil {
		return err
	}
	targetURL := baseURL.ResolveReference(&url.URL{Path: path})

	var body io.Reader
	if payload != nil {
		data, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		body = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(reqCtx, method, targetURL.String(), body)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if clientVersion := strings.TrimSpace(version.Version); clientVersion != "" {
		req.Header.Set(cloudProxyClientVersionHeader, clientVersion)
	}

	if err := cloudProxySignRequest(req.Context(), req); err != nil {
		return err
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return modelShowStatusError(resp, data)
	}

	if out == nil {
		return nil
	}
	return json.Unmarshal(data, out)
}

// modelShowStatusError preserves the important error shape from cloud
// responses, including AuthorizationError for 401s and StatusError otherwise.
func modelShowStatusError(resp *http.Response, body []byte) error {
	if resp.StatusCode == http.StatusUnauthorized {
		err := api.AuthorizationError{
			StatusCode: resp.StatusCode,
			Status:     resp.Status,
		}
		_ = json.Unmarshal(body, &err)
		if err.Status == "" {
			err.Status = resp.Status
		}
		return err
	}

	statusErr := api.StatusError{
		StatusCode: resp.StatusCode,
		Status:     resp.Status,
	}
	if err := json.Unmarshal(body, &statusErr); err != nil || statusErr.ErrorMessage == "" {
		statusErr.ErrorMessage = strings.TrimSpace(string(body))
	}
	return statusErr
}

// loadSnapshots hydrates the in-memory cache from disk. Invalid, partial, or
// missing snapshots are logged and ignored so /api/show continues to work even
// when persistence is unavailable or corrupted.
func (c *modelShowCache) loadSnapshots() {
	c.loadLocalSnapshot()
	c.loadCloudSnapshot()
}

func (c *modelShowCache) loadLocalSnapshot() {
	path, err := modelShowSnapshotPath(modelShowLocalSnapshotFilename)
	if err != nil {
		slog.Warn("failed to resolve local model show snapshot path", "error", err)
		return
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			slog.Warn("failed to read local model show snapshot", "path", path, "error", err)
		}
		return
	}

	var snap modelShowLocalSnapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		slog.Warn("failed to parse local model show snapshot", "path", path, "error", err)
		return
	}

	count := 0
	c.mu.Lock()
	for _, entry := range snap.Entries {
		if entry.Model == "" || entry.Digest == "" || entry.Response == nil {
			continue
		}
		if entry.Response.ModelInfo == nil {
			entry.Response.ModelInfo = map[string]any{}
		}
		key := modelShowLocalKey{Model: entry.Model, Digest: entry.Digest, Verbose: entry.Verbose}
		entry.Response = cloneShowResponse(entry.Response)
		c.local[key] = entry
		count++
	}
	c.mu.Unlock()

	slog.Debug("loaded local model show snapshot", "path", path, "count", count)
}

func (c *modelShowCache) loadCloudSnapshot() {
	path, err := modelShowSnapshotPath(modelShowCloudSnapshotFilename)
	if err != nil {
		slog.Warn("failed to resolve cloud model show snapshot path", "error", err)
		return
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			slog.Warn("failed to read cloud model show snapshot", "path", path, "error", err)
		}
		return
	}

	var snap modelShowCloudSnapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		slog.Warn("failed to parse cloud model show snapshot", "path", path, "error", err)
		return
	}

	count := 0
	c.mu.Lock()
	for _, entry := range snap.Entries {
		entry.Model = modelShowNormalizeCloudModel(entry.Model)
		if entry.Model == "" || entry.Response == nil {
			continue
		}
		if entry.Response.ModelInfo == nil {
			entry.Response.ModelInfo = map[string]any{}
		}
		key := modelShowCloudKey{Model: entry.Model, Verbose: entry.Verbose}
		entry.Response = cloneShowResponse(entry.Response)
		c.cloud[key] = entry
		count++
	}
	c.mu.Unlock()

	slog.Debug("loaded cloud model show snapshot", "path", path, "count", count)
}

func (c *modelShowCache) persistLocalSnapshot() error {
	entries := c.localSnapshotEntries()
	return writeModelShowSnapshot(modelShowLocalSnapshotFilename, modelShowLocalSnapshot{Entries: entries})
}

func (c *modelShowCache) persistCloudSnapshot() error {
	entries := c.cloudSnapshotEntries()
	return writeModelShowSnapshot(modelShowCloudSnapshotFilename, modelShowCloudSnapshot{Entries: entries})
}

func (c *modelShowCache) localSnapshotEntries() []modelShowLocalEntry {
	c.mu.RLock()
	entries := make([]modelShowLocalEntry, 0, len(c.local))
	for _, entry := range c.local {
		entry.Response = cloneShowResponse(entry.Response)
		entries = append(entries, entry)
	}
	c.mu.RUnlock()

	slices.SortFunc(entries, func(a, b modelShowLocalEntry) int {
		if c := strings.Compare(a.Model, b.Model); c != 0 {
			return c
		}
		if c := strings.Compare(a.Digest, b.Digest); c != 0 {
			return c
		}
		return compareBool(a.Verbose, b.Verbose)
	})
	return entries
}

func (c *modelShowCache) cloudSnapshotEntries() []modelShowCloudEntry {
	c.mu.RLock()
	entries := make([]modelShowCloudEntry, 0, len(c.cloud))
	for _, entry := range c.cloud {
		entry.Response = cloneShowResponse(entry.Response)
		entries = append(entries, entry)
	}
	c.mu.RUnlock()

	slices.SortFunc(entries, func(a, b modelShowCloudEntry) int {
		if c := strings.Compare(a.Model, b.Model); c != 0 {
			return c
		}
		return compareBool(a.Verbose, b.Verbose)
	})
	return entries
}

// writeModelShowSnapshot writes one snapshot using a temp file, fsync, close,
// and rename sequence. This matches the recommendations cache pattern and
// avoids leaving a partially written JSON file at the final path.
func writeModelShowSnapshot(filename string, payload any) error {
	path, err := modelShowSnapshotPath(filename)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}

	tmp, err := os.CreateTemp(filepath.Dir(path), ".model-show-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}

	return os.Rename(tmpPath, path)
}

func modelShowSnapshotPath(filename string) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "cache", modelShowSnapshotDirectoryComponent, filename), nil
}

// modelShowLocalKeyForRequest normalizes a local show request to the canonical
// on-disk model name and current manifest digest. The manifest digest is what
// makes local entries content-addressable.
func modelShowLocalKeyForRequest(req api.ShowRequest) (modelShowLocalKey, error) {
	name := model.ParseName(req.Model)
	if !name.IsValid() {
		return modelShowLocalKey{}, model.Unqualified(name)
	}
	name, err := getExistingName(name)
	if err != nil {
		return modelShowLocalKey{}, err
	}

	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return modelShowLocalKey{}, err
	}

	return modelShowLocalKey{
		Model:   name.String(),
		Digest:  mf.Digest(),
		Verbose: req.Verbose,
	}, nil
}

func modelShowCloudKeyForModel(modelName string, verbose bool) modelShowCloudKey {
	return modelShowCloudKey{
		Model:   modelShowNormalizeCloudModel(modelName),
		Verbose: verbose,
	}
}

// modelShowNormalizeCloudModel strips explicit cloud source syntax, including
// legacy "-cloud" tags, so :cloud and -cloud forms share a cache entry.
func modelShowNormalizeCloudModel(modelName string) string {
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		return ""
	}
	if base, stripped := modelref.StripCloudSourceTag(modelName); stripped {
		return strings.TrimSpace(base)
	}
	return modelName
}

// modelShowManifestIsRemote checks whether a manifest represents a local stub
// for a remote model. Startup hydration skips these so the local content cache
// does not store entries whose freshness is governed by cloud state.
func modelShowManifestIsRemote(mf *manifest.Manifest) bool {
	if mf == nil || mf.Config.Digest == "" {
		return false
	}

	f, err := mf.Config.Open()
	if err != nil {
		slog.Warn("failed to open manifest config while checking model show cache eligibility", "error", err)
		return false
	}
	defer f.Close()

	var cfg model.ConfigV2
	if err := json.NewDecoder(f).Decode(&cfg); err != nil {
		slog.Warn("failed to decode manifest config while checking model show cache eligibility", "error", err)
		return false
	}

	return cfg.RemoteHost != "" || cfg.RemoteModel != ""
}

// cloneShowResponse deep-copies mutable fields of api.ShowResponse before
// storing or returning cached entries. The response contains maps and slices,
// and some handlers mutate ModelInfo before writing JSON.
func cloneShowResponse(in *api.ShowResponse) *api.ShowResponse {
	if in == nil {
		return nil
	}

	out := *in
	out.Details.Families = slices.Clone(in.Details.Families)
	out.Messages = cloneMessages(in.Messages)
	out.Capabilities = slices.Clone(in.Capabilities)
	out.ModelInfo = cloneAnyMap(in.ModelInfo)
	out.ProjectorInfo = cloneAnyMap(in.ProjectorInfo)
	out.Tensors = cloneTensors(in.Tensors)
	return &out
}

func cloneMessages(in []api.Message) []api.Message {
	if in == nil {
		return nil
	}
	out := make([]api.Message, len(in))
	for i, msg := range in {
		out[i] = msg
		if msg.Images != nil {
			out[i].Images = make([]api.ImageData, len(msg.Images))
			for j, image := range msg.Images {
				out[i].Images[j] = slices.Clone(image)
			}
		}
		out[i].ToolCalls = slices.Clone(msg.ToolCalls)
	}
	return out
}

func cloneTensors(in []api.Tensor) []api.Tensor {
	if in == nil {
		return nil
	}
	out := make([]api.Tensor, len(in))
	for i, tensor := range in {
		out[i] = tensor
		out[i].Shape = slices.Clone(tensor.Shape)
	}
	return out
}

func cloneAnyMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = cloneAny(v)
	}
	return out
}

func cloneAny(v any) any {
	switch v := v.(type) {
	case map[string]any:
		return cloneAnyMap(v)
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = cloneAny(item)
		}
		return out
	case []string:
		return slices.Clone(v)
	case []bool:
		return slices.Clone(v)
	case []int:
		return slices.Clone(v)
	case []int8:
		return slices.Clone(v)
	case []int16:
		return slices.Clone(v)
	case []int32:
		return slices.Clone(v)
	case []int64:
		return slices.Clone(v)
	case []uint:
		return slices.Clone(v)
	case []uint8:
		return slices.Clone(v)
	case []uint16:
		return slices.Clone(v)
	case []uint32:
		return slices.Clone(v)
	case []uint64:
		return slices.Clone(v)
	case []float32:
		return slices.Clone(v)
	case []float64:
		return slices.Clone(v)
	default:
		return v
	}
}

func compareBool(a, b bool) int {
	switch {
	case a == b:
		return 0
	case !a && b:
		return -1
	default:
		return 1
	}
}
