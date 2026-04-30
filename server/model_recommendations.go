package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

const (
	modelRecommendationsURL = "https://ollama.com/api/experimental/model-recommendations"
)

var (
	modelRecommendationsRefreshInterval     = 4 * time.Hour
	modelRecommendationsFetchTimeout        = 3 * time.Second
	modelRecommendationsReadRefreshCooldown = 5 * time.Second
	modelRecommendationsBackoffSteps        = []time.Duration{
		5 * time.Minute,
		15 * time.Minute,
		time.Hour,
		4 * time.Hour,
	}

	errModelRecommendationsNoCloud = errors.New("cloud disabled")
)

type modelRecommendationsCache struct {
	mu                   sync.RWMutex
	recommendations      []api.ModelRecommendation
	refreshing           bool
	nextReadRefreshAfter time.Time
	once                 sync.Once
	client               *http.Client
}

func newModelRecommendationsCache() *modelRecommendationsCache {
	return &modelRecommendationsCache{
		recommendations: cloneModelRecommendations(defaultModelRecommendations),
		client:          http.DefaultClient,
	}
}

func (c *modelRecommendationsCache) Start(ctx context.Context) {
	c.once.Do(func() {
		slog.Debug("starting model recommendations cache",
			"default_recommendations", len(defaultModelRecommendations),
			"refresh_interval", modelRecommendationsRefreshInterval.String(),
			"fetch_timeout", modelRecommendationsFetchTimeout.String(),
		)
		go c.run(ctx)
	})
}

func (c *modelRecommendationsCache) Get() []api.ModelRecommendation {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return cloneModelRecommendations(c.recommendations)
}

func (c *modelRecommendationsCache) GetSWR(ctx context.Context) []api.ModelRecommendation {
	recs := c.Get()
	c.triggerRefreshOnRead(ctx)
	return recs
}

func (c *modelRecommendationsCache) set(recs []api.ModelRecommendation) {
	c.mu.Lock()
	c.recommendations = cloneModelRecommendations(recs)
	c.mu.Unlock()
}

func (c *modelRecommendationsCache) beginRefresh() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.refreshing {
		return false
	}
	c.refreshing = true
	return true
}

func (c *modelRecommendationsCache) beginReadRefresh() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := time.Now()
	if c.refreshing || now.Before(c.nextReadRefreshAfter) {
		return false
	}

	c.refreshing = true
	return true
}

func (c *modelRecommendationsCache) endRefresh() {
	c.mu.Lock()
	c.refreshing = false
	c.mu.Unlock()
}

func (c *modelRecommendationsCache) endReadRefresh() {
	c.mu.Lock()
	c.refreshing = false
	c.nextReadRefreshAfter = time.Now().Add(modelRecommendationsReadRefreshCooldown)
	c.mu.Unlock()
}

func (c *modelRecommendationsCache) refreshIfIdle(ctx context.Context) (bool, error) {
	if !c.beginRefresh() {
		return false, nil
	}
	defer c.endRefresh()
	return true, c.refresh(ctx)
}

func (c *modelRecommendationsCache) triggerRefreshOnRead(ctx context.Context) {
	if !c.beginReadRefresh() {
		return
	}
	if ctx == nil {
		ctx = context.Background()
	}
	ctx = context.WithoutCancel(ctx)

	slog.Debug("triggering model recommendations refresh on read")
	go func() {
		defer c.endReadRefresh()

		if err := c.refresh(ctx); err != nil {
			switch {
			case errors.Is(err, errModelRecommendationsNoCloud):
				slog.Debug("skipping model recommendations read refresh because cloud is disabled")
			default:
				slog.Warn("model recommendations read refresh failed", "error", err)
			}
		}
	}()
}

func (c *modelRecommendationsCache) run(ctx context.Context) {
	c.loadSnapshot()

	failures := 0
	for {
		started, err := c.refreshIfIdle(ctx)
		switch {
		case !started:
			failures = 0
			slog.Debug("skipping timer model recommendations refresh because refresh is already running")
		case err == nil:
			failures = 0
		case errors.Is(err, errModelRecommendationsNoCloud):
			failures = 0
			slog.Debug("skipping model recommendations refresh because cloud is disabled")
		default:
			failures++
			slog.Warn("model recommendations refresh failed", "error", err)
		}

		var wait time.Duration
		if failures == 0 {
			wait = withJitter(modelRecommendationsRefreshInterval)
		} else {
			wait = withJitter(modelRecommendationsBackoffSteps[min(failures-1, len(modelRecommendationsBackoffSteps)-1)])
		}
		slog.Info("model recommendations cache sleep scheduled", "wait", wait.String(), "consecutive_failures", failures)

		select {
		case <-ctx.Done():
			slog.Debug("stopping model recommendations cache")
			return
		case <-time.After(wait):
		}
	}
}

func (c *modelRecommendationsCache) refresh(ctx context.Context) error {
	if envconfig.NoCloud() {
		return errModelRecommendationsNoCloud
	}
	slog.Debug("refreshing model recommendations from remote", "url", modelRecommendationsURL)

	reqCtx, cancel := context.WithTimeout(ctx, modelRecommendationsFetchTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, modelRecommendationsURL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var payload api.ModelRecommendationsResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return err
	}

	recs, err := validateModelRecommendations(payload.Recommendations)
	if err != nil {
		return err
	}

	c.set(recs)
	slog.Debug("model recommendations refreshed", "count", len(recs))
	if err := c.persistSnapshot(recs); err != nil {
		slog.Warn("failed to persist model recommendations snapshot", "error", err)
	}
	return nil
}

func (c *modelRecommendationsCache) loadSnapshot() {
	path, err := modelRecommendationsSnapshotPath()
	if err != nil {
		slog.Warn("failed to resolve model recommendations snapshot path", "error", err)
		return
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			slog.Warn("failed to read model recommendations snapshot", "path", path, "error", err)
		} else {
			slog.Debug("model recommendations snapshot not found", "path", path)
		}
		return
	}

	var snap api.ModelRecommendationsResponse
	if err := json.Unmarshal(data, &snap); err != nil {
		slog.Warn("failed to parse model recommendations snapshot", "path", path, "error", err)
		return
	}

	recs, err := validateModelRecommendations(snap.Recommendations)
	if err != nil {
		slog.Warn("ignoring invalid model recommendations snapshot", "path", path, "error", err)
		return
	}

	c.set(recs)
	slog.Debug("loaded model recommendations snapshot", "path", path, "count", len(recs))
}

func (c *modelRecommendationsCache) persistSnapshot(recs []api.ModelRecommendation) error {
	path, err := modelRecommendationsSnapshotPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	payload := api.ModelRecommendationsResponse{Recommendations: recs}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}

	tmp, err := os.CreateTemp(filepath.Dir(path), ".model-recommendations-*.tmp")
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

	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	slog.Debug("persisted model recommendations snapshot", "path", path, "count", len(recs))
	return nil
}

func modelRecommendationsSnapshotPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "cache", "model-recommendations.json"), nil
}

func validateModelRecommendations(recs []api.ModelRecommendation) ([]api.ModelRecommendation, error) {
	if len(recs) == 0 {
		return nil, errors.New("empty recommendations")
	}

	seen := make(map[string]struct{}, len(recs))
	valid := make([]api.ModelRecommendation, 0, len(recs))
	for _, rec := range recs {
		rec.Model = strings.TrimSpace(rec.Model)
		rec.Description = strings.TrimSpace(rec.Description)

		if rec.Model == "" {
			return nil, errors.New("recommendation missing model")
		}
		if _, ok := seen[rec.Model]; ok {
			return nil, fmt.Errorf("duplicate recommendation %q", rec.Model)
		}
		seen[rec.Model] = struct{}{}

		if isCloudRecommendation(rec.Model) && (rec.ContextLength <= 0 || rec.MaxOutputTokens <= 0) {
			slog.Warn("dropping cloud recommendation missing limits", "model", rec.Model)
			continue
		}
		valid = append(valid, rec)
	}

	if len(valid) == 0 {
		return nil, errors.New("no valid recommendations")
	}

	return valid, nil
}

func isCloudRecommendation(modelName string) bool {
	return strings.HasSuffix(modelName, ":cloud") || strings.HasSuffix(modelName, "-cloud")
}

func withJitter(d time.Duration) time.Duration {
	if d <= 0 {
		return d
	}
	// jitter in range [0.8x, 1.2x]
	factor := 0.8 + rand.Float64()*0.4
	return time.Duration(float64(d) * factor)
}

func cloneModelRecommendations(in []api.ModelRecommendation) []api.ModelRecommendation {
	out := make([]api.ModelRecommendation, len(in))
	copy(out, in)
	return out
}

var defaultModelRecommendations = []api.ModelRecommendation{
	{
		Model:           "kimi-k2.6:cloud",
		Description:     "State-of-the-art coding, long-horizon execution, and multimodal agent swarm capability",
		ContextLength:   262_144,
		MaxOutputTokens: 262_144,
	},
	{
		Model:           "glm-5.1:cloud",
		Description:     "Reasoning and code generation",
		ContextLength:   202_752,
		MaxOutputTokens: 131_072,
	},
	{
		Model:           "qwen3.5:cloud",
		Description:     "Reasoning, coding, and agentic tool use with vision",
		ContextLength:   262_144,
		MaxOutputTokens: 32_768,
	},
	{
		Model:           "minimax-m2.7:cloud",
		Description:     "Fast, efficient coding and real-world productivity",
		ContextLength:   204_800,
		MaxOutputTokens: 128_000,
	},
	{
		Model:       "gemma4",
		Description: "Reasoning and code generation locally",
		VRAMBytes:   12 * format.GigaByte,
	},
	{
		Model:       "qwen3.5",
		Description: "Reasoning, coding, and visual understanding locally",
		VRAMBytes:   14 * format.GigaByte,
	},
}
