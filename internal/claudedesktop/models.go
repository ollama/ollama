// Package claudedesktop resolves saved Claude Desktop routes for both server
// and launcher. It never edits Claude's profile or creates inference requests.
package claudedesktop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/internal/launchconfig"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/internal/proxy"
)

const IntegrationName = "claude-desktop"

type catalogLoad struct {
	done   chan struct{}
	models []proxy.ClaudeDesktopModel
}

type Resolver struct {
	Client       *api.Client
	HTTP         *http.Client
	CloudURL     string
	Load         func(string) (*launchconfig.Integration, error)
	Sign         func(context.Context, []byte) (string, error)
	mu           sync.Mutex
	catalog      []proxy.ClaudeDesktopModel
	updated      time.Time
	catalogState proxy.ClaudeDesktopAccessState
	loading      map[proxy.ClaudeDesktopAccessState]*catalogLoad
}

func NewResolver(endpoint *url.URL) *Resolver {
	cloudURL := os.Getenv("OLLAMA_DOT_COM_URL")
	if cloudURL == "" {
		cloudURL = "https://ollama.com"
	}
	return &Resolver{
		Client: api.NewClient(endpoint, http.DefaultClient),
		HTTP:   &http.Client{Timeout: 3 * time.Second}, CloudURL: cloudURL,
		Load: launchconfig.LoadIntegration, Sign: auth.Sign,
	}
}

func (r *Resolver) Models(ctx context.Context, _ []proxy.ClaudeDesktopModel) ([]proxy.ClaudeDesktopModel, error) {
	cfg, err := r.Load(IntegrationName)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return r.Resolve(ctx, cfg)
}

// DefaultMappings uses app-specific recommendations where supplied. Other
// slots inherit the Sonnet default, or the standard compatibility default if
// the recommendations omit it. It does not change the saved model selection.
func (r *Resolver) DefaultMappings(ctx context.Context) (map[string]string, error) {
	models := r.cloudCatalog(ctx)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	defaults := proxy.DefaultClaudeDesktopMappingsForModels(models)
	standard := defaults["claude-sonnet-5"]
	if standard == "" {
		standard = proxy.DefaultClaudeDesktopMappings()["claude-sonnet-5"]
	}
	for _, route := range proxy.ClaudeDesktopRoutes() {
		if defaults[route.ID] == "" {
			defaults[route.ID] = standard
		}
	}
	return defaults, nil
}

func (r *Resolver) Resolve(ctx context.Context, cfg *launchconfig.Integration) ([]proxy.ClaudeDesktopModel, error) {
	if cfg == nil {
		return nil, nil
	}
	names := cfg.Models
	if len(cfg.Aliases) > 0 {
		names = nil
		known := make(map[string]bool)
		for _, route := range proxy.ClaudeDesktopRoutes() {
			known[route.ID] = true
		}
		for route, name := range cfg.Aliases {
			if !known[route] {
				return nil, fmt.Errorf("unknown Claude Desktop route %q", route)
			}
			if strings.TrimSpace(name) != "" {
				names = append(names, strings.TrimSpace(name))
			}
		}
	}
	if len(names) == 0 {
		return nil, nil
	}
	for _, name := range names {
		if strings.TrimSpace(name) == "" || name == "Default Ollama model" {
			return nil, errors.New("Claude Desktop has no valid saved model")
		}
	}
	localNames, err := r.LocalModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("list installed Claude models: %w", err)
	}
	installed := make(map[string]bool)
	for _, name := range localNames {
		installed[name] = true
	}
	needsCloud := false
	for _, name := range names {
		if modelref.HasExplicitCloudSource(name) {
			needsCloud = true
		} else if !installed[name] {
			for _, model := range proxy.DefaultClaudeDesktopModels() {
				needsCloud = needsCloud || model.Cloud && (model.Name == name || model.OllamaModel == name)
			}
		}
	}
	var available []proxy.ClaudeDesktopModel
	if needsCloud {
		available = r.cloudCatalog(ctx)
	}
	// Installed local names override any similarly named recommendation.
	for _, name := range names {
		if installed[name] && !modelref.HasExplicitCloudSource(name) {
			available = append(available, proxy.SelectClaudeDesktopModels(nil, []string{name})...)
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(cfg.Aliases) > 0 {
		return proxy.MapClaudeDesktopModels(available, cfg.Aliases), nil
	}
	return proxy.SelectClaudeDesktopModels(available, names), nil
}

func (r *Resolver) LocalModels(ctx context.Context) ([]string, error) {
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	response, err := r.Client.List(ctx)
	if err != nil {
		return nil, err
	}
	return LocalModelNames(response), nil
}

func LocalModelNames(response *api.ListResponse) []string {
	var names []string
	if response == nil {
		return names
	}
	for _, model := range response.Models {
		if model.RemoteModel != "" || model.RemoteHost != "" || modelref.HasExplicitCloudSource(model.Name) || modelref.HasExplicitCloudSource(model.Model) {
			continue
		}
		for _, name := range []string{model.Name, model.Model} {
			if name != "" {
				names = append(names, name, strings.TrimSuffix(name, ":latest"))
			}
		}
	}
	return names
}

func ResolveAccessState(ctx context.Context, status func(context.Context) (*api.StatusResponse, error), whoami func(context.Context) (*api.UserResponse, error)) (proxy.ClaudeDesktopAccessState, error) {
	state := proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudUnknown, Account: proxy.ClaudeDesktopAccountUnknown}
	checkCtx, cancel := context.WithTimeout(ctx, time.Second)
	cloud, err := status(checkCtx)
	cancel()
	if err != nil {
		return state, fmt.Errorf("check Ollama cloud: %w", err)
	}
	if cloud == nil {
		return state, errors.New("Ollama cloud status is unavailable")
	}
	if cloud.Cloud.Disabled {
		state.Cloud = proxy.ClaudeDesktopCloudOff
		return state, nil
	}
	state.Cloud = proxy.ClaudeDesktopCloudOn
	checkCtx, cancel = context.WithTimeout(ctx, time.Second)
	defer cancel()
	user, err := whoami(checkCtx)
	if err != nil {
		var authErr api.AuthorizationError
		if errors.As(err, &authErr) && authErr.StatusCode == http.StatusUnauthorized {
			state.Account = proxy.ClaudeDesktopAccountSignedOut
			return state, nil
		}
		return state, err
	}
	if user == nil || strings.TrimSpace(user.Name) == "" {
		state.Account = proxy.ClaudeDesktopAccountSignedOut
		return state, nil
	}
	state.Account, state.Plan = proxy.ClaudeDesktopAccountSignedIn, strings.TrimSpace(user.Plan)
	state.AccountID = "name:" + strings.TrimSpace(user.Name)
	if user.ID != uuid.Nil {
		state.AccountID = user.ID.String()
	}
	return state, nil
}

func (r *Resolver) AccessState(ctx context.Context) (proxy.ClaudeDesktopAccessState, error) {
	return ResolveAccessState(ctx, r.Client.CloudStatusExperimental, r.Client.Whoami)
}

func (r *Resolver) Validate(ctx context.Context, models []proxy.ClaudeDesktopModel) error {
	if len(models) == 0 {
		return errors.New("select at least one Claude Desktop model")
	}
	local, err := r.LocalModels(ctx)
	if err != nil {
		return err
	}
	installed := make(map[string]bool)
	for _, name := range local {
		installed[name] = true
	}
	state := proxy.ClaudeDesktopAccessState{}
	for _, model := range models {
		if model.Cloud {
			state, err = r.AccessState(ctx)
			if err != nil {
				return err
			}
			break
		}
	}
	for _, model := range models {
		access := proxy.EvaluateClaudeDesktopModelAccess(model, state, installed[model.OllamaModel], true)
		if access.Availability != proxy.ClaudeDesktopAvailabilityAvailable {
			return fmt.Errorf("Claude model %s is unavailable: %s", model.OllamaModel, access.Reason)
		}
	}
	return nil
}

func (r *Resolver) signedRequest(ctx context.Context, endpoint string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	query := req.URL.Query()
	query.Set("ts", strconv.FormatInt(time.Now().Unix(), 10))
	req.URL.RawQuery = query.Encode()
	signature, err := r.Sign(ctx, []byte(fmt.Sprintf("%s,%s", req.Method, req.URL.RequestURI())))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", signature)
	return req, nil
}

func (r *Resolver) cloudCatalog(ctx context.Context) []proxy.ClaudeDesktopModel {
	models := proxy.UnverifyClaudeDesktopCloudEntitlements(proxy.DefaultClaudeDesktopModels())
	state, stateErr := r.AccessState(ctx)
	r.mu.Lock()
	if stateErr != nil || state.Cloud != proxy.ClaudeDesktopCloudOn || state.Account != proxy.ClaudeDesktopAccountSignedIn {
		r.catalog, r.updated = nil, time.Time{}
		r.mu.Unlock()
		return models
	}
	if state == r.catalogState && time.Since(r.updated) < time.Minute {
		cached := append([]proxy.ClaudeDesktopModel(nil), r.catalog...)
		r.mu.Unlock()
		return cached
	}
	pending := r.loading[state]
	if pending == nil {
		if r.loading == nil {
			r.loading = make(map[proxy.ClaudeDesktopAccessState]*catalogLoad)
		}
		pending = &catalogLoad{done: make(chan struct{})}
		r.loading[state] = pending
		go r.loadCloudCatalog(context.WithoutCancel(ctx), state, pending)
	}
	r.mu.Unlock()
	select {
	case <-ctx.Done():
		return models
	case <-pending.done:
		return append([]proxy.ClaudeDesktopModel(nil), pending.models...)
	}
}

func (r *Resolver) loadCloudCatalog(ctx context.Context, state proxy.ClaudeDesktopAccessState, pending *catalogLoad) {
	// The two cloud lookups share a bounded lifetime, independent of any
	// caller's cancellation. Every caller can stop waiting independently.
	ctx, cancel := context.WithTimeout(ctx, 6*time.Second)
	defer cancel()
	models := proxy.UnverifyClaudeDesktopCloudEntitlements(proxy.DefaultClaudeDesktopModels())
	loadedRecommendations, loadedInventory := false, false
	if req, err := r.signedRequest(ctx, strings.TrimRight(r.CloudURL, "/")+"/api/experimental/model-recommendations?app=claude-desktop"); err == nil {
		if recommended, err := proxy.FetchClaudeDesktopModels(r.HTTP, req); err == nil {
			models = recommended
			loadedRecommendations = true
		}
	}
	if req, err := r.signedRequest(ctx, strings.TrimRight(r.CloudURL, "/")+"/api/tags"); err == nil {
		if response, err := r.HTTP.Do(req); err == nil {
			var inventory api.ListResponse
			if response.StatusCode == http.StatusOK && json.NewDecoder(http.MaxBytesReader(nil, response.Body, 1<<20)).Decode(&inventory) == nil {
				var names []string
				for _, model := range inventory.Models {
					names = append(names, model.Name, model.Model)
				}
				cloud := proxy.ClaudeDesktopModelsFromCloudInventory(names)
				models = proxy.VerifyClaudeDesktopModelsWithCloudInventory(models, cloud)
				models = append(cloud, models...)
				loadedInventory = true
			}
			_ = response.Body.Close()
		}
	}
	if loadedRecommendations || loadedInventory {
		// Account identity can change while the signed requests are in flight.
		// Discard results that no longer belong to the state starting this load.
		if current, err := r.AccessState(ctx); err != nil || current != state {
			models = proxy.UnverifyClaudeDesktopCloudEntitlements(proxy.DefaultClaudeDesktopModels())
			loadedRecommendations, loadedInventory = false, false
		}
	}
	r.mu.Lock()
	if loadedRecommendations && loadedInventory {
		r.catalog, r.updated, r.catalogState = models, time.Now(), state
	}
	pending.models = append([]proxy.ClaudeDesktopModel(nil), models...)
	delete(r.loading, state)
	close(pending.done)
	r.mu.Unlock()
}
