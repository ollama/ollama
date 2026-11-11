package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/internal/proxy"
	modelpkg "github.com/ollama/ollama/types/model"
)

// ChatGPTModelSource supplies the same app-specific catalog to Settings and CLI.
// Discovery never pulls models, signs in, or makes inference requests.
type ChatGPTModelSource struct {
	Client          *api.Client
	Recommendations func(context.Context) ([]api.ModelRecommendation, error)
	CloudModels     func(context.Context) ([]string, error)
	AccessState     func(context.Context) (proxy.ClaudeDesktopAccessState, error)
	Attempts        int
	RetryWait       time.Duration
}

func NewChatGPTModelSource(client *api.Client) *ChatGPTModelSource {
	baseURL := strings.TrimRight(os.Getenv("OLLAMA_DOT_COM_URL"), "/")
	if baseURL == "" {
		baseURL = "https://ollama.com"
	}
	httpClient := &http.Client{Timeout: 3 * time.Second}
	load := func(ctx context.Context, path string, result any) error {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+path, nil)
		if err != nil {
			return err
		}
		query := req.URL.Query()
		query.Set("ts", strconv.FormatInt(time.Now().Unix(), 10))
		req.URL.RawQuery = query.Encode()
		signature, err := auth.Sign(ctx, []byte(fmt.Sprintf("%s,%s", req.Method, req.URL.RequestURI())))
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", signature)
		resp, err := httpClient.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("ChatGPT model discovery: status %d", resp.StatusCode)
		}
		return json.NewDecoder(io.LimitReader(resp.Body, (1<<20)+1)).Decode(result)
	}
	return &ChatGPTModelSource{
		Client: client, Attempts: 20, RetryWait: 250 * time.Millisecond,
		Recommendations: func(ctx context.Context) ([]api.ModelRecommendation, error) {
			var response api.ModelRecommendationsResponse
			err := load(ctx, "/api/experimental/model-recommendations?app=codex-desktop", &response)
			return response.Recommendations, err
		},
		CloudModels: func(ctx context.Context) ([]string, error) {
			var response api.ListResponse
			if err := load(ctx, "/api/tags", &response); err != nil {
				return nil, err
			}
			var names []string
			for _, model := range response.Models {
				for _, name := range []string{model.Name, model.Model} {
					if name = strings.TrimSpace(name); name != "" {
						names = append(names, name)
					}
				}
			}
			models := proxy.ClaudeDesktopModelsFromCloudInventory(names)
			names = nil
			for _, model := range models {
				names = append(names, model.OllamaModel)
			}
			return names, nil
		},
		AccessState: func(ctx context.Context) (proxy.ClaudeDesktopAccessState, error) {
			return resolveChatGPTAccessState(ctx, client.CloudStatusExperimental, client.Whoami)
		},
	}
}

func (source *ChatGPTModelSource) Inventory(ctx context.Context) (ChatGPTModelInventory, error) {
	if err := ctx.Err(); err != nil {
		return ChatGPTModelInventory{}, err
	}
	recommendations, err := source.Recommendations(ctx)
	if err != nil {
		slog.Debug("could not load ChatGPT model recommendations", "error", err)
	}
	var access proxy.ClaudeDesktopAccessState
	accessKnown := false
	var last ChatGPTModelInventory
	for attempt := range max(source.Attempts, 1) {
		if !accessKnown {
			if resolved, err := source.AccessState(ctx); err == nil {
				access, accessKnown = resolved, true
			}
		}
		var listed []api.ListModelResponse
		listKnown := false
		if response, err := source.Client.List(ctx); err == nil {
			listed, listKnown = response.Models, true
		}
		cloud, cloudErr := source.CloudModels(ctx)
		last = BuildChatGPTModelInventory(recommendations, listed, cloud, access, accessKnown, listKnown, cloudErr == nil)
		if err := ctx.Err(); err != nil {
			return ChatGPTModelInventory{}, err
		}
		if len(last.Available) > 0 && (accessKnown || attempt+1 == max(source.Attempts, 1)) {
			return last, nil
		}
		if attempt+1 == max(source.Attempts, 1) {
			break
		}
		timer := time.NewTimer(source.RetryWait)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ChatGPTModelInventory{}, ctx.Err()
		case <-timer.C:
		}
	}
	if len(last.Catalog) > 0 {
		return last, nil
	}
	return ChatGPTModelInventory{}, errors.New("no Ollama models are available for ChatGPT")
}

func (source *ChatGPTModelSource) Resolve(ctx context.Context, selected []string, reconnect bool) (string, []LaunchModel, error) {
	inventory, err := source.Inventory(ctx)
	if err != nil {
		return "", nil, err
	}
	defaults := ChatGPTDefaultModels(inventory)
	if len(selected) == 0 {
		selected = ChatGPTModelNames(defaults)
	}
	var models []LaunchModel
	if reconnect {
		_, models, err = ReconcileChatGPTModels(selected, inventory.Available, defaults)
	} else {
		_, models, err = SelectChatGPTModels(selected, inventory.Available)
	}
	if err != nil {
		return "", nil, err
	}
	return chatGPTPreferredPrimary(inventory.DefaultPrimary, models), source.Hydrate(ctx, models), nil
}

func (source *ChatGPTModelSource) Hydrate(ctx context.Context, models []LaunchModel) []LaunchModel {
	hydrated := cloneLaunchModels(models)
	for i := range hydrated {
		response, err := source.Client.Show(ctx, &api.ShowRequest{Model: hydrated[i].Name})
		if err != nil {
			continue
		}
		if len(response.Capabilities) > 0 {
			hydrated[i].Capabilities = append([]modelpkg.Capability(nil), response.Capabilities...)
		}
		if response.Details.Family != "" || len(response.Details.Families) > 0 {
			hydrated[i].Details = response.Details
		}
		if response.Thinking.Valid() {
			hydrated[i].Thinking = response.Thinking.Clone()
		}
	}
	return hydrated
}

func resolveChatGPTAccessState(ctx context.Context, status func(context.Context) (*api.StatusResponse, error), whoami func(context.Context) (*api.UserResponse, error)) (proxy.ClaudeDesktopAccessState, error) {
	state := proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudUnknown, Account: proxy.ClaudeDesktopAccountUnknown}
	checkCtx, cancel := context.WithTimeout(ctx, time.Second)
	cloud, err := status(checkCtx)
	cancel()
	if err != nil {
		return state, fmt.Errorf("check Ollama cloud: %w", err)
	}
	if cloud != nil && cloud.Cloud.Disabled {
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
	return state, nil
}
