package server

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/internal/proxy"
)

func newClaudeDesktopProxy() (*proxy.ClaudeDesktop, error) {
	home, err := os.UserHomeDir()
	var catalogPath string
	if err == nil {
		catalogPath = filepath.Join(home, ".ollama", proxy.ClaudeDesktopRoutingCatalogFilename)
	}
	endpoint := envconfig.ConnectableHost()
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	client := api.NewClient(endpoint, &http.Client{Transport: transport})
	return proxy.NewClaudeDesktop(proxy.ClaudeDesktopConfig{
		PathPrefix:         proxy.ClaudeDesktopPathPrefix,
		RoutingCatalogPath: catalogPath,
		OllamaURL:          endpoint.String(),
		ResolveAccessState: func(ctx context.Context) (proxy.ClaudeDesktopAccessState, error) {
			return claudeDesktopAccessState(ctx, client)
		},
		ListLocalModels: func(ctx context.Context) ([]string, error) {
			ctx, cancel := context.WithTimeout(ctx, time.Second)
			defer cancel()
			response, err := client.List(ctx)
			if err != nil {
				return nil, err
			}
			var names []string
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
			return names, nil
		},
	})
}

func claudeDesktopAccessState(ctx context.Context, client *api.Client) (proxy.ClaudeDesktopAccessState, error) {
	state := proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudUnknown, Account: proxy.ClaudeDesktopAccountUnknown}
	statusCtx, cancel := context.WithTimeout(ctx, time.Second)
	status, err := client.CloudStatusExperimental(statusCtx)
	cancel()
	if err != nil {
		return state, err
	}
	if status.Cloud.Disabled {
		state.Cloud = proxy.ClaudeDesktopCloudOff
		return state, nil
	}
	state.Cloud = proxy.ClaudeDesktopCloudOn
	accountCtx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	user, err := client.Whoami(accountCtx)
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
