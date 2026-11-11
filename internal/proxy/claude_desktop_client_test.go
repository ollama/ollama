package proxy_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/internal/claudedesktop"
	"github.com/ollama/ollama/internal/launchconfig"
	"github.com/ollama/ollama/internal/proxy"
)

func TestClaudeObserversShareColdCloudFallback(t *testing.T) {
	cloudStarted := make(chan struct{})
	cloudStopped := make(chan struct{})
	var cloudRequests atomic.Int32
	cloud := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		if cloudRequests.Add(1) == 1 {
			close(cloudStarted)
		}
		select {
		case <-r.Context().Done():
		case <-cloudStopped:
		}
	}))
	defer cloud.Close()
	defer close(cloudStopped)
	var gateway *proxy.ClaudeDesktop
	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, proxy.ClaudeDesktopPathPrefix+"/") {
			gateway.ServeHTTP(w, r)
			return
		}
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[]}`)
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":false}}`)
		case "/api/me":
			fmt.Fprint(w, `{"name":"fixture-user","plan":"pro"}`)
		default:
			t.Errorf("unexpected fixture request: %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer local.Close()
	endpoint, err := url.Parse(local.URL)
	if err != nil {
		t.Fatal(err)
	}
	mappings := map[string]string{"claude-sonnet-5": "glm-5.2:cloud"}
	resolver := claudedesktop.NewResolver(endpoint)
	resolver.CloudURL = cloud.URL
	resolver.Load = func(string) (*launchconfig.Integration, error) {
		return &launchconfig.Integration{Aliases: mappings}, nil
	}
	resolver.Sign = func(context.Context, []byte) (string, error) { return "synthetic-signature", nil }
	gateway, err = proxy.NewClaudeDesktop(proxy.ClaudeDesktopConfig{
		PathPrefix:         proxy.ClaudeDesktopPathPrefix,
		OllamaURL:          local.URL,
		RefreshModels:      resolver.Models,
		ResolveAccessState: resolver.AccessState,
		ListLocalModels:    resolver.LocalModels,
	})
	if err != nil {
		t.Fatal(err)
	}
	observer := proxy.NewClaudeDesktopClient(proxy.ClaudeDesktopBaseURL(local.URL))
	defer observer.Close(t.Context())
	second := proxy.NewClaudeDesktopClient(observer.BaseURL)
	defer second.Close(t.Context())
	models := proxy.MapClaudeDesktopModels(nil, mappings)
	firstResult := make(chan error, 1)
	go func() { firstResult <- observer.SetModelsContext(t.Context(), models) }()
	select {
	case <-cloudStarted:
	case <-time.After(time.Second):
		t.Fatal("first observer did not begin cloud discovery")
	}
	// A canceled waiter must leave promptly without canceling the active lookup.
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	started := time.Now()
	_, err = resolver.Resolve(ctx, &launchconfig.Integration{Aliases: mappings})
	cancel()
	if !errors.Is(err, context.DeadlineExceeded) || time.Since(started) > time.Second {
		t.Fatalf("queued resolver ignored cancellation: %v, elapsed %v", err, time.Since(started))
	}
	secondErr := second.SetModelsContext(t.Context(), models)
	if firstErr := <-firstResult; firstErr != nil || secondErr != nil {
		t.Fatalf("concurrent observers could not verify fallback mappings: %v / %v", firstErr, secondErr)
	}
	if got := cloudRequests.Load(); got != 2 {
		t.Fatalf("concurrent observers made %d cloud requests, want one shared pair", got)
	}
	// Cancellation of a fresh lookup must return an error, rather than fallback.
	ctx, cancel = context.WithTimeout(t.Context(), 20*time.Millisecond)
	_, err = resolver.Resolve(ctx, &launchconfig.Integration{Aliases: mappings})
	cancel()
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("canceled resolver returned fallback instead of its deadline: %v", err)
	}
	// A caller may still cancel a slow refresh before the observer's own limit.
	ctx, cancel = context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	started = time.Now()
	if _, err := observer.Status(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("caller deadline = %v", err)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("observer ignored caller deadline: %v", elapsed)
	}
}
