package claudedesktop

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/internal/launchconfig"
	"github.com/ollama/ollama/internal/proxy"
)

func TestResolverSharesDiscoveryAfterFirstCallerCancels(t *testing.T) {
	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[]}`)
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":false}}`)
		case "/api/me":
			fmt.Fprint(w, `{"name":"fixture-user","plan":"pro"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer local.Close()
	started, release := make(chan struct{}), make(chan struct{})
	releaseCatalog := sync.OnceFunc(func() { close(release) })
	var cloudRequests atomic.Int32
	cloud := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if cloudRequests.Add(1) == 1 {
			close(started)
		}
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[{"model":"test:cloud","required_plan":"pro"}]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"test"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer cloud.Close()
	defer releaseCatalog()
	endpoint, _ := url.Parse(local.URL)
	resolver := NewResolver(endpoint)
	resolver.CloudURL = cloud.URL
	resolver.Sign = func(context.Context, []byte) (string, error) { return "synthetic", nil }
	cfg := &launchconfig.Integration{Aliases: map[string]string{"claude-sonnet-5": "test:cloud"}}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	firstResult := make(chan error, 1)
	go func() {
		_, err := resolver.Resolve(ctx, cfg)
		firstResult <- err
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("first caller did not begin discovery")
	}
	secondResult := make(chan error, 1)
	go func() {
		models, err := resolver.Resolve(t.Context(), cfg)
		if err == nil {
			err = resolver.Validate(t.Context(), models)
		}
		secondResult <- err
	}()
	// Let the second caller join the active discovery before canceling its owner.
	time.Sleep(20 * time.Millisecond)
	cancel()
	if err := <-firstResult; !errors.Is(err, context.Canceled) {
		t.Fatalf("first caller ignored cancellation: %v", err)
	}
	releaseCatalog()
	if err := <-secondResult; err != nil {
		t.Fatalf("second caller could not use shared discovery: %v", err)
	}
	if got := cloudRequests.Load(); got != 2 {
		t.Fatalf("shared discovery made %d cloud requests, want one pair", got)
	}
}

func TestResolverReusesSavedLocalMappingsWithoutCloud(t *testing.T) {
	cloudCalls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			fmt.Fprint(w, `{"models":[{"name":"local-one:latest"},{"name":"local-two:latest"}]}`)
			return
		}
		cloudCalls++
		http.Error(w, "unexpected cloud lookup", 500)
	}))
	defer server.Close()
	endpoint, _ := url.Parse(server.URL)
	resolver := NewResolver(endpoint)
	resolver.Sign = func(context.Context, []byte) (string, error) {
		t.Fatal("local configuration touched signing keys")
		return "", nil
	}
	for _, test := range []struct {
		name   string
		cfg    launchconfig.Integration
		routes []string
		names  []string
	}{
		{"explicit overrides legacy", launchconfig.Integration{Models: []string{"stale:cloud"}, Aliases: map[string]string{"claude-sonnet-5": "local-one", "claude-opus-5": "local-two"}}, []string{"claude-opus-5", "claude-sonnet-5"}, []string{"local-two", "local-one"}},
		{"sparse duplicate targets", launchconfig.Integration{Aliases: map[string]string{"claude-sonnet-5": "local-one", "claude-haiku-4-5-20251001": "local-one"}}, []string{"claude-sonnet-5", "claude-haiku-4-5-20251001"}, []string{"local-one", "local-one"}},
		{"legacy ordered models", launchconfig.Integration{Models: []string{"local-two", "local-one"}}, []string{"claude-fable-5", "claude-opus-5"}, []string{"local-two", "local-one"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			models, err := resolver.Resolve(context.Background(), &test.cfg)
			if err != nil || len(models) != len(test.routes) {
				t.Fatalf("resolve = %+v, %v", models, err)
			}
			for i, model := range models {
				if model.GatewayID() != test.routes[i] || model.OllamaModel != test.names[i] || model.Cloud {
					t.Fatalf("model %d = %+v", i, model)
				}
			}
			if err := resolver.Validate(context.Background(), models); err != nil {
				t.Fatal(err)
			}
		})
	}
	models, err := resolver.Resolve(context.Background(), &launchconfig.Integration{Models: []string{"uninstalled-local"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := resolver.Validate(context.Background(), models); err == nil {
		t.Fatal("missing local model was accepted")
	}
	if cloudCalls != 0 {
		t.Fatalf("cloud calls = %d", cloudCalls)
	}
	if _, err := resolver.Resolve(context.Background(), &launchconfig.Integration{Aliases: map[string]string{"unknown-route": "local-one"}}); err == nil {
		t.Fatal("unknown route was accepted")
	}
}

func TestResolverVerifiesCloudInventoryAndFailsClosed(t *testing.T) {
	cloudEnabled := true
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[]}`)
		case "/api/status":
			fmt.Fprintf(w, `{"cloud":{"disabled":%t}}`, !cloudEnabled)
		case "/api/me":
			fmt.Fprint(w, `{"name":"test-user","plan":"pro"}`)
		default:
			t.Errorf("unexpected local path %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	cloud := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "synthetic-signature" || r.URL.Query().Get("ts") == "" {
			t.Error("missing signed request")
		}
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			if r.URL.Query().Get("app") != IntegrationName {
				t.Error("missing app recommendation policy")
			}
			fmt.Fprint(w, `{"recommendations":[{"model":"test-cloud:cloud","required_plan":"pro","max_output_tokens":8192},{"model":"other-cloud:cloud","required_plan":"pro"}],"mappings":{"claude-sonnet-5":{"model":"test-cloud:cloud"},"claude-opus-5":{"model":"other-cloud:cloud"}}}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"test-cloud"},{"name":"other-cloud"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer cloud.Close()
	endpoint, _ := url.Parse(server.URL)
	resolver := NewResolver(endpoint)
	resolver.CloudURL = cloud.URL
	resolver.Sign = func(context.Context, []byte) (string, error) { return "synthetic-signature", nil }
	models, err := resolver.Resolve(context.Background(), &launchconfig.Integration{Aliases: map[string]string{"claude-sonnet-5": "test-cloud:cloud"}})
	if err != nil || len(models) != 1 || !models[0].AccountCloud {
		t.Fatalf("cloud selection = %+v, %v", models, err)
	}
	if err := resolver.Validate(context.Background(), models); err != nil {
		t.Fatal(err)
	}
	defaults, err := resolver.DefaultMappings(context.Background())
	wantedDefaults := map[string]string{
		"claude-fable-5":            "test-cloud:cloud",
		"claude-opus-5":             "other-cloud:cloud",
		"claude-sonnet-5":           "test-cloud:cloud",
		"claude-haiku-4-5-20251001": "test-cloud:cloud",
		"claude-sonnet-4-6":         "test-cloud:cloud",
	}
	if err != nil || !maps.Equal(defaults, wantedDefaults) {
		t.Fatalf("recommended defaults = %v, %v", defaults, err)
	}
	cloudEnabled = false
	if err := resolver.Validate(context.Background(), models); err == nil {
		t.Fatal("cloud disabled state did not revoke routing")
	}
}

func TestResolverDefaultMappingsUsesExistingFallbackWithoutCloudSignIn(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/status" {
			t.Errorf("unexpected default lookup: %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
	}))
	defer server.Close()
	endpoint, _ := url.Parse(server.URL)
	resolver := NewResolver(endpoint)
	resolver.Sign = func(context.Context, []byte) (string, error) {
		t.Fatal("fallback default lookup touched signing keys")
		return "", nil
	}
	defaults, err := resolver.DefaultMappings(context.Background())
	if err != nil || len(defaults) != proxy.MaxClaudeDesktopModels {
		t.Fatalf("fallback defaults = %v, %v", defaults, err)
	}
	for _, route := range proxy.ClaudeDesktopRoutes() {
		if defaults[route.ID] != "gemma4:31b-cloud" {
			t.Fatalf("fallback for %s = %q", route.ID, defaults[route.ID])
		}
	}
}

func TestResolverRetriesCloudDiscovery(t *testing.T) {
	for _, first := range []string{"signed-out", "cloud-off", "recommendations-unavailable", "inventory-unavailable"} {
		t.Run(first, func(t *testing.T) {
			var ready atomic.Bool
			var cloudRequests atomic.Int32
			local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/tags":
					fmt.Fprint(w, `{"models":[]}`)
				case "/api/status":
					fmt.Fprintf(w, `{"cloud":{"disabled":%t}}`, !ready.Load() && first == "cloud-off")
				case "/api/me":
					if !ready.Load() && first == "signed-out" {
						http.Error(w, "signed out", http.StatusUnauthorized)
						return
					}
					fmt.Fprint(w, `{"name":"fixture-user","plan":"pro"}`)
				default:
					t.Errorf("unexpected local request %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer local.Close()
			cloud := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				cloudRequests.Add(1)
				switch r.URL.Path {
				case "/api/experimental/model-recommendations":
					if !ready.Load() && first == "recommendations-unavailable" {
						http.Error(w, "unavailable", http.StatusServiceUnavailable)
						return
					}
					fmt.Fprint(w, `{"recommendations":[{"model":"test-cloud","required_plan":"pro"}]}`)
				case "/api/tags":
					if !ready.Load() && first == "inventory-unavailable" {
						http.Error(w, "unavailable", http.StatusServiceUnavailable)
						return
					}
					fmt.Fprint(w, `{"models":[{"name":"test-cloud"}]}`)
				default:
					t.Errorf("unexpected cloud request %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer cloud.Close()
			endpoint, _ := url.Parse(local.URL)
			resolver := NewResolver(endpoint)
			resolver.CloudURL = cloud.URL
			resolver.Sign = func(context.Context, []byte) (string, error) { return "synthetic", nil }
			if _, err := resolver.DefaultMappings(context.Background()); err != nil {
				t.Fatal(err)
			}
			ready.Store(true)
			cfg := &launchconfig.Integration{Aliases: map[string]string{"claude-sonnet-5": "test-cloud:cloud"}}
			models, err := resolver.Resolve(context.Background(), cfg)
			if err != nil {
				t.Fatal(err)
			}
			if len(models) != 1 || !models[0].AccountCloud {
				t.Fatalf("retry did not refresh account inventory: %+v", models)
			}
			if err := resolver.Validate(context.Background(), models); err != nil {
				t.Fatalf("same resolver still denies retry: %v", err)
			}
			requests := cloudRequests.Load()
			if _, err := resolver.Resolve(context.Background(), cfg); err != nil {
				t.Fatal(err)
			}
			if cloudRequests.Load() != requests {
				t.Fatal("successful cloud catalog was not cached")
			}
		})
	}
}

func TestResolverUsesConfiguredCloudBase(t *testing.T) {
	expected := "http://127.0.0.1:43123/staging"
	t.Setenv("OLLAMA_DOT_COM_URL", expected)
	endpoint, _ := url.Parse("http://127.0.0.1:0")
	if got := NewResolver(endpoint).CloudURL; got != expected {
		t.Fatalf("resolver ignored cloud base override: %s", got)
	}
}

func TestResolverRefreshesInventoryAfterSamePlanAccountSwitch(t *testing.T) {
	for _, withID := range []bool{true, false} {
		t.Run(fmt.Sprintf("account_id_%t", withID), func(t *testing.T) {
			var account, cloudRequests atomic.Int32
			local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/tags":
					fmt.Fprint(w, `{"models":[]}`)
				case "/api/status":
					fmt.Fprint(w, `{"cloud":{"disabled":false}}`)
				case "/api/me":
					if withID {
						fmt.Fprintf(w, `{"id":"00000000-0000-4000-8000-%012d","name":"fixture-user","plan":"pro"}`, account.Load()+1)
					} else {
						fmt.Fprintf(w, `{"name":"fixture-user-%d","plan":"pro"}`, account.Load())
					}
				default:
					http.NotFound(w, r)
				}
			}))
			defer local.Close()
			cloud := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				cloudRequests.Add(1)
				switch r.URL.Path {
				case "/api/experimental/model-recommendations":
					fmt.Fprint(w, `{"recommendations":[{"model":"shared:cloud"}]}`)
				case "/api/tags":
					fmt.Fprintf(w, `{"models":[{"name":"private-%d"}]}`, account.Load())
				default:
					http.NotFound(w, r)
				}
			}))
			defer cloud.Close()
			endpoint, _ := url.Parse(local.URL)
			resolver := NewResolver(endpoint)
			resolver.CloudURL = cloud.URL
			resolver.Sign = func(context.Context, []byte) (string, error) { return "synthetic", nil }
			cfg := &launchconfig.Integration{Aliases: map[string]string{"claude-sonnet-5": "private-0:cloud"}}
			models, err := resolver.Resolve(context.Background(), cfg)
			if err != nil || len(models) != 1 || !models[0].AccountCloud {
				t.Fatalf("first account models = %+v, %v", models, err)
			}
			if err := resolver.Validate(context.Background(), models); err != nil {
				t.Fatal(err)
			}
			requests := cloudRequests.Load()
			account.Store(1)
			models, err = resolver.Resolve(context.Background(), cfg)
			if err != nil || cloudRequests.Load() == requests {
				t.Fatalf("account switch did not refresh inventory: %v, requests=%d", err, cloudRequests.Load())
			}
			if err := resolver.Validate(context.Background(), models); err == nil {
				t.Fatal("previous account model was accepted")
			}
			requests = cloudRequests.Load()
			cfg.Aliases["claude-sonnet-5"] = "private-1:cloud"
			models, err = resolver.Resolve(context.Background(), cfg)
			if err != nil || len(models) != 1 || !models[0].AccountCloud || cloudRequests.Load() != requests {
				t.Fatalf("current account catalog was not reused: %+v, %v", models, err)
			}
			if err := resolver.Validate(context.Background(), models); err != nil {
				t.Fatal(err)
			}
			// Changing accounts during discovery must not put the new account's
			// inventory in the cache under the identity captured before signing.
			account.Store(0)
			signStarted, allowSign := make(chan struct{}), make(chan struct{})
			signalSign := sync.OnceFunc(func() { close(signStarted) })
			releaseSign := sync.OnceFunc(func() { close(allowSign) })
			defer releaseSign()
			resolver.Sign = func(context.Context, []byte) (string, error) {
				signalSign()
				<-allowSign
				return "synthetic", nil
			}
			firstDone := make(chan error, 1)
			go func() {
				_, err := resolver.Resolve(t.Context(), cfg)
				firstDone <- err
			}()
			select {
			case <-signStarted:
			case <-time.After(time.Second):
				t.Fatal("discovery did not begin signing")
			}
			account.Store(1)
			releaseSign()
			if err := <-firstDone; err != nil {
				t.Fatal(err)
			}
			account.Store(0)
			requests = cloudRequests.Load()
			models, err = resolver.Resolve(t.Context(), cfg)
			if err != nil || cloudRequests.Load() == requests {
				t.Fatalf("switched inventory was reused under the old account: %v", err)
			}
			if err := resolver.Validate(t.Context(), models); err == nil {
				t.Fatal("other account's model was accepted after switching back")
			}
		})
	}
}
