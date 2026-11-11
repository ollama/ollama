package claudedesktop

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/internal/launchconfig"
)

func serveResolverCatalog(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/api/experimental/model-recommendations":
		fmt.Fprint(w, `{"recommendations":[{"model":"test:cloud","required_plan":"pro"}]}`)
	case "/api/tags":
		fmt.Fprint(w, `{"models":[{"name":"test"}]}`)
	default:
		http.NotFound(w, r)
	}
}

func newResolverFixture(t *testing.T, local map[string]http.HandlerFunc, cloud http.HandlerFunc) *Resolver {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if handler := local[r.URL.Path]; handler != nil {
			handler(w, r)
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
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	endpoint, _ := url.Parse(server.URL)
	if cloud == nil {
		cloud = serveResolverCatalog
	}
	server = httptest.NewServer(http.StripPrefix("/cloud", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "synthetic" || r.URL.Query().Get("ts") == "" ||
			r.URL.Path == "/api/experimental/model-recommendations" && r.URL.Query().Get("app") != IntegrationName {
			t.Error("missing signed app discovery request")
		}
		cloud(w, r)
	})))
	t.Cleanup(server.Close)
	t.Setenv("OLLAMA_DOT_COM_URL", server.URL+"/cloud")
	resolver := NewResolver(endpoint)
	resolver.Sign = func(context.Context, []byte) (string, error) { return "synthetic", nil }
	return resolver
}

func TestResolverSharesCloudDiscovery(t *testing.T) {
	for _, cancelFirst := range []bool{false, true} {
		t.Run(fmt.Sprintf("cancel_first_%t", cancelFirst), func(t *testing.T) {
			started, release := make(chan struct{}), make(chan struct{})
			unblock := sync.OnceFunc(func() { close(release) })
			var requests atomic.Int32
			resolver := newResolverFixture(t, nil, func(w http.ResponseWriter, r *http.Request) {
				if requests.Add(1) == 1 {
					close(started)
				}
				select {
				case <-release:
					serveResolverCatalog(w, r)
				case <-r.Context().Done():
				}
			})
			t.Cleanup(unblock)
			if !cancelFirst {
				resolver.HTTP.Timeout = 200 * time.Millisecond
			}
			cfg := &launchconfig.Integration{Aliases: map[string]string{"claude-sonnet-5": "test:cloud"}}
			type result struct {
				verified bool
				err      error
			}
			resolve := func(ctx context.Context) result {
				models, err := resolver.Resolve(ctx, cfg)
				if err == nil && (len(models) != 1 || models[0].OllamaModel != "test:cloud") {
					err = fmt.Errorf("unexpected routes: %+v", models)
				}
				return result{len(models) == 1 && models[0].AccountCloud, err}
			}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			first, second := make(chan result, 1), make(chan result, 1)
			go func() { first <- resolve(ctx) }()
			select {
			case <-started:
			case <-time.After(time.Second):
				t.Fatal("discovery did not begin")
			}
			go func() { second <- resolve(t.Context()) }()
			checkCancellation := func() {
				t.Helper()
				ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
				defer cancel()
				before := time.Now()
				if got := resolve(ctx); !errors.Is(got.err, context.DeadlineExceeded) || time.Since(before) > 200*time.Millisecond {
					t.Fatalf("resolver ignored cancellation: %v, elapsed %v", got.err, time.Since(before))
				}
			}
			checkCancellation() // Another canceled waiter must leave both lookups intact.
			if cancelFirst {
				cancel()
				unblock()
			}
			if got := <-first; cancelFirst && !errors.Is(got.err, context.Canceled) || !cancelFirst && (got.err != nil || got.verified) {
				t.Fatalf("first caller = %+v", got)
			}
			if got := <-second; got.err != nil || got.verified != cancelFirst {
				t.Fatalf("healthy caller = %+v", got)
			}
			if got := requests.Load(); got != 2 {
				t.Fatalf("shared discovery made %d requests, want one pair", got)
			}
			if !cancelFirst {
				checkCancellation() // A fresh failed lookup must also return its caller's deadline.
			}
		})
	}
}

func TestResolverReusesSavedLocalMappingsWithoutCloud(t *testing.T) {
	unexpected := func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("unexpected account lookup: %s", r.URL.Path)
		http.Error(w, "unexpected lookup", http.StatusInternalServerError)
	}
	resolver := newResolverFixture(t, map[string]http.HandlerFunc{
		"/api/tags": func(w http.ResponseWriter, _ *http.Request) {
			fmt.Fprint(w, `{"models":[{"name":"local-one:latest"},{"name":"local-two:latest"}]}`)
		},
		"/api/status": unexpected, "/api/me": unexpected,
	}, nil)
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
			models, err := resolver.Resolve(t.Context(), &test.cfg)
			if err != nil || len(models) != len(test.routes) {
				t.Fatalf("resolve = %+v, %v", models, err)
			}
			for i, model := range models {
				if model.GatewayID() != test.routes[i] || model.OllamaModel != test.names[i] || model.Cloud {
					t.Fatalf("model %d = %+v", i, model)
				}
			}
		})
	}
	if _, err := resolver.Resolve(t.Context(), &launchconfig.Integration{Aliases: map[string]string{"unknown-route": "local-one"}}); err == nil {
		t.Fatal("unknown route was accepted")
	}
}

func TestResolverRetriesCloudDiscovery(t *testing.T) {
	for _, first := range []string{"signed-out", "cloud-off", "recommendations-unavailable", "inventory-unavailable"} {
		t.Run(first, func(t *testing.T) {
			var ready atomic.Bool
			var cloudRequests atomic.Int32
			resolver := newResolverFixture(t, map[string]http.HandlerFunc{
				"/api/status": func(w http.ResponseWriter, _ *http.Request) {
					fmt.Fprintf(w, `{"cloud":{"disabled":%t}}`, !ready.Load() && first == "cloud-off")
				},
				"/api/me": func(w http.ResponseWriter, r *http.Request) {
					if !ready.Load() && first == "signed-out" {
						http.Error(w, "signed out", http.StatusUnauthorized)
						return
					}
					fmt.Fprint(w, `{"name":"fixture-user","plan":"pro"}`)
				},
			}, func(w http.ResponseWriter, r *http.Request) {
				cloudRequests.Add(1)
				if !ready.Load() && (first == "recommendations-unavailable" && r.URL.Path == "/api/experimental/model-recommendations" || first == "inventory-unavailable" && r.URL.Path == "/api/tags") {
					http.Error(w, "unavailable", http.StatusServiceUnavailable)
					return
				}
				serveResolverCatalog(w, r)
			})
			cfg := &launchconfig.Integration{Aliases: map[string]string{"claude-sonnet-5": "test:cloud"}}
			if _, err := resolver.Resolve(t.Context(), cfg); err != nil {
				t.Fatal(err)
			}
			before := cloudRequests.Load()
			ready.Store(true)
			models, err := resolver.Resolve(t.Context(), cfg)
			if err != nil || len(models) != 1 || !models[0].AccountCloud || cloudRequests.Load() == before {
				t.Fatalf("retry did not refresh inventory: %+v, %v", models, err)
			}
			before = cloudRequests.Load()
			if _, err := resolver.Resolve(t.Context(), cfg); err != nil || cloudRequests.Load() != before {
				t.Fatalf("successful cloud catalog was not cached: %v", err)
			}
			if first == "cloud-off" {
				ready.Store(false)
				if models, err := resolver.Resolve(t.Context(), cfg); err != nil || len(models) != 1 || models[0].AccountCloud {
					t.Fatalf("cloud disabled state retained verified inventory: %+v, %v", models, err)
				}
			}
		})
	}
}

func TestResolverRefreshesInventoryAfterSamePlanAccountSwitch(t *testing.T) {
	for _, withID := range []bool{true, false} {
		t.Run(fmt.Sprintf("account_id_%t", withID), func(t *testing.T) {
			var account, cloudRequests atomic.Int32
			var switchDuringLoad atomic.Bool
			resolver := newResolverFixture(t, map[string]http.HandlerFunc{
				"/api/me": func(w http.ResponseWriter, _ *http.Request) {
					if withID {
						fmt.Fprintf(w, `{"id":"00000000-0000-4000-8000-%012d","name":"fixture-user","plan":"pro"}`, account.Load()+1)
					} else {
						fmt.Fprintf(w, `{"name":"fixture-user-%d","plan":"pro"}`, account.Load())
					}
				},
			}, func(w http.ResponseWriter, r *http.Request) {
				cloudRequests.Add(1)
				if switchDuringLoad.Swap(false) {
					account.Store(1)
				}
				if r.URL.Path == "/api/tags" {
					fmt.Fprintf(w, `{"models":[{"name":"private-%d"}]}`, account.Load())
					return
				}
				serveResolverCatalog(w, r)
			})
			cfg := &launchconfig.Integration{Aliases: map[string]string{"claude-sonnet-5": "private-0:cloud"}}
			if models, err := resolver.Resolve(t.Context(), cfg); err != nil || len(models) != 1 || !models[0].AccountCloud {
				t.Fatalf("first account models = %+v, %v", models, err)
			}
			before := cloudRequests.Load()
			account.Store(1)
			if models, err := resolver.Resolve(t.Context(), cfg); err != nil || cloudRequests.Load() == before || len(models) != 1 || models[0].AccountCloud {
				t.Fatalf("account switch retained old inventory: %+v, %v", models, err)
			}
			cfg.Aliases["claude-sonnet-5"] = "private-1:cloud"
			before = cloudRequests.Load()
			if models, err := resolver.Resolve(t.Context(), cfg); err != nil || len(models) != 1 || !models[0].AccountCloud || cloudRequests.Load() != before {
				t.Fatalf("current account catalog was not reused: %+v, %v", models, err)
			}
			// Reassociate the same credential with another account during discovery.
			account.Store(0)
			switchDuringLoad.Store(true)
			if models, err := resolver.Resolve(t.Context(), cfg); err != nil || len(models) != 1 || models[0].AccountCloud {
				t.Fatalf("lookup accepted another account's inventory: %+v, %v", models, err)
			}
			account.Store(0)
			before = cloudRequests.Load()
			if models, err := resolver.Resolve(t.Context(), cfg); err != nil || cloudRequests.Load() == before || len(models) != 1 || models[0].AccountCloud {
				t.Fatalf("switched inventory was cached under the old account: %+v, %v", models, err)
			}
		})
	}
}
