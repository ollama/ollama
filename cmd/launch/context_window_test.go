package launch

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestResolveLaunchContextLoadsAndUsesRunningContext(t *testing.T) {
	var generateCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/generate":
			generateCalls.Add(1)
			var request api.GenerateRequest
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Fatalf("decode generate request: %v", err)
			}
			if request.Model != "qwen3.8:27b-mlx" || len(request.Options) != 0 {
				t.Fatalf("generate request = %#v, want model-only load", request)
			}
			_ = json.NewEncoder(w).Encode(api.GenerateResponse{Done: true})
		case "/api/ps":
			_ = json.NewEncoder(w).Encode(api.ProcessResponse{Models: []api.ProcessModelResponse{{
				Name:          "qwen3.8:27b-mlx",
				ContextLength: 65_536,
			}}})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	base, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	got := resolveLaunchContext(context.Background(), api.NewClient(base, server.Client()), LaunchModel{
		Name:          "qwen3.8:27b-mlx",
		ContextLength: 262_144,
	}, true)
	if got.ContextLength != 65_536 || !got.RuntimeVerified {
		t.Fatalf("resolution = %#v, want verified 65536", got)
	}
	if generateCalls.Load() != 1 {
		t.Fatalf("generate calls = %d, want 1", generateCalls.Load())
	}
}

func TestResolveLaunchContextConfigurationOnlyDoesNotLoad(t *testing.T) {
	var generateCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/generate":
			generateCalls.Add(1)
			http.Error(w, "must not load", http.StatusInternalServerError)
		case "/api/ps":
			_ = json.NewEncoder(w).Encode(api.ProcessResponse{})
		case "/api/show":
			_ = json.NewEncoder(w).Encode(api.ShowResponse{ModelInfo: map[string]any{
				"qwen3_5.context_length": float64(262_144),
			}})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	base, _ := url.Parse(server.URL)
	got := resolveLaunchContext(context.Background(), api.NewClient(base, server.Client()), LaunchModel{Name: "qwen3.8:27b-mlx"}, false)
	if got.ContextLength != 262_144 || got.RuntimeVerified {
		t.Fatalf("resolution = %#v, want unverified native metadata 262144", got)
	}
	if generateCalls.Load() != 0 {
		t.Fatalf("generate calls = %d, want 0", generateCalls.Load())
	}
}

func TestResolveLaunchContextBoundsExplicitNumCtxByNativeCapacity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/ps":
			_ = json.NewEncoder(w).Encode(api.ProcessResponse{})
		case "/api/show":
			_ = json.NewEncoder(w).Encode(api.ShowResponse{
				Parameters: "temperature 0.2\nnum_ctx 524288\ntop_p 0.9",
				ModelInfo:  map[string]any{"qwen3_5.context_length": float64(262_144)},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	base, _ := url.Parse(server.URL)
	got := resolveLaunchContext(context.Background(), api.NewClient(base, server.Client()), LaunchModel{Name: "qwen3.8:27b-mlx"}, false)
	if got.ContextLength != 262_144 {
		t.Fatalf("context = %d, want native-bounded 262144", got.ContextLength)
	}
}

func TestResolveLaunchContextFallsBackToInventoryWhenAPIsFail(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	}))
	t.Cleanup(server.Close)

	base, _ := url.Parse(server.URL)
	got := resolveLaunchContext(context.Background(), api.NewClient(base, server.Client()), LaunchModel{
		Name:          "qwen3.8:27b-mlx",
		ContextLength: 131_072,
	}, true)
	if got.ContextLength != 131_072 || got.RuntimeVerified {
		t.Fatalf("resolution = %#v, want unverified inventory fallback 131072", got)
	}
}

func TestResolveLaunchContextCloudDoesNotCallLocalServer(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		http.Error(w, "unexpected", http.StatusInternalServerError)
	}))
	t.Cleanup(server.Close)

	base, _ := url.Parse(server.URL)
	got := resolveLaunchContext(context.Background(), api.NewClient(base, server.Client()), LaunchModel{
		Name:          "qwen3.5:cloud",
		Remote:        true,
		ContextLength: 262_144,
	}, true)
	if got.ContextLength != 262_144 {
		t.Fatalf("context = %d, want cloud metadata 262144", got.ContextLength)
	}
	if calls.Load() != 0 {
		t.Fatalf("local API calls = %d, want 0", calls.Load())
	}
}

func TestProcessContextWindowMatchesLatestAlias(t *testing.T) {
	got := processContextWindow("ornith", &api.ProcessResponse{
		Models: []api.ProcessModelResponse{
			{Name: "other:latest", Model: "other:latest", ContextLength: 32768},
			{Name: "ornith:latest", Model: "ornith:latest", ContextLength: 262144},
		},
	})
	if got != 262144 {
		t.Fatalf("context window = %d, want 262144", got)
	}
}
