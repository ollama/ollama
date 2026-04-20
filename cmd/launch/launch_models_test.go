package launch

import (
	"io"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestMain(m *testing.M) {
	launchModelsURL = ""
	os.Exit(m.Run())
}

func TestFetchRecommendedModels(t *testing.T) {
	oldURL := launchModelsURL
	oldClient := launchModelsHTTPClient
	t.Cleanup(func() {
		launchModelsURL = oldURL
		launchModelsHTTPClient = oldClient
	})

	var gotUserAgent string
	launchModelsURL = "https://ollama.com/api/experimental/launch-models"
	launchModelsHTTPClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/api/experimental/launch-models" {
			return &http.Response{
				StatusCode: http.StatusNotFound,
				Body:       io.NopCloser(strings.NewReader("not found")),
				Header:     make(http.Header),
			}, nil
		}
		gotUserAgent = r.UserAgent()
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(testLaunchModelsJSON)),
			Header:     make(http.Header),
		}, nil
	})}

	got := fetchRecommendedModels(t.Context())
	want := []string{"kimi-k2.5:cloud", "glm-5.1:cloud", "qwen3.5:cloud", "minimax-m2.7:cloud", "gemma4", "qwen3.5"}
	if gotNames := names(got); !slices.Equal(gotNames, want) {
		t.Fatalf("models = %v, want %v", gotNames, want)
	}
	if !strings.HasPrefix(gotUserAgent, "ollama/") {
		t.Fatalf("User-Agent = %q, want ollama prefix", gotUserAgent)
	}
	if limit, ok := lookupCloudModelLimit("kimi-k2.5:cloud"); !ok || limit.Context != 262_144 || limit.Output != 262_144 {
		t.Fatalf("kimi limit = %+v, %v; want context/output limits", limit, ok)
	}
}

func TestFetchRecommendedModelsFallback(t *testing.T) {
	oldURL := launchModelsURL
	oldClient := launchModelsHTTPClient
	t.Cleanup(func() {
		launchModelsURL = oldURL
		launchModelsHTTPClient = oldClient
	})

	launchModelsURL = "https://ollama.com/api/experimental/launch-models"
	launchModelsHTTPClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusServiceUnavailable,
			Body:       io.NopCloser(strings.NewReader("unavailable")),
			Header:     make(http.Header),
		}, nil
	})}

	got := fetchRecommendedModels(t.Context())
	gotNames := names(got)
	wantNames := names(defaultRecommendedModels)
	if !slices.Equal(gotNames, wantNames) {
		t.Fatalf("models = %v, want fallback %v", gotNames, wantNames)
	}
}

func TestLoadSelectableModelsCloudDisabledSkipsRemoteRecommendations(t *testing.T) {
	oldURL := launchModelsURL
	oldClient := launchModelsHTTPClient
	t.Cleanup(func() {
		launchModelsURL = oldURL
		launchModelsHTTPClient = oldClient
	})

	remoteCalled := false
	launchModelsURL = "https://ollama.com/api/experimental/launch-models"
	launchModelsHTTPClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		remoteCalled = true
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(testLaunchModelsJSON)),
			Header:     make(http.Header),
		}, nil
	})}

	baseURL, err := url.Parse("http://ollama.test")
	if err != nil {
		t.Fatal(err)
	}

	apiClient := api.NewClient(baseURL, &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path == "/api/status" {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(`{"cloud":{"disabled":true,"source":"config"}}`)),
				Header:     make(http.Header),
			}, nil
		}
		return &http.Response{
			StatusCode: http.StatusNotFound,
			Body:       io.NopCloser(strings.NewReader(`{"error":"not found"}`)),
			Header:     make(http.Header),
		}, nil
	})})

	client := &launcherClient{
		apiClient:       apiClient,
		inventoryLoaded: true,
	}
	items, _, err := client.loadSelectableModels(t.Context(), nil, "", "no models")
	if err != nil {
		t.Fatalf("loadSelectableModels returned error: %v", err)
	}
	if remoteCalled {
		t.Fatal("expected cloud-disabled launch to skip remote recommendations fetch")
	}
	want := []string{"gemma4", "qwen3.5"}
	if got := names(items); !slices.Equal(got, want) {
		t.Fatalf("models = %v, want %v", got, want)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}
