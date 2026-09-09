package launch

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestLaunchRetryCommand(t *testing.T) {
	if got, want := launchRetryCommand("claude")("some-model:cloud"), "ollama launch claude --model some-model:cloud"; got != want {
		t.Fatalf("launchRetryCommand(claude) = %q, want %q", got, want)
	}
	if got, want := launchRetryCommand("")("some-model:cloud"), "ollama launch --model some-model:cloud"; got != want {
		t.Fatalf("launchRetryCommand(\"\") = %q, want %q", got, want)
	}
}

func newMissingModelClient(t *testing.T) *api.Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, `{"error":"model not found"}`)
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)
	u, _ := url.Parse(srv.URL)
	return api.NewClient(u, srv.Client())
}

func TestShowOrPullWithPolicy_CloudSuggestionRename(t *testing.T) {
	client := newMissingModelClient(t)

	oldHook := DefaultCloudSuggestPull
	t.Cleanup(func() { DefaultCloudSuggestPull = oldHook })

	var hookModel, hookRetry string
	DefaultCloudSuggestPull = func(ctx context.Context, c *api.Client, model string, retryCommand func(string) string) (string, error) {
		hookModel = model
		if retryCommand != nil {
			hookRetry = retryCommand(model + ":cloud")
		}
		return model + ":cloud", nil
	}

	resolved, err := showOrPullWithPolicy(context.Background(), client, "some-model", missingModelAutoPull, false, launchRetryCommand("claude"))
	if err != nil {
		t.Fatalf("showOrPullWithPolicy returned error: %v", err)
	}
	if resolved != "some-model:cloud" {
		t.Fatalf("resolved = %q, want %q", resolved, "some-model:cloud")
	}
	if hookModel != "some-model" {
		t.Fatalf("hook model = %q, want %q", hookModel, "some-model")
	}
	if want := "ollama launch claude --model some-model:cloud"; hookRetry != want {
		t.Fatalf("hook retry command = %q, want %q", hookRetry, want)
	}
}

func TestShowOrPullWithPolicy_CloudSuggestionErrorWrapped(t *testing.T) {
	client := newMissingModelClient(t)

	oldHook := DefaultCloudSuggestPull
	t.Cleanup(func() { DefaultCloudSuggestPull = oldHook })

	DefaultCloudSuggestPull = func(ctx context.Context, c *api.Client, model string, retryCommand func(string) string) (string, error) {
		return "", errors.New("boom")
	}

	_, err := showOrPullWithPolicy(context.Background(), client, "some-model", missingModelAutoPull, false, nil)
	if err == nil {
		t.Fatal("showOrPullWithPolicy returned nil, want an error")
	}
	if !strings.Contains(err.Error(), "failed to pull some-model") || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("error = %q, want it to wrap the hook error with the pull context", err)
	}
}
