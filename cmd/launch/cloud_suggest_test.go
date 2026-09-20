package launch

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
)

// newCloudSuggestTestClient serves a daemon where "some-model" doesn't exist
// (locally or in the registry), "some-model:cloud" does, and the user is
// signed in. It records pull requests in pulled.
func newCloudSuggestTestClient(t *testing.T, pulled *[]string) *api.Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			var req api.ShowRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if req.Model == "some-model:cloud" {
				if err := json.NewEncoder(w).Encode(api.ShowResponse{RemoteModel: "some-model"}); err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
				}
				return
			}
			w.WriteHeader(http.StatusNotFound)
			if err := json.NewEncoder(w).Encode(map[string]string{"error": "model not found"}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		case "/api/pull":
			var req api.PullRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			*pulled = append(*pulled, req.Model+req.Name)
			if err := json.NewEncoder(w).Encode(map[string]string{
				"error": "pull model manifest: file does not exist",
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		case "/api/me":
			if err := json.NewEncoder(w).Encode(api.UserResponse{Name: "tester"}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		case "/api/tags":
			if err := json.NewEncoder(w).Encode(api.ListResponse{}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	u, _ := url.Parse(srv.URL)
	return api.NewClient(u, srv.Client())
}

func TestReadyModel_CloudSuggestionRename(t *testing.T) {
	var pulled []string
	client := newCloudSuggestTestClient(t, &pulled)

	oldHook := DefaultCloudSuggest
	t.Cleanup(func() { DefaultCloudSuggest = oldHook })

	var hookModel string
	var hookErr error
	DefaultCloudSuggest = func(ctx context.Context, c *api.Client, model string, pullErr error) (string, error) {
		hookModel = model
		hookErr = pullErr
		return model + ":cloud", nil
	}

	c := &launcherClient{apiClient: client, policy: LaunchPolicy{MissingModel: LaunchMissingModelAutoPull}}
	resolved, err := c.readyModel(context.Background(), "some-model", "ollama launch", "claude")
	if err != nil {
		t.Fatalf("readyModel returned error: %v", err)
	}
	if resolved != "some-model:cloud" {
		t.Fatalf("resolved = %q, want %q", resolved, "some-model:cloud")
	}
	if hookModel != "some-model" {
		t.Fatalf("hook model = %q, want %q", hookModel, "some-model")
	}
	if hookErr == nil || !strings.Contains(hookErr.Error(), "failed to pull some-model") {
		t.Fatalf("hook pull error = %v, want the wrapped pull failure", hookErr)
	}
	if len(pulled) != 1 {
		t.Fatalf("pull requests = %v, want just the original attempt", pulled)
	}
}

// TestResolveRunModel_CloudSuggestionPersistsResolvedModel covers the
// last_model lifecycle across an accepted suggestion: a saved model that no
// longer resolves routes to the picker, picking a default-tag name whose
// pull fails accepts the ":cloud" variant, the resolved name is what gets
// persisted, and a rerun reuses it without prompting again.
func TestResolveRunModel_CloudSuggestionPersistsResolvedModel(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	var pulled []string
	client := newCloudSuggestTestClient(t, &pulled)

	if err := config.SetLastModel("some-model"); err != nil {
		t.Fatal(err)
	}

	oldSelector, oldHook := DefaultSingleSelector, DefaultCloudSuggest
	t.Cleanup(func() { DefaultSingleSelector, DefaultCloudSuggest = oldSelector, oldHook })

	var selectorCalls, hookCalls int
	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		selectorCalls++
		return "some-model", nil
	}
	DefaultCloudSuggest = func(ctx context.Context, c *api.Client, model string, pullErr error) (string, error) {
		hookCalls++
		return model + ":cloud", nil
	}

	c := &launcherClient{apiClient: client, policy: LaunchPolicy{MissingModel: LaunchMissingModelAutoPull}}
	model, err := c.resolveRunModel(context.Background(), RunModelRequest{})
	if err != nil {
		t.Fatalf("resolveRunModel returned error: %v", err)
	}
	if model != "some-model:cloud" {
		t.Fatalf("model = %q, want %q", model, "some-model:cloud")
	}
	if selectorCalls != 1 || hookCalls != 1 {
		t.Fatalf("selector calls = %d, hook calls = %d, want 1 and 1", selectorCalls, hookCalls)
	}
	if got := config.LastModel(); got != "some-model:cloud" {
		t.Fatalf("saved last model = %q, want %q", got, "some-model:cloud")
	}

	// Rerunning with the persisted cloud name must not open the picker or
	// re-prompt the suggestion.
	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		t.Error("unexpected model picker on rerun")
		return "", ErrCancelled
	}
	DefaultCloudSuggest = func(ctx context.Context, c *api.Client, model string, pullErr error) (string, error) {
		t.Error("unexpected cloud suggestion on rerun")
		return "", pullErr
	}

	rerun := &launcherClient{apiClient: client, policy: LaunchPolicy{MissingModel: LaunchMissingModelAutoPull}}
	model, err = rerun.resolveRunModel(context.Background(), RunModelRequest{})
	if err != nil {
		t.Fatalf("rerun resolveRunModel returned error: %v", err)
	}
	if model != "some-model:cloud" {
		t.Fatalf("rerun model = %q, want %q", model, "some-model:cloud")
	}
}

func TestReadyModel_CloudSuggestionErrorSurfaces(t *testing.T) {
	var pulled []string
	client := newCloudSuggestTestClient(t, &pulled)

	oldHook := DefaultCloudSuggest
	t.Cleanup(func() { DefaultCloudSuggest = oldHook })

	suggestErr := errors.New("augmented pull error")
	DefaultCloudSuggest = func(ctx context.Context, c *api.Client, model string, pullErr error) (string, error) {
		return "", suggestErr
	}

	c := &launcherClient{apiClient: client, policy: LaunchPolicy{MissingModel: LaunchMissingModelAutoPull}}
	if _, err := c.readyModel(context.Background(), "some-model", "ollama launch", ""); !errors.Is(err, suggestErr) {
		t.Fatalf("readyModel error = %v, want the hook's error", err)
	}
}

func TestReadyModel_NoHookKeepsPullError(t *testing.T) {
	var pulled []string
	client := newCloudSuggestTestClient(t, &pulled)

	oldHook := DefaultCloudSuggest
	t.Cleanup(func() { DefaultCloudSuggest = oldHook })
	DefaultCloudSuggest = nil

	c := &launcherClient{apiClient: client, policy: LaunchPolicy{MissingModel: LaunchMissingModelAutoPull}}
	_, err := c.readyModel(context.Background(), "some-model", "ollama launch", "")
	if err == nil || !strings.Contains(err.Error(), "failed to pull some-model") {
		t.Fatalf("readyModel error = %v, want the plain pull failure", err)
	}
}
