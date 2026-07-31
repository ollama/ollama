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
