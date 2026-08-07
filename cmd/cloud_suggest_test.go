package cmd

import (
	"cmp"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/types/model"
)

func TestCloudSuggestionCandidate(t *testing.T) {
	notFoundErr := errors.New("pull model manifest: file does not exist")
	suggestedErr := errors.New("pull model manifest: file does not exist\n\nTry one of these models:\n  some-model:cloud")

	tests := []struct {
		name     string
		model    string
		pullErr  error
		insecure bool
		want     string
		wantOK   bool
	}{
		{name: "default tag not found", model: "some-model", pullErr: notFoundErr, want: "some-model:cloud", wantOK: true},
		{name: "composes with server tag suggestions", model: "some-model", pullErr: suggestedErr, want: "some-model:cloud", wantOK: true},
		{name: "namespaced default tag", model: "user/some-model", pullErr: notFoundErr, want: "user/some-model:cloud", wantOK: true},
		{name: "nil error", model: "some-model", pullErr: nil},
		{name: "unrelated error", model: "some-model", pullErr: errors.New("boom")},
		{name: "insecure registry", model: "some-model", pullErr: notFoundErr, insecure: true},
		{name: "explicit tag", model: "some-model:9b", pullErr: notFoundErr},
		{name: "explicit latest tag", model: "some-model:latest", pullErr: notFoundErr},
		{name: "explicit cloud source", model: "some-model:cloud", pullErr: notFoundErr},
		{name: "explicit legacy cloud tag", model: "some-model:9b-cloud", pullErr: notFoundErr},
		{name: "explicit local source", model: "some-model:local", pullErr: notFoundErr},
		{name: "custom registry host", model: "internal.example.com/team/private-model", pullErr: notFoundErr},
		{name: "custom registry host with port", model: "registry.example.com:5000/team/private-model", pullErr: notFoundErr},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := cloudSuggestionCandidate(tt.model, tt.pullErr, tt.insecure)
			if ok != tt.wantOK {
				t.Fatalf("cloudSuggestionCandidate(%q) ok = %v, want %v", tt.model, ok, tt.wantOK)
			}
			if got != tt.want {
				t.Fatalf("cloudSuggestionCandidate(%q) = %q, want %q", tt.model, got, tt.want)
			}
		})
	}
}

// stubCloudSuggest replaces the TTY check and confirmation prompt for the
// duration of the test. If confirm is nil, any prompt fails the test.
func stubCloudSuggest(t *testing.T, interactive bool, confirm func(prompt string) (bool, error)) *[]string {
	t.Helper()

	oldTTY, oldConfirm := isInteractiveTerminal, confirmCloudSuggestion
	t.Cleanup(func() {
		isInteractiveTerminal, confirmCloudSuggestion = oldTTY, oldConfirm
	})

	isInteractiveTerminal = func() bool { return interactive }

	prompts := &[]string{}
	confirmCloudSuggestion = func(prompt string) (bool, error) {
		*prompts = append(*prompts, prompt)
		if confirm == nil {
			t.Errorf("unexpected cloud suggestion prompt: %q", prompt)
			return false, nil
		}
		return confirm(prompt)
	}
	return prompts
}

type cloudSuggestServer struct {
	cloudName   string // model name whose show/pull succeeds (e.g. "some-model:cloud")
	cloudExists bool   // whether showing/pulling cloudName succeeds
	pullErr     string // error message for failing pulls

	showModels     []string
	pullModels     []string
	generateModels []string
}

// start serves mock /api/show, /api/pull, /api/tags, and /api/generate
// endpoints: only cloudName is known (when cloudExists), and pulling any other
// model fails with pullErr streamed the way real servers do (an in-band error
// under HTTP 200).
func (s *cloudSuggestServer) start(t *testing.T) {
	t.Helper()

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/show" && r.Method == http.MethodPost:
			var req api.ShowRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			name := cmp.Or(req.Model, req.Name)
			s.showModels = append(s.showModels, name)
			if s.cloudExists && name == s.cloudName {
				if err := json.NewEncoder(w).Encode(api.ShowResponse{
					Capabilities: []model.Capability{model.CapabilityCompletion},
					RemoteModel:  strings.TrimSuffix(s.cloudName, ":cloud"),
				}); err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
				}
				return
			}
			w.WriteHeader(http.StatusNotFound)
			if err := json.NewEncoder(w).Encode(map[string]string{
				"error": "model '" + name + "' not found",
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		case r.URL.Path == "/api/pull" && r.Method == http.MethodPost:
			var req api.PullRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			name := cmp.Or(req.Model, req.Name)
			s.pullModels = append(s.pullModels, name)
			var body any
			if s.cloudExists && name == s.cloudName {
				body = api.ProgressResponse{Status: "success"}
			} else {
				body = map[string]string{"error": s.pullErr}
			}
			if err := json.NewEncoder(w).Encode(body); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		case r.URL.Path == "/api/tags" && r.Method == http.MethodGet:
			if err := json.NewEncoder(w).Encode(api.ListResponse{
				Models: []api.ListModelResponse{{Name: s.cloudName}},
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		case r.URL.Path == "/api/generate" && r.Method == http.MethodPost:
			var req api.GenerateRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			s.generateModels = append(s.generateModels, req.Model)
			if err := json.NewEncoder(w).Encode(api.GenerateResponse{Done: true}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		default:
			http.NotFound(w, r)
		}
	}))

	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Cleanup(mockServer.Close)
}

func newCloudSuggestServer(t *testing.T) *cloudSuggestServer {
	t.Helper()
	s := &cloudSuggestServer{
		cloudName:   "some-model:cloud",
		cloudExists: true,
		pullErr:     "pull model manifest: file does not exist",
	}
	s.start(t)
	return s
}

func newPullTestCmd(t *testing.T) *cobra.Command {
	t.Helper()
	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().Bool("insecure", false, "")
	return cmd
}

func newRunTestCmd(t *testing.T) *cobra.Command {
	t.Helper()
	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("truncate", false, "")
	cmd.Flags().Int("dimensions", 0, "")
	cmd.Flags().Bool("verbose", false, "")
	cmd.Flags().Bool("insecure", false, "")
	cmd.Flags().Bool("nowordwrap", false, "")
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")
	return cmd
}

func TestPullHandler_SuccessfulPullNoSuggestion(t *testing.T) {
	server := newCloudSuggestServer(t)
	server.cloudName = "some-model" // the requested model itself pulls fine
	stubCloudSuggest(t, true, nil)

	if err := PullHandler(newPullTestCmd(t), []string{"some-model"}); err != nil {
		t.Fatalf("PullHandler returned error: %v", err)
	}
	if want := []string{"some-model"}; !slices.Equal(server.pullModels, want) {
		t.Fatalf("pulled models = %v, want %v", server.pullModels, want)
	}
	if len(server.showModels) != 0 {
		t.Fatalf("show models = %v, want no probe after a successful pull", server.showModels)
	}
}

func TestPullHandler_CloudSuggestionAccepted(t *testing.T) {
	server := newCloudSuggestServer(t)
	prompts := stubCloudSuggest(t, true, func(string) (bool, error) { return true, nil })

	if err := PullHandler(newPullTestCmd(t), []string{"some-model"}); err != nil {
		t.Fatalf("PullHandler returned error: %v", err)
	}

	if want := []string{"some-model", "some-model:cloud"}; !slices.Equal(server.pullModels, want) {
		t.Fatalf("pulled models = %v, want %v", server.pullModels, want)
	}
	if len(*prompts) != 1 || !strings.Contains((*prompts)[0], `"some-model:cloud"`) {
		t.Fatalf("prompts = %v, want one prompt mentioning some-model:cloud", *prompts)
	}
}

func TestPullHandler_CloudSuggestionDeclined(t *testing.T) {
	server := newCloudSuggestServer(t)
	stubCloudSuggest(t, true, func(string) (bool, error) { return false, nil })

	err := PullHandler(newPullTestCmd(t), []string{"some-model"})
	if err == nil {
		t.Fatal("PullHandler returned nil, want an error")
	}
	if !strings.Contains(err.Error(), "pull model manifest: file does not exist") {
		t.Fatalf("error = %q, want it to contain the original pull error", err)
	}
	if strings.Contains(err.Error(), "Try:") {
		t.Fatalf("error = %q, want no non-interactive hint after declining", err)
	}
	if want := []string{"some-model"}; !slices.Equal(server.pullModels, want) {
		t.Fatalf("pulled models = %v, want %v", server.pullModels, want)
	}
}

func TestPullHandler_CloudSuggestionCancelled(t *testing.T) {
	server := newCloudSuggestServer(t)
	stubCloudSuggest(t, true, func(string) (bool, error) { return false, launch.ErrCancelled })

	err := PullHandler(newPullTestCmd(t), []string{"some-model"})
	if err == nil {
		t.Fatal("PullHandler returned nil, want an error")
	}
	if errors.Is(err, launch.ErrCancelled) {
		t.Fatalf("error = %v, want the original pull error rather than ErrCancelled", err)
	}
	if !strings.Contains(err.Error(), "pull model manifest: file does not exist") {
		t.Fatalf("error = %q, want it to contain the original pull error", err)
	}
	if want := []string{"some-model"}; !slices.Equal(server.pullModels, want) {
		t.Fatalf("pulled models = %v, want %v", server.pullModels, want)
	}
}

func TestPullHandler_CloudSuggestionNonInteractive(t *testing.T) {
	server := newCloudSuggestServer(t)
	stubCloudSuggest(t, false, nil)

	err := PullHandler(newPullTestCmd(t), []string{"some-model"})
	if err == nil {
		t.Fatal("PullHandler returned nil, want an error")
	}
	if !strings.Contains(err.Error(), "pull model manifest: file does not exist") {
		t.Fatalf("error = %q, want it to contain the original pull error", err)
	}
	if !strings.Contains(err.Error(), "ollama pull some-model:cloud") {
		t.Fatalf("error = %q, want it to hint at 'ollama pull some-model:cloud'", err)
	}
	if want := []string{"some-model"}; !slices.Equal(server.pullModels, want) {
		t.Fatalf("pulled models = %v, want %v", server.pullModels, want)
	}
}

func TestPullHandler_CloudSuggestionNoCloudTag(t *testing.T) {
	server := newCloudSuggestServer(t)
	server.cloudExists = false
	stubCloudSuggest(t, true, nil)

	err := PullHandler(newPullTestCmd(t), []string{"some-model"})
	if err == nil || err.Error() != "pull model manifest: file does not exist" {
		t.Fatalf("error = %v, want the unmodified pull error", err)
	}
	if want := []string{"some-model:cloud"}; !slices.Equal(server.showModels, want) {
		t.Fatalf("show models = %v, want the cloud existence probe %v", server.showModels, want)
	}
}

func TestPullHandler_CloudSuggestionExplicitTag(t *testing.T) {
	server := newCloudSuggestServer(t)
	stubCloudSuggest(t, true, nil)

	err := PullHandler(newPullTestCmd(t), []string{"some-model:9b"})
	if err == nil || err.Error() != "pull model manifest: file does not exist" {
		t.Fatalf("error = %v, want the unmodified pull error", err)
	}
	if len(server.showModels) != 0 {
		t.Fatalf("show models = %v, want no cloud probe for explicitly tagged models", server.showModels)
	}
}

func TestPullHandler_CloudSuggestionExplicitCloud(t *testing.T) {
	server := newCloudSuggestServer(t)
	server.cloudExists = false // make the explicit :cloud pull fail too
	stubCloudSuggest(t, true, nil)

	err := PullHandler(newPullTestCmd(t), []string{"some-model:cloud"})
	if err == nil || err.Error() != "pull model manifest: file does not exist" {
		t.Fatalf("error = %v, want the unmodified pull error", err)
	}
	if len(server.showModels) != 0 {
		t.Fatalf("show models = %v, want no probe for explicit :cloud requests", server.showModels)
	}
}

func TestPullHandler_CloudSuggestionInsecure(t *testing.T) {
	server := newCloudSuggestServer(t)
	stubCloudSuggest(t, true, nil)

	cmd := newPullTestCmd(t)
	if err := cmd.Flags().Set("insecure", "true"); err != nil {
		t.Fatal(err)
	}

	err := PullHandler(cmd, []string{"some-model"})
	if err == nil || err.Error() != "pull model manifest: file does not exist" {
		t.Fatalf("error = %v, want the unmodified pull error", err)
	}
	if len(server.showModels) != 0 {
		t.Fatalf("show models = %v, want no probe for --insecure pulls", server.showModels)
	}
}

func TestPullHandler_CloudSuggestionUnrelatedError(t *testing.T) {
	server := newCloudSuggestServer(t)
	server.pullErr = "boom"
	stubCloudSuggest(t, true, nil)

	err := PullHandler(newPullTestCmd(t), []string{"some-model"})
	if err == nil || err.Error() != "boom" {
		t.Fatalf("error = %v, want the unmodified pull error %q", err, "boom")
	}
	if len(server.showModels) != 0 {
		t.Fatalf("show models = %v, want no probe for unrelated pull errors", server.showModels)
	}
}

func TestLaunchCloudSuggestHookWired(t *testing.T) {
	server := newCloudSuggestServer(t)
	stubCloudSuggest(t, true, func(string) (bool, error) { return true, nil })

	if launch.DefaultCloudSuggest == nil {
		t.Fatal("launch.DefaultCloudSuggest is not wired")
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		t.Fatal(err)
	}

	pullErr := errors.New("failed to pull some-model: pull model manifest: file does not exist")
	resolved, err := launch.DefaultCloudSuggest(t.Context(), client, "some-model", pullErr)
	if err != nil {
		t.Fatalf("DefaultCloudSuggest returned error: %v", err)
	}
	if resolved != "some-model:cloud" {
		t.Fatalf("resolved = %q, want %q", resolved, "some-model:cloud")
	}
	if len(server.pullModels) != 0 {
		t.Fatalf("pulled models = %v, want none (the launch flow owns pulling)", server.pullModels)
	}
	if want := []string{"some-model:cloud"}; !slices.Equal(server.showModels, want) {
		t.Fatalf("show models = %v, want the cloud existence probe %v", server.showModels, want)
	}
}

func TestLaunchCloudSuggestHookNonInteractiveHint(t *testing.T) {
	newCloudSuggestServer(t)
	stubCloudSuggest(t, false, nil)

	client, err := api.ClientFromEnvironment()
	if err != nil {
		t.Fatal(err)
	}

	pullErr := errors.New("failed to pull some-model: pull model manifest: file does not exist")
	_, err = launch.DefaultCloudSuggest(t.Context(), client, "some-model", pullErr)
	if err == nil {
		t.Fatal("DefaultCloudSuggest returned nil, want an error")
	}
	if !errors.Is(err, pullErr) {
		t.Fatalf("error = %q, want it to wrap the original pull error", err)
	}
	if !strings.Contains(err.Error(), "--model some-model:cloud") {
		t.Fatalf("error = %q, want it to hint at '--model some-model:cloud'", err)
	}
}

func TestLaunchCloudSuggestHookDeclinedKeepsOriginalError(t *testing.T) {
	newCloudSuggestServer(t)
	stubCloudSuggest(t, true, func(string) (bool, error) { return false, launch.ErrCancelled })

	client, err := api.ClientFromEnvironment()
	if err != nil {
		t.Fatal(err)
	}

	pullErr := errors.New("failed to pull some-model: pull model manifest: file does not exist")
	_, err = launch.DefaultCloudSuggest(t.Context(), client, "some-model", pullErr)
	if err != pullErr {
		t.Fatalf("error = %v, want the original pull error unchanged", err)
	}
}

func TestRunHandler_CloudSuggestionAccepted_RunsCloudModel(t *testing.T) {
	server := newCloudSuggestServer(t)
	stubCloudSuggest(t, true, func(string) (bool, error) { return true, nil })

	if err := RunHandler(newRunTestCmd(t), []string{"some-model", "hi"}); err != nil {
		t.Fatalf("RunHandler returned error: %v", err)
	}

	if want := []string{"some-model", "some-model:cloud"}; !slices.Equal(server.pullModels, want) {
		t.Fatalf("pulled models = %v, want %v", server.pullModels, want)
	}
	if want := []string{"some-model:cloud"}; !slices.Equal(server.generateModels, want) {
		t.Fatalf("generate models = %v, want %v", server.generateModels, want)
	}
}

func TestRunHandler_CloudSuggestionDeclined_ReturnsNotFound(t *testing.T) {
	server := newCloudSuggestServer(t)
	stubCloudSuggest(t, true, func(string) (bool, error) { return false, nil })

	err := RunHandler(newRunTestCmd(t), []string{"some-model", "hi"})
	if err == nil {
		t.Fatal("RunHandler returned nil, want an error")
	}
	if !strings.Contains(err.Error(), "pull model manifest: file does not exist") {
		t.Fatalf("error = %q, want it to contain the original pull error", err)
	}
	if len(server.generateModels) != 0 {
		t.Fatalf("generate models = %v, want none after declining", server.generateModels)
	}
}

func TestRunHandler_CloudSuggestionNonInteractive_Hint(t *testing.T) {
	server := newCloudSuggestServer(t)
	stubCloudSuggest(t, false, nil)

	err := RunHandler(newRunTestCmd(t), []string{"some-model", "hi"})
	if err == nil {
		t.Fatal("RunHandler returned nil, want an error")
	}
	if !strings.Contains(err.Error(), "ollama run some-model:cloud") {
		t.Fatalf("error = %q, want it to hint at 'ollama run some-model:cloud'", err)
	}
	if len(server.generateModels) != 0 {
		t.Fatalf("generate models = %v, want none in non-interactive mode", server.generateModels)
	}
}
