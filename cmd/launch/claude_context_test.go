package launch

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestClaudePrepareServingContext(t *testing.T) {
	for _, tt := range []struct {
		name, body string
		status     int
		want       int
	}{
		{name: "known unloaded model", body: `{"serving_context_length":65536,"model_info":{"gemma4.context_length":262144}}`, want: 65536},
		{name: "unknown automatic context", body: `{"model_info":{"gemma4.context_length":262144}}`},
		{name: "old server default is not model capacity", body: `{"context_length":65536,"model_info":{"gemma4.context_length":262144}}`},
		{name: "negative serving limit", body: `{"serving_context_length":-1}`},
		{name: "contradicts architecture maximum", body: `{"serving_context_length":524288,"model_info":{"gemma4.context_length":262144}}`},
		{name: "remote alias", body: `{"serving_context_length":65536,"remote_host":"https://ollama.com","remote_model":"remote"}`},
		{name: "missing model", status: http.StatusNotFound, body: `{"error":"not found"}`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", "http://different.invalid:11434")
			t.Setenv("OLLAMA_CONTEXT_LENGTH", "999999")
			oldConfirm := DefaultConfirmPrompt
			DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) {
				t.Fatal("context discovery must not add a launch confirmation")
				return false, nil
			}
			t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })
			calls := 0
			transport := contextTestTransport(func(req *http.Request) (*http.Response, error) {
				calls++
				if req.URL.Host != "selected.test:12345" || req.URL.Path != "/api/show" {
					t.Fatalf("unexpected metadata route: %s", req.URL)
				}
				var show api.ShowRequest
				if err := json.NewDecoder(req.Body).Decode(&show); err != nil {
					t.Fatal(err)
				}
				if show.Model != "gemma4" || len(show.Options) != 0 {
					t.Fatalf("show request = %+v", show)
				}
				status := tt.status
				if status == 0 {
					status = http.StatusOK
				}
				return &http.Response{StatusCode: status, Body: io.NopCloser(strings.NewReader(tt.body)), Header: make(http.Header)}, nil
			})
			u, _ := url.Parse("http://selected.test:12345")
			client := &launcherClient{apiClient: api.NewClient(u, &http.Client{Transport: transport})}
			input := []LaunchModel{{Name: "other", ServingContextLength: 8192}, {Name: "gemma4:latest", ContextLength: 262144, ServingContextLength: 12345}}
			got, err := client.prepareLaunchModelsForRun(context.Background(), &Claude{}, "gemma4", input)
			if err != nil {
				t.Fatal(err)
			}
			if calls != 1 || got[1].ServingContextLength != tt.want || got[1].ContextLength != 262144 || got[0].ServingContextLength != 8192 {
				t.Fatalf("models = %+v, calls = %d, want selected serving context %d", got, calls, tt.want)
			}
			if input[1].ServingContextLength != 12345 {
				t.Fatal("mutated model inventory")
			}
		})
	}
}

func TestClaudePrepareServingContextSkipsRemote(t *testing.T) {
	client := &launcherClient{apiClient: api.NewClient(&url.URL{Scheme: "http", Host: "selected.test"}, &http.Client{Transport: contextTestTransport(func(*http.Request) (*http.Response, error) {
		t.Fatal("remote model should keep existing cloud metadata path")
		return nil, nil
	})})}
	for _, model := range []LaunchModel{{Name: "example:cloud"}, {Name: "remote-alias", Remote: true}} {
		if _, err := client.prepareLaunchModelsForRun(context.Background(), &Claude{}, model.Name, []LaunchModel{model}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestClaudeLocalContextLeavesCompactionPreferences(t *testing.T) {
	t.Setenv("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "150000")
	for _, length := range []int{65536, 196608} {
		env := (&Claude{}).modelEnvVars("local", []LaunchModel{{Name: "local", ContextLength: 262144, ServingContextLength: length}})
		for _, entry := range env {
			if strings.HasPrefix(entry, "CLAUDE_CODE_AUTO_COMPACT_WINDOW=") {
				t.Fatal("local metadata must not override compaction selection", entry)
			}
		}
	}
}
