package launch

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestOpenCodeRunContentLoadedContext(t *testing.T) {
	for _, fromEdit := range []bool{false, true} {
		t.Run(map[bool]string{false: "saved configuration", true: "fresh edit"}[fromEdit], func(t *testing.T) {
			setTestHome(t, t.TempDir())
			models := []LaunchModel{
				{Name: "selected", ContextLength: 131072, Details: api.ModelDetails{ContextLength: 131072}},
				{Name: "other", ContextLength: 65536, MaxOutputTokens: 4096},
				{Name: "cloud-alias", Remote: true, ContextLength: 262144, MaxOutputTokens: 32768},
			}
			o := &OpenCode{}
			var loaded []string
			previous := openCodeLoadedContextLength
			openCodeLoadedContextLength = func(name string) (int, error) {
				loaded = append(loaded, name)
				return 8192, nil
			}
			t.Cleanup(func() { openCodeLoadedContextLength = previous })
			if fromEdit {
				if err := o.Edit(models); err != nil {
					t.Fatal(err)
				}
				if len(loaded) != 0 {
					t.Fatalf("Edit loaded models: %v", loaded)
				}
			}
			before := o.resolveContent("selected", models)
			after, err := o.runContent("selected", models)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(loaded, []string{"selected"}) {
				t.Fatalf("loaded = %v, want selected only", loaded)
			}
			var expected, got map[string]any
			if err := json.Unmarshal([]byte(before), &expected); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(after), &got); err != nil {
				t.Fatal(err)
			}
			entry := expected["provider"].(map[string]any)["ollama"].(map[string]any)["models"].(map[string]any)["selected"].(map[string]any)
			entry["limit"] = map[string]any{"context": float64(8192), "output": float64(2048)}
			if !reflect.DeepEqual(got, expected) {
				t.Fatalf("runtime config changed beyond selected limits:\ngot %s", after)
			}
			if models[0].ContextLength != 131072 || models[0].Details.ContextLength != 131072 || o.resolveContent("selected", models) != before {
				t.Fatal("runtime limits mutated inventory or cached configuration")
			}
		})
	}
}

func TestOpenCodeRunContentSkipsCloud(t *testing.T) {
	setTestHome(t, t.TempDir())
	previous := openCodeLoadedContextLength
	openCodeLoadedContextLength = func(name string) (int, error) {
		t.Fatalf("cloud model %q was loaded locally", name)
		return 0, nil
	}
	t.Cleanup(func() { openCodeLoadedContextLength = previous })
	for _, primary := range []LaunchModel{
		{Name: "glm-5:cloud", ContextLength: 202752, MaxOutputTokens: 131072},
		{Name: "cloud-alias", Remote: true, ContextLength: 131072, MaxOutputTokens: 8192},
	} {
		o := &OpenCode{}
		models := []LaunchModel{primary, {Name: "local", ContextLength: 32768}}
		before := o.resolveContent(primary.Name, models)
		after, err := o.runContent(primary.Name, models)
		if err != nil || after != before {
			t.Fatalf("cloud config changed: %s, %v", after, err)
		}
	}
}

func TestOpenCodeRunContentOutputBudget(t *testing.T) {
	setTestHome(t, t.TempDir())
	previous := openCodeLoadedContextLength
	t.Cleanup(func() { openCodeLoadedContextLength = previous })
	for _, tt := range []struct {
		context, knownOutput, wantOutput int
	}{{1024, 0, 256}, {2048, 0, 512}, {16384, 0, 4096}, {65536, 0, 8192}, {16384, 1024, 1024}} {
		openCodeLoadedContextLength = func(string) (int, error) { return tt.context, nil }
		content, err := (&OpenCode{}).runContent("local", []LaunchModel{{Name: "local", ContextLength: 131072, MaxOutputTokens: tt.knownOutput}})
		if err != nil {
			t.Fatal(err)
		}
		var cfg struct {
			Provider map[string]struct {
				Models map[string]struct {
					Limit struct{ Context, Output int }
				}
			}
		}
		if err := json.Unmarshal([]byte(content), &cfg); err != nil {
			t.Fatal(err)
		}
		limit := cfg.Provider["ollama"].Models["local"].Limit
		if limit.Context != tt.context || limit.Output != tt.wantOutput {
			t.Fatalf("limit = %+v, want context %d output %d", limit, tt.context, tt.wantOutput)
		}
	}
}

func TestOpenCodeRunContentUnknownAllocationKeepsConfig(t *testing.T) {
	setTestHome(t, t.TempDir())
	previous := openCodeLoadedContextLength
	openCodeLoadedContextLength = func(string) (int, error) { return 0, nil }
	t.Cleanup(func() { openCodeLoadedContextLength = previous })
	for _, knownOutput := range []int{0, 4096} {
		o := &OpenCode{}
		models := []LaunchModel{{Name: "local", ContextLength: 131072, MaxOutputTokens: knownOutput}}
		before := o.resolveContent("local", models)
		warning := captureStderr(t, func() {
			content, err := o.runContent("local", models)
			if err != nil || content != before {
				t.Fatalf("unknown allocation changed launch config: %q, %v", content, err)
			}
		})
		if strings.Count(warning, "Warning:") != 1 || !strings.Contains(warning, "local") {
			t.Fatalf("expected one warning identifying the model, got %q", warning)
		}
	}
}

func TestOpenCodeRunContentLoadFailurePreservesCause(t *testing.T) {
	setTestHome(t, t.TempDir())
	previous := openCodeLoadedContextLength
	loadErr := errors.New("model requires more system memory")
	openCodeLoadedContextLength = func(string) (int, error) { return 0, loadErr }
	t.Cleanup(func() { openCodeLoadedContextLength = previous })
	content, err := (&OpenCode{}).runContent("local", []LaunchModel{{Name: "local"}})
	if content != "" || !errors.Is(err, loadErr) || !strings.Contains(err.Error(), "local") {
		t.Fatalf("load error lost its cause or allowed launch: %q, %v", content, err)
	}
}

func TestOpenCodeRunContentLatestAlias(t *testing.T) {
	setTestHome(t, t.TempDir())
	previous := openCodeLoadedContextLength
	openCodeLoadedContextLength = func(name string) (int, error) {
		if name != "local:latest" {
			t.Fatalf("loaded %q, want canonical selected model", name)
		}
		return 8192, nil
	}
	t.Cleanup(func() { openCodeLoadedContextLength = previous })
	if _, err := (&OpenCode{}).runContent("local", []LaunchModel{{Name: "local:latest"}}); err != nil {
		t.Fatal(err)
	}
}

func TestOpenCodeConfigureOnlySkipsLoad(t *testing.T) {
	setTestHome(t, t.TempDir())
	previousLoad, previousConfirm := openCodeLoadedContextLength, DefaultConfirmPrompt
	openCodeLoadedContextLength = func(name string) (int, error) {
		t.Fatalf("configuration-only loaded %q", name)
		return 0, nil
	}
	DefaultConfirmPrompt = func(prompt string, _ ConfirmOptions) (bool, error) {
		if prompt != "Launch OpenCode now?" {
			t.Fatalf("unexpected prompt: %s", prompt)
		}
		return false, nil
	}
	t.Cleanup(func() {
		openCodeLoadedContextLength, DefaultConfirmPrompt = previousLoad, previousConfirm
	})
	o := &OpenCode{}
	models := []LaunchModel{{Name: "local", ContextLength: 131072}}
	if err := o.Edit(models); err != nil {
		t.Fatal(err)
	}
	if err := launchAfterConfiguration("opencode", o, "local", models, IntegrationLaunchRequest{ConfigureOnly: true}); err != nil {
		t.Fatal(err)
	}
}

func TestOpenCodeRunForwardsArgsWithoutPreload(t *testing.T) {
	setTestHome(t, t.TempDir())
	bin := t.TempDir()
	writeFakeBinary(t, bin, "opencode")
	t.Setenv("PATH", bin)
	previous := openCodeLoadedContextLength
	openCodeLoadedContextLength = func(model string) (int, error) {
		t.Errorf("unexpected load of %q", model)
		return 0, errors.New("model does not fit in memory")
	}
	t.Cleanup(func() { openCodeLoadedContextLength = previous })
	models := []LaunchModel{{Name: "local"}, {Name: "other"}}
	o := &OpenCode{}
	before := o.resolveContent("local", models)
	for _, args := range [][]string{
		{"--help"},
		{"--version"},
		{"serve"},
		{"debug", "config"},
		{"run", "hello"},
		{"--model", "ollama/other"},
		{"run", "-m", "other/model", "hello"},
		{"--continue"},
		{"--session", "session-id"},
		{"--agent", "custom"},
		{"project"},
		{"--new-option"},
	} {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			original := append([]string(nil), args...)
			content, err := o.runContent("local", models, args...)
			if err != nil || content != before {
				t.Fatalf("forwarded arguments changed the configuration: %q, %v", content, err)
			}
			if err := o.Run("local", models, args); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(args, original) {
				t.Fatal("modified caller-owned arguments")
			}
		})
	}
}

type openCodeContentRunner struct {
	OpenCode
	content string
}

func (o *openCodeContentRunner) Run(model string, models []LaunchModel, args []string) (err error) {
	o.content, err = o.runContent(model, models, args...)
	return err
}

func TestOpenCodeLaunchRecommendationOutputCeiling(t *testing.T) {
	for _, metadataAvailable := range []bool{true, false} {
		t.Run(fmt.Sprintf("context metadata available=%t", metadataAvailable), func(t *testing.T) {
			setTestHome(t, t.TempDir())
			bin := t.TempDir()
			writeFakeBinary(t, bin, "opencode")
			t.Setenv("PATH", bin)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/experimental/model-recommendations":
					fmt.Fprint(w, `{"recommendations":[{"model":"local","context_length":131072,"max_output_tokens":1024}]}`)
				case "/api/tags":
					fmt.Fprint(w, `{"models":[{"name":"local:latest","details":{"context_length":131072}}]}`)
				case "/api/show":
					fmt.Fprint(w, `{}`)
				case "/api/status":
					fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
				case "/api/generate":
					fmt.Fprint(w, `{"done":true,"done_reason":"load"}`)
				case "/api/ps":
					if !metadataAvailable {
						w.WriteHeader(http.StatusServiceUnavailable)
						return
					}
					fmt.Fprint(w, `{"models":[{"name":"local:latest","context_length":65536}]}`)
				default:
					t.Errorf("unexpected request: %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer server.Close()
			t.Setenv("OLLAMA_HOST", server.URL)
			client, err := newLauncherClient(defaultLaunchPolicy(false, false))
			if err != nil {
				t.Fatal(err)
			}
			o := &openCodeContentRunner{}
			if err := client.launchEditorIntegration(t.Context(), "opencode", o, o, nil, IntegrationLaunchRequest{ModelOverride: "local"}); err != nil {
				t.Fatal(err)
			}
			var cfg map[string]any
			if err := json.Unmarshal([]byte(o.content), &cfg); err != nil {
				t.Fatal(err)
			}
			entry := cfg["provider"].(map[string]any)["ollama"].(map[string]any)["models"].(map[string]any)["local:latest"].(map[string]any)
			if metadataAvailable {
				limit := entry["limit"].(map[string]any)
				if limit["context"] != float64(65536) || limit["output"] != float64(1024) {
					t.Fatalf("lost the known smaller output ceiling: %s", o.content)
				}
			} else if o.content != o.configContent || entry["limit"] != nil {
				t.Fatalf("recommendations altered fallback configuration: %s", o.content)
			}
			if client.inventory.models[0].ContextLength != 131072 || client.inventory.models[0].MaxOutputTokens != 0 {
				t.Fatal("runtime output ceiling mutated the original inventory")
			}
		})
	}
}
