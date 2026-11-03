package launch

import (
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/types/model"
)

func TestPiThinkingConfig(t *testing.T) {
	for _, tc := range []struct {
		name      string
		thinking  *api.ModelRecommendationThinking
		enabled   map[string]any
		reasoning bool
	}{
		{"named", &api.ModelRecommendationThinking{Values: []any{"low", "high", "max"}, Default: "max"}, map[string]any{"low": "low", "high": "high", "max": "max"}, true},
		{"mixed off and named", &api.ModelRecommendationThinking{Values: []any{false, "low", "high", "max"}, Default: "high"}, map[string]any{"off": "none", "low": "low", "high": "high", "max": "max"}, true},
		{"boolean", &api.ModelRecommendationThinking{Values: []any{false, true}, Default: true}, map[string]any{"off": "none", "high": "high"}, true},
		{"always on", &api.ModelRecommendationThinking{Values: []any{true}, Default: true}, map[string]any{"high": "high"}, true},
		{"disabled", &api.ModelRecommendationThinking{Values: []any{false}, Default: false}, map[string]any{"off": "none"}, false},
		{"all named levels", &api.ModelRecommendationThinking{Values: []any{"minimal", "low", "medium", "high", "xhigh", "max"}, Default: "medium"}, map[string]any{"minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}, true},
		// The OpenAI endpoint preserves strings for mixed descriptors. A string
		// cannot request boolean true in this case, so do not invent a high alias.
		{"mixed on and named", &api.ModelRecommendationThinking{Values: []any{false, true, "max"}, Default: true}, map[string]any{"off": "none", "max": "max"}, true},
		{"unknown alongside known", &api.ModelRecommendationThinking{Values: []any{"future", "max"}, Default: "future"}, map[string]any{"max": "max"}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := createConfig(LaunchModel{Name: "custom", Thinking: tc.thinking})
			want := map[string]any{"off": nil, "minimal": nil, "low": nil, "medium": nil, "high": nil, "xhigh": nil, "max": nil}
			for k, v := range tc.enabled {
				want[k] = v
			}
			if diff := cmp.Diff(want, cfg["thinkingLevelMap"]); diff != "" {
				t.Errorf("thinkingLevelMap (-want +got):\n%s", diff)
			}
			for level, effort := range tc.enabled {
				got, err := openai.ThinkingFromReasoningEffort(effort.(string), tc.thinking)
				var want any = effort
				if level == "off" {
					want = false
				} else if tc.name == "boolean" || tc.name == "always on" {
					want = true
				}
				if err != nil || got == nil || got.Value != want {
					t.Errorf("%s round trip: got %v, %v; want %v", level, got, err, want)
				}
			}
			if cfg["reasoning"] != tc.reasoning {
				t.Errorf("reasoning=%v, want %v", cfg["reasoning"], tc.reasoning)
			}
		})
	}
}

func TestPiThinkingDiscoveryTimeout(t *testing.T) {
	client, err := newLauncherClient(defaultLaunchPolicy(false, false))
	if err != nil {
		t.Fatal(err)
	}
	transport := &thinkingDeadlineTransport{t: t}
	client.apiClient = api.NewClient(&url.URL{Scheme: "http", Host: "thinking.test"}, &http.Client{Transport: transport})
	client.recommendationsLoaded = true
	client.inventory = newModelInventory(client.apiClient)
	client.inventory.loaded = true
	thinking := &api.ModelRecommendationThinking{Values: []any{false, true}, Default: true}
	client.inventory.models = []LaunchModel{{Name: "custom-local", Thinking: thinking}}
	models := client.resolveRunModels(t.Context(), "pi", []string{"custom-local"})
	if len(models) != 1 || transport.calls != 1 {
		t.Fatalf("models=%v show calls=%d", models, transport.calls)
	}
	if diff := cmp.Diff(thinking, models[0].Thinking); diff != "" {
		t.Fatalf("failed discovery lost existing metadata:\n%s", diff)
	}
}

func TestPiThinkingConfigUnknown(t *testing.T) {
	for _, thinking := range []*api.ModelRecommendationThinking{
		nil,
		{Values: []any{"low"}, Default: "missing"},
		{Values: []any{"future"}, Default: "future"},
		{Values: []any{false, true, "future"}, Default: true},
	} {
		cfg := createConfig(LaunchModel{Name: "custom", Thinking: thinking, Capabilities: []model.Capability{model.CapabilityThinking}})
		if _, ok := cfg["thinkingLevelMap"]; ok {
			t.Errorf("unknown controls produced a map: %v", cfg)
		}
		if cfg["reasoning"] != true {
			t.Errorf("lost capability fallback: %v", cfg)
		}
	}
}

func TestPiEditRefreshesThinkingControls(t *testing.T) {
	home := t.TempDir()
	setTestHome(t, home)
	dir := filepath.Join(home, ".pi", "agent")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "models.json")
	original := `{"providers":{"other":{"models":[{"id":"keep"}]},"ollama":{"models":[{"id":"managed","_launch":true,"contextWindow":32768,"compat":{"supportsDeveloperRole":false}},{"id":"manual","contextWindow":8192,"reasoning":true,"thinkingLevelMap":{"high":"custom"}}]}}}`
	if err := os.WriteFile(path, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}
	settingsPath := filepath.Join(dir, "settings.json")
	if err := os.WriteFile(settingsPath, []byte(`{"modelThinkingLevels":{"ollama/managed":"low"},"packages":["keep"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	thinking := &api.ModelRecommendationThinking{Values: []any{"low", "high", "max"}, Default: "max"}
	models := []LaunchModel{{Name: "managed", Thinking: thinking}, {Name: "manual", Thinking: thinking}}
	read := func(path string) map[string]any {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var cfg map[string]any
		if err := json.Unmarshal(data, &cfg); err != nil {
			t.Fatal(err)
		}
		return cfg
	}
	before := read(path)
	var first map[string]any
	for i := 0; i < 2; i++ {
		if err := (&Pi{}).Edit(models); err != nil {
			t.Fatal(err)
		}
		cfg := read(path)
		providers := cfg["providers"].(map[string]any)
		entries := providers["ollama"].(map[string]any)["models"].([]any)
		if len(entries) != 2 {
			t.Fatalf("model count=%d, want 2", len(entries))
		}
		managed := entries[0].(map[string]any)
		want := createConfig(models[0])["thinkingLevelMap"]
		if want == nil {
			t.Fatal("expected thinking controls")
		}
		if diff := cmp.Diff(want, managed["thinkingLevelMap"]); diff != "" {
			t.Errorf("managed thinking (-want +got):\n%s", diff)
		}
		if managed["reasoning"] != true || managed["contextWindow"] != float64(32768) || managed["compat"].(map[string]any)["supportsDeveloperRole"] != false {
			t.Errorf("unexpected managed entry: %v", managed)
		}
		oldProviders := before["providers"].(map[string]any)
		oldManual := oldProviders["ollama"].(map[string]any)["models"].([]any)[1]
		if diff := cmp.Diff(oldManual, entries[1]); diff != "" {
			t.Errorf("manual entry changed:\n%s", diff)
		}
		if diff := cmp.Diff(oldProviders["other"], providers["other"]); diff != "" {
			t.Errorf("other provider changed:\n%s", diff)
		}
		if i == 0 {
			first = cfg
		} else if diff := cmp.Diff(first, cfg); diff != "" {
			t.Errorf("not idempotent:\n%s", diff)
		}
	}
	settings := read(settingsPath)
	if settings["modelThinkingLevels"].(map[string]any)["ollama/managed"] != "low" || settings["packages"].([]any)[0] != "keep" {
		t.Errorf("user settings changed: %v", settings)
	}
	// Failed discovery must not erase a previously known map.
	if err := (&Pi{}).Edit([]LaunchModel{{Name: "managed"}, {Name: "manual"}}); err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(first, read(path)); diff != "" {
		t.Errorf("missing metadata changed models:\n%s", diff)
	}
}
