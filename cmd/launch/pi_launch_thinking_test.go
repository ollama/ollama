package launch

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/cmd/internal/fileutil"
)

func TestLaunchIntegrationPiThinkingControls(t *testing.T) {
	for _, existing := range []bool{false, true} {
		t.Run(fmt.Sprintf("existing=%t", existing), func(t *testing.T) {
			testPiThinkingLaunch(t, existing)
		})
	}
}

// Exercise the real launch entry point, discovery, and on-disk Pi configuration.
// Kept separate so the generated config can also be checked with an installed Pi.
func testPiThinkingLaunch(t *testing.T, existing bool) map[string]any {
	t.Helper()
	home := t.TempDir()
	setLaunchTestHome(t, home)
	withLauncherHooks(t)
	withInteractiveSession(t, true)
	DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return false, nil }
	path := filepath.Join(home, ".pi", "agent", "models.json")
	if existing {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		data := `{"providers":{"ollama":{"models":[{"id":"custom-local:latest","_launch":true,"contextWindow":32768,"customField":"keep"},{"id":"manual:latest","reasoning":true,"thinkingLevelMap":{"high":"high"},"customField":"user"}]}}}`
		if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	var discoveries atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			// A recommendation is weaker than the actual /show response and
			// must not overwrite its result on a later regular launch.
			fmt.Fprint(w, `{"recommendations":[{"model":"custom-local:latest","thinking":{"values":[false,true],"default":true}}]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"custom-local:latest","capabilities":["completion","tools"],"details":{"context_length":32768}}]}`)
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
		case "/api/show":
			discoveries.Add(1)
			// The full set of controls comes from Show even without a thinking capability.
			fmt.Fprint(w, `{"capabilities":["completion","tools"],"thinking":{"values":[false,"low","high","max"],"default":"max"}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)
	DefaultSingleSelector = func(string, []SelectionItem, string) (string, error) {
		return "custom-local:latest", nil
	}
	if err := LaunchIntegration(t.Context(), IntegrationLaunchRequest{Name: "pi", ForceConfigure: true, ConfigureOnly: true}); err != nil {
		t.Fatal(err)
	}
	if got := discoveries.Load(); got != 1 {
		t.Fatalf("setup /api/show calls = %d, want 1", got)
	}
	cfg, err := fileutil.ReadJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	provider := cfg["providers"].(map[string]any)["ollama"].(map[string]any)
	entries := provider["models"].([]any)
	count := 1
	if existing {
		count = 2
	}
	if len(entries) != count {
		t.Fatalf("got %d entries, want %d", len(entries), count)
	}
	managed := entries[count-1].(map[string]any)
	want := map[string]any{"off": "none", "minimal": nil, "low": "low", "medium": nil, "high": "high", "xhigh": nil, "max": "max"}
	if diff := cmp.Diff(want, managed["thinkingLevelMap"]); diff != "" {
		t.Fatalf("thinking map (-want +got):\n%s", diff)
	}
	if managed["reasoning"] != true {
		t.Fatalf("reasoning=%v", managed["reasoning"])
	}
	if existing {
		if managed["contextWindow"] != float64(32768) || managed["customField"] != "keep" {
			t.Fatalf("lost managed fields: %v", managed)
		}
		var manual map[string]any
		if err := json.Unmarshal([]byte(`{"id":"manual:latest","reasoning":true,"thinkingLevelMap":{"high":"high"},"customField":"user"}`), &manual); err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(manual, entries[0]); diff != "" {
			t.Fatalf("user config changed:\n%s", diff)
		}
	}
	if runtime.GOOS != "windows" {
		bin := t.TempDir()
		for name, data := range map[string]string{
			"pi":           "#!/bin/sh\nexit 0\n",
			"npm":          "#!/bin/sh\nexit 1\n",
			"package.json": `{"name":"@earendil-works/pi-coding-agent"}`,
		} {
			if err := os.WriteFile(filepath.Join(bin, name), []byte(data), 0o755); err != nil {
				t.Fatal(err)
			}
		}
		t.Setenv("PATH", bin)
		DefaultSingleSelector = func(string, []SelectionItem, string) (string, error) {
			t.Fatal("regular launch unexpectedly opened model selection")
			return "", nil
		}
		if err := LaunchIntegration(t.Context(), IntegrationLaunchRequest{Name: "pi"}); err != nil {
			t.Fatal(err)
		}
		if got := discoveries.Load(); got != 1 {
			t.Fatalf("regular launch added %d /api/show calls, want 0", got-1)
		}
		after, err := fileutil.ReadJSON(path)
		if err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(cfg, after); diff != "" {
			t.Fatalf("regular launch changed saved model config (-want +got):\n%s", diff)
		}
	}
	return cfg
}

func TestLaunchIntegrationPiThinkingCatalogAndUpgrade(t *testing.T) {
	t.Run("explicit config discovers thinking catalog models only", func(t *testing.T) {
		home := t.TempDir()
		setLaunchTestHome(t, home)
		withLauncherHooks(t)
		withInteractiveSession(t, true)
		DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return false, nil }
		DefaultSingleSelector = func(string, []SelectionItem, string) (string, error) {
			return "primary:latest", nil
		}

		var (
			mu    sync.Mutex
			shown []string
		)
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/experimental/model-recommendations":
				fmt.Fprint(w, `{"recommendations":[]}`)
			case "/api/tags":
				fmt.Fprint(w, `{"models":[
					{"name":"primary:latest","capabilities":["completion","thinking"],"details":{"context_length":32768}},
					{"name":"secondary:latest","capabilities":["completion","thinking"],"details":{"context_length":32768}},
					{"name":"plain:latest","capabilities":["completion","tools"],"details":{"context_length":32768}}
				]}`)
			case "/api/status":
				fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
			case "/api/show":
				var req struct {
					Model string `json:"model"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err == nil {
					mu.Lock()
					shown = append(shown, req.Model)
					mu.Unlock()
				}
				fmt.Fprint(w, `{"capabilities":["completion","thinking"],"thinking":{"values":[false,"low","high","max"],"default":"max"}}`)
			default:
				http.NotFound(w, r)
			}
		}))
		defer server.Close()
		t.Setenv("OLLAMA_HOST", server.URL)

		if err := LaunchIntegration(t.Context(), IntegrationLaunchRequest{Name: "pi", ForceConfigure: true, ConfigureOnly: true}); err != nil {
			t.Fatal(err)
		}
		if len(shown) != 2 {
			t.Fatalf("/api/show called for %v, want only primary:latest and secondary:latest", shown)
		}
		cfg, err := fileutil.ReadJSON(filepath.Join(home, ".pi", "agent", "models.json"))
		if err != nil {
			t.Fatal(err)
		}
		entries := cfg["providers"].(map[string]any)["ollama"].(map[string]any)["models"].([]any)
		byID := make(map[string]map[string]any, len(entries))
		for _, entry := range entries {
			m := entry.(map[string]any)
			byID[m["id"].(string)] = m
		}
		for _, id := range []string{"primary:latest", "secondary:latest"} {
			levels, _ := byID[id]["thinkingLevelMap"].(map[string]any)
			if levels["max"] != "max" {
				t.Fatalf("%s thinkingLevelMap = %v, want max", id, levels)
			}
		}
		if _, ok := byID["plain:latest"]["thinkingLevelMap"]; ok {
			t.Fatalf("plain:latest unexpectedly got thinkingLevelMap: %v", byID["plain:latest"])
		}
	})

	t.Run("regular launch fills legacy recommended entry without show", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("requires shell stubs")
		}
		home := t.TempDir()
		setLaunchTestHome(t, home)
		withLauncherHooks(t)
		withInteractiveSession(t, true)
		DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return false, nil }
		DefaultSingleSelector = func(string, []SelectionItem, string) (string, error) {
			t.Fatal("regular launch unexpectedly opened model selection")
			return "", nil
		}

		dir := filepath.Join(home, ".pi", "agent")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		modelsPath := filepath.Join(dir, "models.json")
		initial := `{"providers":{"ollama":{"models":[
			{"id":"legacy-rec:latest","_launch":true,"contextWindow":32768,"reasoning":true},
			{"id":"saved-show:latest","_launch":true,"contextWindow":32768,"reasoning":true,"thinkingLevelMap":{"off":"none","minimal":null,"low":"low","medium":null,"high":"high","xhigh":null,"max":"max"}}
		]}}}`
		if err := os.WriteFile(modelsPath, []byte(initial), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "settings.json"), []byte(`{"defaultProvider":"ollama","defaultModel":"legacy-rec:latest"}`), 0o600); err != nil {
			t.Fatal(err)
		}

		bin := t.TempDir()
		for name, data := range map[string]string{
			"pi":           "#!/bin/sh\nexit 0\n",
			"npm":          "#!/bin/sh\nexit 1\n",
			"package.json": `{"name":"@earendil-works/pi-coding-agent"}`,
		} {
			if err := os.WriteFile(filepath.Join(bin, name), []byte(data), 0o755); err != nil {
				t.Fatal(err)
			}
		}
		t.Setenv("PATH", bin)

		var shows atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/experimental/model-recommendations":
				fmt.Fprint(w, `{"recommendations":[
					{"model":"legacy-rec:latest","thinking":{"values":["low","high","max"],"default":"max"}},
					{"model":"saved-show:latest","thinking":{"values":[false,true],"default":true}}
				]}`)
			case "/api/tags":
				fmt.Fprint(w, `{"models":[
					{"name":"legacy-rec:latest","capabilities":["completion","thinking"],"details":{"context_length":32768}},
					{"name":"saved-show:latest","capabilities":["completion","thinking"],"details":{"context_length":32768}}
				]}`)
			case "/api/status":
				fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
			case "/api/show":
				shows.Add(1)
				http.Error(w, "unexpected /api/show on regular launch", http.StatusInternalServerError)
			default:
				http.NotFound(w, r)
			}
		}))
		defer server.Close()
		t.Setenv("OLLAMA_HOST", server.URL)

		if err := LaunchIntegration(t.Context(), IntegrationLaunchRequest{Name: "pi"}); err != nil {
			t.Fatal(err)
		}
		if got := shows.Load(); got != 0 {
			t.Fatalf("regular launch made %d /api/show calls, want 0", got)
		}
		cfg, err := fileutil.ReadJSON(modelsPath)
		if err != nil {
			t.Fatal(err)
		}
		entries := cfg["providers"].(map[string]any)["ollama"].(map[string]any)["models"].([]any)
		byID := make(map[string]map[string]any, len(entries))
		for _, entry := range entries {
			m := entry.(map[string]any)
			byID[m["id"].(string)] = m
		}
		if got := byID["legacy-rec:latest"]["thinkingLevelMap"].(map[string]any)["max"]; got != "max" {
			t.Fatalf("legacy-rec:latest thinkingLevelMap = %v, want max", byID["legacy-rec:latest"]["thinkingLevelMap"])
		}
		if got := byID["saved-show:latest"]["thinkingLevelMap"].(map[string]any)["max"]; got != "max" {
			t.Fatalf("saved-show:latest lost saved thinkingLevelMap: %v", byID["saved-show:latest"]["thinkingLevelMap"])
		}
	})

	t.Run("explicit config keeps saved controls over weaker recommendation", func(t *testing.T) {
		for _, tc := range []struct {
			name             string
			inspect          string
			primaryShowFails bool
			wantShowCalls    int32
		}{
			{"primary show 503 preserves saved map", "primary:latest", true, 1},
			{"secondary with weaker recommendation still queries show", "secondary:latest", false, 1},
		} {
			t.Run(tc.name, func(t *testing.T) {
				home := t.TempDir()
				setLaunchTestHome(t, home)
				withLauncherHooks(t)
				withInteractiveSession(t, true)
				DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return false, nil }
				DefaultSingleSelector = func(string, []SelectionItem, string) (string, error) {
					return "primary:latest", nil
				}
				dir := filepath.Join(home, ".pi", "agent")
				if err := os.MkdirAll(dir, 0o755); err != nil {
					t.Fatal(err)
				}
				modelsPath := filepath.Join(dir, "models.json")
				saved := `{"providers":{"ollama":{"models":[
					{"id":"primary:latest","_launch":true,"contextWindow":32768,"reasoning":true,"thinkingLevelMap":{"off":null,"minimal":null,"low":"low","medium":null,"high":"high","xhigh":null,"max":"max"}},
					{"id":"secondary:latest","_launch":true,"contextWindow":32768,"reasoning":true,"thinkingLevelMap":{"off":null,"minimal":null,"low":"low","medium":null,"high":"high","xhigh":null,"max":"max"}}
				]}}}`
				if err := os.WriteFile(modelsPath, []byte(saved), 0o600); err != nil {
					t.Fatal(err)
				}

				var inspectShows atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					switch r.URL.Path {
					case "/api/experimental/model-recommendations":
						fmt.Fprintf(w, `{"recommendations":[{"model":%q,"thinking":{"values":[false,true],"default":true}}]}`, tc.inspect)
					case "/api/tags":
						fmt.Fprint(w, `{"models":[
							{"name":"primary:latest","capabilities":["completion","thinking"],"details":{"context_length":32768}},
							{"name":"secondary:latest","capabilities":["completion","thinking"],"details":{"context_length":32768}}
						]}`)
					case "/api/status":
						fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
					case "/api/show":
						var req struct {
							Model string `json:"model"`
						}
						if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
							http.Error(w, "invalid request", http.StatusBadRequest)
							return
						}
						if req.Model == tc.inspect {
							inspectShows.Add(1)
						}
						if tc.primaryShowFails && req.Model == "primary:latest" {
							http.Error(w, "temporary error", http.StatusServiceUnavailable)
							return
						}
						fmt.Fprint(w, `{"capabilities":["completion","thinking"],"thinking":{"values":["low","high","max"],"default":"max"}}`)
					default:
						http.NotFound(w, r)
					}
				}))
				defer server.Close()
				t.Setenv("OLLAMA_HOST", server.URL)

				if err := LaunchIntegration(t.Context(), IntegrationLaunchRequest{Name: "pi", ForceConfigure: true, ConfigureOnly: true}); err != nil {
					t.Fatal(err)
				}
				if got := inspectShows.Load(); got != tc.wantShowCalls {
					t.Fatalf("%s /api/show calls = %d, want %d", tc.inspect, got, tc.wantShowCalls)
				}
				cfg, err := fileutil.ReadJSON(modelsPath)
				if err != nil {
					t.Fatal(err)
				}
				entries := cfg["providers"].(map[string]any)["ollama"].(map[string]any)["models"].([]any)
				for _, entry := range entries {
					m := entry.(map[string]any)
					if m["id"] != tc.inspect {
						continue
					}
					levels := m["thinkingLevelMap"].(map[string]any)
					want := map[string]any{"off": nil, "minimal": nil, "low": "low", "medium": nil, "high": "high", "xhigh": nil, "max": "max"}
					if diff := cmp.Diff(want, levels); diff != "" {
						t.Fatalf("%s thinkingLevelMap (-want +got):\n%s", tc.inspect, diff)
					}
					return
				}
				t.Fatalf("missing %s", tc.inspect)
			})
		}
	})
}
