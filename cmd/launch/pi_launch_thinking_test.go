package launch

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"custom-local:latest","capabilities":["completion","tools"]},{"name":"manual:latest","capabilities":["completion","tools"]}]}`)
		case "/api/show":
			discoveries.Add(1)
			// Deliberately no thinking capability or recommendation: the controls
			// must come from Show, not a hardcoded name or capability flag.
			fmt.Fprint(w, `{"capabilities":["completion","tools"],"thinking":{"values":[false,"low","high","max"],"default":"max"}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)
	DefaultMultiSelector = func(string, []SelectionItem, []string) ([]string, error) {
		models := []string{"custom-local:latest"}
		if existing {
			models = append(models, "manual:latest")
		}
		return models, nil
	}
	if err := LaunchIntegration(t.Context(), IntegrationLaunchRequest{Name: "pi", ForceConfigure: true, ConfigureOnly: true}); err != nil {
		t.Fatal(err)
	}
	if discoveries.Load() == 0 {
		t.Fatal("thinking discovery was not called")
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
	managed := entries[0].(map[string]any)
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
		if diff := cmp.Diff(manual, entries[1]); diff != "" {
			t.Fatalf("user config changed:\n%s", diff)
		}
	}
	return cfg
}
