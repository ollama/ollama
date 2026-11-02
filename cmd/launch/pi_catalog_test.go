package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	modelpkg "github.com/ollama/ollama/types/model"
)

func TestPiPrimaryMetadataBudget(t *testing.T) {
	for _, complete := range []bool{false, true} {
		t.Run(fmt.Sprint(complete), func(t *testing.T) {
			home := t.TempDir()
			setLaunchTestHome(t, home)
			withLauncherHooks(t)
			oldTimeout := piModelDiscoveryTimeout
			piModelDiscoveryTimeout = 50 * time.Millisecond
			t.Cleanup(func() { piModelDiscoveryTimeout = oldTimeout })
			if err := config.SaveIntegration("pi", []string{"primary"}); err != nil {
				t.Fatal(err)
			}
			dir := filepath.Join(home, ".pi", "agent")
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			original := `{"providers":{"ollama":{"models":[{"id":"primary","_launch":true,"contextWindow":8192}]}}}`
			path := filepath.Join(dir, "models.json")
			if err := os.WriteFile(path, []byte(original), 0o644); err != nil {
				t.Fatal(err)
			}
			DefaultSingleSelector = func(string, []SelectionItem, string) (string, error) { return "primary", nil }
			DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return false, nil }
			stop := make(chan struct{})
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/status":
					fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
				case "/api/experimental/model-recommendations":
					fmt.Fprint(w, `{"recommendations":[{"model":"primary"}]}`)
				case "/api/tags":
					if complete {
						fmt.Fprint(w, `{"models":[{"name":"primary","details":{"context_length":8192},"capabilities":["completion","tools"]}]}`)
					} else {
						fmt.Fprint(w, `{"models":[{"name":"primary"}]}`)
					}
				case "/api/show":
					select {
					case <-r.Context().Done():
					case <-stop:
					}
				default:
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			defer close(stop)
			t.Setenv("OLLAMA_HOST", srv.URL)
			c, err := newLauncherClient(defaultLaunchPolicy(true, false))
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			finished := make(chan error, 1)
			go func() {
				saved, _ := config.LoadIntegration("pi")
				finished <- c.launchPiIntegration(ctx, &Pi{}, saved, IntegrationLaunchRequest{ConfigureOnly: true})
			}()
			select {
			case err := <-finished:
				if err != nil {
					t.Fatalf("partial discovery should succeed: %v", err)
				}
			case <-time.After(time.Second):
				cancel()
				<-finished
				t.Fatal("primary /show exceeded the metadata budget")
			}
			if !complete {
				data, err := fileutil.ReadJSON(path)
				if err != nil {
					t.Fatal(err)
				}
				models := data["providers"].(map[string]any)["ollama"].(map[string]any)["models"].([]any)
				if len(models) != 1 || models[0].(map[string]any)["contextWindow"] != float64(8192) {
					t.Fatalf("known metadata changed on timeout: %v", models)
				}
			}
		})
	}
}

func TestPiCatalogSync(t *testing.T) {
	home := t.TempDir()
	setLaunchTestHome(t, home)
	withLauncherHooks(t)
	withInteractiveSession(t, true)
	if err := config.SaveIntegration("pi", []string{"primary", "gone", "unused"}); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(home, ".pi", "agent")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{
		"models.json":   `{"providers":{"ollama":{"models":[{"id":"gone","_launch":true},{"id":"manual","contextWindow":8192},{"id":"secondary","contextWindow":2048,"custom":true}]},"other":{"models":[{"id":"keep"}]}}}`,
		"settings.json": `{"defaultProvider":"ollama","defaultModel":"primary","editorPaddingX":3}`,
		"history.jsonl": "preserve history exactly\n",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	DefaultSingleSelector = func(string, []SelectionItem, string) (string, error) {
		return "primary", nil
	}
	DefaultConfirmPrompt = func(prompt string, _ ConfirmOptions) (bool, error) {
		if prompt != "Launch Pi now?" {
			t.Errorf("unexpected confirmation: %s", prompt)
		}
		return false, nil
	}
	var shown []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[{"model":"not-downloaded"}]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"primary:latest","details":{"context_length":65536},"capabilities":["completion","tools"]},{"name":"secondary","details":{"context_length":32768},"capabilities":["completion","vision"]},{"name":"embeddings","details":{"context_length":8192},"capabilities":["embedding"]},{"name":"legacy-remote"}]}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			shown = append(shown, req.Model)
			if req.Model == "legacy-remote" {
				fmt.Fprint(w, `{"remote_model":"hidden","capabilities":["completion","tools"],"model_info":{"test.context_length":8192}}`)
				return
			}
			if req.Model != "primary" {
				t.Errorf("unexpected lookup for a catalog model: %s", req.Model)
			}
			fmt.Fprint(w, `{}`)
		default:
			t.Errorf("unexpected request: %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)
	c, err := newLauncherClient(defaultLaunchPolicy(true, false))
	if err != nil {
		t.Fatal(err)
	}
	saved, _ := config.LoadIntegration("pi")
	if err := c.launchPiIntegration(t.Context(), &Pi{}, saved, IntegrationLaunchRequest{ConfigureOnly: true, ModelOverride: "remote:cloud"}); err != nil {
		t.Fatal(err)
	}
	if len(shown) == 0 || slices.ContainsFunc(shown, func(name string) bool { return name != "primary" && name != "legacy-remote" }) {
		t.Fatalf("unexpected metadata lookups: %v", shown)
	}
	saved, _ = config.LoadIntegration("pi")
	if !slices.Equal(saved.Models, []string{"primary"}) {
		t.Fatalf("saved models = %v", saved.Models)
	}
	models, _ := fileutil.ReadJSON(filepath.Join(dir, "models.json"))
	providers := models["providers"].(map[string]any)
	entries := providers["ollama"].(map[string]any)["models"].([]any)
	if len(entries) != 3 {
		t.Fatalf("catalog = %v; want primary and the two manual entries", entries)
	}
	byID := make(map[string]map[string]any)
	for _, entry := range entries {
		model := entry.(map[string]any)
		byID[model["id"].(string)] = model
	}
	if byID["primary"]["contextWindow"] != float64(65536) || byID["secondary"]["contextWindow"] != float64(2048) || byID["secondary"]["custom"] != true {
		t.Fatalf("unexpected metadata/manual override: %v", entries)
	}
	if providers["other"].(map[string]any)["models"].([]any)[0].(map[string]any)["id"] != "keep" {
		t.Fatal("unrelated provider changed")
	}
	settings, _ := fileutil.ReadJSON(filepath.Join(dir, "settings.json"))
	if settings["defaultModel"] != "primary" || settings["editorPaddingX"] != float64(3) {
		t.Fatalf("unexpected settings: %v", settings)
	}
	history, _ := os.ReadFile(filepath.Join(dir, "history.jsonl"))
	if string(history) != "preserve history exactly\n" {
		t.Fatal("history changed")
	}
}

func TestPiLauncherStateUsesOnlyStartingModel(t *testing.T) {
	for _, tc := range []struct {
		primary       string
		cloudDisabled bool
		usable        bool
	}{
		{primary: "primary", usable: true},
		{primary: "missing"},
		{primary: "embeddings"},
		{primary: "remote:cloud", cloudDisabled: true},
		{primary: "remote:cloud", usable: true},
		{primary: "alias", cloudDisabled: true},
		{primary: "alias", usable: true},
	} {
		t.Run(fmt.Sprintf("%s/cloud-disabled=%t", tc.primary, tc.cloudDisabled), func(t *testing.T) {
			setLaunchTestHome(t, t.TempDir())
			if err := config.SaveIntegration("pi", []string{tc.primary, "secondary"}); err != nil {
				t.Fatal(err)
			}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/status":
					fmt.Fprintf(w, `{"cloud":{"disabled":%t}}`, tc.cloudDisabled)
				case "/api/tags":
					fmt.Fprint(w, `{"models":[{"name":"primary","capabilities":["completion","tools"]},{"name":"secondary","capabilities":["completion","tools"]},{"name":"embeddings","capabilities":["embedding"]},{"name":"alias","remote_model":"remote","capabilities":["completion","tools"]}]}`)
				case "/api/show":
					var req apiShowRequest
					_ = json.NewDecoder(r.Body).Decode(&req)
					switch req.Model {
					case "missing":
						http.Error(w, `{"error":"missing"}`, http.StatusNotFound)
					case "embeddings":
						fmt.Fprint(w, `{"capabilities":["embedding"]}`)
					case "alias", "remote:cloud":
						fmt.Fprint(w, `{"remote_model":"remote","capabilities":["completion","tools"]}`)
					default:
						fmt.Fprint(w, `{"capabilities":["completion","tools"]}`)
					}
				default:
					t.Errorf("unexpected request: %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)
			c, err := newLauncherClient(defaultLaunchPolicy(true, false))
			if err != nil {
				t.Fatal(err)
			}
			current, usable, err := c.launcherModelState(t.Context(), "pi", true)
			if err != nil || current != tc.primary || usable != tc.usable {
				t.Fatalf("launcher state = %s/%t, %v; want %s/%t", current, usable, err, tc.primary, tc.usable)
			}
		})
	}
}

func TestPiRemoteAliasAuthAndPayment(t *testing.T) {
	for _, payment := range []bool{false, true} {
		t.Run(fmt.Sprint(payment), func(t *testing.T) {
			home := t.TempDir()
			setLaunchTestHome(t, home)
			withLauncherHooks(t)
			if err := config.SaveIntegration("pi", []string{"primary"}); err != nil {
				t.Fatal(err)
			}
			requestedAuth, requestedPayment := false, false
			DefaultSignIn = func(model, _ string) (string, error) {
				requestedAuth = true
				if model != "alias" {
					t.Errorf("sign-in target = %s", model)
				}
				return "", ErrCancelled
			}
			DefaultConfirmPrompt = func(prompt string, _ ConfirmOptions) (bool, error) {
				requestedPayment = true
				if !strings.Contains(prompt, "paid:cloud") || !strings.Contains(prompt, "usage credits") {
					t.Errorf("unexpected payment prompt: %s", prompt)
				}
				return false, nil
			}
			DefaultSingleSelector = func(string, []SelectionItem, string) (string, error) {
				return "", ErrCancelled
			}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/status":
					fmt.Fprint(w, `{"cloud":{"disabled":false}}`)
				case "/api/tags":
					fmt.Fprint(w, `{"models":[{"name":"primary","capabilities":["completion","tools"]},{"name":"alias","remote_model":"paid","capabilities":["completion","tools"]}]}`)
				case "/api/show":
					fmt.Fprint(w, `{"remote_model":"paid","capabilities":["completion","tools"]}`)
				case "/api/experimental/model-recommendations":
					if payment {
						fmt.Fprint(w, `{"recommendations":[{"model":"paid:cloud","context_length":65536,"max_output_tokens":8192,"required_plan":"pro"}]}`)
					} else {
						fmt.Fprint(w, `{"recommendations":[{"model":"primary"}]}`)
					}
				case "/api/me":
					if payment {
						fmt.Fprint(w, `{"name":"fixture","plan":"free"}`)
					} else {
						w.WriteHeader(http.StatusUnauthorized)
						fmt.Fprint(w, `{"error":"unauthorized","signin_url":"https://example.com/signin"}`)
					}
				default:
					t.Errorf("unexpected request: %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)
			c, err := newLauncherClient(defaultLaunchPolicy(true, false))
			if err != nil {
				t.Fatal(err)
			}
			saved, _ := config.LoadIntegration("pi")
			err = c.launchPiIntegration(t.Context(), &Pi{}, saved, IntegrationLaunchRequest{ModelOverride: "alias"})
			if !errors.Is(err, ErrCancelled) || requestedPayment != payment || requestedAuth == payment {
				t.Fatalf("auth=%v, payment=%v, error=%v", requestedAuth, requestedPayment, err)
			}
			saved, _ = config.LoadIntegration("pi")
			if !slices.Equal(saved.Models, []string{"primary"}) {
				t.Fatalf("saved model changed after cancellation: %v", saved.Models)
			}
			if _, err := os.Stat(filepath.Join(home, ".pi")); !os.IsNotExist(err) {
				t.Fatalf("Pi config written after cancellation: %v", err)
			}
		})
	}
}

func TestPiCatalogInventoryErrorPreservesConfig(t *testing.T) {
	for _, corrupt := range []bool{false, true} {
		t.Run(fmt.Sprint(corrupt), func(t *testing.T) {
			home := t.TempDir()
			setLaunchTestHome(t, home)
			withLauncherHooks(t)
			if err := config.SaveIntegration("pi", []string{"primary", "secondary"}); err != nil {
				t.Fatal(err)
			}
			if corrupt {
				if err := os.WriteFile(filepath.Join(home, ".ollama", "config.json"), []byte("{invalid"), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			dir := filepath.Join(home, ".pi", "agent")
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			original := map[string]string{
				"models.json":   `{"providers":{"ollama":{"models":[{"id":"primary","_launch":true},{"id":"secondary","_launch":true}]}}}`,
				"settings.json": `{"defaultProvider":"ollama","defaultModel":"secondary","theme":"dark"}`,
			}
			for name, content := range original {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			selected := false
			DefaultSingleSelector = func(string, []SelectionItem, string) (string, error) {
				selected = true
				return "primary", nil
			}
			DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return false, nil }
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if !corrupt {
					http.Error(w, `{"error":"unavailable"}`, http.StatusServiceUnavailable)
					return
				}
				switch r.URL.Path {
				case "/api/status":
					fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
				case "/api/experimental/model-recommendations":
					fmt.Fprint(w, `{"recommendations":[{"model":"primary"}]}`)
				case "/api/tags":
					fmt.Fprint(w, `{"models":[{"name":"primary","details":{"context_length":8192},"capabilities":["completion","tools"]}]}`)
				default:
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)
			if err := LaunchIntegration(t.Context(), IntegrationLaunchRequest{Name: "pi", ConfigureOnly: true}); err == nil {
				t.Fatal("expected config or inventory error")
			}
			for name, want := range original {
				data, err := os.ReadFile(filepath.Join(dir, name))
				if err != nil || string(data) != want {
					t.Errorf("%s changed after failed setup: %s, %v", name, data, err)
				}
			}
			if corrupt {
				data, err := os.ReadFile(filepath.Join(home, ".ollama", "config.json"))
				if err != nil || string(data) != "{invalid" || selected {
					t.Fatalf("invalid config changed or picker opened: %s, %v, selected=%t", data, err, selected)
				}
			} else {
				saved, _ := config.LoadIntegration("pi")
				if !slices.Equal(saved.Models, []string{"primary", "secondary"}) {
					t.Fatalf("saved models changed: %v", saved.Models)
				}
			}
		})
	}
}

func TestPiShowBudgetDoesNotChargeStatus(t *testing.T) {
	setLaunchTestHome(t, t.TempDir())
	withLauncherHooks(t)
	oldTimeout := modelShowTimeout
	modelShowTimeout = 40 * time.Millisecond
	t.Cleanup(func() { modelShowTimeout = oldTimeout })
	if err := config.SaveIntegration("pi", []string{"alias"}); err != nil {
		t.Fatal(err)
	}
	DefaultSingleSelector = func(string, []SelectionItem, string) (string, error) { return "alias", nil }
	DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return false, nil }
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			time.Sleep(100 * time.Millisecond)
			fmt.Fprint(w, `{"cloud":{"disabled":false}}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"alias","remote_model":"remote","capabilities":["completion","tools"]}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"remote_model":"remote","capabilities":["completion","tools"],"model_info":{"test.context_length":8192}}`)
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[{"model":"alias"}]}`)
		case "/api/me":
			fmt.Fprint(w, `{"name":"fixture"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	c, err := newLauncherClient(defaultLaunchPolicy(true, false))
	if err != nil {
		t.Fatal(err)
	}
	current, usable, stateErr := c.launcherModelState(ctx, "pi", true)
	if stateErr != nil || current != "alias" || !usable {
		t.Errorf("status latency made the root primary unusable: %s/%t, %v", current, usable, stateErr)
	}
	if err := LaunchIntegration(ctx, IntegrationLaunchRequest{Name: "pi", ConfigureOnly: true}); err != nil {
		t.Fatalf("status latency was charged to /show: %v", err)
	}
}

func TestPiCatalogMetadataFallback(t *testing.T) {
	for _, status := range []int{http.StatusNotFound, http.StatusUnauthorized, http.StatusServiceUnavailable} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			home := t.TempDir()
			setLaunchTestHome(t, home)
			dir := filepath.Join(home, ".pi", "agent")
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			original := `{"providers":{"ollama":{"models":[{"id":"secondary:cloud","_launch":true,"contextWindow":8192,"input":["text","image"],"reasoning":true}]}}}`
			if err := os.WriteFile(filepath.Join(dir, "models.json"), []byte(original), 0o644); err != nil {
				t.Fatal(err)
			}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/status":
					fmt.Fprint(w, `{"cloud":{"disabled":false}}`)
				case "/api/experimental/model-recommendations":
					fmt.Fprint(w, `{"recommendations":[{"model":"primary"}]}`)
				case "/api/tags":
					fmt.Fprint(w, `{"models":[{"name":"primary","details":{"context_length":32768},"capabilities":["completion","tools"]},{"name":"secondary:cloud","remote_model":"secondary"},{"name":"legacy"}]}`)
				case "/api/show":
					var req apiShowRequest
					_ = json.NewDecoder(r.Body).Decode(&req)
					if req.Model == "legacy" {
						fmt.Fprint(w, `{"capabilities":["completion","tools","vision"],"model_info":{"test.context_length":16384}}`)
					} else {
						http.Error(w, `{"error":"fixture"}`, status)
					}
				default:
					t.Errorf("unexpected request: %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)
			c, err := newLauncherClient(defaultLaunchPolicy(true, false))
			if err != nil {
				t.Fatal(err)
			}
			models, err := c.piModelCatalog(t.Context(), LaunchModel{Name: "primary"}, modelShowTimeout)
			if err != nil {
				t.Fatal(err)
			}
			if err := (&Pi{}).Edit(models); err != nil {
				t.Fatal(err)
			}
			cfg, _ := fileutil.ReadJSON(filepath.Join(dir, "models.json"))
			entries := cfg["providers"].(map[string]any)["ollama"].(map[string]any)["models"].([]any)
			byID := make(map[string]map[string]any)
			for _, raw := range entries {
				entry := raw.(map[string]any)
				byID[entry["id"].(string)] = entry
			}
			if entry := byID["legacy"]; entry["contextWindow"] != float64(16384) || !slices.Equal([]any{"text", "image"}, entry["input"].([]any)) {
				t.Fatalf("legacy metadata fallback = %v", entry)
			}
			entry := byID["secondary:cloud"]
			if status == http.StatusNotFound {
				if entry != nil {
					t.Fatalf("confirmed missing model remains: %v", entry)
				}
			} else if entry["contextWindow"] != float64(8192) || entry["reasoning"] != true || !slices.Equal([]any{"text", "image"}, entry["input"].([]any)) {
				t.Fatalf("known metadata lost after %d: %v", status, entry)
			}
		})
	}
}

func TestPiEditPreservesInvalidConfig(t *testing.T) {
	home := t.TempDir()
	setLaunchTestHome(t, home)
	dir := filepath.Join(home, ".pi", "agent")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	models := `{"providers":{"ollama":{"models":[{"id":"missing","_launch":true}]}}}`
	if err := os.WriteFile(filepath.Join(dir, "models.json"), []byte(models), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "settings.json"), []byte("{invalid"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := (&Pi{}).Edit([]LaunchModel{{Name: "chosen"}}); err == nil {
		t.Fatal("expected invalid settings error")
	}
	data, err := os.ReadFile(filepath.Join(dir, "models.json"))
	if err != nil || string(data) != models {
		t.Fatalf("models changed before invalid settings were detected: %s, %v", data, err)
	}
}

// A selected primary that was already resolved must not be looked up again by
// catalog enrichment. The second response deliberately changes to 404 to make
// the duplicate RPC's user-visible consequence deterministic.
func TestPiResolvedPrimaryIsNotShownTwice(t *testing.T) {
	setLaunchTestHome(t, t.TempDir())
	var showCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"primary"}]}`)
		case "/api/show":
			if showCalls.Add(1) == 1 {
				fmt.Fprint(w, `{"capabilities":["completion"]}`)
				return
			}
			http.Error(w, `{"error":"gone"}`, http.StatusNotFound)
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[{"model":"primary"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)
	c, err := newLauncherClient(defaultLaunchPolicy(true, false))
	if err != nil {
		t.Fatal(err)
	}
	remaining := modelShowTimeout
	primary, err := c.piTargetModel(t.Context(), "primary", &remaining)
	if err != nil || !primary.HasCapability(modelpkg.CapabilityCompletion) {
		t.Fatalf("target resolution = %+v, %v", primary, err)
	}
	models, err := c.piModelCatalog(t.Context(), primary, remaining)
	if err != nil {
		t.Fatalf("resolved primary was invalidated by catalog enrichment: %v", err)
	}
	if showCalls.Load() != 1 || len(models) == 0 || models[0].Name != "primary" {
		t.Fatalf("show calls=%d models=%v", showCalls.Load(), models)
	}
}

// launchModelMatches and catalog de-duplication treat id and id:latest as the
// same model. Metadata preservation must use the same identity rule.
func TestPiMetadataFallbackNormalizesLatest(t *testing.T) {
	home := t.TempDir()
	setLaunchTestHome(t, home)
	dir := filepath.Join(home, ".pi", "agent")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "models.json"), []byte(`{"providers":{"ollama":{"models":[{"id":"secondary","_launch":true,"contextWindow":8192,"input":["text","image"],"reasoning":true}]}}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"primary","details":{"context_length":32768},"capabilities":["completion"]},{"name":"secondary:latest"}]}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Model == "secondary:latest" {
				http.Error(w, `{"error":"temporary"}`, http.StatusServiceUnavailable)
				return
			}
			http.NotFound(w, r)
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[{"model":"primary"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)
	c, err := newLauncherClient(defaultLaunchPolicy(true, false))
	if err != nil {
		t.Fatal(err)
	}
	models, err := c.piModelCatalog(t.Context(), LaunchModel{Name: "primary", ContextLength: 32768, Capabilities: []modelpkg.Capability{modelpkg.CapabilityCompletion}}, modelShowTimeout)
	if err != nil {
		t.Fatal(err)
	}
	if err := (&Pi{}).Edit(models); err != nil {
		t.Fatal(err)
	}
	cfg, err := fileutil.ReadJSON(filepath.Join(dir, "models.json"))
	if err != nil {
		t.Fatal(err)
	}
	entries := cfg["providers"].(map[string]any)["ollama"].(map[string]any)["models"].([]any)
	var secondary map[string]any
	for _, raw := range entries {
		entry := raw.(map[string]any)
		if strings.TrimSuffix(entry["id"].(string), ":latest") == "secondary" {
			secondary = entry
		}
	}
	if secondary == nil || secondary["contextWindow"] != float64(8192) || secondary["reasoning"] != true {
		t.Fatalf("known fields were not preserved across :latest identity: %v", secondary)
	}
	input, _ := secondary["input"].([]any)
	if len(input) != 2 || input[1] != "image" {
		t.Fatalf("known input capability was not preserved: %v", secondary)
	}
}

// Both gpt-oss:20b:cloud and gpt-oss:20b-cloud are supported spellings for
// the same cloud source. Required-plan guidance must not depend on spelling.
func TestPiTaggedAliasRetainsPlanGuidance(t *testing.T) {
	withLauncherHooks(t)
	prompted := false
	DefaultConfirmPrompt = func(prompt string, _ ConfirmOptions) (bool, error) {
		prompted = true
		if !strings.Contains(prompt, "usage credits") {
			t.Fatalf("unexpected prompt: %s", prompt)
		}
		return false, nil
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[{"model":"gpt-oss:20b-cloud","context_length":131072,"max_output_tokens":131072,"required_plan":"pro"}]}`)
		case "/api/me":
			fmt.Fprint(w, `{"name":"fixture","plan":"free"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)
	c, err := newLauncherClient(defaultLaunchPolicy(true, false))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"gpt-oss:20b:cloud", "gpt-oss:20b-cloud"} {
		prompted = false
		err = c.confirmCloudModelPayment(t.Context(), name)
		if !errors.Is(err, errModelPaymentDeclined) || !prompted {
			t.Fatalf("equivalent tagged cloud name skipped plan guidance: name=%s prompted=%t err=%v", name, prompted, err)
		}
	}
}

func TestPiManualEquivalentPrimaryKeepsItsID(t *testing.T) {
	home := t.TempDir()
	setLaunchTestHome(t, home)
	dir := filepath.Join(home, ".pi", "agent")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "models.json"), []byte(`{"providers":{"ollama":{"models":[{"id":"primary:latest","contextWindow":4096,"custom":true}]}}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	primary, err := (&Pi{}).writeConfig([]LaunchModel{{Name: "primary", ContextLength: 32768}}, false)
	if err != nil || primary != "primary:latest" {
		t.Fatalf("configured primary = %q, %v", primary, err)
	}
	settings, _ := fileutil.ReadJSON(filepath.Join(dir, "settings.json"))
	if settings["defaultModel"] != primary {
		t.Fatalf("Pi default does not name the preserved entry: %v", settings)
	}
	cfg, _ := fileutil.ReadJSON(filepath.Join(dir, "models.json"))
	entries := cfg["providers"].(map[string]any)["ollama"].(map[string]any)["models"].([]any)
	if len(entries) != 1 || entries[0].(map[string]any)["custom"] != true {
		t.Fatalf("manual equivalent was duplicated or rewritten: %v", entries)
	}
}
