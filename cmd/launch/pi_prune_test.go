package launch

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
)

func TestPiPruneSavedModels(t *testing.T) {
	for _, status := range []int{http.StatusNotFound, http.StatusUnauthorized, http.StatusServiceUnavailable} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			home := t.TempDir()
			setLaunchTestHome(t, home)
			if err := config.SaveIntegration("pi", []string{"missing"}); err != nil {
				t.Fatal(err)
			}
			dir := filepath.Join(home, ".pi", "agent")
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			for name, content := range map[string]string{
				"models.json":   `{"providers":{"ollama":{"models":[{"id":"missing","_launch":true},{"id":"manual","contextWindow":8192}]},"other":{"models":[{"id":"keep"}]}}}`,
				"settings.json": `{"defaultProvider":"ollama","defaultModel":"missing","editorPaddingX":3}`,
			} {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(status)
				fmt.Fprint(w, `{"error":"fixture"}`)
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)
			client, err := api.ClientFromEnvironment()
			if err != nil {
				t.Fatal(err)
			}
			saved, err := config.LoadIntegration("pi")
			if err != nil {
				t.Fatal(err)
			}
			updated, err := (&Pi{}).pruneSavedModels(t.Context(), client, saved)
			if err != nil {
				t.Fatal(err)
			}
			want := []string{"missing"}
			if status == http.StatusNotFound {
				want = nil
			}
			persisted, err := config.LoadIntegration("pi")
			if err != nil || !slices.Equal(updated.Models, want) || !slices.Equal(persisted.Models, want) {
				t.Fatalf("saved models = %v, %v, %v; want %v", updated, persisted, err, want)
			}
			models, _ := fileutil.ReadJSON(filepath.Join(dir, "models.json"))
			providers := models["providers"].(map[string]any)
			entries := providers["ollama"].(map[string]any)["models"].([]any)
			if len(entries) != len(want)+1 || entries[len(entries)-1].(map[string]any)["id"] != "manual" {
				t.Fatalf("unexpected managed/manual models: %v", entries)
			}
			if providers["other"].(map[string]any)["models"].([]any)[0].(map[string]any)["id"] != "keep" {
				t.Fatal("unrelated provider changed")
			}
			settings, _ := fileutil.ReadJSON(filepath.Join(dir, "settings.json"))
			if (settings["defaultModel"] == nil) != (status == http.StatusNotFound) || settings["editorPaddingX"] != float64(3) {
				t.Fatalf("unexpected settings: %v", settings)
			}
		})
	}
}

func TestPiRemoveModelsPreservesInvalidConfig(t *testing.T) {
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
	if err := (&Pi{}).removeModels([]string{"missing"}); err == nil {
		t.Fatal("expected invalid settings error")
	}
	data, err := os.ReadFile(filepath.Join(dir, "models.json"))
	if err != nil || !json.Valid(data) || string(data) != models {
		t.Fatalf("models changed before invalid settings were detected: %s, %v", data, err)
	}
}
