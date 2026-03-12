package config

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestOpenclawIntegration(t *testing.T) {
	c := &Openclaw{}

	t.Run("String", func(t *testing.T) {
		if got := c.String(); got != "OpenClaw" {
			t.Errorf("String() = %q, want %q", got, "OpenClaw")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = c
	})

	t.Run("implements Editor", func(t *testing.T) {
		var _ Editor = c
	})
}

func TestOpenclawRunPassthroughArgs(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("PATH", tmpDir)

	if err := integrationOnboarded("openclaw"); err != nil {
		t.Fatal(err)
	}

	configDir := filepath.Join(tmpDir, ".openclaw")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{
		"wizard": {"lastRunAt": "2026-01-01T00:00:00Z"}
	}`), 0o644); err != nil {
		t.Fatal(err)
	}

	bin := filepath.Join(tmpDir, "openclaw")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$HOME/invocations.log\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	c := &Openclaw{}
	if err := c.Run("llama3.2", []string{"gateway", "--someflag"}); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, "invocations.log"))
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected exactly 1 invocation, got %d: %v", len(lines), lines)
	}
	if lines[0] != "gateway --someflag" {
		t.Fatalf("invocation = %q, want %q", lines[0], "gateway --someflag")
	}
}

func TestOpenclawRunFirstLaunchPersistence(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	oldHook := DefaultConfirmPrompt
	DefaultConfirmPrompt = func(prompt string) (bool, error) {
		return true, nil
	}
	defer func() { DefaultConfirmPrompt = oldHook }()

	t.Run("success persists onboarding flag", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("PATH", tmpDir)

		configDir := filepath.Join(tmpDir, ".openclaw")
		if err := os.MkdirAll(configDir, 0o755); err != nil {
			t.Fatal(err)
		}
		// Mark OpenClaw onboarding complete so Run takes passthrough path directly.
		if err := os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{
			"wizard": {"lastRunAt": "2026-01-01T00:00:00Z"}
		}`), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(tmpDir, "openclaw"), []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
			t.Fatal(err)
		}

		c := &Openclaw{}
		if err := c.Run("llama3.2", []string{"gateway", "--status"}); err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		integrationConfig, err := loadIntegration("openclaw")
		if err != nil {
			t.Fatalf("loadIntegration() error = %v", err)
		}
		if !integrationConfig.Onboarded {
			t.Fatal("expected onboarding flag to be persisted after successful run")
		}
	})

	t.Run("failure does not persist onboarding flag", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("PATH", tmpDir)

		configDir := filepath.Join(tmpDir, ".openclaw")
		if err := os.MkdirAll(configDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{
			"wizard": {"lastRunAt": "2026-01-01T00:00:00Z"}
		}`), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(tmpDir, "openclaw"), []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
			t.Fatal(err)
		}

		c := &Openclaw{}
		if err := c.Run("llama3.2", []string{"gateway", "--status"}); err == nil {
			t.Fatal("expected run failure")
		}
		integrationConfig, err := loadIntegration("openclaw")
		if err == nil && integrationConfig.Onboarded {
			t.Fatal("expected onboarding flag to remain unset after failed run")
		}
	})
}

func TestOpenclawEdit(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".openclaw")
	configPath := filepath.Join(configDir, "openclaw.json")

	cleanup := func() { os.RemoveAll(configDir) }

	t.Run("fresh install", func(t *testing.T) {
		cleanup()
		if err := c.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		assertOpenclawModelExists(t, configPath, "llama3.2")
		assertOpenclawPrimaryModel(t, configPath, "ollama/llama3.2")
	})

	t.Run("multiple models - first is primary", func(t *testing.T) {
		cleanup()
		if err := c.Edit([]string{"llama3.2", "mistral"}); err != nil {
			t.Fatal(err)
		}
		assertOpenclawModelExists(t, configPath, "llama3.2")
		assertOpenclawModelExists(t, configPath, "mistral")
		assertOpenclawPrimaryModel(t, configPath, "ollama/llama3.2")
	})

	t.Run("preserve other providers", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":{"providers":{"anthropic":{"apiKey":"xxx"}}}}`), 0o644)
		if err := c.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		models := cfg["models"].(map[string]any)
		providers := models["providers"].(map[string]any)
		if providers["anthropic"] == nil {
			t.Error("anthropic provider was removed")
		}
	})

	t.Run("preserve top-level keys", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"theme":"dark","mcp":{"servers":{}}}`), 0o644)
		if err := c.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		if cfg["theme"] != "dark" {
			t.Error("theme was removed")
		}
		if cfg["mcp"] == nil {
			t.Error("mcp was removed")
		}
	})

	t.Run("preserve user customizations on models", func(t *testing.T) {
		cleanup()
		c.Edit([]string{"llama3.2"})

		// User adds custom field
		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		models := cfg["models"].(map[string]any)
		providers := models["providers"].(map[string]any)
		ollama := providers["ollama"].(map[string]any)
		modelList := ollama["models"].([]any)
		entry := modelList[0].(map[string]any)
		entry["customField"] = "user-value"
		configData, _ := json.MarshalIndent(cfg, "", "  ")
		os.WriteFile(configPath, configData, 0o644)

		// Re-run Edit
		c.Edit([]string{"llama3.2"})

		data, _ = os.ReadFile(configPath)
		json.Unmarshal(data, &cfg)
		models = cfg["models"].(map[string]any)
		providers = models["providers"].(map[string]any)
		ollama = providers["ollama"].(map[string]any)
		modelList = ollama["models"].([]any)
		entry = modelList[0].(map[string]any)
		if entry["customField"] != "user-value" {
			t.Error("custom field was lost")
		}
	})

	t.Run("edit replaces models list", func(t *testing.T) {
		cleanup()
		c.Edit([]string{"llama3.2", "mistral"})
		c.Edit([]string{"llama3.2"})

		assertOpenclawModelExists(t, configPath, "llama3.2")
		assertOpenclawModelNotExists(t, configPath, "mistral")
	})

	t.Run("empty models is no-op", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		original := `{"existing":"data"}`
		os.WriteFile(configPath, []byte(original), 0o644)

		c.Edit([]string{})

		data, _ := os.ReadFile(configPath)
		if string(data) != original {
			t.Error("empty models should not modify file")
		}
	})

	t.Run("corrupted JSON treated as empty", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{corrupted`), 0o644)

		if err := c.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		if err := json.Unmarshal(data, &cfg); err != nil {
			t.Error("result should be valid JSON")
		}
	})

	t.Run("wrong type models section", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":"not a map"}`), 0o644)

		if err := c.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		assertOpenclawModelExists(t, configPath, "llama3.2")
	})
}

func TestOpenclawModels(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	t.Run("no config returns nil", func(t *testing.T) {
		if models := c.Models(); len(models) > 0 {
			t.Errorf("expected nil/empty, got %v", models)
		}
	})

	t.Run("returns all ollama models", func(t *testing.T) {
		configDir := filepath.Join(tmpDir, ".openclaw")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{
			"models":{"providers":{"ollama":{"models":[
				{"id":"llama3.2"},
				{"id":"mistral"}
			]}}}
		}`), 0o644)

		models := c.Models()
		if len(models) != 2 {
			t.Errorf("expected 2 models, got %v", models)
		}
	})
}

// Helper functions
func assertOpenclawModelExists(t *testing.T, path, model string) {
	t.Helper()
	data, _ := os.ReadFile(path)
	var cfg map[string]any
	json.Unmarshal(data, &cfg)
	models := cfg["models"].(map[string]any)
	providers := models["providers"].(map[string]any)
	ollama := providers["ollama"].(map[string]any)
	modelList := ollama["models"].([]any)
	for _, m := range modelList {
		if entry, ok := m.(map[string]any); ok {
			if entry["id"] == model {
				return
			}
		}
	}
	t.Errorf("model %s not found", model)
}

func assertOpenclawModelNotExists(t *testing.T, path, model string) {
	t.Helper()
	data, _ := os.ReadFile(path)
	var cfg map[string]any
	json.Unmarshal(data, &cfg)
	models, _ := cfg["models"].(map[string]any)
	providers, _ := models["providers"].(map[string]any)
	ollama, _ := providers["ollama"].(map[string]any)
	modelList, _ := ollama["models"].([]any)
	for _, m := range modelList {
		if entry, ok := m.(map[string]any); ok {
			if entry["id"] == model {
				t.Errorf("model %s should not exist", model)
			}
		}
	}
}

func assertOpenclawPrimaryModel(t *testing.T, path, expected string) {
	t.Helper()
	data, _ := os.ReadFile(path)
	var cfg map[string]any
	json.Unmarshal(data, &cfg)
	agents := cfg["agents"].(map[string]any)
	defaults := agents["defaults"].(map[string]any)
	model := defaults["model"].(map[string]any)
	if model["primary"] != expected {
		t.Errorf("primary model = %v, want %v", model["primary"], expected)
	}
}

func TestOpenclawPaths(t *testing.T) {
	c := &Openclaw{}

	t.Run("returns path when config exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		configDir := filepath.Join(tmpDir, ".openclaw")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{}`), 0o644)

		paths := c.Paths()
		if len(paths) != 1 {
			t.Errorf("expected 1 path, got %d", len(paths))
		}
	})

	t.Run("returns nil when config missing", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		if paths := c.Paths(); paths != nil {
			t.Errorf("expected nil, got %v", paths)
		}
	})
}

func TestOpenclawModelsEdgeCases(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".openclaw")
	configPath := filepath.Join(configDir, "openclaw.json")
	cleanup := func() { os.RemoveAll(configDir) }

	t.Run("corrupted JSON returns nil", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{corrupted`), 0o644)
		if models := c.Models(); models != nil {
			t.Errorf("expected nil, got %v", models)
		}
	})

	t.Run("wrong type at models level", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":"string"}`), 0o644)
		if models := c.Models(); models != nil {
			t.Errorf("expected nil, got %v", models)
		}
	})

	t.Run("wrong type at providers level", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":{"providers":"string"}}`), 0o644)
		if models := c.Models(); models != nil {
			t.Errorf("expected nil, got %v", models)
		}
	})

	t.Run("wrong type at ollama level", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":{"providers":{"ollama":"string"}}}`), 0o644)
		if models := c.Models(); models != nil {
			t.Errorf("expected nil, got %v", models)
		}
	})

	t.Run("model entry missing id", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":{"providers":{"ollama":{"models":[{"name":"test"}]}}}}`), 0o644)
		if len(c.Models()) != 0 {
			t.Error("expected empty for missing id")
		}
	})

	t.Run("model id is not string", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":{"providers":{"ollama":{"models":[{"id":123}]}}}}`), 0o644)
		if len(c.Models()) != 0 {
			t.Error("expected empty for non-string id")
		}
	})
}

func TestOpenclawEditSchemaFields(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configPath := filepath.Join(tmpDir, ".openclaw", "openclaw.json")

	if err := c.Edit([]string{"llama3.2"}); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(configPath)
	var cfg map[string]any
	json.Unmarshal(data, &cfg)
	models := cfg["models"].(map[string]any)
	providers := models["providers"].(map[string]any)
	ollama := providers["ollama"].(map[string]any)
	modelList := ollama["models"].([]any)
	entry := modelList[0].(map[string]any)

	// Verify base schema fields (always set regardless of API availability)
	if entry["id"] != "llama3.2" {
		t.Errorf("id = %v, want llama3.2", entry["id"])
	}
	if entry["name"] != "llama3.2" {
		t.Errorf("name = %v, want llama3.2", entry["name"])
	}
	if entry["input"] == nil {
		t.Error("input should be set")
	}
	cost := entry["cost"].(map[string]any)
	if cost["cacheRead"] == nil {
		t.Error("cost.cacheRead should be set")
	}
	if cost["cacheWrite"] == nil {
		t.Error("cost.cacheWrite should be set")
	}
}

func TestOpenclawEditModelNames(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configPath := filepath.Join(tmpDir, ".openclaw", "openclaw.json")
	cleanup := func() { os.RemoveAll(filepath.Join(tmpDir, ".openclaw")) }

	t.Run("model with colon tag", func(t *testing.T) {
		cleanup()
		if err := c.Edit([]string{"llama3.2:70b"}); err != nil {
			t.Fatal(err)
		}
		assertOpenclawModelExists(t, configPath, "llama3.2:70b")
		assertOpenclawPrimaryModel(t, configPath, "ollama/llama3.2:70b")
	})

	t.Run("model with slash", func(t *testing.T) {
		cleanup()
		if err := c.Edit([]string{"library/model:tag"}); err != nil {
			t.Fatal(err)
		}
		assertOpenclawModelExists(t, configPath, "library/model:tag")
		assertOpenclawPrimaryModel(t, configPath, "ollama/library/model:tag")
	})

	t.Run("model with hyphen", func(t *testing.T) {
		cleanup()
		if err := c.Edit([]string{"test-model"}); err != nil {
			t.Fatal(err)
		}
		assertOpenclawModelExists(t, configPath, "test-model")
	})
}

func TestOpenclawEditAgentsPreservation(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".openclaw")
	configPath := filepath.Join(configDir, "openclaw.json")
	cleanup := func() { os.RemoveAll(configDir) }

	t.Run("preserve other agent defaults", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"agents":{"defaults":{"model":{"primary":"old"},"temperature":0.7}}}`), 0o644)

		c.Edit([]string{"llama3.2"})

		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		agents := cfg["agents"].(map[string]any)
		defaults := agents["defaults"].(map[string]any)
		if defaults["temperature"] != 0.7 {
			t.Error("temperature setting was lost")
		}
	})

	t.Run("preserve other agents besides defaults", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"agents":{"defaults":{},"custom-agent":{"foo":"bar"}}}`), 0o644)

		c.Edit([]string{"llama3.2"})

		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		agents := cfg["agents"].(map[string]any)
		if agents["custom-agent"] == nil {
			t.Error("custom-agent was lost")
		}
	})
}

const testOpenclawFixture = `{
  "theme": "dark",
  "mcp": {"servers": {"custom": {"enabled": true}}},
  "models": {
    "providers": {
      "anthropic": {"apiKey": "xxx"},
      "ollama": {
        "baseUrl": "http://127.0.0.1:11434",
        "models": [{"id": "old-model", "customField": "preserved"}]
      }
    }
  },
  "agents": {
    "defaults": {"model": {"primary": "old"}, "temperature": 0.7},
    "custom-agent": {"foo": "bar"}
  }
}`

func TestOpenclawEdit_RoundTrip(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".openclaw")
	configPath := filepath.Join(configDir, "openclaw.json")

	os.MkdirAll(configDir, 0o755)
	os.WriteFile(configPath, []byte(testOpenclawFixture), 0o644)

	if err := c.Edit([]string{"llama3.2", "mistral"}); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(configPath)
	var cfg map[string]any
	json.Unmarshal(data, &cfg)

	// Verify top-level preserved
	if cfg["theme"] != "dark" {
		t.Error("theme not preserved")
	}
	mcp := cfg["mcp"].(map[string]any)
	servers := mcp["servers"].(map[string]any)
	if servers["custom"] == nil {
		t.Error("mcp.servers.custom not preserved")
	}

	// Verify other providers preserved
	models := cfg["models"].(map[string]any)
	providers := models["providers"].(map[string]any)
	if providers["anthropic"] == nil {
		t.Error("anthropic provider not preserved")
	}

	// Verify agents preserved
	agents := cfg["agents"].(map[string]any)
	if agents["custom-agent"] == nil {
		t.Error("custom-agent not preserved")
	}
	defaults := agents["defaults"].(map[string]any)
	if defaults["temperature"] != 0.7 {
		t.Error("temperature not preserved")
	}
}

func TestOpenclawEdit_Idempotent(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".openclaw")
	configPath := filepath.Join(configDir, "openclaw.json")

	os.MkdirAll(configDir, 0o755)
	os.WriteFile(configPath, []byte(testOpenclawFixture), 0o644)

	c.Edit([]string{"llama3.2", "mistral"})
	firstData, _ := os.ReadFile(configPath)

	c.Edit([]string{"llama3.2", "mistral"})
	secondData, _ := os.ReadFile(configPath)

	if string(firstData) != string(secondData) {
		t.Error("repeated edits with same models produced different results")
	}
}

func TestOpenclawEdit_MultipleConsecutiveEdits(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".openclaw")
	configPath := filepath.Join(configDir, "openclaw.json")

	os.MkdirAll(configDir, 0o755)
	os.WriteFile(configPath, []byte(testOpenclawFixture), 0o644)

	for i := range 10 {
		models := []string{"model-a", "model-b"}
		if i%2 == 0 {
			models = []string{"model-x", "model-y", "model-z"}
		}
		if err := c.Edit(models); err != nil {
			t.Fatalf("edit %d failed: %v", i, err)
		}
	}

	data, _ := os.ReadFile(configPath)
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("file is not valid JSON after multiple edits: %v", err)
	}

	if cfg["theme"] != "dark" {
		t.Error("theme lost after multiple edits")
	}
}

func TestOpenclawEdit_BackupCreated(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".openclaw")
	configPath := filepath.Join(configDir, "openclaw.json")
	backupDir := filepath.Join(os.TempDir(), "ollama-backups")

	os.MkdirAll(configDir, 0o755)
	uniqueMarker := fmt.Sprintf("test-marker-%d", os.Getpid())
	original := fmt.Sprintf(`{"theme": "%s"}`, uniqueMarker)
	os.WriteFile(configPath, []byte(original), 0o644)

	if err := c.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	backups, _ := filepath.Glob(filepath.Join(backupDir, "openclaw.json.*"))
	foundBackup := false
	for _, backup := range backups {
		data, _ := os.ReadFile(backup)
		if string(data) == original {
			foundBackup = true
			break
		}
	}

	if !foundBackup {
		t.Error("backup with original content not found")
	}
}

func TestOpenclawClawdbotAlias(t *testing.T) {
	for _, alias := range []string{"clawdbot", "moltbot"} {
		t.Run(alias+" alias resolves to Openclaw runner", func(t *testing.T) {
			r, ok := integrations[alias]
			if !ok {
				t.Fatalf("%s not found in integrations", alias)
			}
			if _, ok := r.(*Openclaw); !ok {
				t.Errorf("%s integration is %T, want *Openclaw", alias, r)
			}
		})

		t.Run(alias+" is hidden from selector", func(t *testing.T) {
			if !integrationAliases[alias] {
				t.Errorf("%s should be in integrationAliases", alias)
			}
		})
	}
}

func TestOpenclawLegacyPaths(t *testing.T) {
	c := &Openclaw{}

	t.Run("falls back to legacy clawdbot path", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		legacyDir := filepath.Join(tmpDir, ".clawdbot")
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(legacyDir, "clawdbot.json"), []byte(`{}`), 0o644)

		paths := c.Paths()
		if len(paths) != 1 {
			t.Fatalf("expected 1 path, got %d", len(paths))
		}
		if paths[0] != filepath.Join(legacyDir, "clawdbot.json") {
			t.Errorf("expected legacy path, got %s", paths[0])
		}
	})

	t.Run("prefers new path over legacy", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		newDir := filepath.Join(tmpDir, ".openclaw")
		legacyDir := filepath.Join(tmpDir, ".clawdbot")
		os.MkdirAll(newDir, 0o755)
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(newDir, "openclaw.json"), []byte(`{}`), 0o644)
		os.WriteFile(filepath.Join(legacyDir, "clawdbot.json"), []byte(`{}`), 0o644)

		paths := c.Paths()
		if len(paths) != 1 {
			t.Fatalf("expected 1 path, got %d", len(paths))
		}
		if paths[0] != filepath.Join(newDir, "openclaw.json") {
			t.Errorf("expected new path, got %s", paths[0])
		}
	})

	t.Run("Models reads from legacy path", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		legacyDir := filepath.Join(tmpDir, ".clawdbot")
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(legacyDir, "clawdbot.json"), []byte(`{
			"models":{"providers":{"ollama":{"models":[{"id":"llama3.2"}]}}}
		}`), 0o644)

		models := c.Models()
		if len(models) != 1 || models[0] != "llama3.2" {
			t.Errorf("expected [llama3.2], got %v", models)
		}
	})

	t.Run("Models prefers new path over legacy", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		newDir := filepath.Join(tmpDir, ".openclaw")
		legacyDir := filepath.Join(tmpDir, ".clawdbot")
		os.MkdirAll(newDir, 0o755)
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(newDir, "openclaw.json"), []byte(`{
			"models":{"providers":{"ollama":{"models":[{"id":"new-model"}]}}}
		}`), 0o644)
		os.WriteFile(filepath.Join(legacyDir, "clawdbot.json"), []byte(`{
			"models":{"providers":{"ollama":{"models":[{"id":"legacy-model"}]}}}
		}`), 0o644)

		models := c.Models()
		if len(models) != 1 || models[0] != "new-model" {
			t.Errorf("expected [new-model], got %v", models)
		}
	})

	t.Run("Edit reads new path over legacy when both exist", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		newDir := filepath.Join(tmpDir, ".openclaw")
		legacyDir := filepath.Join(tmpDir, ".clawdbot")
		os.MkdirAll(newDir, 0o755)
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(newDir, "openclaw.json"), []byte(`{"theme":"new"}`), 0o644)
		os.WriteFile(filepath.Join(legacyDir, "clawdbot.json"), []byte(`{"theme":"legacy"}`), 0o644)

		if err := c.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(filepath.Join(newDir, "openclaw.json"))
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		if cfg["theme"] != "new" {
			t.Errorf("expected theme from new config, got %v", cfg["theme"])
		}
	})

	t.Run("Edit migrates from legacy config", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		legacyDir := filepath.Join(tmpDir, ".clawdbot")
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(legacyDir, "clawdbot.json"), []byte(`{"theme":"dark"}`), 0o644)

		if err := c.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}

		// Should write to new path
		newPath := filepath.Join(tmpDir, ".openclaw", "openclaw.json")
		data, err := os.ReadFile(newPath)
		if err != nil {
			t.Fatal("expected new config file to be created")
		}
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		if cfg["theme"] != "dark" {
			t.Error("legacy theme setting was not migrated")
		}
	})
}

func TestOpenclawEdit_CreatesDirectoryIfMissing(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".openclaw")

	if _, err := os.Stat(configDir); !os.IsNotExist(err) {
		t.Fatal("directory should not exist before test")
	}

	if err := c.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(configDir); os.IsNotExist(err) {
		t.Fatal("directory was not created")
	}
}

func TestOpenclawOnboarded(t *testing.T) {
	c := &Openclaw{}

	t.Run("returns false when no config exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		if c.onboarded() {
			t.Error("expected false when no config exists")
		}
	})

	t.Run("returns false when config exists but no wizard section", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		configDir := filepath.Join(tmpDir, ".openclaw")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{"theme":"dark"}`), 0o644)

		if c.onboarded() {
			t.Error("expected false when no wizard section")
		}
	})

	t.Run("returns false when wizard section exists but no lastRunAt", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		configDir := filepath.Join(tmpDir, ".openclaw")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{"wizard":{}}`), 0o644)

		if c.onboarded() {
			t.Error("expected false when wizard.lastRunAt is missing")
		}
	})

	t.Run("returns false when wizard.lastRunAt is empty string", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		configDir := filepath.Join(tmpDir, ".openclaw")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{"wizard":{"lastRunAt":""}}`), 0o644)

		if c.onboarded() {
			t.Error("expected false when wizard.lastRunAt is empty")
		}
	})

	t.Run("returns true when wizard.lastRunAt is set", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		configDir := filepath.Join(tmpDir, ".openclaw")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{"wizard":{"lastRunAt":"2024-01-01T00:00:00Z"}}`), 0o644)

		if !c.onboarded() {
			t.Error("expected true when wizard.lastRunAt is set")
		}
	})

	t.Run("checks legacy clawdbot path", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		legacyDir := filepath.Join(tmpDir, ".clawdbot")
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(legacyDir, "clawdbot.json"), []byte(`{"wizard":{"lastRunAt":"2024-01-01T00:00:00Z"}}`), 0o644)

		if !c.onboarded() {
			t.Error("expected true when legacy config has wizard.lastRunAt")
		}
	})

	t.Run("prefers new path over legacy", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		newDir := filepath.Join(tmpDir, ".openclaw")
		legacyDir := filepath.Join(tmpDir, ".clawdbot")
		os.MkdirAll(newDir, 0o755)
		os.MkdirAll(legacyDir, 0o755)
		// New path has no wizard marker
		os.WriteFile(filepath.Join(newDir, "openclaw.json"), []byte(`{}`), 0o644)
		// Legacy has wizard marker
		os.WriteFile(filepath.Join(legacyDir, "clawdbot.json"), []byte(`{"wizard":{"lastRunAt":"2024-01-01T00:00:00Z"}}`), 0o644)

		if c.onboarded() {
			t.Error("expected false - should prefer new path which has no wizard marker")
		}
	})

	t.Run("handles corrupted JSON gracefully", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		configDir := filepath.Join(tmpDir, ".openclaw")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{corrupted`), 0o644)

		if c.onboarded() {
			t.Error("expected false for corrupted JSON")
		}
	})

	t.Run("handles wrong type for wizard section", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		configDir := filepath.Join(tmpDir, ".openclaw")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{"wizard":"not a map"}`), 0o644)

		if c.onboarded() {
			t.Error("expected false when wizard is wrong type")
		}
	})
}

func TestOpenclawGatewayInfo(t *testing.T) {
	c := &Openclaw{}

	t.Run("returns defaults when no config exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		token, port := c.gatewayInfo()
		if token != "" {
			t.Errorf("expected empty token, got %q", token)
		}
		if port != defaultGatewayPort {
			t.Errorf("expected default port %d, got %d", defaultGatewayPort, port)
		}
	})

	t.Run("reads token and port from config", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		configDir := filepath.Join(tmpDir, ".openclaw")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{
			"gateway": {
				"port": 9999,
				"auth": {"mode": "token", "token": "my-secret"}
			}
		}`), 0o644)

		token, port := c.gatewayInfo()
		if token != "my-secret" {
			t.Errorf("expected token %q, got %q", "my-secret", token)
		}
		if port != 9999 {
			t.Errorf("expected port 9999, got %d", port)
		}
	})

	t.Run("uses default port when not in config", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		configDir := filepath.Join(tmpDir, ".openclaw")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{
			"gateway": {"auth": {"token": "tok"}}
		}`), 0o644)

		token, port := c.gatewayInfo()
		if token != "tok" {
			t.Errorf("expected token %q, got %q", "tok", token)
		}
		if port != defaultGatewayPort {
			t.Errorf("expected default port %d, got %d", defaultGatewayPort, port)
		}
	})

	t.Run("falls back to legacy clawdbot config", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		legacyDir := filepath.Join(tmpDir, ".clawdbot")
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(legacyDir, "clawdbot.json"), []byte(`{
			"gateway": {"port": 12345, "auth": {"token": "legacy-token"}}
		}`), 0o644)

		token, port := c.gatewayInfo()
		if token != "legacy-token" {
			t.Errorf("expected token %q, got %q", "legacy-token", token)
		}
		if port != 12345 {
			t.Errorf("expected port 12345, got %d", port)
		}
	})

	t.Run("handles corrupted JSON gracefully", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		configDir := filepath.Join(tmpDir, ".openclaw")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{corrupted`), 0o644)

		token, port := c.gatewayInfo()
		if token != "" {
			t.Errorf("expected empty token, got %q", token)
		}
		if port != defaultGatewayPort {
			t.Errorf("expected default port, got %d", port)
		}
	})

	t.Run("handles missing gateway section", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		configDir := filepath.Join(tmpDir, ".openclaw")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "openclaw.json"), []byte(`{"theme":"dark"}`), 0o644)

		token, port := c.gatewayInfo()
		if token != "" {
			t.Errorf("expected empty token, got %q", token)
		}
		if port != defaultGatewayPort {
			t.Errorf("expected default port, got %d", port)
		}
	})
}

func TestPrintOpenclawReady(t *testing.T) {
	t.Run("includes port in URL", func(t *testing.T) {
		var buf bytes.Buffer
		old := os.Stderr
		r, w, _ := os.Pipe()
		os.Stderr = w

		printOpenclawReady("openclaw", "", 9999, false)

		w.Close()
		os.Stderr = old
		buf.ReadFrom(r)

		output := buf.String()
		if !strings.Contains(output, "localhost:9999") {
			t.Errorf("expected port 9999 in output, got:\n%s", output)
		}
		if strings.Contains(output, "#token=") {
			t.Error("should not include token fragment when token is empty")
		}
	})

	t.Run("URL-escapes token", func(t *testing.T) {
		var buf bytes.Buffer
		old := os.Stderr
		r, w, _ := os.Pipe()
		os.Stderr = w

		printOpenclawReady("openclaw", "my token&special=chars", defaultGatewayPort, false)

		w.Close()
		os.Stderr = old
		buf.ReadFrom(r)

		output := buf.String()
		escaped := url.QueryEscape("my token&special=chars")
		if !strings.Contains(output, "#token="+escaped) {
			t.Errorf("expected URL-escaped token %q in output, got:\n%s", escaped, output)
		}
	})

	t.Run("simple token is not mangled", func(t *testing.T) {
		var buf bytes.Buffer
		old := os.Stderr
		r, w, _ := os.Pipe()
		os.Stderr = w

		printOpenclawReady("openclaw", "ollama", defaultGatewayPort, false)

		w.Close()
		os.Stderr = old
		buf.ReadFrom(r)

		output := buf.String()
		if !strings.Contains(output, "#token=ollama") {
			t.Errorf("expected #token=ollama in output, got:\n%s", output)
		}
	})

	t.Run("includes web UI hint", func(t *testing.T) {
		var buf bytes.Buffer
		old := os.Stderr
		r, w, _ := os.Pipe()
		os.Stderr = w

		printOpenclawReady("openclaw", "", defaultGatewayPort, false)

		w.Close()
		os.Stderr = old
		buf.ReadFrom(r)

		output := buf.String()
		if !strings.Contains(output, "Open the Web UI") {
			t.Errorf("expected web UI hint in output, got:\n%s", output)
		}
	})

	t.Run("first launch shows quick start tips", func(t *testing.T) {
		var buf bytes.Buffer
		old := os.Stderr
		r, w, _ := os.Pipe()
		os.Stderr = w

		printOpenclawReady("openclaw", "ollama", defaultGatewayPort, true)

		w.Close()
		os.Stderr = old
		buf.ReadFrom(r)

		output := buf.String()
		for _, want := range []string{"/help", "channels", "skills", "gateway"} {
			if !strings.Contains(output, want) {
				t.Errorf("expected %q in first-launch output, got:\n%s", want, output)
			}
		}
	})

	t.Run("subsequent launch shows single tip", func(t *testing.T) {
		var buf bytes.Buffer
		old := os.Stderr
		r, w, _ := os.Pipe()
		os.Stderr = w

		printOpenclawReady("openclaw", "ollama", defaultGatewayPort, false)

		w.Close()
		os.Stderr = old
		buf.ReadFrom(r)

		output := buf.String()
		if !strings.Contains(output, "Tip:") {
			t.Errorf("expected single tip line, got:\n%s", output)
		}
		if strings.Contains(output, "Quick start") {
			t.Errorf("should not show quick start on subsequent launch")
		}
	})
}

func TestOpenclawModelConfig(t *testing.T) {
	t.Run("nil client returns base config", func(t *testing.T) {
		cfg, _ := openclawModelConfig(context.Background(), nil, "llama3.2")

		if cfg["id"] != "llama3.2" {
			t.Errorf("id = %v, want llama3.2", cfg["id"])
		}
		if cfg["name"] != "llama3.2" {
			t.Errorf("name = %v, want llama3.2", cfg["name"])
		}
		if cfg["cost"] == nil {
			t.Error("cost should be set")
		}
		// Should not have capability fields without API
		if _, ok := cfg["reasoning"]; ok {
			t.Error("reasoning should not be set without API")
		}
		if _, ok := cfg["contextWindow"]; ok {
			t.Error("contextWindow should not be set without API")
		}
	})

	t.Run("sets vision input when model has vision capability", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":["vision"],"model_info":{"llama.context_length":4096}}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg, _ := openclawModelConfig(context.Background(), client, "llava:7b")

		input, ok := cfg["input"].([]any)
		if !ok || len(input) != 2 {
			t.Errorf("input = %v, want [text image]", cfg["input"])
		}
	})

	t.Run("sets text-only input when model lacks vision", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":["completion"],"model_info":{}}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg, _ := openclawModelConfig(context.Background(), client, "llama3.2")

		input, ok := cfg["input"].([]any)
		if !ok || len(input) != 1 {
			t.Errorf("input = %v, want [text]", cfg["input"])
		}
		if _, ok := cfg["reasoning"]; ok {
			t.Error("reasoning should not be set for non-thinking model")
		}
	})

	t.Run("sets reasoning when model has thinking capability", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":["thinking"],"model_info":{}}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg, _ := openclawModelConfig(context.Background(), client, "qwq")

		if cfg["reasoning"] != true {
			t.Error("expected reasoning = true for thinking model")
		}
	})

	t.Run("extracts context window from model info", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":[],"model_info":{"llama.context_length":131072}}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg, _ := openclawModelConfig(context.Background(), client, "llama3.2")

		if cfg["contextWindow"] != 131072 {
			t.Errorf("contextWindow = %v, want 131072", cfg["contextWindow"])
		}
	})

	t.Run("handles all capabilities together", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":["vision","thinking"],"model_info":{"qwen3.context_length":32768}}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg, _ := openclawModelConfig(context.Background(), client, "qwen3-vision")

		input, ok := cfg["input"].([]any)
		if !ok || len(input) != 2 {
			t.Errorf("input = %v, want [text image]", cfg["input"])
		}
		if cfg["reasoning"] != true {
			t.Error("expected reasoning = true")
		}
		if cfg["contextWindow"] != 32768 {
			t.Errorf("contextWindow = %v, want 32768", cfg["contextWindow"])
		}
	})

	t.Run("returns base config when show fails", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, `{"error":"model not found"}`)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg, _ := openclawModelConfig(context.Background(), client, "missing-model")

		if cfg["id"] != "missing-model" {
			t.Errorf("id = %v, want missing-model", cfg["id"])
		}
		// Should still have input (default)
		if cfg["input"] == nil {
			t.Error("input should always be set")
		}
		if _, ok := cfg["reasoning"]; ok {
			t.Error("reasoning should not be set when show fails")
		}
		if _, ok := cfg["contextWindow"]; ok {
			t.Error("contextWindow should not be set when show fails")
		}
	})

	t.Run("times out slow show and returns base config", func(t *testing.T) {
		oldTimeout := openclawModelShowTimeout
		openclawModelShowTimeout = 50 * time.Millisecond
		t.Cleanup(func() { openclawModelShowTimeout = oldTimeout })

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				time.Sleep(300 * time.Millisecond)
				fmt.Fprintf(w, `{"capabilities":["thinking"],"model_info":{"llama.context_length":4096}}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		start := time.Now()
		cfg, _ := openclawModelConfig(context.Background(), client, "slow-model")
		elapsed := time.Since(start)
		if elapsed >= 250*time.Millisecond {
			t.Fatalf("openclawModelConfig took too long: %v", elapsed)
		}
		if cfg["id"] != "slow-model" {
			t.Errorf("id = %v, want slow-model", cfg["id"])
		}
		if _, ok := cfg["reasoning"]; ok {
			t.Error("reasoning should not be set on timeout")
		}
		if _, ok := cfg["contextWindow"]; ok {
			t.Error("contextWindow should not be set on timeout")
		}
	})

	t.Run("skips zero context length", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":[],"model_info":{"llama.context_length":0}}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg, _ := openclawModelConfig(context.Background(), client, "test-model")

		if _, ok := cfg["contextWindow"]; ok {
			t.Error("contextWindow should not be set for zero value")
		}
	})

	t.Run("cloud model uses hardcoded limits", func(t *testing.T) {
		// Use a model name that's in cloudModelLimits and make the server
		// report it as a remote/cloud model
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":[],"model_info":{},"remote_model":"minimax-m2.5"}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg, isCloud := openclawModelConfig(context.Background(), client, "minimax-m2.5:cloud")

		if !isCloud {
			t.Error("expected isCloud = true for cloud model")
		}
		if cfg["contextWindow"] != 204_800 {
			t.Errorf("contextWindow = %v, want 204800", cfg["contextWindow"])
		}
		if cfg["maxTokens"] != 128_000 {
			t.Errorf("maxTokens = %v, want 128000", cfg["maxTokens"])
		}
	})

	t.Run("cloud model with vision capability gets image input", func(t *testing.T) {
		// Regression test: cloud models must not skip capability detection.
		// A cloud model that reports vision capability should have input: [text, image].
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":["vision"],"model_info":{},"remote_model":"qwen3-vl"}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg, isCloud := openclawModelConfig(context.Background(), client, "qwen3-vl:235b-cloud")

		if !isCloud {
			t.Error("expected isCloud = true for cloud vision model")
		}
		input, ok := cfg["input"].([]any)
		if !ok || len(input) != 2 {
			t.Errorf("input = %v, want [text image] for cloud vision model", cfg["input"])
		}
	})

	t.Run("cloud model with thinking capability gets reasoning flag", func(t *testing.T) {
		// Regression test: cloud models must not skip capability detection.
		// A cloud model that reports thinking capability should have reasoning: true.
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":["thinking"],"model_info":{},"remote_model":"qwq-cloud"}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg, isCloud := openclawModelConfig(context.Background(), client, "qwq:cloud")

		if !isCloud {
			t.Error("expected isCloud = true for cloud thinking model")
		}
		if cfg["reasoning"] != true {
			t.Error("expected reasoning = true for cloud thinking model")
		}
	})
}

func TestIntegrationOnboarded(t *testing.T) {
	t.Run("returns false when not set", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		integrationConfig, err := loadIntegration("openclaw")
		if err == nil && integrationConfig.Onboarded {
			t.Error("expected false for fresh config")
		}
	})

	t.Run("returns true after integrationOnboarded", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		os.MkdirAll(filepath.Join(tmpDir, ".ollama"), 0o755)

		if err := integrationOnboarded("openclaw"); err != nil {
			t.Fatal(err)
		}
		integrationConfig, err := loadIntegration("openclaw")
		if err != nil || !integrationConfig.Onboarded {
			t.Error("expected true after integrationOnboarded")
		}
	})

	t.Run("is case insensitive", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		os.MkdirAll(filepath.Join(tmpDir, ".ollama"), 0o755)

		if err := integrationOnboarded("OpenClaw"); err != nil {
			t.Fatal(err)
		}
		integrationConfig, err := loadIntegration("openclaw")
		if err != nil || !integrationConfig.Onboarded {
			t.Error("expected true when set with different case")
		}
	})

	t.Run("preserves existing integration data", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		os.MkdirAll(filepath.Join(tmpDir, ".ollama"), 0o755)

		if err := SaveIntegration("openclaw", []string{"llama3.2", "mistral"}); err != nil {
			t.Fatal(err)
		}
		if err := integrationOnboarded("openclaw"); err != nil {
			t.Fatal(err)
		}

		// Verify onboarded is set
		integrationConfig, err := loadIntegration("openclaw")
		if err != nil || !integrationConfig.Onboarded {
			t.Error("expected true after integrationOnboarded")
		}

		// Verify models are preserved
		model := IntegrationModel("openclaw")
		if model != "llama3.2" {
			t.Errorf("expected first model llama3.2, got %q", model)
		}
	})
}
