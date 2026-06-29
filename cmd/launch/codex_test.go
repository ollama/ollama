package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	modelpkg "github.com/ollama/ollama/types/model"
)

func TestCodexIntegration(t *testing.T) {
	c := &Codex{}

	t.Run("implements runner", func(t *testing.T) {
		var _ Runner = c
	})
	t.Run("implements restore", func(t *testing.T) {
		var _ RestorableIntegration = c
		var _ RestoreSuccessIntegration = c
		var _ RestoreInstallCheckSkipper = c
	})
}

func TestCodexArgs(t *testing.T) {
	c := &Codex{}
	catalogPath := filepath.Join("tmp", "model.json")
	managedArgs := []string{
		"--profile", "ollama-launch",
		"-c", fmt.Sprintf("%s=%q", codexRootModelProviderKey, codexProfileName),
		"-c", fmt.Sprintf("model_providers.%s.name=%q", codexProfileName, codexProviderName),
		"-c", fmt.Sprintf("model_providers.%s.base_url=%q", codexProfileName, codexBaseURL()),
		"-c", fmt.Sprintf("model_providers.%s.wire_api=%q", codexProfileName, "responses"),
		"-c", fmt.Sprintf("%s=%q", codexRootModelCatalogJSONKey, catalogPath),
	}

	tests := []struct {
		name  string
		model string
		args  []string
		want  []string
	}{
		{"with model", "llama3.2", nil, append(slices.Clone(managedArgs), "-m", "llama3.2")},
		{"empty model", "", nil, managedArgs},
		{"with sandbox flag", "llama3.2", []string{"--sandbox", "workspace-write"}, append(append(slices.Clone(managedArgs), "-m", "llama3.2"), "--sandbox", "workspace-write")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := c.args(tt.model, catalogPath, tt.args)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, tt.want) {
				t.Errorf("args(%q, %v) = %v, want %v", tt.model, tt.args, got, tt.want)
			}
		})
	}
}

func TestCodexArgsRejectManagedProfile(t *testing.T) {
	c := &Codex{}
	for _, extra := range [][]string{
		{"-p", "myprofile"},
		{"-pmyprofile"},
		{"--profile", "myprofile"},
		{"--profile=myprofile"},
	} {
		t.Run(strings.Join(extra, " "), func(t *testing.T) {
			_, err := c.args("llama3.2", "", extra)
			if err == nil || !strings.Contains(err.Error(), "manages --profile") {
				t.Fatalf("args error = %v, want profile conflict", err)
			}
		})
	}
}

func TestCodexArgsRejectManagedOverrides(t *testing.T) {
	c := &Codex{}
	for _, extra := range [][]string{
		{"-m", "other"},
		{"-mother"},
		{"--model", "other"},
		{"--model=other"},
		{"-c", `model_catalog_json="/tmp/other.json"`},
		{"--config", `model_provider="openai"`},
		{"--config=model_providers.ollama-launch.base_url=\"http://other.invalid/v1/\""},
	} {
		t.Run(strings.Join(extra, " "), func(t *testing.T) {
			_, err := c.args("llama3.2", "", extra)
			if err == nil {
				t.Fatalf("args error = nil, want managed config conflict")
			}
		})
	}
}

func TestWriteCodexProfileConfig(t *testing.T) {
	t.Run("creates new file when none exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		profilePath := filepath.Join(tmpDir, "ollama-launch.config.toml")
		catalogPath := filepath.Join(tmpDir, "model.json")

		if err := writeCodexProfileConfig(profilePath, "llama3.2", catalogPath); err != nil {
			t.Fatal(err)
		}

		data, err := os.ReadFile(profilePath)
		if err != nil {
			t.Fatal(err)
		}

		content := string(data)
		for _, want := range []string{
			`model = "llama3.2"`,
			`model_provider = "ollama-launch"`,
			fmt.Sprintf("model_catalog_json = %q", catalogPath),
			"[model_providers.ollama-launch]",
			`name = "Ollama"`,
			`base_url = "http://127.0.0.1:11434/v1/"`,
			`wire_api = "responses"`,
		} {
			if !strings.Contains(content, want) {
				t.Errorf("missing %q in:\n%s", want, content)
			}
		}
		if got, ok := codexRootStringValueOK(content, "profile"); ok {
			t.Fatalf("legacy root profile should not be generated, got %q in:\n%s", got, content)
		}
		if strings.Contains(content, "[profiles.ollama-launch]") {
			t.Fatalf("legacy profile section should not be generated, got:\n%s", content)
		}
		if err := codexValidateConfigText(content); err != nil {
			t.Fatalf("generated config should be valid TOML: %v\n%s", err, content)
		}
	})

	t.Run("overwrites owned profile and backs up previous profile", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		profilePath := filepath.Join(tmpDir, ".codex", "ollama-launch.config.toml")
		if err := os.MkdirAll(filepath.Dir(profilePath), 0o755); err != nil {
			t.Fatal(err)
		}
		existing := "# original-codex-profile-backup-marker\nmodel = \"old\"\nmodel_provider = \"old-provider\"\n"
		if err := os.WriteFile(profilePath, []byte(existing), 0o644); err != nil {
			t.Fatal(err)
		}

		if err := writeCodexProfileConfig(profilePath, "llama3.2", ""); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(profilePath)
		content := string(data)
		if strings.Contains(content, "old-provider") {
			t.Fatalf("profile should be replaced, got:\n%s", content)
		}
		assertBackupContains(t, filepath.Join(fileutil.BackupDir(), "ollama-launch.config.toml.*"), "original-codex-profile-backup-marker")
	})

	t.Run("uses custom OLLAMA_HOST", func(t *testing.T) {
		t.Setenv("OLLAMA_HOST", "http://myhost:9999")
		tmpDir := t.TempDir()
		profilePath := filepath.Join(tmpDir, "ollama-launch.config.toml")

		if err := writeCodexProfileConfig(profilePath, "llama3.2", ""); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(profilePath)
		content := string(data)

		if !strings.Contains(content, "myhost:9999/v1/") {
			t.Errorf("expected custom host in URL, got:\n%s", content)
		}
	})

	t.Run("uses connectable host for unspecified bind address", func(t *testing.T) {
		t.Setenv("OLLAMA_HOST", "http://0.0.0.0:11434")
		tmpDir := t.TempDir()
		profilePath := filepath.Join(tmpDir, "ollama-launch.config.toml")

		if err := writeCodexProfileConfig(profilePath, "", ""); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(profilePath)
		content := string(data)

		if strings.Contains(content, "0.0.0.0") {
			t.Fatalf("config should not write bind-only host, got:\n%s", content)
		}
		if !strings.Contains(content, "127.0.0.1:11434/v1/") {
			t.Fatalf("expected connectable loopback URL, got:\n%s", content)
		}
	})
}

func TestEnsureCodexConfig(t *testing.T) {
	t.Run("creates .codex dir, profile config, and model catalog", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		if err := ensureCodexConfig("llama3.2", launchModelsFromNames([]string{"llama3.2"})); err != nil {
			t.Fatal(err)
		}

		configPath := filepath.Join(tmpDir, ".codex", "config.toml")
		if _, err := os.Stat(configPath); !os.IsNotExist(err) {
			t.Fatalf("root config.toml should not be created by CLI config refresh, err=%v", err)
		}

		profilePath := filepath.Join(tmpDir, ".codex", "ollama-launch.config.toml")
		data, err := os.ReadFile(profilePath)
		if err != nil {
			t.Fatalf("profile config not created: %v", err)
		}
		content := string(data)
		if strings.Contains(content, "[profiles.ollama-launch]") {
			t.Fatalf("legacy profile section should not be generated, got:\n%s", content)
		}
		if got := codexRootStringValue(content, "model"); got != "llama3.2" {
			t.Fatalf("profile model = %q, want llama3.2 in:\n%s", got, content)
		}
		if got := codexRootStringValue(content, "model_provider"); got != codexProfileName {
			t.Fatalf("profile model_provider = %q, want %q in:\n%s", got, codexProfileName, content)
		}
		catalogPath := filepath.Join(tmpDir, ".codex", "model.json")
		if got := codexRootStringValue(content, "model_catalog_json"); got != catalogPath {
			t.Fatalf("profile model_catalog_json = %q, want %q in:\n%s", got, catalogPath, content)
		}
		if got := codexSectionStringValue(content, codexProviderHeader(), "base_url"); !strings.Contains(got, "/v1/") {
			t.Fatalf("provider base_url = %q, want /v1/ URL", got)
		}

		data, err = os.ReadFile(catalogPath)
		if err != nil {
			t.Fatalf("model.json not created: %v", err)
		}
		if !strings.Contains(string(data), `"slug": "llama3.2"`) {
			t.Error("missing model catalog entry for selected model")
		}
	})

	t.Run("writes requested local alias as catalog slug", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		models := []LaunchModel{
			{Name: "gemma4:latest", ContextLength: 65_536, Details: api.ModelDetails{Format: "gguf"}},
		}
		if err := ensureCodexConfig("gemma4", models); err != nil {
			t.Fatal(err)
		}

		catalogPath := filepath.Join(tmpDir, ".codex", "model.json")
		data, err := os.ReadFile(catalogPath)
		if err != nil {
			t.Fatalf("model.json not created: %v", err)
		}

		var catalog struct {
			Models []map[string]any `json:"models"`
		}
		if err := json.Unmarshal(data, &catalog); err != nil {
			t.Fatalf("model catalog should be valid JSON: %v", err)
		}
		if len(catalog.Models) != 1 {
			t.Fatalf("catalog model count = %d, want 1", len(catalog.Models))
		}
		if got := catalog.Models[0]["slug"]; got != "gemma4" {
			t.Fatalf("catalog slug = %v, want gemma4", got)
		}
		if got := catalog.Models[0]["context_window"]; got != float64(65_536) {
			t.Fatalf("context_window = %v, want 65536", got)
		}
	})

	t.Run("is idempotent", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		if err := ensureCodexConfig("llama3.2", launchModelsFromNames([]string{"llama3.2"})); err != nil {
			t.Fatal(err)
		}
		if err := ensureCodexConfig("llama3.2", launchModelsFromNames([]string{"llama3.2"})); err != nil {
			t.Fatal(err)
		}

		configPath := filepath.Join(tmpDir, ".codex", "config.toml")
		if _, err := os.Stat(configPath); !os.IsNotExist(err) {
			t.Fatalf("root config.toml should not be created by CLI config refresh, err=%v", err)
		}
		profilePath := filepath.Join(tmpDir, ".codex", "ollama-launch.config.toml")
		data, err := os.ReadFile(profilePath)
		if err != nil {
			t.Fatal(err)
		}
		content := string(data)

		if strings.Contains(content, "[profiles.ollama-launch]") {
			t.Fatalf("legacy profile section should not be generated, got:\n%s", content)
		}
		if strings.Count(content, "[model_providers.ollama-launch]") != 1 {
			t.Errorf("expected exactly one [model_providers.ollama-launch] section after two calls, got %d", strings.Count(content, "[model_providers.ollama-launch]"))
		}
	})

	t.Run("cleans legacy root profile that conflicts with --profile", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		configPath := filepath.Join(tmpDir, ".codex", "config.toml")
		if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
			t.Fatal(err)
		}
		existing := "" +
			`profile = "ollama-launch"` + "\n" +
			`model = "gpt-5.5"` + "\n" +
			`model_provider = "openai"` + "\n\n" +
			"[profiles.ollama-launch]\n" +
			`model = "old-local"` + "\n" +
			`model_provider = "ollama-launch"` + "\n\n" +
			"[profiles.default]\n" +
			`model = "gpt-5.5"` + "\n"
		if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
			t.Fatal(err)
		}

		if err := ensureCodexConfig("llama3.2", launchModelsFromNames([]string{"llama3.2"})); err != nil {
			t.Fatal(err)
		}

		data, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatal(err)
		}
		content := string(data)
		if got, ok := codexRootStringValueOK(content, codexRootProfileKey); ok {
			t.Fatalf("legacy root profile should be removed, got %q in:\n%s", got, content)
		}
		if strings.Contains(content, codexProfileHeader()) {
			t.Fatalf("legacy profile table should be removed, got:\n%s", content)
		}
		for _, want := range []string{
			`model = "gpt-5.5"`,
			`model_provider = "openai"`,
			"[profiles.default]",
		} {
			if !strings.Contains(content, want) {
				t.Fatalf("expected %q to be preserved in:\n%s", want, content)
			}
		}

		profilePath := filepath.Join(tmpDir, ".codex", "ollama-launch.config.toml")
		profileData, err := os.ReadFile(profilePath)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(profileData), `model = "llama3.2"`) {
			t.Fatalf("managed profile was not written with selected model:\n%s", profileData)
		}
		assertBackupContains(t, filepath.Join(fileutil.BackupDir(), "config.toml.*"), `profile = "ollama-launch"`)
	})
}

func TestCodexRestoreRemovesCLIProfileAndCatalogWithoutChangingUserRootConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	userConfig := "" +
		`model = "gpt-5.5"` + "\n" +
		`model_provider = "openai"` + "\n\n" +
		"[model_providers.openai]\n" +
		`name = "OpenAI"` + "\n"
	if err := os.WriteFile(configPath, []byte(userConfig), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := ensureCodexConfig("llama3.2", launchModelsFromNames([]string{"llama3.2"})); err != nil {
		t.Fatal(err)
	}

	if err := (&Codex{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	profilePath := filepath.Join(tmpDir, ".codex", "ollama-launch.config.toml")
	if _, err := os.Stat(profilePath); !os.IsNotExist(err) {
		t.Fatalf("CLI profile should be removed, got err=%v", err)
	}
	catalogPath := filepath.Join(tmpDir, ".codex", "model.json")
	if _, err := os.Stat(catalogPath); !os.IsNotExist(err) {
		t.Fatalf("CLI catalog should be removed, got err=%v", err)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != userConfig {
		t.Fatalf("user root config should be unchanged, got:\n%s", data)
	}
}

func TestCodexRestoreDoesNotRewriteRootConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	catalogPath := filepath.Join(tmpDir, ".codex", "model.json")
	profilePath := filepath.Join(tmpDir, ".codex", "ollama-launch.config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	legacyConfig := "" +
		`profile = "ollama-launch"` + "\n" +
		`model = "llama3.2"` + "\n" +
		`model_provider = "ollama-launch"` + "\n" +
		fmt.Sprintf("model_catalog_json = %q\n\n", catalogPath) +
		"[model_providers.ollama-launch]\n" +
		`name = "Ollama"` + "\n" +
		`base_url = "http://127.0.0.1:11434/v1/"` + "\n" +
		`wire_api = "responses"` + "\n\n" +
		"[profiles.ollama-launch]\n" +
		`model = "llama3.2"` + "\n\n" +
		"[tools]\n" +
		`web_search = true` + "\n"
	if err := os.WriteFile(configPath, []byte(legacyConfig), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalogPath, []byte(`{"models":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(profilePath, []byte(`model_provider = "ollama-launch"`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&Codex{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != legacyConfig {
		t.Fatalf("root config should be left untouched, got:\n%s", data)
	}
	if _, err := os.Stat(profilePath); !os.IsNotExist(err) {
		t.Fatalf("CLI profile should be removed, got err=%v", err)
	}
	if _, err := os.Stat(catalogPath); err != nil {
		t.Fatalf("CLI catalog should be left while root config references it: %v", err)
	}
}

func TestCodexRestoreDoesNotTouchCodexAppConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	cliCatalogPath := filepath.Join(tmpDir, ".codex", "model.json")
	appCatalogPath := filepath.Join(tmpDir, ".codex", codexAppModelCatalogFilename)
	cliProfilePath := filepath.Join(tmpDir, ".codex", "ollama-launch.config.toml")
	appProfilePath := filepath.Join(tmpDir, ".codex", codexAppProfileName+".config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	appManagedConfig := "" +
		`model = "llama3.2"` + "\n" +
		fmt.Sprintf("model_provider = %q\n", codexAppProfileName) +
		fmt.Sprintf("model_catalog_json = %q\n\n", appCatalogPath) +
		codexProviderHeaderFor(codexAppProfileName) + "\n" +
		`name = "Ollama"` + "\n" +
		`base_url = "http://127.0.0.1:11434/v1/"` + "\n" +
		`wire_api = "responses"` + "\n\n" +
		codexProviderHeader() + "\n" +
		`name = "Ollama"` + "\n" +
		`base_url = "http://127.0.0.1:11434/v1/"` + "\n" +
		`wire_api = "responses"` + "\n"
	if err := os.WriteFile(configPath, []byte(appManagedConfig), 0o644); err != nil {
		t.Fatal(err)
	}
	restoreState := fmt.Sprintf(`{"had_profile":false,"had_model":true,"model":"qwen3:8b","had_model_provider":true,"model_provider":%q,"had_model_catalog_json":true,"model_catalog_json":%q}`, codexProfileName, cliCatalogPath)
	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(restoreState), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{cliCatalogPath, appCatalogPath, cliProfilePath, appProfilePath} {
		if err := os.WriteFile(path, []byte(`{"models":[]}`), 0o644); err != nil {
			t.Fatalf("write %s: %v", path, err)
		}
	}

	if err := (&Codex{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != appManagedConfig {
		t.Fatalf("Codex App root config should be left untouched, got:\n%s", data)
	}
	if _, err := os.Stat(cliProfilePath); !os.IsNotExist(err) {
		t.Fatalf("CLI profile should be removed, got err=%v", err)
	}
	if _, err := os.Stat(cliCatalogPath); !os.IsNotExist(err) {
		t.Fatalf("CLI catalog should be removed when root config does not reference it, got err=%v", err)
	}
	for _, path := range []string{appCatalogPath, appProfilePath, codexAppRestoreStatePath()} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("%s should be left untouched, got err=%v", path, err)
		}
	}
}

func TestLaunchIntegrationCodexRestoreDoesNotRequireInstalledCLI(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("PATH", tmpDir)

	profilePath := filepath.Join(tmpDir, ".codex", "ollama-launch.config.toml")
	if err := os.MkdirAll(filepath.Dir(profilePath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(profilePath, []byte(`model_provider = "ollama-launch"`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "codex", Restore: true}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}
	if _, err := os.Stat(profilePath); !os.IsNotExist(err) {
		t.Fatalf("CLI restore should run without codex installed and remove profile, got err=%v", err)
	}
}

func assertBackupContains(t *testing.T, pattern, marker string) {
	t.Helper()
	backups, err := filepath.Glob(pattern)
	if err != nil {
		t.Fatal(err)
	}
	for _, backupPath := range backups {
		data, err := os.ReadFile(backupPath)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(data), marker) {
			return
		}
	}
	t.Fatalf("backup matching %q with marker %q not found", pattern, marker)
}

func TestModelInfoContextLength(t *testing.T) {
	tests := []struct {
		name      string
		modelInfo map[string]any
		want      int
	}{
		{"float64 value", map[string]any{"qwen3_5_moe.context_length": float64(262144)}, 262144},
		{"int value", map[string]any{"llama.context_length": 131072}, 131072},
		{"no context_length key", map[string]any{"llama.embedding_length": float64(4096)}, 0},
		{"empty map", map[string]any{}, 0},
		{"nil map", nil, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, _ := modelInfoContextLength(tt.modelInfo)
			if got != tt.want {
				t.Errorf("modelInfoContextLength() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestBuildCodexModelEntryContextWindow(t *testing.T) {
	tests := []struct {
		name          string
		model         LaunchModel
		envContextLen string
		wantContext   int
	}{
		{
			name: "inventory context length as fallback",
			model: LaunchModel{
				Name:          "llama3.2",
				ContextLength: 131072,
				Details:       api.ModelDetails{Format: "gguf"},
			},
			wantContext: 131072,
		},
		{
			name: "details context length is used when model context is empty",
			model: LaunchModel{
				Name:    "llama3.2",
				Details: api.ModelDetails{Format: "gguf", ContextLength: 131072},
			},
			wantContext: 131072,
		},
		{
			name: "OLLAMA_CONTEXT_LENGTH overrides local gguf inventory context",
			model: LaunchModel{
				Name:          "llama3.2",
				ContextLength: 131072,
				Details:       api.ModelDetails{Format: "gguf"},
			},
			envContextLen: "64000",
			wantContext:   64000,
		},
		{
			name: "safetensors uses inventory context only",
			model: LaunchModel{
				Name:          "llama3.2",
				ContextLength: 131072,
				Details:       api.ModelDetails{Format: "safetensors"},
			},
			envContextLen: "64000",
			wantContext:   131072,
		},
		{
			name: "cloud model uses hardcoded limits",
			model: LaunchModel{
				Name:          "qwen3.5:cloud",
				ContextLength: 131072,
				Details:       api.ModelDetails{Format: "gguf"},
			},
			envContextLen: "64000",
			wantContext:   262144,
		},
		{
			name: "unknown cloud model without metadata uses fallback context",
			model: LaunchModel{
				Name: "deepseek-v4-pro:cloud",
			},
			envContextLen: "64000",
			wantContext:   codexFallbackContextWindow,
		},
		{
			name: "vision capability without reasoning advertisement",
			model: LaunchModel{
				Name:          "llama3.2",
				ContextLength: 131072,
				Details:       api.ModelDetails{Format: "gguf"},
				Capabilities:  []modelpkg.Capability{modelpkg.CapabilityVision, modelpkg.CapabilityThinking},
			},
			wantContext: 131072,
		},
		{
			name:        "missing metadata uses fallback context",
			model:       LaunchModel{Name: "llama3.2"},
			wantContext: codexFallbackContextWindow,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envContextLen != "" {
				t.Setenv("OLLAMA_CONTEXT_LENGTH", tt.envContextLen)
			} else {
				t.Setenv("OLLAMA_CONTEXT_LENGTH", "")
			}

			entry := buildCodexModelEntry(tt.model)

			gotContext, _ := entry["context_window"].(int)
			if gotContext != tt.wantContext {
				t.Errorf("context_window = %d, want %d", gotContext, tt.wantContext)
			}

			if tt.name == "vision capability without reasoning advertisement" {
				modalities, _ := entry["input_modalities"].([]string)
				if !slices.Contains(modalities, "image") {
					t.Error("expected image in input_modalities")
				}
				levels, _ := entry["supported_reasoning_levels"].([]any)
				if len(levels) != 0 {
					t.Errorf("supported_reasoning_levels length = %d, want 0", len(levels))
				}
				if got, _ := entry["supports_reasoning_summaries"].(bool); got {
					t.Error("supports_reasoning_summaries = true, want false")
				}
			}

			if tt.name == "cloud model uses hardcoded limits" {
				truncationPolicy, _ := entry["truncation_policy"].(map[string]any)
				if mode, _ := truncationPolicy["mode"].(string); mode != "tokens" {
					t.Errorf("truncation_policy mode = %q, want %q", mode, "tokens")
				}
			}

			requiredKeys := []string{"slug", "display_name", "shell_type"}
			for _, key := range requiredKeys {
				if _, ok := entry[key]; !ok {
					t.Errorf("missing required key %q", key)
				}
			}
			if _, ok := entry["apply_patch_tool_type"]; ok {
				t.Error("apply_patch_tool_type should be omitted so Codex CLI defaults can handle schema changes")
			}

			if _, err := json.Marshal(entry); err != nil {
				t.Errorf("entry is not JSON serializable: %v", err)
			}
		})
	}
}
