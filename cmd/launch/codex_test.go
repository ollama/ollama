package launch

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/cmd/internal/fileutil"
)

func TestCodexArgs(t *testing.T) {
	c := &Codex{}

	tests := []struct {
		name  string
		model string
		args  []string
		want  []string
	}{
		{"with model", "llama3.2", nil, []string{"--profile", "ollama-launch", "-m", "llama3.2"}},
		{"empty model", "", nil, []string{"--profile", "ollama-launch"}},
		{"with model and extra args", "qwen3.5", []string{"-p", "myprofile"}, []string{"--profile", "ollama-launch", "-m", "qwen3.5", "-p", "myprofile"}},
		{"with sandbox flag", "llama3.2", []string{"--sandbox", "workspace-write"}, []string{"--profile", "ollama-launch", "-m", "llama3.2", "--sandbox", "workspace-write"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := c.args(tt.model, tt.args)
			if !slices.Equal(got, tt.want) {
				t.Errorf("args(%q, %v) = %v, want %v", tt.model, tt.args, got, tt.want)
			}
		})
	}
}

func TestWriteCodexProfile(t *testing.T) {
	t.Run("creates new file when none exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.toml")
		catalogPath := filepath.Join(tmpDir, "model.json")

		if err := writeCodexProfile(configPath, catalogPath); err != nil {
			t.Fatal(err)
		}

		data, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatal(err)
		}

		content := string(data)
		if !strings.Contains(content, "[profiles.ollama-launch]") {
			t.Error("missing [profiles.ollama-launch] header")
		}
		if !strings.Contains(content, "openai_base_url") {
			t.Error("missing openai_base_url key")
		}
		if !strings.Contains(content, "/v1/") {
			t.Error("missing /v1/ suffix in base URL")
		}
		if !strings.Contains(content, `forced_login_method = "api"`) {
			t.Error("missing forced_login_method key")
		}
		if !strings.Contains(content, `model_provider = "ollama-launch"`) {
			t.Error("missing model_provider key")
		}
		if !strings.Contains(content, fmt.Sprintf("model_catalog_json = %q", catalogPath)) {
			t.Error("missing model_catalog_json key")
		}
		if !strings.Contains(content, "[model_providers.ollama-launch]") {
			t.Error("missing [model_providers.ollama-launch] section")
		}
		if !strings.Contains(content, `name = "Ollama"`) {
			t.Error("missing model provider name")
		}
		if err := codexValidateConfigText(content); err != nil {
			t.Fatalf("generated config should be valid TOML: %v\n%s", err, content)
		}
	})

	t.Run("appends profile to existing file without profile", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.toml")
		catalogPath := filepath.Join(tmpDir, "model.json")
		existing := "[some_other_section]\nkey = \"value\"\n"
		os.WriteFile(configPath, []byte(existing), 0o644)

		if err := writeCodexProfile(configPath, catalogPath); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(configPath)
		content := string(data)

		if !strings.Contains(content, "[some_other_section]") {
			t.Error("existing section was removed")
		}
		if !strings.Contains(content, "[profiles.ollama-launch]") {
			t.Error("missing [profiles.ollama-launch] header")
		}
	})

	t.Run("replaces existing profile section", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.toml")
		catalogPath := filepath.Join(tmpDir, "model.json")
		existing := "[profiles.ollama-launch]\nopenai_base_url = \"http://old:1234/v1/\"\n\n[model_providers.ollama-launch]\nname = \"Ollama\"\nbase_url = \"http://old:1234/v1/\"\n"
		os.WriteFile(configPath, []byte(existing), 0o644)

		if err := writeCodexProfile(configPath, catalogPath); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(configPath)
		content := string(data)

		if strings.Contains(content, "old:1234") {
			t.Error("old URL was not replaced")
		}
		if strings.Count(content, "[profiles.ollama-launch]") != 1 {
			t.Errorf("expected exactly one [profiles.ollama-launch] section, got %d", strings.Count(content, "[profiles.ollama-launch]"))
		}
		if strings.Count(content, "[model_providers.ollama-launch]") != 1 {
			t.Errorf("expected exactly one [model_providers.ollama-launch] section, got %d", strings.Count(content, "[model_providers.ollama-launch]"))
		}
		if err := codexValidateConfigText(content); err != nil {
			t.Fatalf("generated config should be valid TOML: %v\n%s", err, content)
		}
	})

	t.Run("replaces equivalent quoted profile table", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.toml")
		existing := "" +
			`profile = "default"` + "\n\n" +
			`[profiles."ollama-launch"]` + "\n" +
			`openai_base_url = "http://old:1234/v1/"` + "\n\n" +
			`[model_providers."ollama-launch"]` + "\n" +
			`name = "Old"` + "\n" +
			`base_url = "http://old:1234/v1/"` + "\n\n" +
			`[profiles.default]` + "\n" +
			`model = "gpt-5.5"` + "\n"
		os.WriteFile(configPath, []byte(existing), 0o644)

		if err := writeCodexProfile(configPath); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(configPath)
		content := string(data)

		if strings.Contains(content, `profiles."ollama-launch"`) {
			t.Fatalf("quoted profile table should be replaced, got:\n%s", content)
		}
		if strings.Contains(content, "old:1234") {
			t.Fatalf("old URL was not replaced, got:\n%s", content)
		}
		if got := codexSectionStringValue(content, codexProfileHeader(), "model_provider"); got != codexProfileName {
			t.Fatalf("profile model_provider = %q, want %q", got, codexProfileName)
		}
		if got := codexSectionStringValue(content, codexProviderHeader(), "base_url"); !strings.Contains(got, "/v1/") {
			t.Fatalf("provider base_url = %q, want /v1/ URL", got)
		}
		if err := codexValidateConfigText(content); err != nil {
			t.Fatalf("generated config should be valid TOML: %v\n%s", err, content)
		}
	})

	t.Run("rejects invalid existing toml without writing", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.toml")
		existing := "profile = \n"
		os.WriteFile(configPath, []byte(existing), 0o644)

		err := writeCodexProfile(configPath)
		if err == nil || !strings.Contains(err.Error(), "invalid Codex config TOML") {
			t.Fatalf("writeCodexProfile error = %v, want invalid TOML", err)
		}

		data, _ := os.ReadFile(configPath)
		if string(data) != existing {
			t.Fatalf("invalid config should be left untouched, got:\n%s", data)
		}
	})

	t.Run("rejects malformed existing toml variants without writing", func(t *testing.T) {
		tests := map[string]string{
			"duplicate root key":  "profile = \"default\"\nprofile = \"other\"\n",
			"unterminated string": "model = \"gpt-5.5\n",
			"bad table":           "[profiles.ollama-launch\nmodel = \"llama3.2\"\n",
			"duplicate table key": "[profiles.ollama-launch]\nmodel = \"a\"\nmodel = \"b\"\n",
		}
		for name, existing := range tests {
			t.Run(name, func(t *testing.T) {
				tmpDir := t.TempDir()
				configPath := filepath.Join(tmpDir, "config.toml")
				if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
					t.Fatal(err)
				}

				err := writeCodexProfile(configPath)
				if err == nil || !strings.Contains(err.Error(), "invalid Codex config TOML") {
					t.Fatalf("writeCodexProfile error = %v, want invalid TOML", err)
				}

				data, _ := os.ReadFile(configPath)
				if string(data) != existing {
					t.Fatalf("invalid config should be left untouched, got:\n%s", data)
				}
			})
		}
	})

	t.Run("backs up previous config before overwrite", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		configPath := filepath.Join(tmpDir, ".codex", "config.toml")
		if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
			t.Fatal(err)
		}
		existing := "# original-codex-backup-marker\n[profiles.default]\nmodel = \"gpt-5.5\"\n"
		if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
			t.Fatal(err)
		}

		if err := writeCodexProfile(configPath); err != nil {
			t.Fatal(err)
		}

		assertBackupContains(t, filepath.Join(fileutil.BackupDir(), "config.toml.*"), "original-codex-backup-marker")
	})

	t.Run("updates equivalent quoted root keys", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.toml")
		existing := "" +
			`"profile" = "default"` + "\n" +
			`"model" = "gpt-5.5"` + "\n" +
			`"model_provider" = "openai"` + "\n\n" +
			`[profiles.default]` + "\n" +
			`model = "gpt-5.5"` + "\n"
		os.WriteFile(configPath, []byte(existing), 0o644)

		err := writeCodexLaunchProfile(configPath, codexLaunchProfileOptions{
			activate:           true,
			setRootModelConfig: true,
			model:              "llama3.2",
		})
		if err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(configPath)
		content := string(data)
		for key, want := range map[string]string{
			"profile":        codexProfileName,
			"model":          "llama3.2",
			"model_provider": codexProfileName,
		} {
			if got := codexRootStringValue(content, key); got != want {
				t.Fatalf("root %s = %q, want %q in:\n%s", key, got, want, content)
			}
		}
		if strings.Contains(content, `"profile"`) || strings.Contains(content, `"model_provider"`) {
			t.Fatalf("quoted root keys should be rewritten once, got:\n%s", content)
		}
		if err := codexValidateConfigText(content); err != nil {
			t.Fatalf("generated config should be valid TOML: %v\n%s", err, content)
		}
	})

	t.Run("replaces profile while preserving following sections", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.toml")
		catalogPath := filepath.Join(tmpDir, "model.json")
		existing := "[profiles.ollama-launch]\nopenai_base_url = \"http://old:1234/v1/\"\n[another_section]\nfoo = \"bar\"\n"
		os.WriteFile(configPath, []byte(existing), 0o644)

		if err := writeCodexProfile(configPath, catalogPath); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(configPath)
		content := string(data)

		if strings.Contains(content, "old:1234") {
			t.Error("old URL was not replaced")
		}
		if !strings.Contains(content, "[another_section]") {
			t.Error("following section was removed")
		}
		if !strings.Contains(content, "foo = \"bar\"") {
			t.Error("following section content was removed")
		}
	})

	t.Run("appends newline to file not ending with newline", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.toml")
		catalogPath := filepath.Join(tmpDir, "model.json")
		existing := "[other]\nkey = \"val\""
		os.WriteFile(configPath, []byte(existing), 0o644)

		if err := writeCodexProfile(configPath, catalogPath); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(configPath)
		content := string(data)

		if !strings.Contains(content, "[profiles.ollama-launch]") {
			t.Error("missing [profiles.ollama-launch] header")
		}
		// Should not have double blank lines from missing trailing newline
		if strings.Contains(content, "\n\n\n") {
			t.Error("unexpected triple newline in output")
		}
	})

	t.Run("uses custom OLLAMA_HOST", func(t *testing.T) {
		t.Setenv("OLLAMA_HOST", "http://myhost:9999")
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.toml")
		catalogPath := filepath.Join(tmpDir, "model.json")

		if err := writeCodexProfile(configPath, catalogPath); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(configPath)
		content := string(data)

		if !strings.Contains(content, "myhost:9999/v1/") {
			t.Errorf("expected custom host in URL, got:\n%s", content)
		}
	})

	t.Run("uses connectable host for unspecified bind address", func(t *testing.T) {
		t.Setenv("OLLAMA_HOST", "http://0.0.0.0:11434")
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.toml")

		if err := writeCodexProfile(configPath); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(configPath)
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
	t.Run("creates .codex dir and config.toml", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		if err := ensureCodexConfig("llama3.2"); err != nil {
			t.Fatal(err)
		}

		configPath := filepath.Join(tmpDir, ".codex", "config.toml")
		data, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatalf("config.toml not created: %v", err)
		}

		content := string(data)
		if !strings.Contains(content, "[profiles.ollama-launch]") {
			t.Error("missing [profiles.ollama-launch] header")
		}
		if !strings.Contains(content, "openai_base_url") {
			t.Error("missing openai_base_url key")
		}

		catalogPath := filepath.Join(tmpDir, ".codex", "model.json")
		data, err = os.ReadFile(catalogPath)
		if err != nil {
			t.Fatalf("model.json not created: %v", err)
		}
		if !strings.Contains(string(data), `"slug": "llama3.2"`) {
			t.Error("missing model catalog entry for selected model")
		}
	})

	t.Run("is idempotent", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		if err := ensureCodexConfig("llama3.2"); err != nil {
			t.Fatal(err)
		}
		if err := ensureCodexConfig("llama3.2"); err != nil {
			t.Fatal(err)
		}

		configPath := filepath.Join(tmpDir, ".codex", "config.toml")
		data, _ := os.ReadFile(configPath)
		content := string(data)

		if strings.Count(content, "[profiles.ollama-launch]") != 1 {
			t.Errorf("expected exactly one [profiles.ollama-launch] section after two calls, got %d", strings.Count(content, "[profiles.ollama-launch]"))
		}
		if strings.Count(content, "[model_providers.ollama-launch]") != 1 {
			t.Errorf("expected exactly one [model_providers.ollama-launch] section after two calls, got %d", strings.Count(content, "[model_providers.ollama-launch]"))
		}
	})
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

func TestParseNumCtx(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		want       int
	}{
		{"num_ctx set", "num_ctx 8192", 8192},
		{"num_ctx with other params", "temperature 0.7\nnum_ctx 4096\ntop_p 0.9", 4096},
		{"no num_ctx", "temperature 0.7\ntop_p 0.9", 0},
		{"empty string", "", 0},
		{"malformed value", "num_ctx abc", 0},
		{"float value", "num_ctx 8192.0", 8192},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseNumCtx(tt.parameters); got != tt.want {
				t.Errorf("parseNumCtx(%q) = %d, want %d", tt.parameters, got, tt.want)
			}
		})
	}
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
		modelName     string
		showResponse  string
		envContextLen string
		wantContext   int
	}{
		{
			name:      "architectural context length as fallback",
			modelName: "llama3.2",
			showResponse: `{
				"model_info": {"llama.context_length": 131072},
				"details": {"format": "gguf"}
			}`,
			wantContext: 131072,
		},
		{
			name:      "OLLAMA_CONTEXT_LENGTH overrides architectural",
			modelName: "llama3.2",
			showResponse: `{
				"model_info": {"llama.context_length": 131072},
				"details": {"format": "gguf"}
			}`,
			envContextLen: "64000",
			wantContext:   64000,
		},
		{
			name:      "num_ctx overrides OLLAMA_CONTEXT_LENGTH",
			modelName: "llama3.2",
			showResponse: `{
				"model_info": {"llama.context_length": 131072},
				"parameters": "num_ctx 8192",
				"details": {"format": "gguf"}
			}`,
			envContextLen: "64000",
			wantContext:   8192,
		},
		{
			name:      "num_ctx overrides architectural",
			modelName: "llama3.2",
			showResponse: `{
				"model_info": {"llama.context_length": 131072},
				"parameters": "num_ctx 32768",
				"details": {"format": "gguf"}
			}`,
			wantContext: 32768,
		},
		{
			name:      "safetensors uses architectural context only",
			modelName: "llama3.2",
			showResponse: `{
				"model_info": {"llama.context_length": 131072},
				"parameters": "num_ctx 8192",
				"details": {"format": "safetensors"}
			}`,
			envContextLen: "64000",
			wantContext:   131072,
		},
		{
			name:      "cloud model uses hardcoded limits",
			modelName: "qwen3.5:cloud",
			showResponse: `{
				"model_info": {"qwen3_5_moe.context_length": 131072},
				"details": {"format": "gguf"}
			}`,
			envContextLen: "64000",
			wantContext:   262144,
		},
		{
			name:      "unknown cloud model uses fallback context",
			modelName: "deepseek-v4-pro:cloud",
			showResponse: `{
				"model_info": {"deepseek.context_length": 131072},
				"details": {"format": "gguf"}
			}`,
			envContextLen: "64000",
			wantContext:   codexFallbackContextWindow,
		},
		{
			name:      "vision capability without reasoning advertisement",
			modelName: "llama3.2",
			showResponse: `{
				"model_info": {"llama.context_length": 131072},
				"details": {"format": "gguf"},
				"capabilities": ["vision", "thinking"]
			}`,
			wantContext: 131072,
		},
		{
			name:      "system prompt passed through",
			modelName: "llama3.2",
			showResponse: `{
				"model_info": {"llama.context_length": 131072},
				"details": {"format": "gguf"},
				"system": "You are a helpful assistant."
			}`,
			wantContext: 131072,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/show":
					fmt.Fprint(w, tt.showResponse)
				default:
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)

			if tt.envContextLen != "" {
				t.Setenv("OLLAMA_CONTEXT_LENGTH", tt.envContextLen)
			} else {
				t.Setenv("OLLAMA_CONTEXT_LENGTH", "")
			}

			entry := buildCodexModelEntry(tt.modelName)

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

			if tt.name == "system prompt passed through" {
				if got, _ := entry["base_instructions"].(string); got != "You are a helpful assistant." {
					t.Errorf("base_instructions = %q, want %q", got, "You are a helpful assistant.")
				}
			}

			if tt.name == "cloud model uses hardcoded limits" {
				truncationPolicy, _ := entry["truncation_policy"].(map[string]any)
				if mode, _ := truncationPolicy["mode"].(string); mode != "tokens" {
					t.Errorf("truncation_policy mode = %q, want %q", mode, "tokens")
				}
			}

			requiredKeys := []string{"slug", "display_name", "apply_patch_tool_type", "shell_type"}
			for _, key := range requiredKeys {
				if _, ok := entry[key]; !ok {
					t.Errorf("missing required key %q", key)
				}
			}

			if _, err := json.Marshal(entry); err != nil {
				t.Errorf("entry is not JSON serializable: %v", err)
			}
		})
	}
}
