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
		catalogPath := filepath.Join(tmpDir, codexCatalogFileName)

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
	})

	t.Run("appends profile to existing file without profile", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.toml")
		catalogPath := filepath.Join(tmpDir, codexCatalogFileName)
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
		catalogPath := filepath.Join(tmpDir, codexCatalogFileName)
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
	})

	t.Run("replaces profile while preserving following sections", func(t *testing.T) {
		tmpDir := t.TempDir()
		configPath := filepath.Join(tmpDir, "config.toml")
		catalogPath := filepath.Join(tmpDir, codexCatalogFileName)
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
		catalogPath := filepath.Join(tmpDir, codexCatalogFileName)
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
		catalogPath := filepath.Join(tmpDir, codexCatalogFileName)

		if err := writeCodexProfile(configPath, catalogPath); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(configPath)
		content := string(data)

		if !strings.Contains(content, "myhost:9999/v1/") {
			t.Errorf("expected custom host in URL, got:\n%s", content)
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

		catalogPath := filepath.Join(tmpDir, ".codex", codexCatalogFileName)
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

	t.Run("does not overwrite user's default model catalog", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		userCatalogPath := filepath.Join(tmpDir, ".codex", "model.json")
		if err := os.MkdirAll(filepath.Dir(userCatalogPath), 0o755); err != nil {
			t.Fatal(err)
		}
		original := `{"models":[{"slug":"user-custom"}]}`
		if err := os.WriteFile(userCatalogPath, []byte(original), 0o644); err != nil {
			t.Fatal(err)
		}

		if err := ensureCodexConfig("llama3.2"); err != nil {
			t.Fatal(err)
		}

		data, err := os.ReadFile(userCatalogPath)
		if err != nil {
			t.Fatal(err)
		}
		if string(data) != original {
			t.Errorf("default model catalog was modified: got %q want %q", string(data), original)
		}

		ollamaCatalogPath := filepath.Join(tmpDir, ".codex", codexCatalogFileName)
		if _, err := os.Stat(ollamaCatalogPath); err != nil {
			t.Fatalf("ollama catalog not created: %v", err)
		}
	})

	t.Run("merges additional models into ollama catalog", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		if err := ensureCodexConfig("llama3.2"); err != nil {
			t.Fatal(err)
		}
		if err := ensureCodexConfig("qwen3.5"); err != nil {
			t.Fatal(err)
		}

		catalogPath := filepath.Join(tmpDir, ".codex", codexCatalogFileName)
		data, err := os.ReadFile(catalogPath)
		if err != nil {
			t.Fatal(err)
		}

		var catalog struct {
			Models []struct {
				Slug string `json:"slug"`
			} `json:"models"`
		}
		if err := json.Unmarshal(data, &catalog); err != nil {
			t.Fatalf("failed to parse catalog: %v", err)
		}

		if len(catalog.Models) != 2 {
			t.Fatalf("expected 2 merged models, got %d", len(catalog.Models))
		}
		if catalog.Models[0].Slug != "llama3.2" {
			t.Fatalf("first merged model = %q, want %q", catalog.Models[0].Slug, "llama3.2")
		}
		if catalog.Models[1].Slug != "qwen3.5" {
			t.Fatalf("second merged model = %q, want %q", catalog.Models[1].Slug, "qwen3.5")
		}
	})

	t.Run("refreshes existing model entry in place", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		catalogPath := filepath.Join(tmpDir, ".codex", codexCatalogFileName)
		if err := os.MkdirAll(filepath.Dir(catalogPath), 0o755); err != nil {
			t.Fatal(err)
		}
		original := `{
			"models": [
				{"slug":"llama3.2","display_name":"stale","context_window":1},
				{"slug":"qwen3.5","display_name":"keep","context_window":2}
			]
		}`
		if err := os.WriteFile(catalogPath, []byte(original), 0o644); err != nil {
			t.Fatal(err)
		}

		if err := ensureCodexConfig("llama3.2"); err != nil {
			t.Fatal(err)
		}

		data, err := os.ReadFile(catalogPath)
		if err != nil {
			t.Fatal(err)
		}

		var catalog struct {
			Models []struct {
				Slug          string `json:"slug"`
				DisplayName   string `json:"display_name"`
				ContextWindow int    `json:"context_window"`
			} `json:"models"`
		}
		if err := json.Unmarshal(data, &catalog); err != nil {
			t.Fatalf("failed to parse catalog: %v", err)
		}

		if len(catalog.Models) != 2 {
			t.Fatalf("expected 2 models after refresh, got %d", len(catalog.Models))
		}
		if catalog.Models[0].Slug != "llama3.2" {
			t.Fatalf("first refreshed model = %q, want %q", catalog.Models[0].Slug, "llama3.2")
		}
		if catalog.Models[0].DisplayName != "llama3.2" {
			t.Fatalf("refreshed display_name = %q, want %q", catalog.Models[0].DisplayName, "llama3.2")
		}
		if catalog.Models[1].Slug != "qwen3.5" {
			t.Fatalf("preserved second model = %q, want %q", catalog.Models[1].Slug, "qwen3.5")
		}
	})
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
			name:      "vision and thinking capabilities",
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

			if tt.name == "vision and thinking capabilities" {
				modalities, _ := entry["input_modalities"].([]string)
				if !slices.Contains(modalities, "image") {
					t.Error("expected image in input_modalities")
				}
				levels, _ := entry["supported_reasoning_levels"].([]any)
				if len(levels) == 0 {
					t.Error("expected non-empty supported_reasoning_levels")
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
