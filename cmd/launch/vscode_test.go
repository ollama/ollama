package launch

import (
	"database/sql"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

func TestVSCodeIntegration(t *testing.T) {
	v := &VSCode{}

	t.Run("String", func(t *testing.T) {
		if got := v.String(); got != "Visual Studio Code" {
			t.Errorf("String() = %q, want %q", got, "Visual Studio Code")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = v
	})

	t.Run("implements Editor", func(t *testing.T) {
		var _ Editor = v
	})
}

func TestVSCodeEdit(t *testing.T) {
	v := &VSCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("XDG_CONFIG_HOME", "")
	clmPath := testVSCodePath(t, tmpDir, "chatLanguageModels.json")

	tests := []struct {
		name     string
		setup    string // initial chatLanguageModels.json content, empty means no file
		models   []string
		validate func(t *testing.T, data []byte)
	}{
		{
			name:   "fresh install",
			models: []string{"llama3.2"},
			validate: func(t *testing.T, data []byte) {
				assertOllamaVendorConfigured(t, data)
			},
		},
		{
			name:   "preserve other vendor entries",
			setup:  `[{"vendor": "azure", "name": "Azure", "url": "https://example.com"}]`,
			models: []string{"llama3.2"},
			validate: func(t *testing.T, data []byte) {
				var entries []map[string]any
				json.Unmarshal(data, &entries)
				if len(entries) != 2 {
					t.Errorf("expected 2 entries, got %d", len(entries))
				}
				// Check Azure entry preserved
				found := false
				for _, e := range entries {
					if v, _ := e["vendor"].(string); v == "azure" {
						found = true
					}
				}
				if !found {
					t.Error("azure vendor entry was not preserved")
				}
				assertOllamaVendorConfigured(t, data)
			},
		},
		{
			name:   "update existing ollama entry",
			setup:  `[{"vendor": "ollama", "name": "Ollama", "url": "http://old:11434"}]`,
			models: []string{"llama3.2"},
			validate: func(t *testing.T, data []byte) {
				assertOllamaVendorConfigured(t, data)
			},
		},
		{
			name:   "empty models is no-op",
			setup:  `[{"vendor": "azure", "name": "Azure"}]`,
			models: []string{},
			validate: func(t *testing.T, data []byte) {
				if string(data) != `[{"vendor": "azure", "name": "Azure"}]` {
					t.Error("empty models should not modify file")
				}
			},
		},
		{
			name:   "corrupted JSON treated as empty",
			setup:  `{corrupted json`,
			models: []string{"llama3.2"},
			validate: func(t *testing.T, data []byte) {
				var entries []map[string]any
				if err := json.Unmarshal(data, &entries); err != nil {
					t.Errorf("result is not valid JSON: %v", err)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			os.RemoveAll(filepath.Dir(clmPath))

			if tt.setup != "" {
				os.MkdirAll(filepath.Dir(clmPath), 0o755)
				os.WriteFile(clmPath, []byte(tt.setup), 0o644)
			}

			if err := v.Edit(tt.models); err != nil {
				t.Fatal(err)
			}

			data, _ := os.ReadFile(clmPath)
			tt.validate(t, data)
		})
	}
}

func TestVSCodeEditCleansUpOldSettings(t *testing.T) {
	v := &VSCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("XDG_CONFIG_HOME", "")
	settingsPath := testVSCodePath(t, tmpDir, "settings.json")

	// Create settings.json with old byok setting
	os.MkdirAll(filepath.Dir(settingsPath), 0o755)
	os.WriteFile(settingsPath, []byte(`{"github.copilot.chat.byok.ollamaEndpoint": "http://old:11434", "ollama.launch.configured": true, "editor.fontSize": 14}`), 0o644)

	if err := v.Edit([]string{"llama3.2"}); err != nil {
		t.Fatal(err)
	}

	// Verify old settings were removed
	data, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatal(err)
	}

	var settings map[string]any
	json.Unmarshal(data, &settings)
	if _, ok := settings["github.copilot.chat.byok.ollamaEndpoint"]; ok {
		t.Error("github.copilot.chat.byok.ollamaEndpoint should have been removed")
	}
	if _, ok := settings["ollama.launch.configured"]; ok {
		t.Error("ollama.launch.configured should have been removed")
	}
	if settings["editor.fontSize"] != float64(14) {
		t.Error("editor.fontSize should have been preserved")
	}
}

func TestVSCodePaths(t *testing.T) {
	v := &VSCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("XDG_CONFIG_HOME", "")
	clmPath := testVSCodePath(t, tmpDir, "chatLanguageModels.json")

	t.Run("no file returns nil", func(t *testing.T) {
		os.Remove(clmPath)
		if paths := v.Paths(); paths != nil {
			t.Errorf("expected nil, got %v", paths)
		}
	})

	t.Run("existing file returns path", func(t *testing.T) {
		os.MkdirAll(filepath.Dir(clmPath), 0o755)
		os.WriteFile(clmPath, []byte(`[]`), 0o644)

		if paths := v.Paths(); len(paths) != 1 {
			t.Errorf("expected 1 path, got %d", len(paths))
		}
	})
}

// testVSCodePath returns the expected VS Code config path for the given file in tests.
func testVSCodePath(t *testing.T, tmpDir, filename string) string {
	t.Helper()
	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(tmpDir, "Library", "Application Support", "Code", "User", filename)
	case "windows":
		t.Setenv("APPDATA", tmpDir)
		return filepath.Join(tmpDir, "Code", "User", filename)
	default:
		return filepath.Join(tmpDir, ".config", "Code", "User", filename)
	}
}

func assertOllamaVendorConfigured(t *testing.T, data []byte) {
	t.Helper()
	var entries []map[string]any
	if err := json.Unmarshal(data, &entries); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	for _, entry := range entries {
		if vendor, _ := entry["vendor"].(string); vendor == "ollama" {
			if name, _ := entry["name"].(string); name != "Ollama" {
				t.Errorf("expected name \"Ollama\", got %q", name)
			}
			if url, _ := entry["url"].(string); url == "" {
				t.Error("url not set")
			}
			return
		}
	}
	t.Error("no ollama vendor entry found")
}

func TestShowInModelPicker(t *testing.T) {
	v := &VSCode{}

	// helper to create a state DB with optional seed data
	setupDB := func(t *testing.T, tmpDir string, seedPrefs map[string]bool, seedCache []map[string]any) string {
		t.Helper()
		dbDir := filepath.Join(tmpDir, "globalStorage")
		os.MkdirAll(dbDir, 0o755)
		dbPath := filepath.Join(dbDir, "state.vscdb")

		db, err := sql.Open("sqlite3", dbPath)
		if err != nil {
			t.Fatal(err)
		}
		defer db.Close()

		if _, err := db.Exec("CREATE TABLE ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)"); err != nil {
			t.Fatal(err)
		}
		if seedPrefs != nil {
			data, _ := json.Marshal(seedPrefs)
			db.Exec("INSERT INTO ItemTable (key, value) VALUES ('chatModelPickerPreferences', ?)", string(data))
		}
		if seedCache != nil {
			data, _ := json.Marshal(seedCache)
			db.Exec("INSERT INTO ItemTable (key, value) VALUES ('chat.cachedLanguageModels.v2', ?)", string(data))
		}
		return dbPath
	}

	// helper to read prefs back from DB
	readPrefs := func(t *testing.T, dbPath string) map[string]bool {
		t.Helper()
		db, err := sql.Open("sqlite3", dbPath)
		if err != nil {
			t.Fatal(err)
		}
		defer db.Close()

		var raw string
		if err := db.QueryRow("SELECT value FROM ItemTable WHERE key = 'chatModelPickerPreferences'").Scan(&raw); err != nil {
			t.Fatal(err)
		}
		prefs := make(map[string]bool)
		json.Unmarshal([]byte(raw), &prefs)
		return prefs
	}

	t.Run("fresh DB creates table and shows models", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("XDG_CONFIG_HOME", "")
		if runtime.GOOS == "windows" {
			t.Setenv("APPDATA", tmpDir)
		}

		err := v.ShowInModelPicker([]string{"llama3.2"})
		if err != nil {
			t.Fatal(err)
		}

		dbPath := testVSCodePath(t, tmpDir, filepath.Join("globalStorage", "state.vscdb"))
		prefs := readPrefs(t, dbPath)
		if !prefs["ollama/Ollama/llama3.2"] {
			t.Error("expected llama3.2 to be shown")
		}
		if !prefs["ollama/Ollama/llama3.2:latest"] {
			t.Error("expected llama3.2:latest to be shown")
		}
	})

	t.Run("configured models are shown", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("XDG_CONFIG_HOME", "")
		dbPath := setupDB(t, testVSCodePath(t, tmpDir, ""), nil, nil)

		err := v.ShowInModelPicker([]string{"llama3.2", "qwen3:8b"})
		if err != nil {
			t.Fatal(err)
		}

		prefs := readPrefs(t, dbPath)
		if !prefs["ollama/Ollama/llama3.2"] {
			t.Error("expected llama3.2 to be shown")
		}
		if !prefs["ollama/Ollama/qwen3:8b"] {
			t.Error("expected qwen3:8b to be shown")
		}
	})

	t.Run("removed models are hidden", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("XDG_CONFIG_HOME", "")
		dbPath := setupDB(t, testVSCodePath(t, tmpDir, ""), map[string]bool{
			"ollama/Ollama/llama3.2":        true,
			"ollama/Ollama/llama3.2:latest": true,
			"ollama/Ollama/mistral":         true,
			"ollama/Ollama/mistral:latest":  true,
		}, nil)

		// Only configure llama3.2 — mistral should get hidden
		err := v.ShowInModelPicker([]string{"llama3.2"})
		if err != nil {
			t.Fatal(err)
		}

		prefs := readPrefs(t, dbPath)
		if !prefs["ollama/Ollama/llama3.2"] {
			t.Error("expected llama3.2 to stay shown")
		}
		if prefs["ollama/Ollama/mistral"] {
			t.Error("expected mistral to be hidden")
		}
		if prefs["ollama/Ollama/mistral:latest"] {
			t.Error("expected mistral:latest to be hidden")
		}
	})

	t.Run("non-ollama prefs are preserved", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("XDG_CONFIG_HOME", "")
		dbPath := setupDB(t, testVSCodePath(t, tmpDir, ""), map[string]bool{
			"copilot/gpt-4o": true,
		}, nil)

		err := v.ShowInModelPicker([]string{"llama3.2"})
		if err != nil {
			t.Fatal(err)
		}

		prefs := readPrefs(t, dbPath)
		if !prefs["copilot/gpt-4o"] {
			t.Error("expected copilot/gpt-4o to stay shown")
		}
	})

	t.Run("uses cached numeric IDs when available", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("XDG_CONFIG_HOME", "")
		cache := []map[string]any{
			{
				"identifier": "ollama/Ollama/4",
				"metadata":   map[string]any{"vendor": "ollama", "name": "llama3.2"},
			},
		}
		dbPath := setupDB(t, testVSCodePath(t, tmpDir, ""), nil, cache)

		err := v.ShowInModelPicker([]string{"llama3.2"})
		if err != nil {
			t.Fatal(err)
		}

		prefs := readPrefs(t, dbPath)
		if !prefs["ollama/Ollama/4"] {
			t.Error("expected numeric ID ollama/Ollama/4 to be shown")
		}
		// Name-based fallback should also be set
		if !prefs["ollama/Ollama/llama3.2"] {
			t.Error("expected name-based ID to also be shown")
		}
	})

	t.Run("empty models is no-op", func(t *testing.T) {
		err := v.ShowInModelPicker([]string{})
		if err != nil {
			t.Fatal(err)
		}
	})

	t.Run("previously hidden model is re-shown when configured", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("XDG_CONFIG_HOME", "")
		dbPath := setupDB(t, testVSCodePath(t, tmpDir, ""), map[string]bool{
			"ollama/Ollama/llama3.2":        false,
			"ollama/Ollama/llama3.2:latest": false,
		}, nil)

		// Ollama config is authoritative — should override the hidden state
		err := v.ShowInModelPicker([]string{"llama3.2"})
		if err != nil {
			t.Fatal(err)
		}

		prefs := readPrefs(t, dbPath)
		if !prefs["ollama/Ollama/llama3.2"] {
			t.Error("expected llama3.2 to be re-shown")
		}
	})
}

func TestParseCopilotChatVersion(t *testing.T) {
	tests := []struct {
		name          string
		output        string
		wantInstalled bool
		wantVersion   string
	}{
		{
			name:          "found among other extensions",
			output:        "ms-python.python@2024.1.1\ngithub.copilot-chat@0.40.1\ngithub.copilot@1.200.0\n",
			wantInstalled: true,
			wantVersion:   "0.40.1",
		},
		{
			name:          "only extension",
			output:        "GitHub.copilot-chat@0.41.0\n",
			wantInstalled: true,
			wantVersion:   "0.41.0",
		},
		{
			name:          "not installed",
			output:        "ms-python.python@2024.1.1\ngithub.copilot@1.200.0\n",
			wantInstalled: false,
		},
		{
			name:          "empty output",
			output:        "",
			wantInstalled: false,
		},
		{
			name:          "case insensitive match",
			output:        "GitHub.Copilot-Chat@0.39.0\n",
			wantInstalled: true,
			wantVersion:   "0.39.0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			installed, version := parseCopilotChatVersion(tt.output)
			if installed != tt.wantInstalled {
				t.Errorf("installed = %v, want %v", installed, tt.wantInstalled)
			}
			if installed && version != tt.wantVersion {
				t.Errorf("version = %q, want %q", version, tt.wantVersion)
			}
		})
	}
}

func TestCompareVersions(t *testing.T) {
	tests := []struct {
		a, b string
		want int
	}{
		{"0.40.1", "0.40.1", 0},
		{"0.40.2", "0.40.1", 1},
		{"0.40.0", "0.40.1", -1},
		{"0.41.0", "0.40.1", 1},
		{"0.39.9", "0.40.1", -1},
		{"1.0.0", "0.40.1", 1},
		{"0.40", "0.40.1", -1},
		{"0.40.1.1", "0.40.1", 1},
	}

	for _, tt := range tests {
		t.Run(tt.a+"_vs_"+tt.b, func(t *testing.T) {
			got := compareVersions(tt.a, tt.b)
			if got != tt.want {
				t.Errorf("compareVersions(%q, %q) = %d, want %d", tt.a, tt.b, got, tt.want)
			}
		})
	}
}
