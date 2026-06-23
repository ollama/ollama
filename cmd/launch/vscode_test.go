package launch

import (
	"database/sql"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	_ "github.com/mattn/go-sqlite3"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
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

	t.Run("implements ManagedSingleModel", func(t *testing.T) {
		var _ ManagedSingleModel = v
	})
}

func TestVSCodeConfigure(t *testing.T) {
	v := &VSCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("XDG_CONFIG_HOME", "")
	clmPath := testVSCodePath(t, tmpDir, "chatLanguageModels.json")
	settingsPath := testVSCodePath(t, tmpDir, "settings.json")

	tests := []struct {
		name         string
		setup        string // initial chatLanguageModels.json content, empty means no file
		model        string
		validate     func(t *testing.T, clmData []byte, settingsData []byte)
		wantSettings bool
	}{
		{
			name:         "fresh install",
			model:        "llama3.2",
			wantSettings: true,
			validate: func(t *testing.T, clmData []byte, settingsData []byte) {
				assertOllamaProviderConfigured(t, clmData)
				assertSettingsConfigured(t, settingsData, nil)
			},
		},
		{
			name:         "preserve other vendor entries",
			setup:        `[{"vendor": "azure", "name": "Azure", "url": "https://example.com"}]`,
			model:        "llama3.2",
			wantSettings: true,
			validate: func(t *testing.T, clmData []byte, settingsData []byte) {
				var entries []map[string]any
				json.Unmarshal(clmData, &entries)
				if len(entries) != 2 {
					t.Errorf("expected 2 entries, got %d", len(entries))
				}
				foundAzure := false
				for _, entry := range entries {
					if vendor, _ := entry["vendor"].(string); vendor == "azure" {
						foundAzure = true
					}
				}
				if !foundAzure {
					t.Fatalf("expected azure entry to be preserved, got %#v", entries)
				}
				assertOllamaProviderConfigured(t, clmData)
				assertSettingsConfigured(t, settingsData, nil)
			},
		},
		{
			name:         "update existing ollama entry",
			setup:        `[{"vendor": "ollama", "name": "Ollama", "url": "http://old:11434"}]`,
			model:        "llama3.2",
			wantSettings: true,
			validate: func(t *testing.T, clmData []byte, settingsData []byte) {
				assertOllamaProviderConfigured(t, clmData)
				assertSettingsConfigured(t, settingsData, nil)
			},
		},
		{
			name:         "remove legacy, current, and old custom endpoint ollama entries",
			setup:        `[{"vendor": "ollama", "name": "Ollama", "url": "http://old:11434"},{"vendor": "ollama-vscode", "name": "Ollama", "url": "http://older:11434"},{"vendor": "customendpoint", "name": "Ollama", "url": "http://127.0.0.1:11434"},{"vendor": "azure", "name": "Azure"}]`,
			model:        "llama3.2",
			wantSettings: true,
			validate: func(t *testing.T, clmData []byte, settingsData []byte) {
				var entries []map[string]any
				if err := json.Unmarshal(clmData, &entries); err != nil {
					t.Fatalf("invalid JSON: %v", err)
				}

				if len(entries) != 2 {
					t.Fatalf("expected non-Ollama entry plus one configured provider, got %#v", entries)
				}
				managed := 0
				for _, entry := range entries {
					if isManagedOllamaProviderEntry(entry) {
						managed++
					}
				}
				if managed != 1 {
					t.Fatalf("expected exactly one managed Ollama provider entry, got %#v", entries)
				}
				assertOllamaProviderConfigured(t, clmData)
				assertSettingsConfigured(t, settingsData, nil)
			},
		},
		{
			name:         "empty model is no-op",
			setup:        `[{"vendor": "azure", "name": "Azure"}]`,
			model:        "",
			wantSettings: false,
			validate: func(t *testing.T, clmData []byte, settingsData []byte) {
				if string(clmData) != `[{"vendor": "azure", "name": "Azure"}]` {
					t.Error("empty model should not modify file")
				}
				if len(settingsData) != 0 {
					t.Fatalf("empty model should not create settings, got %q", string(settingsData))
				}
			},
		},
		{
			name:         "corrupted JSON treated as empty",
			setup:        `{corrupted json`,
			model:        "llama3.2",
			wantSettings: true,
			validate: func(t *testing.T, clmData []byte, settingsData []byte) {
				assertOllamaProviderConfigured(t, clmData)
				assertSettingsConfigured(t, settingsData, nil)
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

			if err := v.Configure(tt.model); err != nil {
				t.Fatal(err)
			}

			clmData, _ := os.ReadFile(clmPath)
			settingsData, _ := os.ReadFile(settingsPath)
			if tt.wantSettings && len(settingsData) == 0 {
				t.Fatalf("expected settings.json to be written")
			}
			tt.validate(t, clmData, settingsData)
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

	if err := v.Configure("llama3.2"); err != nil {
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
	if settings["ollama.endpoint"] != envconfig.Host().String() {
		t.Errorf("ollama.endpoint = %v, want %q", settings["ollama.endpoint"], envconfig.Host().String())
	}
	if settings["editor.fontSize"] != float64(14) {
		t.Error("editor.fontSize should have been preserved")
	}
}

func TestVSCodeEdit_CreatesDistinctBackupsForManagedFiles(t *testing.T) {
	v := &VSCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("XDG_CONFIG_HOME", "")

	clmPath := testVSCodePath(t, tmpDir, "chatLanguageModels.json")
	settingsPath := testVSCodePath(t, tmpDir, "settings.json")
	backupDir := fileutil.BackupDir()

	if err := os.MkdirAll(filepath.Dir(clmPath), 0o755); err != nil {
		t.Fatal(err)
	}

	clmOriginal := `[{"vendor":"ollama","name":"Ollama","url":"http://old:11434"}]`
	settingsOriginal := `{"github.copilot.chat.byok.ollamaEndpoint":"http://old:11434","ollama.launch.configured":true,"editor.fontSize":14}`
	if err := os.WriteFile(clmPath, []byte(clmOriginal), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(settingsPath, []byte(settingsOriginal), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := v.Configure("llama3.2"); err != nil {
		t.Fatal(err)
	}

	assertBackupMatches := func(pattern, want string) {
		t.Helper()
		backups, err := filepath.Glob(filepath.Join(backupDir, pattern))
		if err != nil {
			t.Fatalf("glob %q failed: %v", pattern, err)
		}
		for _, backup := range backups {
			data, err := os.ReadFile(backup)
			if err == nil && string(data) == want {
				return
			}
		}
		t.Fatalf("backup matching %q with expected content not found", pattern)
	}

	assertBackupMatches(filepath.Join("vscode", "chatLanguageModels.json.*"), clmOriginal)
	assertBackupMatches(filepath.Join("vscode", "settings.json.*"), settingsOriginal)
}

func TestVSCodePaths(t *testing.T) {
	v := &VSCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("XDG_CONFIG_HOME", "")
	clmPath := testVSCodePath(t, tmpDir, "chatLanguageModels.json")
	settingsPath := testVSCodePath(t, tmpDir, "settings.json")

	t.Run("no file returns nil", func(t *testing.T) {
		os.Remove(clmPath)
		os.Remove(settingsPath)
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

	t.Run("settings file is included", func(t *testing.T) {
		os.Remove(clmPath)
		os.MkdirAll(filepath.Dir(settingsPath), 0o755)
		os.WriteFile(settingsPath, []byte(`{}`), 0o644)

		if paths := v.Paths(); len(paths) != 1 || paths[0] != settingsPath {
			t.Errorf("expected settings path, got %v", paths)
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

func assertChatLanguageModelsCleaned(t *testing.T, data []byte) {
	t.Helper()
	var entries []map[string]any
	if err := json.Unmarshal(data, &entries); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	for _, entry := range entries {
		if isManagedOllamaProviderEntry(entry) {
			t.Fatalf("expected managed Ollama provider entries to be removed, got %#v", entries)
		}
	}
}

func assertOllamaProviderConfigured(t *testing.T, data []byte) {
	t.Helper()
	var entries []map[string]any
	if err := json.Unmarshal(data, &entries); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	var managed []map[string]any
	for _, entry := range entries {
		if isManagedOllamaProviderEntry(entry) {
			managed = append(managed, entry)
		}
	}
	if len(managed) != 1 {
		t.Fatalf("expected exactly one managed Ollama provider, got %#v", entries)
	}
	entry := managed[0]
	if vendor, _ := entry["vendor"].(string); vendor != vscodeOllamaVendor {
		t.Fatalf("vendor = %q, want %q", vendor, vscodeOllamaVendor)
	}
	if name, _ := entry["name"].(string); name != vscodeOllamaName {
		t.Fatalf("name = %q, want %q", name, vscodeOllamaName)
	}
	if url, _ := entry["url"].(string); url != envconfig.Host().String() {
		t.Fatalf("url = %q, want %q", url, envconfig.Host().String())
	}
}

func assertSettingsConfigured(t *testing.T, data []byte, extras map[string]any) {
	t.Helper()
	var settings map[string]any
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatalf("invalid settings JSON: %v", err)
	}

	if settings["ollama.endpoint"] != envconfig.Host().String() {
		t.Fatalf("ollama.endpoint = %v, want %q", settings["ollama.endpoint"], envconfig.Host().String())
	}
	if _, ok := settings["github.copilot.chat.byok.ollamaEndpoint"]; ok {
		t.Fatal("github.copilot.chat.byok.ollamaEndpoint should have been removed")
	}
	if _, ok := settings["ollama.launch.configured"]; ok {
		t.Fatal("ollama.launch.configured should have been removed")
	}
	for key, want := range extras {
		if settings[key] != want {
			t.Fatalf("%s = %v, want %v", key, settings[key], want)
		}
	}
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

	t.Run("fresh DB creates table and selects model", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("XDG_CONFIG_HOME", "")
		if runtime.GOOS == "windows" {
			t.Setenv("APPDATA", tmpDir)
		}

		err := v.ShowInModelPicker("llama3.2")
		if err != nil {
			t.Fatal(err)
		}

		dbPath := testVSCodePath(t, tmpDir, filepath.Join("globalStorage", "state.vscdb"))
		prefs := readPrefs(t, dbPath)
		if len(prefs) != 0 {
			t.Fatalf("expected no model picker overrides, got %#v", prefs)
		}
		assertSelectedChatModel(t, dbPath, vscodeOllamaVendor+"/"+vscodeOllamaName+"/llama3.2:latest")
	})

	t.Run("selected model is stored", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("XDG_CONFIG_HOME", "")
		dbPath := setupDB(t, testVSCodePath(t, tmpDir, ""), nil, nil)

		err := v.ShowInModelPicker("llama3.2")
		if err != nil {
			t.Fatal(err)
		}

		prefs := readPrefs(t, dbPath)
		if len(prefs) != 0 {
			t.Fatalf("expected no model picker overrides, got %#v", prefs)
		}
		assertSelectedChatModel(t, dbPath, vscodeOllamaVendor+"/"+vscodeOllamaName+"/llama3.2:latest")
	})

	t.Run("old launch-managed model visibility overrides are cleared", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("XDG_CONFIG_HOME", "")
		dbPath := setupDB(t, testVSCodePath(t, tmpDir, ""), map[string]bool{
			vscodeOllamaVendor + "/llama3.2":        true,
			vscodeOllamaVendor + "/llama3.2:latest": true,
			vscodeOllamaVendor + "/mistral":         true,
			vscodeOllamaVendor + "/mistral:latest":  true,
			legacyVSCodeOllamaVendor + "/llama3.2":  true,
		}, nil)

		err := v.ShowInModelPicker("llama3.2")
		if err != nil {
			t.Fatal(err)
		}

		prefs := readPrefs(t, dbPath)
		for id := range prefs {
			if isManagedOllamaModelID(id) {
				t.Fatalf("expected managed Ollama overrides to be cleared, got %#v", prefs)
			}
		}
	})

	t.Run("non-ollama prefs are preserved", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("XDG_CONFIG_HOME", "")
		dbPath := setupDB(t, testVSCodePath(t, tmpDir, ""), map[string]bool{
			"copilot/gpt-4o": true,
		}, nil)

		err := v.ShowInModelPicker("llama3.2")
		if err != nil {
			t.Fatal(err)
		}

		prefs := readPrefs(t, dbPath)
		if !prefs["copilot/gpt-4o"] {
			t.Error("expected copilot/gpt-4o to stay shown")
		}
	})

	t.Run("uses cached IDs when available", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("XDG_CONFIG_HOME", "")
		cache := []map[string]any{
			{
				"identifier": vscodeOllamaVendor + "/4",
				"metadata":   map[string]any{"vendor": vscodeOllamaVendor, "name": "llama3.2"},
			},
		}
		dbPath := setupDB(t, testVSCodePath(t, tmpDir, ""), nil, cache)

		err := v.ShowInModelPicker("llama3.2")
		if err != nil {
			t.Fatal(err)
		}

		prefs := readPrefs(t, dbPath)
		if len(prefs) != 0 {
			t.Fatalf("expected no model picker overrides, got %#v", prefs)
		}
		assertSelectedChatModel(t, dbPath, vscodeOllamaVendor+"/4")
	})

	t.Run("keeps current named cache entries and removes legacy named entries", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("XDG_CONFIG_HOME", "")
		cache := []map[string]any{
			{
				"identifier": vscodeOllamaVendor + "/qwen3.6-latest-8e5d0015",
				"metadata":   map[string]any{"vendor": vscodeOllamaVendor, "name": "qwen3.6:latest"},
			},
			{
				"identifier": vscodeOllamaVendor + "/" + vscodeOllamaName + "/qwen3.6-latest-8e5d0015",
				"metadata":   map[string]any{"vendor": vscodeOllamaVendor, "name": "qwen3.6:latest", "detail": vscodeOllamaName},
			},
			{
				"identifier": legacyVSCodeOllamaVendor + "/" + vscodeOllamaName + "/qwen3.6-latest-8e5d0015",
				"metadata":   map[string]any{"vendor": legacyVSCodeOllamaVendor, "name": "qwen3.6:latest", "detail": vscodeOllamaName},
			},
		}
		dbPath := setupDB(t, testVSCodePath(t, tmpDir, ""), nil, cache)

		err := v.ShowInModelPicker("qwen3.6:latest")
		if err != nil {
			t.Fatal(err)
		}

		assertSelectedChatModel(t, dbPath, vscodeOllamaVendor+"/"+vscodeOllamaName+"/qwen3.6-latest-8e5d0015")
		assertNoStaleCachedOllamaEntries(t, dbPath)
		assertCachedLanguageModelPresent(t, dbPath, vscodeOllamaVendor+"/"+vscodeOllamaName+"/qwen3.6-latest-8e5d0015")
	})

	t.Run("empty model is no-op", func(t *testing.T) {
		err := v.ShowInModelPicker("")
		if err != nil {
			t.Fatal(err)
		}
	})

	t.Run("previously hidden selected model uses extension default visibility", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("XDG_CONFIG_HOME", "")
		dbPath := setupDB(t, testVSCodePath(t, tmpDir, ""), map[string]bool{
			vscodeOllamaVendor + "/llama3.2":        false,
			vscodeOllamaVendor + "/llama3.2:latest": false,
		}, nil)

		err := v.ShowInModelPicker("llama3.2")
		if err != nil {
			t.Fatal(err)
		}

		prefs := readPrefs(t, dbPath)
		for id := range prefs {
			if isManagedOllamaModelID(id) {
				t.Fatalf("expected managed Ollama overrides to be cleared, got %#v", prefs)
			}
		}
	})
}

func assertSelectedChatModel(t *testing.T, dbPath, want string) {
	t.Helper()
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	var selected string
	if err := db.QueryRow("SELECT value FROM ItemTable WHERE key = 'chat.currentLanguageModel.panel'").Scan(&selected); err != nil {
		t.Fatal(err)
	}
	if selected != want {
		t.Fatalf("selected model = %q, want %q", selected, want)
	}

	var isDefault string
	if err := db.QueryRow("SELECT value FROM ItemTable WHERE key = 'chat.currentLanguageModel.panel.isDefault'").Scan(&isDefault); err != nil {
		t.Fatal(err)
	}
	if isDefault != "false" {
		t.Fatalf("selected model default flag = %q, want false", isDefault)
	}

	var recentJSON string
	if err := db.QueryRow("SELECT value FROM ItemTable WHERE key = 'chatModelRecentlyUsed'").Scan(&recentJSON); err != nil {
		t.Fatal(err)
	}
	var recent []string
	if err := json.Unmarshal([]byte(recentJSON), &recent); err != nil {
		t.Fatal(err)
	}
	if len(recent) == 0 || recent[0] != want {
		t.Fatalf("recent models = %#v, want %q first", recent, want)
	}
}

func assertNoStaleCachedOllamaEntries(t *testing.T, dbPath string) {
	t.Helper()
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	var cacheJSON string
	if err := db.QueryRow("SELECT value FROM ItemTable WHERE key = 'chat.cachedLanguageModels.v2'").Scan(&cacheJSON); err != nil {
		t.Fatal(err)
	}

	var cached []map[string]any
	if err := json.Unmarshal([]byte(cacheJSON), &cached); err != nil {
		t.Fatal(err)
	}

	for _, entry := range cached {
		if isStaleCachedOllamaModelEntry(entry) {
			t.Fatalf("stale cached Ollama entry still present: %#v", entry)
		}
	}
}

func assertCachedLanguageModelPresent(t *testing.T, dbPath, want string) {
	t.Helper()
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	var cacheJSON string
	if err := db.QueryRow("SELECT value FROM ItemTable WHERE key = 'chat.cachedLanguageModels.v2'").Scan(&cacheJSON); err != nil {
		t.Fatal(err)
	}

	var cached []map[string]any
	if err := json.Unmarshal([]byte(cacheJSON), &cached); err != nil {
		t.Fatal(err)
	}

	for _, entry := range cached {
		if id, _ := entry["identifier"].(string); id == want {
			return
		}
	}
	t.Fatalf("cached language model %q not found in %#v", want, cached)
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

func TestHasVSCodeExtension(t *testing.T) {
	output := "github.copilot-chat@0.41.0\nollama.ollama@0.0.1\nms-python.python\n"

	if !hasVSCodeExtension(output, "ollama.ollama") {
		t.Fatal("expected Ollama extension to be detected")
	}
	if !hasVSCodeExtension(output, "Ollama.Ollama") {
		t.Fatal("expected extension detection to be case-insensitive")
	}
	if hasVSCodeExtension(output, "ollama.missing") {
		t.Fatal("unexpected missing extension detected")
	}
	if !hasVSCodeExtension(output, "ms-python.python") {
		t.Fatal("expected extension without version to be detected")
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
