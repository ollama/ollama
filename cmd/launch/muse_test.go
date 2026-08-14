package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/ollama/ollama/cmd/internal/fileutil"
)

// museGeneratedSettings is the launch-owned view of the file muse reads.
type museGeneratedSettings struct {
	SchemaVersion int    `json:"schema_version"`
	Provider      string `json:"provider"`
	Model         string `json:"model"`
	Transport     struct {
		BaseURL string `json:"base_url"`
		Auth    string `json:"auth"`
	} `json:"endpoint_transport"`
	ModelCatalog []museCatalogRow `json:"model_catalog"`
	MCPServers   map[string]any   `json:"mcp_servers"`
	TUI          map[string]any   `json:"tui"`
}

func readMuseSettings(t *testing.T) museGeneratedSettings {
	t.Helper()
	path, err := museSettingsPath()
	if err != nil {
		t.Fatalf("museSettingsPath() error = %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read generated settings: %v", err)
	}
	var settings museGeneratedSettings
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatalf("generated settings is not valid JSON: %v", err)
	}
	return settings
}

// stubMuseLoadedContext pins the default model's loaded-context probe so tests
// never load a model on a live server; 0 means "probe failed, keep inventory".
func stubMuseLoadedContext(t *testing.T, n int) {
	t.Helper()
	prev := museLoadedContextLength
	museLoadedContextLength = func(string) int { return n }
	t.Cleanup(func() { museLoadedContextLength = prev })
}

func TestMuseWriteSettings_BuildsCatalog(t *testing.T) {
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_HOST", "127.0.0.1:11434")

	models := []LaunchModel{
		{Name: "gpt-oss:20b", ContextLength: 131072, MaxOutputTokens: 32768},
		{Name: "qwen3:8b"},
	}
	if err := writeMuseSettings(models); err != nil {
		t.Fatalf("writeMuseSettings() error = %v", err)
	}

	settings := readMuseSettings(t)
	if settings.Model != "gpt-oss:20b" {
		t.Errorf("model = %q, want gpt-oss:20b", settings.Model)
	}
	if settings.Provider != museProviderID {
		t.Errorf("provider = %q, want %q", settings.Provider, museProviderID)
	}
	if want := "http://127.0.0.1:11434/v1"; settings.Transport.BaseURL != want {
		t.Errorf("base_url = %q, want %q", settings.Transport.BaseURL, want)
	}
	// Anything but "none" makes muse demand a credential it will never need.
	if settings.Transport.Auth != "none" {
		t.Errorf("auth = %q, want none", settings.Transport.Auth)
	}

	if len(settings.ModelCatalog) != 2 {
		t.Fatalf("model_catalog has %d rows, want 2", len(settings.ModelCatalog))
	}

	first := settings.ModelCatalog[0]
	if first.ModelID != "gpt-oss:20b" || first.DisplayOrder != 0 || !first.IsDefault {
		t.Errorf("first row = %+v, want gpt-oss:20b as the default row", first)
	}
	if first.ContextLimit != 131072 || first.OutputLimit != 32768 {
		t.Errorf("first row limits = %d/%d, want 131072/32768", first.ContextLimit, first.OutputLimit)
	}

	second := settings.ModelCatalog[1]
	if second.ModelID != "qwen3:8b" || second.DisplayOrder != 1 || second.IsDefault {
		t.Errorf("second row = %+v, want qwen3:8b as a non-default row", second)
	}
	if second.ContextLimit != museFallbackContextLimit || second.OutputLimit != museFallbackOutputLimit {
		t.Errorf("second row limits = %d/%d, want the fallbacks %d/%d",
			second.ContextLimit, second.OutputLimit, museFallbackContextLimit, museFallbackOutputLimit)
	}

	// Rows that disagree with the session's provider or profile are dropped by
	// muse, which then falls back to a catalog fetch Ollama cannot serve.
	for _, row := range settings.ModelCatalog {
		if row.ProviderID != museProviderID {
			t.Errorf("row %q provider_id = %q, want %q", row.ModelID, row.ProviderID, museProviderID)
		}
		if row.ProfileID != museProfileID {
			t.Errorf("row %q profile_id = %q, want %q", row.ModelID, row.ProfileID, museProfileID)
		}
		if row.Visibility != "visible" {
			t.Errorf("row %q visibility = %q, want visible", row.ModelID, row.Visibility)
		}
	}
}

func TestMuseWriteSettings_ClampsOutputLimitToContext(t *testing.T) {
	setTestHome(t, t.TempDir())

	if err := writeMuseSettings([]LaunchModel{{Name: "tiny:1b", ContextLength: 4096}}); err != nil {
		t.Fatalf("writeMuseSettings() error = %v", err)
	}

	row := readMuseSettings(t).ModelCatalog[0]
	if row.ContextLimit != 4096 || row.OutputLimit != 4096 {
		t.Errorf("limits = %d/%d, want 4096/4096", row.ContextLimit, row.OutputLimit)
	}
}

// TestMuseApplyLoadedContext pins the property that the launched model's row
// carries the context the server actually loaded it with, not the trained
// maximum from the inventory: muse budgets prompt packing and compaction
// against this row, and the loaded size is what requests really get.
func TestMuseApplyLoadedContext(t *testing.T) {
	setTestHome(t, t.TempDir())
	stubMuseLoadedContext(t, 8192)

	models := []LaunchModel{
		{Name: "gpt-oss:20b", ContextLength: 131072},
		{Name: "qwen3:8b", ContextLength: 131072},
	}
	if err := writeMuseSettings(museApplyLoadedContext(models)); err != nil {
		t.Fatalf("writeMuseSettings() error = %v", err)
	}

	rows := readMuseSettings(t).ModelCatalog
	if rows[0].ContextLimit != 8192 || rows[0].OutputLimit != 8192 {
		t.Errorf("default row limits = %d/%d, want the loaded 8192/8192", rows[0].ContextLimit, rows[0].OutputLimit)
	}
	// Only the launched model is preloaded; other rows keep inventory values.
	if rows[1].ContextLimit != 131072 {
		t.Errorf("second row context = %d, want the inventory 131072", rows[1].ContextLimit)
	}
	// The probe result lands in a copy, not the caller's slice.
	if models[0].ContextLength != 131072 {
		t.Errorf("caller's model mutated to ContextLength=%d", models[0].ContextLength)
	}
}

// TestMuseApplyLoadedContext_SkipsRemoteAndEmpty: a remote (cloud) launch
// target must not be preloaded, and an empty selection passes through.
func TestMuseApplyLoadedContext_SkipsRemoteAndEmpty(t *testing.T) {
	prev := museLoadedContextLength
	museLoadedContextLength = func(string) int {
		t.Fatal("probe must not run for a remote model")
		return 0
	}
	t.Cleanup(func() { museLoadedContextLength = prev })

	models := museApplyLoadedContext([]LaunchModel{{Name: "big:cloud", Remote: true, ContextLength: 65536}})
	if models[0].ContextLength != 65536 {
		t.Errorf("remote model context = %d, want untouched 65536", models[0].ContextLength)
	}
	if got := museApplyLoadedContext(nil); got != nil {
		t.Errorf("nil models = %v, want nil", got)
	}
}

func TestMuseWriteSettings_KeepsUserPreferences(t *testing.T) {
	home := t.TempDir()
	setTestHome(t, home)
	t.Setenv("XDG_CONFIG_HOME", "")

	userSettings := map[string]any{
		"schema_version": 2,
		"provider":       "echo",
		"model":          "muse-large",
		"endpoint_transport": map[string]any{
			"base_url": "https://api.meta.ai/v1",
			"auth":     "bearer",
		},
		"mcp_servers": map[string]any{"github": map[string]any{"transport": "stdio"}},
		"tui":         map[string]any{"theme": "dark"},
	}
	userPath := filepath.Join(home, ".config", "muse", "settings.json")
	if err := os.MkdirAll(filepath.Dir(userPath), 0o755); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(userSettings)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(userPath, data, 0o644); err != nil {
		t.Fatal(err)
	}

	if err := writeMuseSettings([]LaunchModel{{Name: "gpt-oss:20b"}}); err != nil {
		t.Fatalf("writeMuseSettings() error = %v", err)
	}

	settings := readMuseSettings(t)
	if settings.MCPServers["github"] == nil {
		t.Error("mcp_servers was not carried over from the user's settings")
	}
	if settings.TUI["theme"] != "dark" {
		t.Errorf("tui.theme = %v, want dark", settings.TUI["theme"])
	}
	if settings.SchemaVersion != 2 {
		t.Errorf("schema_version = %d, want the user's 2", settings.SchemaVersion)
	}
	if settings.Provider != museProviderID {
		t.Errorf("provider = %q, want it replaced with %q", settings.Provider, museProviderID)
	}
	if settings.Model != "gpt-oss:20b" {
		t.Errorf("model = %q, want gpt-oss:20b", settings.Model)
	}
	if strings.Contains(settings.Transport.BaseURL, "meta.ai") {
		t.Errorf("base_url = %q, want it repointed at Ollama", settings.Transport.BaseURL)
	}

	// The user's own settings must be left exactly as they were.
	after, err := os.ReadFile(userPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(data) {
		t.Errorf("user settings were modified:\n got: %s\nwant: %s", after, data)
	}
}

func TestMuseWriteSettings_KeepsWhatMusePersisted(t *testing.T) {
	setTestHome(t, t.TempDir())

	if err := writeMuseSettings([]LaunchModel{{Name: "gpt-oss:20b"}}); err != nil {
		t.Fatalf("writeMuseSettings() error = %v", err)
	}

	// Muse writes its own settings back into the config root it was handed.
	settingsPath, err := museSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	settings, err := fileutil.ReadJSON(settingsPath)
	if err != nil {
		t.Fatal(err)
	}
	settings["tui"] = map[string]any{"foreign_context_notice_shown": true}
	data, err := json.Marshal(settings)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(settingsPath, data, 0o644); err != nil {
		t.Fatal(err)
	}

	if err := writeMuseSettings([]LaunchModel{{Name: "qwen3:8b"}}); err != nil {
		t.Fatalf("writeMuseSettings() error = %v", err)
	}

	got := readMuseSettings(t)
	if got.TUI["foreign_context_notice_shown"] != true {
		t.Errorf("tui = %v, want muse's persisted settings kept", got.TUI)
	}
	if got.Model != "qwen3:8b" {
		t.Errorf("model = %q, want qwen3:8b", got.Model)
	}
	if diff := compareStrings(museCatalogModelIDs(got.ModelCatalog), []string{"qwen3:8b"}); diff != "" {
		t.Errorf("model_catalog mismatch: %s", diff)
	}
}

func museCatalogModelIDs(rows []museCatalogRow) []string {
	ids := make([]string, 0, len(rows))
	for _, row := range rows {
		ids = append(ids, row.ModelID)
	}
	return ids
}

func TestMuseRunModels(t *testing.T) {
	selection := []LaunchModel{
		{Name: "gpt-oss:20b"},
		{Name: "qwen3:8b"},
	}

	tests := []struct {
		name    string
		primary string
		models  []LaunchModel
		want    []string
	}{
		{
			name:    "primary already first",
			primary: "gpt-oss:20b",
			models:  selection,
			want:    []string{"gpt-oss:20b", "qwen3:8b"},
		},
		{
			name:    "primary moves to front",
			primary: "qwen3:8b",
			models:  selection,
			want:    []string{"qwen3:8b", "gpt-oss:20b"},
		},
		{
			name:    "primary outside the selection",
			primary: "llama3.2",
			models:  selection,
			want:    []string{"llama3.2", "gpt-oss:20b", "qwen3:8b"},
		},
		{
			name:    "no selection",
			primary: "llama3.2",
			want:    []string{"llama3.2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := launchModelNames(museRunModels(tt.primary, tt.models))
			if diff := compareStrings(got, tt.want); diff != "" {
				t.Errorf("museRunModels(%q) mismatch: %s", tt.primary, diff)
			}
		})
	}
}

func TestMusePathsAndModels(t *testing.T) {
	setTestHome(t, t.TempDir())
	m := &Muse{}

	if paths := m.Paths(); paths != nil {
		t.Errorf("Paths() before configuring = %v, want nil", paths)
	}
	if models := m.Models(); models != nil {
		t.Errorf("Models() before configuring = %v, want nil", models)
	}

	if err := m.Edit([]LaunchModel{{Name: "gpt-oss:20b"}, {Name: "qwen3:8b"}}); err != nil {
		t.Fatalf("Edit() error = %v", err)
	}

	settingsPath, err := museSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if diff := compareStrings(m.Paths(), []string{settingsPath}); diff != "" {
		t.Errorf("Paths() mismatch: %s", diff)
	}
	if diff := compareStrings(m.Models(), []string{"gpt-oss:20b", "qwen3:8b"}); diff != "" {
		t.Errorf("Models() mismatch: %s", diff)
	}
}

func TestMuseRun_PointsMuseAtLaunchConfig(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	stubMuseLoadedContext(t, 0)
	t.Setenv("OLLAMA_HOST", "127.0.0.1:11434")

	logPath := filepath.Join(tmpDir, "muse-invocation.log")
	script := fmt.Sprintf(`#!/bin/sh
printf "%%s\n" "$XDG_CONFIG_HOME" >> %q
for arg in "$@"; do
  printf "%%s\n" "$arg" >> %q
done
exit 0
`, logPath, logPath)
	if err := os.WriteFile(filepath.Join(tmpDir, "muse"), []byte(script), 0o755); err != nil {
		t.Fatalf("failed to write fake muse: %v", err)
	}
	t.Setenv("PATH", tmpDir)

	m := &Muse{}
	if err := m.Run("qwen3:8b", testLaunchModels("gpt-oss:20b", "qwen3:8b"), []string{"--trust-workspace"}); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read invocation log: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("invocation log = %v, want the config home and one arg", lines)
	}

	configHome, err := museConfigHome()
	if err != nil {
		t.Fatal(err)
	}
	if lines[0] != configHome {
		t.Errorf("XDG_CONFIG_HOME = %q, want %q", lines[0], configHome)
	}
	if lines[1] != "--trust-workspace" {
		t.Errorf("extra args = %v, want [--trust-workspace]", lines[1:])
	}

	// Run configures muse itself, so the launched model leads the catalog even
	// when Edit never ran.
	settings := readMuseSettings(t)
	if settings.Model != "qwen3:8b" {
		t.Errorf("model = %q, want qwen3:8b", settings.Model)
	}
	if len(settings.ModelCatalog) != 2 || !settings.ModelCatalog[0].IsDefault ||
		settings.ModelCatalog[0].ModelID != "qwen3:8b" {
		t.Errorf("model_catalog = %+v, want qwen3:8b first and default", settings.ModelCatalog)
	}
}

func TestMuseRun_PreservesPreEditBackup(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binary")
	}

	home := t.TempDir()
	setTestHome(t, home)
	stubMuseLoadedContext(t, 8192)

	writeFakeBinary(t, home, "muse")
	t.Setenv("PATH", home)

	settingsPath, err := museSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(settingsPath), 0o755); err != nil {
		t.Fatal(err)
	}
	original := []byte(`{"schema_version":1,"mcp_servers":{"original":{"transport":"stdio"}}}`)
	if err := os.WriteFile(settingsPath, original, 0o644); err != nil {
		t.Fatal(err)
	}

	m := &Muse{}
	models := []LaunchModel{{Name: "gpt-oss:20b", ContextLength: 131072}}
	if err := m.Edit(models); err != nil {
		t.Fatalf("Edit() error = %v", err)
	}
	if err := m.Run("gpt-oss:20b", models, nil); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	settings := readMuseSettings(t)
	if settings.ModelCatalog[0].ContextLimit != 8192 {
		t.Fatalf("context limit = %d, want loaded context 8192", settings.ModelCatalog[0].ContextLimit)
	}

	backups, err := filepath.Glob(filepath.Join(fileutil.BackupDir(), "muse", "settings.json.*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(backups) != 1 {
		t.Fatalf("backup count = %d, want 1", len(backups))
	}

	data, err := os.ReadFile(backups[0])
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != string(original) {
		t.Fatalf("pre-Edit settings backup = %s, want %s", data, original)
	}
}

func TestMuseRun_RequiresModel(t *testing.T) {
	setTestHome(t, t.TempDir())
	if err := (&Muse{}).Run("", nil, nil); err == nil {
		t.Error("Run() without a model = nil, want an error")
	}
}

func TestMuseSupported(t *testing.T) {
	oldGOOS := museGOOS
	t.Cleanup(func() { museGOOS = oldGOOS })

	m := &Muse{}
	for _, goos := range []string{"darwin", "linux"} {
		museGOOS = goos
		if err := m.Supported(); err != nil {
			t.Errorf("Supported() on %s = %v, want nil", goos, err)
		}
	}

	museGOOS = "windows"
	if err := m.Supported(); err == nil {
		t.Error("Supported() on windows = nil, want an error")
	}
}

func TestMuseBaseSettings_MalformedLaunchFileFails(t *testing.T) {
	setTestHome(t, t.TempDir())

	settingsPath, err := museSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(settingsPath), 0o755); err != nil {
		t.Fatal(err)
	}
	malformed := []byte("{ this is not json")
	if err := os.WriteFile(settingsPath, malformed, 0o644); err != nil {
		t.Fatal(err)
	}

	if err := writeMuseSettings([]LaunchModel{{Name: "gpt-oss:20b"}}); err == nil {
		t.Fatal("writeMuseSettings() = nil, want parse error for malformed launch-owned settings")
	}

	data, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != string(malformed) {
		t.Fatalf("malformed settings were rewritten to %q", data)
	}
}

func TestEnsureMuseInstalled(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Muse is not supported on Windows")
	}

	withConfirm := func(t *testing.T, fn func(prompt string) (bool, error)) {
		t.Helper()
		oldConfirm := DefaultConfirmPrompt
		DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
			return fn(prompt)
		}
		t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })
	}

	stubInstaller := func(t *testing.T, script string) {
		t.Helper()
		oldCommand := museInstallCommand
		// Absolute shell path and explicit PATH: these tests clear PATH to
		// hide any real muse, which also hides the script's own utilities.
		museInstallCommand = []string{"/bin/sh", "-c", "PATH=/usr/bin:/bin; " + script}
		t.Cleanup(func() { museInstallCommand = oldCommand })
	}

	t.Run("already installed skips prompt", func(t *testing.T) {
		home := t.TempDir()
		setTestHome(t, home)
		t.Setenv("PATH", t.TempDir())
		binDir := filepath.Join(home, ".local", "bin")
		if err := os.MkdirAll(binDir, 0o755); err != nil {
			t.Fatal(err)
		}
		writeFakeBinary(t, binDir, "muse")

		withConfirm(t, func(prompt string) (bool, error) {
			t.Fatalf("did not expect prompt, got %q", prompt)
			return false, nil
		})

		bin, err := ensureMuseInstalled()
		if err != nil {
			t.Fatalf("ensureMuseInstalled() error = %v", err)
		}
		if bin != filepath.Join(binDir, "muse") {
			t.Fatalf("bin = %q, want %q", bin, filepath.Join(binDir, "muse"))
		}
	})

	t.Run("installs after confirmation and verifies binary", func(t *testing.T) {
		home := t.TempDir()
		setTestHome(t, home)
		t.Setenv("PATH", t.TempDir())
		binDir := filepath.Join(home, ".local", "bin")
		stubInstaller(t, fmt.Sprintf("mkdir -p %q && printf '#!/bin/sh\n' > %q && chmod +x %q",
			binDir, filepath.Join(binDir, "muse"), filepath.Join(binDir, "muse")))

		prompted := false
		withConfirm(t, func(prompt string) (bool, error) {
			prompted = true
			return true, nil
		})

		bin, err := ensureMuseInstalled()
		if err != nil {
			t.Fatalf("ensureMuseInstalled() error = %v", err)
		}
		if !prompted {
			t.Fatal("expected an install confirmation prompt")
		}
		if bin != filepath.Join(binDir, "muse") {
			t.Fatalf("bin = %q, want %q", bin, filepath.Join(binDir, "muse"))
		}
	})

	t.Run("declined prompt cancels", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		t.Setenv("PATH", t.TempDir())
		stubInstaller(t, "exit 0")

		withConfirm(t, func(prompt string) (bool, error) {
			return false, nil
		})

		if _, err := ensureMuseInstalled(); err == nil {
			t.Fatal("ensureMuseInstalled() = nil, want cancellation error")
		}
	})

	t.Run("installer without binary fails verification", func(t *testing.T) {
		setTestHome(t, t.TempDir())
		t.Setenv("PATH", t.TempDir())
		stubInstaller(t, "exit 0")

		withConfirm(t, func(prompt string) (bool, error) {
			return true, nil
		})

		if _, err := ensureMuseInstalled(); err == nil {
			t.Fatal("ensureMuseInstalled() = nil, want verification error")
		}
	})
}
