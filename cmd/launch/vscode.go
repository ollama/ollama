package launch

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
)

// VSCode implements Runner and Editor for Visual Studio Code integration.
type VSCode struct{}

func (v *VSCode) String() string { return "Visual Studio Code" }

// findBinary returns the path/command to launch VS Code, or "" if not found.
// It checks platform-specific locations only.
func (v *VSCode) findBinary() string {
	var candidates []string
	switch runtime.GOOS {
	case "darwin":
		candidates = []string{
			"/Applications/Visual Studio Code.app",
		}
	case "windows":
		if localAppData := os.Getenv("LOCALAPPDATA"); localAppData != "" {
			candidates = append(candidates, filepath.Join(localAppData, "Programs", "Microsoft VS Code", "bin", "code.cmd"))
		}
	default: // linux
		candidates = []string{
			"/usr/bin/code",
			"/snap/bin/code",
		}
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

// IsRunning reports whether VS Code is currently running.
// Each platform uses a pattern specific enough to avoid matching Cursor or
// other VS Code forks.
func (v *VSCode) IsRunning() bool {
	switch runtime.GOOS {
	case "darwin":
		out, err := exec.Command("pgrep", "-f", "Visual Studio Code.app/Contents/MacOS/Code").Output()
		return err == nil && len(out) > 0
	case "windows":
		// Match VS Code by executable path to avoid matching Cursor or other forks.
		out, err := exec.Command("powershell", "-NoProfile", "-Command",
			`Get-Process Code -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*Microsoft VS Code*' } | Select-Object -First 1`).Output()
		return err == nil && len(strings.TrimSpace(string(out))) > 0
	default:
		// Match VS Code specifically by its install path to avoid matching
		// Cursor (/cursor/) or other forks.
		for _, pattern := range []string{"/usr/share/code/", "/snap/code/"} {
			out, err := exec.Command("pgrep", "-f", pattern).Output()
			if err == nil && len(out) > 0 {
				return true
			}
		}
		return false
	}
}

// Quit gracefully quits VS Code and waits for it to exit so that it flushes
// its in-memory state back to the database.
func (v *VSCode) Quit() {
	if !v.IsRunning() {
		return
	}
	switch runtime.GOOS {
	case "darwin":
		_ = exec.Command("osascript", "-e", `quit app "Visual Studio Code"`).Run()
	case "windows":
		// Kill VS Code by executable path to avoid killing Cursor or other forks.
		_ = exec.Command("powershell", "-NoProfile", "-Command",
			`Get-Process Code -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*Microsoft VS Code*' } | Stop-Process -Force`).Run()
	default:
		for _, pattern := range []string{"/usr/share/code/", "/snap/code/"} {
			_ = exec.Command("pkill", "-f", pattern).Run()
		}
	}
	// Wait for the process to fully exit and flush its state to disk
	// TODO(hoyyeva): update spinner to use bubble tea
	spinnerFrames := []string{"|", "/", "-", "\\"}
	frame := 0
	fmt.Fprintf(os.Stderr, "\033[90mRestarting VS Code... %s\033[0m", spinnerFrames[0])

	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	for range 150 { // 150 ticks × 200ms = 30s timeout
		<-ticker.C
		frame++
		fmt.Fprintf(os.Stderr, "\r\033[90mRestarting VS Code... %s\033[0m", spinnerFrames[frame%len(spinnerFrames)])

		if frame%5 == 0 { // check every ~1s
			if !v.IsRunning() {
				fmt.Fprintf(os.Stderr, "\r\033[K")
				// Give VS Code a moment to finish writing its state DB
				time.Sleep(1 * time.Second)
				return
			}
		}
	}
	fmt.Fprintf(os.Stderr, "\r\033[K")
}

const (
	minCopilotChatVersion = "0.41.0"
	minVSCodeVersion      = "1.113"
)

func (v *VSCode) Run(model string, args []string) error {
	v.checkVSCodeVersion()
	v.checkCopilotChatVersion()

	// Get all configured models (saved by the launcher framework before Run is called)
	models := []string{model}
	if cfg, err := loadStoredIntegrationConfig("vscode"); err == nil && len(cfg.Models) > 0 {
		models = cfg.Models
	}

	// VS Code discovers models from ollama ls. Cloud models that pass Show
	// (the server knows about them) but aren't in ls need to be pulled to
	// register them so VS Code can find them.
	if client, err := api.ClientFromEnvironment(); err == nil {
		v.ensureModelsRegistered(context.Background(), client, models)
	}

	// Warn if the default model doesn't support tool calling
	if client, err := api.ClientFromEnvironment(); err == nil {
		if resp, err := client.Show(context.Background(), &api.ShowRequest{Model: models[0]}); err == nil {
			hasTools := false
			for _, c := range resp.Capabilities {
				if c == "tools" {
					hasTools = true
					break
				}
			}
			if !hasTools {
				fmt.Fprintf(os.Stderr, "Note: %s does not support tool calling and may not appear in the Copilot Chat model picker.\n", models[0])
			}
		}
	}

	v.printModelAccessTip()

	if v.IsRunning() {
		restart, err := ConfirmPrompt("Restart VS Code?")
		if err != nil {
			restart = false
		}
		if restart {
			v.Quit()
			if err := v.ShowInModelPicker(models); err != nil {
				fmt.Fprintf(os.Stderr, "%s  Warning: could not update VS Code model picker: %v%s\n", ansiYellow, err, ansiReset)
			}
			v.FocusVSCode()
		} else {
			fmt.Fprintf(os.Stderr, "\nTo get the latest model configuration, restart VS Code when you're ready.\n")
		}
	} else {
		if err := v.ShowInModelPicker(models); err != nil {
			fmt.Fprintf(os.Stderr, "%s  Warning: could not update VS Code model picker: %v%s\n", ansiYellow, err, ansiReset)
		}
		v.FocusVSCode()
	}

	return nil
}

// ensureModelsRegistered pulls models that the server knows about (Show succeeds)
// but aren't in ollama ls yet. This is needed for cloud models so that VS Code
// can discover them from the Ollama API.
func (v *VSCode) ensureModelsRegistered(ctx context.Context, client *api.Client, models []string) {
	listed, err := client.List(ctx)
	if err != nil {
		return
	}
	registered := make(map[string]bool, len(listed.Models))
	for _, m := range listed.Models {
		registered[m.Name] = true
	}

	for _, model := range models {
		if registered[model] {
			continue
		}
		// Also check without :latest suffix
		if !strings.Contains(model, ":") && registered[model+":latest"] {
			continue
		}
		if err := pullModel(ctx, client, model, false); err != nil {
			fmt.Fprintf(os.Stderr, "%s  Warning: could not register model %s: %v%s\n", ansiYellow, model, err, ansiReset)
		}
	}
}

// FocusVSCode brings VS Code to the foreground.
func (v *VSCode) FocusVSCode() {
	binary := v.findBinary()
	if binary == "" {
		return
	}
	if runtime.GOOS == "darwin" && strings.HasSuffix(binary, ".app") {
		_ = exec.Command("open", "-a", binary).Run()
	} else {
		_ = exec.Command(binary).Start()
	}
}

// printModelAccessTip shows instructions for finding Ollama models in VS Code.
func (v *VSCode) printModelAccessTip() {
	fmt.Fprintf(os.Stderr, "\nTip: To use Ollama models, open Copilot Chat and click the model picker.\n")
	fmt.Fprintf(os.Stderr, "     If you don't see your models, click \"Other models\" to find them.\n\n")
}

func (v *VSCode) Paths() []string {
	if p := v.chatLanguageModelsPath(); fileExists(p) {
		return []string{p}
	}
	return nil
}

func (v *VSCode) Edit(models []string) error {
	if len(models) == 0 {
		return nil
	}

	// Write chatLanguageModels.json with Ollama vendor entry
	clmPath := v.chatLanguageModelsPath()
	if err := os.MkdirAll(filepath.Dir(clmPath), 0o755); err != nil {
		return err
	}

	var entries []map[string]any
	if data, err := os.ReadFile(clmPath); err == nil {
		_ = json.Unmarshal(data, &entries)
	}

	// Remove any existing Ollama entries, preserve others
	filtered := make([]map[string]any, 0, len(entries))
	for _, entry := range entries {
		if vendor, _ := entry["vendor"].(string); vendor != "ollama" {
			filtered = append(filtered, entry)
		}
	}

	// Add new Ollama entry
	filtered = append(filtered, map[string]any{
		"vendor": "ollama",
		"name":   "Ollama",
		"url":    envconfig.Host().String(),
	})

	data, err := json.MarshalIndent(filtered, "", "  ")
	if err != nil {
		return err
	}
	if err := fileutil.WriteWithBackup(clmPath, data, "vscode"); err != nil {
		return err
	}

	// Clean up legacy settings from older Ollama integrations
	v.updateSettings()

	return nil
}

func (v *VSCode) Models() []string {
	if !v.hasOllamaVendor() {
		return nil
	}
	if cfg, err := loadStoredIntegrationConfig("vscode"); err == nil {
		return cfg.Models
	}
	return nil
}

// hasOllamaVendor checks if chatLanguageModels.json contains an Ollama vendor entry.
func (v *VSCode) hasOllamaVendor() bool {
	data, err := os.ReadFile(v.chatLanguageModelsPath())
	if err != nil {
		return false
	}

	var entries []map[string]any
	if err := json.Unmarshal(data, &entries); err != nil {
		return false
	}

	for _, entry := range entries {
		if vendor, _ := entry["vendor"].(string); vendor == "ollama" {
			return true
		}
	}
	return false
}

func (v *VSCode) chatLanguageModelsPath() string {
	return v.vscodePath("chatLanguageModels.json")
}

func (v *VSCode) settingsPath() string {
	return v.vscodePath("settings.json")
}

// updateSettings cleans up legacy settings from older Ollama integrations.
func (v *VSCode) updateSettings() {
	settingsPath := v.settingsPath()
	data, err := os.ReadFile(settingsPath)
	if err != nil {
		return
	}

	var settings map[string]any
	if err := json.Unmarshal(data, &settings); err != nil {
		return
	}

	changed := false
	for _, key := range []string{"github.copilot.chat.byok.ollamaEndpoint", "ollama.launch.configured"} {
		if _, ok := settings[key]; ok {
			delete(settings, key)
			changed = true
		}
	}

	if !changed {
		return
	}

	updated, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return
	}
	_ = fileutil.WriteWithBackup(settingsPath, updated, "vscode")
}

func (v *VSCode) statePath() string {
	return v.vscodePath("globalStorage", "state.vscdb")
}

// ShowInModelPicker ensures the given models are visible in VS Code's Copilot
// Chat model picker. It sets the configured models to true in the picker
// preferences so they appear in the dropdown. Models use the VS Code identifier
// format "ollama/Ollama/<name>".
func (v *VSCode) ShowInModelPicker(models []string) error {
	if len(models) == 0 {
		return nil
	}

	dbPath := v.statePath()
	needsCreate := !fileExists(dbPath)
	if needsCreate {
		if err := os.MkdirAll(filepath.Dir(dbPath), 0o755); err != nil {
			return fmt.Errorf("creating state directory: %w", err)
		}
	}

	db, err := sql.Open("sqlite3", dbPath+"?_busy_timeout=5000")
	if err != nil {
		return fmt.Errorf("opening state database: %w", err)
	}
	defer db.Close()

	// Create the table if this is a fresh DB. Schema must match what VS Code creates.
	if needsCreate {
		if _, err := db.Exec("CREATE TABLE ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)"); err != nil {
			return fmt.Errorf("initializing state database: %w", err)
		}
	}

	// Read existing preferences
	prefs := make(map[string]bool)
	var prefsJSON string
	if err := db.QueryRow("SELECT value FROM ItemTable WHERE key = 'chatModelPickerPreferences'").Scan(&prefsJSON); err == nil {
		_ = json.Unmarshal([]byte(prefsJSON), &prefs)
	}

	// Build name→ID map from VS Code's cached model list.
	// VS Code uses numeric IDs like "ollama/Ollama/4", not "ollama/Ollama/kimi-k2.5:cloud".
	nameToID := make(map[string]string)
	var cacheJSON string
	if err := db.QueryRow("SELECT value FROM ItemTable WHERE key = 'chat.cachedLanguageModels.v2'").Scan(&cacheJSON); err == nil {
		var cached []map[string]any
		if json.Unmarshal([]byte(cacheJSON), &cached) == nil {
			for _, entry := range cached {
				meta, _ := entry["metadata"].(map[string]any)
				if meta == nil {
					continue
				}
				if vendor, _ := meta["vendor"].(string); vendor == "ollama" {
					name, _ := meta["name"].(string)
					id, _ := entry["identifier"].(string)
					if name != "" && id != "" {
						nameToID[name] = id
					}
				}
			}
		}
	}

	// Ollama config is authoritative: always show configured models,
	// hide Ollama models that are no longer in the config.
	configuredIDs := make(map[string]bool)
	for _, m := range models {
		for _, id := range v.modelVSCodeIDs(m, nameToID) {
			prefs[id] = true
			configuredIDs[id] = true
		}
	}
	for id := range prefs {
		if strings.HasPrefix(id, "ollama/") && !configuredIDs[id] {
			prefs[id] = false
		}
	}

	data, _ := json.Marshal(prefs)
	if _, err = db.Exec("INSERT OR REPLACE INTO ItemTable (key, value) VALUES ('chatModelPickerPreferences', ?)", string(data)); err != nil {
		return err
	}

	return nil
}

// modelVSCodeIDs returns all possible VS Code picker IDs for a model name.
func (v *VSCode) modelVSCodeIDs(model string, nameToID map[string]string) []string {
	var ids []string
	if id, ok := nameToID[model]; ok {
		ids = append(ids, id)
	} else if !strings.Contains(model, ":") {
		if id, ok := nameToID[model+":latest"]; ok {
			ids = append(ids, id)
		}
	}
	ids = append(ids, "ollama/Ollama/"+model)
	if !strings.Contains(model, ":") {
		ids = append(ids, "ollama/Ollama/"+model+":latest")
	}
	return ids
}

func (v *VSCode) vscodePath(parts ...string) string {
	home, _ := os.UserHomeDir()
	var base string
	switch runtime.GOOS {
	case "darwin":
		base = filepath.Join(home, "Library", "Application Support", "Code", "User")
	case "windows":
		base = filepath.Join(os.Getenv("APPDATA"), "Code", "User")
	default:
		base = filepath.Join(home, ".config", "Code", "User")
	}
	return filepath.Join(append([]string{base}, parts...)...)
}

// checkVSCodeVersion warns if VS Code is older than minVSCodeVersion.
func (v *VSCode) checkVSCodeVersion() {
	codeCLI := v.findCodeCLI()
	if codeCLI == "" {
		return
	}

	out, err := exec.Command(codeCLI, "--version").Output()
	if err != nil {
		return
	}

	// "code --version" outputs: version\ncommit\narch
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 || lines[0] == "" {
		return
	}
	version := strings.TrimSpace(lines[0])

	if compareVersions(version, minVSCodeVersion) < 0 {
		fmt.Fprintf(os.Stderr, "\n%sWarning: VS Code version (%s) is older than the recommended version (%s)%s\n", ansiYellow, version, minVSCodeVersion, ansiReset)
		fmt.Fprintf(os.Stderr, "Please update VS Code to the latest version.\n\n")
	}
}

// checkCopilotChatVersion warns if the GitHub Copilot Chat extension is
// missing or older than minCopilotChatVersion.
func (v *VSCode) checkCopilotChatVersion() {
	codeCLI := v.findCodeCLI()
	if codeCLI == "" {
		return
	}

	out, err := exec.Command(codeCLI, "--list-extensions", "--show-versions").Output()
	if err != nil {
		return
	}

	installed, version := parseCopilotChatVersion(string(out))
	if !installed {
		fmt.Fprintf(os.Stderr, "\n%sWarning: GitHub Copilot Chat extension is not installed%s\n", ansiYellow, ansiReset)
		fmt.Fprintf(os.Stderr, "Install it in VS Code: Extensions → search \"GitHub Copilot Chat\" → Install\n\n")
		return
	}
	if compareVersions(version, minCopilotChatVersion) < 0 {
		fmt.Fprintf(os.Stderr, "\n%sWarning: GitHub Copilot Chat extension version (%s) is older than the recommended version (%s)%s\n", ansiYellow, version, minCopilotChatVersion, ansiReset)
		fmt.Fprintf(os.Stderr, "Please update it in VS Code: Extensions → search \"GitHub Copilot Chat\" → Update\n\n")
	}
}

// findCodeCLI returns the path to the VS Code CLI for querying extensions.
// On macOS, findBinary may return an .app bundle which can't run --list-extensions,
// so this resolves to the actual CLI binary inside the bundle.
func (v *VSCode) findCodeCLI() string {
	binary := v.findBinary()
	if binary == "" {
		return ""
	}
	if runtime.GOOS == "darwin" && strings.HasSuffix(binary, ".app") {
		bundleCLI := binary + "/Contents/Resources/app/bin/code"
		if _, err := os.Stat(bundleCLI); err == nil {
			return bundleCLI
		}
		return ""
	}
	return binary
}

// parseCopilotChatVersion extracts the version of the GitHub Copilot Chat
// extension from "code --list-extensions --show-versions" output.
func parseCopilotChatVersion(output string) (installed bool, version string) {
	for _, line := range strings.Split(output, "\n") {
		// Format: github.copilot-chat@0.40.1
		if !strings.HasPrefix(strings.ToLower(line), "github.copilot-chat@") {
			continue
		}
		parts := strings.SplitN(line, "@", 2)
		if len(parts) != 2 {
			continue
		}
		return true, strings.TrimSpace(parts[1])
	}
	return false, ""
}

// compareVersions compares two dot-separated version strings.
// Returns -1 if a < b, 0 if a == b, 1 if a > b.
func compareVersions(a, b string) int {
	aParts := strings.Split(a, ".")
	bParts := strings.Split(b, ".")

	maxLen := len(aParts)
	if len(bParts) > maxLen {
		maxLen = len(bParts)
	}

	for i := range maxLen {
		var aNum, bNum int
		if i < len(aParts) {
			aNum, _ = strconv.Atoi(aParts[i])
		}
		if i < len(bParts) {
			bNum, _ = strconv.Atoi(bParts[i])
		}
		if aNum < bNum {
			return -1
		}
		if aNum > bNum {
			return 1
		}
	}
	return 0
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
