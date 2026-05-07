package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/envconfig"
)

const (
	codexAppIntegrationName      = "codex-app"
	codexAppProfileName          = "ollama-launch-codex-app"
	codexAppBundleID             = "com.openai.codex"
	codexAppModelCatalogFilename = "ollama-launch-models.json"
	codexAppRestoreHint          = "To restore your usual Codex profile, run: ollama launch codex-app --restore"
	codexAppConfigurationSuccess = "Codex App profile changed to Ollama."
	codexAppRestoreSuccess       = "Codex App restored to your usual profile."
)

var (
	codexAppGOOS      = runtime.GOOS
	codexAppStat      = os.Stat
	codexAppGlob      = filepath.Glob
	codexAppOpenApp   = defaultCodexAppOpenApp
	codexAppOpenPath  = defaultCodexAppOpenAppPath
	codexAppOpenStart = defaultCodexAppOpenStartAppID
	codexAppQuitApp   = defaultCodexAppQuitApp
	codexAppIsRunning = defaultCodexAppIsRunning
	codexAppRunPath   = defaultCodexAppRunningAppPath
	codexAppStartID   = defaultCodexAppStartAppID
	codexAppSleep     = time.Sleep
)

// CodexApp configures the desktop Codex app with one launch-selected default
// model while leaving model discovery and switching to Codex's Ollama provider.
type CodexApp struct{}

func (c *CodexApp) String() string { return "Codex App" }

func (c *CodexApp) Supported() error { return codexAppSupported() }

func (c *CodexApp) Paths() []string {
	configPath, err := codexConfigPath()
	if err != nil {
		return nil
	}
	return []string{configPath}
}

func (c *CodexApp) Configure(model string) error {
	return c.ConfigureWithModels(model, []string{model})
}

func (c *CodexApp) ConfigureWithModels(primary string, models []string) error {
	primary = strings.TrimSpace(primary)
	if primary == "" {
		return fmt.Errorf("codex-app requires a model")
	}

	configPath, err := codexConfigPath()
	if err != nil {
		return err
	}
	if err := saveCodexAppRestoreState(configPath); err != nil {
		return err
	}
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		return err
	}
	if err := writeCodexAppModelCatalog(catalogPath, codexAppCatalogModelNames(primary, models)); err != nil {
		return err
	}
	return writeCodexLaunchProfile(configPath, codexLaunchProfileOptions{
		activate:           true,
		profileName:        codexAppProfileName,
		setRootModelConfig: true,
		model:              primary,
		modelCatalogPath:   catalogPath,
		backupIntegration:  codexAppIntegrationName,
	})
}

func (c *CodexApp) CurrentModel() string {
	configPath, err := codexConfigPath()
	if err != nil {
		return ""
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}
	text := string(data)
	for _, profileName := range codexAppManagedProfileNames() {
		if codexRootStringValue(text, "model_provider") == profileName {
			baseURL := codexSectionStringValue(text, codexProviderHeaderFor(profileName), "base_url")
			if codexNormalizeURL(baseURL) == codexNormalizeURL(codexBaseURL()) {
				return strings.TrimSpace(codexRootStringValue(text, "model"))
			}
		}
	}

	profileName := codexRootStringValue(text, "profile")
	if !codexAppIsManagedProfileName(profileName) {
		return ""
	}
	if codexSectionStringValue(text, codexProfileHeaderFor(profileName), "model_provider") != profileName {
		return ""
	}
	baseURL := codexSectionStringValue(text, codexProviderHeaderFor(profileName), "base_url")
	if codexNormalizeURL(baseURL) != codexNormalizeURL(codexBaseURL()) {
		return ""
	}
	return strings.TrimSpace(codexSectionStringValue(text, codexProfileHeaderFor(profileName), "model"))
}

func codexAppManagedProfileNames() []string {
	return []string{codexAppProfileName, codexProfileName}
}

func codexAppIsManagedProfileName(profileName string) bool {
	for _, candidate := range codexAppManagedProfileNames() {
		if profileName == candidate {
			return true
		}
	}
	return false
}

func codexAppIsOwnedProfileName(profileName string) bool {
	return profileName == codexAppProfileName
}

func (c *CodexApp) Onboard() error {
	return config.MarkIntegrationOnboarded(codexAppIntegrationName)
}

func (c *CodexApp) RequiresInteractiveOnboarding() bool {
	return false
}

func (c *CodexApp) RestoreHint() string {
	return codexAppRestoreHint
}

func (c *CodexApp) ConfigurationSuccessMessage() string {
	return codexAppConfigurationSuccess + "\n" + codexAppRestoreHint
}

func (c *CodexApp) RestoreSuccessMessage() string {
	return codexAppRestoreSuccess
}

func (c *CodexApp) Run(_ string, args []string) error {
	if err := codexAppSupported(); err != nil {
		return err
	}
	if len(args) > 0 {
		return fmt.Errorf("codex-app does not accept extra arguments")
	}
	return codexAppLaunchOrRestart("Restart Codex to use Ollama?")
}

func (c *CodexApp) Restore() error {
	if err := codexAppSupported(); err != nil {
		return err
	}
	configPath, err := codexConfigPath()
	if err != nil {
		return err
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			return codexAppLaunchOrRestart("Restart Codex to use your usual profile?")
		}
		return err
	}
	text := string(data)
	if err := codexValidateConfigText(text); err != nil {
		return err
	}

	state, stateErr := loadCodexAppRestoreState()
	if stateErr == nil {
		text = codexAppRestoreRootValues(text, state)
	} else if os.IsNotExist(stateErr) {
		text = codexAppRemoveOwnedRootValues(text)
	} else {
		return stateErr
	}
	if !codexAppRootReferencesOwnedConfig(text) {
		text = codexAppRemoveOwnedSections(text)
	}

	if err := codexValidateConfigText(text); err != nil {
		return err
	}
	if err := codexWriteWithBackup(configPath, []byte(text), codexAppIntegrationName); err != nil {
		return err
	}
	codexAppRemoveOwnedCatalogIfUnused(text)
	_ = os.Remove(codexAppRestoreStatePath())
	return codexAppLaunchOrRestart("Restart Codex to use your usual profile?")
}

func codexAppSupported() error {
	switch codexAppGOOS {
	case "darwin", "windows":
		return nil
	default:
		return fmt.Errorf("Codex App launch is only supported on macOS and Windows")
	}
}

func codexAppInstalled() bool {
	if codexAppAppPath() != "" {
		return true
	}
	if codexAppGOOS != "windows" {
		return false
	}
	return codexAppIsRunning() || codexAppStartID() != ""
}

func codexAppModelCatalogPath() (string, error) {
	configPath, err := codexConfigPath()
	if err != nil {
		return "", err
	}
	return filepath.Join(filepath.Dir(configPath), codexAppModelCatalogFilename), nil
}

func writeCodexAppModelCatalog(path string, models []string) error {
	if len(models) == 0 {
		return fmt.Errorf("codex-app model catalog cannot be empty")
	}
	client := api.NewClient(envconfig.Host(), http.DefaultClient)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	baseInstructions := codexAppBaseInstructions()
	entries := make([]map[string]any, 0, len(models))
	for i, model := range models {
		contextWindow := codexAppModelContextWindow(ctx, client, model)
		entries = append(entries, codexAppCatalogEntry(model, contextWindow, i, baseInstructions))
	}

	data, err := json.MarshalIndent(map[string]any{"models": entries}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return codexWriteWithBackup(path, append(data, '\n'), codexAppIntegrationName)
}

func codexAppCatalogModelNames(primary string, fallback []string) []string {
	models := codexAppTagModelNames()
	if len(models) == 0 {
		models = fallback
	}
	return dedupeModelList(append([]string{primary}, models...))
}

func codexAppTagModelNames() []string {
	client := api.NewClient(envconfig.Host(), http.DefaultClient)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	resp, err := client.List(ctx)
	if err != nil {
		return nil
	}

	models := make([]string, 0, len(resp.Models))
	for _, model := range resp.Models {
		name := strings.TrimSpace(model.Name)
		if name != "" {
			models = append(models, name)
		}
	}
	return models
}

func codexAppModelContextWindow(ctx context.Context, client *api.Client, model string) int {
	resp, err := client.Show(ctx, &api.ShowRequest{Model: model})
	if err != nil {
		return 272_000
	}
	if n, ok := modelInfoContextLength(resp.ModelInfo); ok {
		return n
	}
	return 272_000
}

func codexAppCatalogEntry(model string, contextWindow, priority int, baseInstructions string) map[string]any {
	return map[string]any{
		"slug":                             model,
		"display_name":                     model,
		"description":                      "Ollama local model",
		"default_reasoning_level":          nil,
		"supported_reasoning_levels":       []any{},
		"shell_type":                       "default",
		"visibility":                       "list",
		"supported_in_api":                 true,
		"priority":                         priority,
		"additional_speed_tiers":           []any{},
		"availability_nux":                 nil,
		"upgrade":                          nil,
		"base_instructions":                baseInstructions,
		"model_messages":                   nil,
		"supports_reasoning_summaries":     false,
		"default_reasoning_summary":        "auto",
		"support_verbosity":                false,
		"default_verbosity":                nil,
		"apply_patch_tool_type":            nil,
		"web_search_tool_type":             "text",
		"truncation_policy":                map[string]any{"mode": "bytes", "limit": 10_000},
		"supports_parallel_tool_calls":     false,
		"supports_image_detail_original":   false,
		"context_window":                   contextWindow,
		"max_context_window":               contextWindow,
		"auto_compact_token_limit":         nil,
		"effective_context_window_percent": 95,
		"experimental_supported_tools":     []any{},
		"input_modalities":                 []string{"text", "image"},
		"supports_search_tool":             false,
	}
}

func codexAppBaseInstructions() string {
	path, err := codexModelCachePath()
	if err == nil {
		var cached struct {
			Models []struct {
				BaseInstructions string `json:"base_instructions"`
			} `json:"models"`
		}
		if data, readErr := os.ReadFile(path); readErr == nil {
			if json.Unmarshal(data, &cached) == nil {
				for _, model := range cached.Models {
					if strings.TrimSpace(model.BaseInstructions) != "" {
						return model.BaseInstructions
					}
				}
			}
		}
	}
	return "You are Codex, a coding agent. You and the user share the same workspace and collaborate to achieve the user's goals."
}

func codexModelCachePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".codex", "models_cache.json"), nil
}

func codexAppAppPath() string {
	var candidates []string
	switch codexAppGOOS {
	case "darwin":
		candidates = codexAppDarwinAppCandidates()
	case "windows":
		candidates = codexAppWindowsAppCandidates()
	default:
		return ""
	}
	for _, candidate := range candidates {
		if info, err := codexAppStat(candidate); err == nil {
			if codexAppGOOS == "darwin" && !info.IsDir() {
				continue
			}
			if codexAppGOOS == "windows" && info.IsDir() {
				continue
			}
			return candidate
		}
	}
	return ""
}

func codexAppDarwinAppCandidates() []string {
	candidates := []string{"/Applications/Codex.app"}
	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates, filepath.Join(home, "Applications", "Codex.app"))
	}
	return candidates
}

func codexAppWindowsAppCandidates() []string {
	local, err := codexAppLocalAppData()
	if err != nil {
		return nil
	}

	candidates := []string{
		filepath.Join(local, "Programs", "Codex", "Codex.exe"),
		filepath.Join(local, "Programs", "OpenAI Codex", "Codex.exe"),
		filepath.Join(local, "Codex", "Codex.exe"),
		filepath.Join(local, "OpenAI Codex", "Codex.exe"),
		filepath.Join(local, "OpenAI", "Codex", "Codex.exe"),
		filepath.Join(local, "openai-codex-electron", "Codex.exe"),
	}
	for _, pattern := range []string{
		filepath.Join(local, "Programs", "Codex", "app-*", "Codex.exe"),
		filepath.Join(local, "Programs", "OpenAI Codex", "app-*", "Codex.exe"),
		filepath.Join(local, "Codex", "app-*", "Codex.exe"),
		filepath.Join(local, "OpenAI Codex", "app-*", "Codex.exe"),
		filepath.Join(local, "OpenAI", "Codex", "app-*", "Codex.exe"),
		filepath.Join(local, "openai-codex-electron", "app-*", "Codex.exe"),
	} {
		matches, _ := codexAppGlob(pattern)
		candidates = append(candidates, matches...)
	}
	return codexAppDedupePaths(candidates)
}

func codexAppDedupePaths(paths []string) []string {
	out := make([]string, 0, len(paths))
	seen := make(map[string]bool, len(paths))
	for _, path := range paths {
		if strings.TrimSpace(path) == "" {
			continue
		}
		key := strings.ToLower(path)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, path)
	}
	return out
}

func codexAppLocalAppData() (string, error) {
	if local := strings.TrimSpace(os.Getenv("LOCALAPPDATA")); local != "" {
		return local, nil
	}
	if home := strings.TrimSpace(os.Getenv("USERPROFILE")); home != "" {
		return filepath.Join(home, "AppData", "Local"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, "AppData", "Local"), nil
}

func codexAppLaunchOrRestart(prompt string) error {
	if !codexAppIsRunning() {
		return codexAppOpenApp()
	}
	restartAppID := ""
	restartAppPath := ""
	if codexAppGOOS == "windows" {
		restartAppID = codexAppStartID()
		if restartAppID == "" {
			restartAppPath = codexAppRunPath()
		}
	}

	restart, err := ConfirmPrompt(prompt)
	if err != nil {
		return err
	}
	if !restart {
		fmt.Fprintln(os.Stderr, "\nQuit and reopen Codex when you're ready for the profile change to take effect.")
		return nil
	}

	if err := codexAppQuitApp(); err != nil {
		return fmt.Errorf("quit Codex: %w", err)
	}
	if err := waitForCodexAppExit(30 * time.Second); err != nil {
		return err
	}
	if restartAppID != "" {
		return codexAppOpenStart(restartAppID)
	}
	if restartAppPath != "" {
		return codexAppOpenPath(restartAppPath)
	}
	return codexAppOpenApp()
}

func waitForCodexAppExit(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !codexAppIsRunning() {
			return nil
		}
		codexAppSleep(200 * time.Millisecond)
	}
	return fmt.Errorf("Codex did not quit; quit it manually and re-run the command")
}

func defaultCodexAppOpenApp() error {
	switch codexAppGOOS {
	case "windows":
		if path := codexAppAppPath(); path != "" {
			return codexAppOpenPath(path)
		}
		if path := codexAppRunPath(); path != "" {
			return codexAppOpenPath(path)
		}
		if appID := codexAppStartID(); appID != "" {
			return codexAppOpenStart(appID)
		}
		return fmt.Errorf("Codex executable was not found; open Codex manually once and re-run 'ollama launch codex-app'")
	case "darwin":
		if path := codexAppAppPath(); path != "" {
			cmd := exec.Command("open", path)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			return cmd.Run()
		}
		cmd := exec.Command("open", "-b", codexAppBundleID)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		return cmd.Run()
	default:
		return codexAppSupported()
	}
}

func defaultCodexAppOpenAppPath(path string) error {
	switch codexAppGOOS {
	case "windows":
		return exec.Command("powershell.exe", "-NoProfile", "-Command", "Start-Process -FilePath "+quotePowerShellString(path)).Run()
	case "darwin":
		cmd := exec.Command("open", path)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		return cmd.Run()
	default:
		return codexAppSupported()
	}
}

func defaultCodexAppOpenStartAppID(appID string) error {
	return exec.Command("powershell.exe", "-NoProfile", "-Command", "Start-Process "+quotePowerShellString(`shell:AppsFolder\`+appID)).Run()
}

func defaultCodexAppQuitApp() error {
	if codexAppGOOS == "windows" {
		script := `Get-Process Codex -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | ForEach-Object { [void]$_.CloseMainWindow() }`
		scriptErr := exec.Command("powershell.exe", "-NoProfile", "-Command", script).Run()
		codexAppSleep(500 * time.Millisecond)
		if err := defaultCodexAppTerminateProcesses(); err != nil {
			if scriptErr != nil {
				return fmt.Errorf("quit script failed: %v; terminate failed: %w", scriptErr, err)
			}
			return err
		}
		return nil
	}

	scriptErr := exec.Command("osascript", "-e", `tell application "Codex" to quit`).Run()
	if scriptErr != nil {
		scriptErr = exec.Command("osascript", "-e", `tell application id "`+codexAppBundleID+`" to quit`).Run()
	}
	codexAppSleep(500 * time.Millisecond)
	if err := defaultCodexAppTerminateProcesses(); err != nil {
		if scriptErr != nil {
			return fmt.Errorf("quit script failed: %v; terminate failed: %w", scriptErr, err)
		}
		return err
	}
	return nil
}

func defaultCodexAppIsRunning() bool {
	switch codexAppGOOS {
	case "windows":
		return len(codexAppMatchingProcessIDs()) > 0
	case "darwin":
		out, err := exec.Command("osascript", "-e", `tell application "System Events" to exists process "Codex"`).Output()
		if err == nil && strings.TrimSpace(string(out)) == "true" {
			return true
		}
		return len(codexAppMatchingProcessIDs()) > 0
	default:
		return false
	}
}

func defaultCodexAppTerminateProcesses() error {
	pids := codexAppMatchingProcessIDs()
	if codexAppGOOS == "windows" {
		if len(pids) == 0 {
			return nil
		}
		ids := make([]string, 0, len(pids))
		for _, pid := range pids {
			ids = append(ids, strconv.Itoa(pid))
		}
		script := "Stop-Process -Id " + strings.Join(ids, ",") + " -ErrorAction SilentlyContinue"
		return exec.Command("powershell.exe", "-NoProfile", "-Command", script).Run()
	}

	var failures []string
	for _, pid := range pids {
		process, err := os.FindProcess(pid)
		if err != nil {
			failures = append(failures, fmt.Sprintf("%d: %v", pid, err))
			continue
		}
		if err := process.Signal(syscall.SIGTERM); err != nil && !strings.Contains(err.Error(), "process already finished") {
			failures = append(failures, fmt.Sprintf("%d: %v", pid, err))
		}
	}
	if len(failures) > 0 {
		return fmt.Errorf("%s", strings.Join(failures, "; "))
	}
	return nil
}

func codexAppMatchingProcessIDs() []int {
	if codexAppGOOS == "windows" {
		return codexAppWindowsMatchingProcessIDs()
	}

	out, err := exec.Command("ps", "-axo", "pid=,command=").Output()
	if err != nil {
		return nil
	}
	var pids []int
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		pid, err := strconv.Atoi(fields[0])
		if err != nil || pid == os.Getpid() {
			continue
		}
		command := strings.TrimSpace(strings.TrimPrefix(line, fields[0]))
		if codexAppProcessMatches(command) {
			pids = append(pids, pid)
		}
	}
	return pids
}

func codexAppWindowsMatchingProcessIDs() []int {
	script := fmt.Sprintf(`$current = %d; Get-CimInstance Win32_Process -Filter "Name = 'Codex.exe' OR Name = 'codex.exe'" | Where-Object { $_.ProcessId -ne $current -and ((($_.Name -ieq 'Codex.exe') -and (($null -eq $_.CommandLine) -or ($_.CommandLine -notlike '* --type=*'))) -or (($_.Name -ieq 'codex.exe') -and ($_.CommandLine -like '*app-server*'))) } | Select-Object -ExpandProperty ProcessId`, os.Getpid())
	out, err := exec.Command("powershell.exe", "-NoProfile", "-Command", script).Output()
	if err != nil {
		return nil
	}

	var pids []int
	for _, line := range strings.Split(string(out), "\n") {
		pid, err := strconv.Atoi(strings.TrimSpace(line))
		if err == nil && pid != os.Getpid() {
			pids = append(pids, pid)
		}
	}
	return pids
}

func defaultCodexAppRunningAppPath() string {
	if codexAppGOOS != "windows" {
		return ""
	}
	script := `(Get-Process Codex -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 -and $_.Path } | Select-Object -First 1 -ExpandProperty Path)`
	out, err := exec.Command("powershell.exe", "-NoProfile", "-Command", script).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func defaultCodexAppStartAppID() string {
	if codexAppGOOS != "windows" {
		return ""
	}
	script := `(Get-StartApps Codex | Where-Object { $_.Name -eq 'Codex' -or $_.Name -like 'Codex*' } | Select-Object -First 1 -ExpandProperty AppID)`
	out, err := exec.Command("powershell.exe", "-NoProfile", "-Command", script).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func codexAppProcessMatches(command string) bool {
	if strings.Contains(command, `\Codex.exe`) && strings.Contains(command, " --type=") {
		return false
	}
	for _, pattern := range codexAppProcessPatterns() {
		if strings.Contains(command, pattern) {
			return true
		}
	}
	return false
}

func codexAppProcessPatterns() []string {
	return []string{
		"Codex.app/Contents/MacOS/Codex",
		"Codex.app/Contents/Resources/codex app-server",
		`\Codex.exe`,
		`resources\codex.exe app-server`,
		`resources\codex.exe" app-server`,
		`resources\codex.exe" "app-server`,
	}
}

func codexNormalizeURL(raw string) string {
	return strings.TrimRight(strings.TrimSpace(raw), "/")
}

func codexAppRootStillManaged(text string) bool {
	if codexAppIsOwnedProfileName(codexRootStringValue(text, "profile")) {
		return true
	}
	if codexAppIsOwnedProfileName(codexRootStringValue(text, "model_provider")) {
		return true
	}
	return false
}

func codexAppRootReferencesOwnedConfig(text string) bool {
	return codexRootStringValue(text, "profile") == codexAppProfileName ||
		codexRootStringValue(text, "model_provider") == codexAppProfileName
}

func codexAppRootReferencesCatalog(text string) bool {
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		return false
	}
	return codexRootStringValue(text, "model_catalog_json") == catalogPath
}

func codexAppRemoveOwnedSections(text string) string {
	text = codexRemoveSection(text, codexProfileHeaderFor(codexAppProfileName))
	text = codexRemoveSection(text, codexProviderHeaderFor(codexAppProfileName))
	return text
}

func codexAppRemoveOwnedCatalogIfUnused(text string) {
	if codexAppRootReferencesCatalog(text) {
		return
	}
	if catalogPath, err := codexAppModelCatalogPath(); err == nil {
		_ = os.Remove(catalogPath)
	}
}

func codexAppRemoveOwnedRootValues(text string) string {
	if !codexAppRootStillManaged(text) {
		return text
	}
	text = codexRemoveRootValue(text, "profile")
	if codexAppIsOwnedProfileName(codexRootStringValue(text, "model_provider")) {
		text = codexRemoveRootValue(text, "model_provider")
	}
	if catalogPath, err := codexAppModelCatalogPath(); err == nil && codexRootStringValue(text, "model_catalog_json") == catalogPath {
		text = codexRemoveRootValue(text, "model_catalog_json")
	}
	return text
}

func codexAppRestoreRootValues(text string, state codexAppRestoreState) string {
	if !codexAppRootStillManaged(text) {
		return text
	}
	text = codexRestoreRootStringValue(text, "profile", state.HadProfile, state.Profile)
	text = codexRestoreRootStringValue(text, "model", state.HadModel, state.Model)
	text = codexRestoreRootStringValue(text, "model_provider", state.HadModelProvider, state.ModelProvider)
	text = codexRestoreRootStringValue(text, "model_catalog_json", state.HadModelCatalogJSON, state.ModelCatalogJSON)
	return text
}

type codexAppRestoreState struct {
	HadProfile          bool   `json:"had_profile"`
	Profile             string `json:"profile,omitempty"`
	HadModel            bool   `json:"had_model"`
	Model               string `json:"model,omitempty"`
	HadModelProvider    bool   `json:"had_model_provider"`
	ModelProvider       string `json:"model_provider,omitempty"`
	HadModelCatalogJSON bool   `json:"had_model_catalog_json"`
	ModelCatalogJSON    string `json:"model_catalog_json,omitempty"`
}

func saveCodexAppRestoreState(configPath string) error {
	configText := ""
	configExists := false
	if configData, err := os.ReadFile(configPath); err == nil {
		configText = string(configData)
		if err := codexValidateConfigText(configText); err != nil {
			return err
		}
		configExists = true
	} else if !os.IsNotExist(err) {
		return err
	}

	statePath := codexAppRestoreStatePath()
	if stateData, err := os.ReadFile(statePath); err == nil {
		hasRootConfig, err := codexAppRestoreStateHasRootConfig(stateData)
		if err != nil {
			return err
		}
		if hasRootConfig {
			return nil
		}
		if !configExists {
			return nil
		}
		var existing codexAppRestoreState
		if err := json.Unmarshal(stateData, &existing); err != nil {
			return err
		}
		upgraded := codexAppRestoreStateFromText(configText)
		upgraded.HadProfile = existing.HadProfile
		upgraded.Profile = existing.Profile
		return writeCodexAppRestoreState(upgraded)
	} else if !os.IsNotExist(err) {
		return err
	}

	if !configExists {
		return writeCodexAppRestoreState(codexAppRestoreState{})
	}
	return writeCodexAppRestoreState(codexAppRestoreStateFromText(configText))
}

func codexAppRestoreStateHasRootConfig(data []byte) (bool, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return false, err
	}
	_, hasModel := raw["had_model"]
	_, hasModelProvider := raw["had_model_provider"]
	_, hasModelCatalogJSON := raw["had_model_catalog_json"]
	return hasModel && hasModelProvider && hasModelCatalogJSON, nil
}

func codexAppRestoreStateFromText(text string) codexAppRestoreState {
	profile, hadProfile := codexRootStringValueOK(text, "profile")
	model, hadModel := codexRootStringValueOK(text, "model")
	modelProvider, hadModelProvider := codexRootStringValueOK(text, "model_provider")
	modelCatalogJSON, hadModelCatalogJSON := codexRootStringValueOK(text, "model_catalog_json")
	return codexAppRestoreState{
		HadProfile:          hadProfile,
		Profile:             profile,
		HadModel:            hadModel,
		Model:               model,
		HadModelProvider:    hadModelProvider,
		ModelProvider:       modelProvider,
		HadModelCatalogJSON: hadModelCatalogJSON,
		ModelCatalogJSON:    modelCatalogJSON,
	}
}

func codexRestoreRootStringValue(text, key string, hadValue bool, value string) string {
	if hadValue {
		return codexSetRootStringValue(text, key, value)
	}
	return codexRemoveRootValue(text, key)
}

func writeCodexAppRestoreState(state codexAppRestoreState) error {
	path := codexAppRestoreStatePath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return codexWriteWithBackup(path, data, codexAppIntegrationName)
}

func loadCodexAppRestoreState() (codexAppRestoreState, error) {
	data, err := os.ReadFile(codexAppRestoreStatePath())
	if err != nil {
		return codexAppRestoreState{}, err
	}
	var state codexAppRestoreState
	if err := json.Unmarshal(data, &state); err != nil {
		return codexAppRestoreState{}, err
	}
	return state, nil
}

func codexAppRestoreStatePath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "ollama-codex-app-restore.json")
	}
	return filepath.Join(home, ".ollama", "launch", "codex-app-restore.json")
}
