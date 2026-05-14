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
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
	modelpkg "github.com/ollama/ollama/types/model"
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
	codexAppForceQuit = defaultCodexAppForceQuitApp
	codexAppHasWindow = defaultCodexAppHasOpenWindow
	codexAppIsRunning = defaultCodexAppIsRunning
	codexAppRunPath   = defaultCodexAppRunningAppPath
	codexAppStartID   = defaultCodexAppStartAppID
	codexAppCanOpenID = defaultCodexAppCanOpenBundleID
	codexAppSleep     = time.Sleep

	codexAppExitTimeout      = 5 * time.Second
	codexAppForceExitTimeout = 5 * time.Second
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
	if err := writeCodexAppModelCatalog(catalogPath, primary, codexAppCatalogModelNames(primary, models)); err != nil {
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
	parsed, err := codexParseConfig(text)
	if err != nil {
		return ""
	}
	for _, profileName := range codexAppManagedProfileNames() {
		if parsed.RootString(codexRootModelProviderKey) == profileName {
			baseURL := parsed.ProviderString(profileName, "base_url")
			if codexNormalizeURL(baseURL) == codexNormalizeURL(codexBaseURL()) && codexAppCatalogHealthy(parsed, profileName) {
				return strings.TrimSpace(parsed.RootString(codexRootModelKey))
			}
		}
	}

	profileName := parsed.RootString(codexRootProfileKey)
	if !codexAppIsManagedProfileName(profileName) {
		return ""
	}
	if parsed.ProfileString(profileName, codexRootModelProviderKey) != profileName {
		return ""
	}
	baseURL := parsed.ProviderString(profileName, "base_url")
	if codexNormalizeURL(baseURL) != codexNormalizeURL(codexBaseURL()) {
		return ""
	}
	if !codexAppCatalogHealthy(parsed, profileName) {
		return ""
	}
	return strings.TrimSpace(parsed.ProfileString(profileName, codexRootModelKey))
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

func codexAppCatalogHealthy(config codexParsedConfig, profileName string) bool {
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		return false
	}
	if config.RootString(codexRootModelCatalogJSONKey) != catalogPath {
		return false
	}
	if config.ProfileString(profileName, codexRootModelCatalogJSONKey) != catalogPath {
		return false
	}
	data, err := os.ReadFile(catalogPath)
	if err != nil {
		return false
	}
	var catalog struct {
		Models []json.RawMessage `json:"models"`
	}
	if err := json.Unmarshal(data, &catalog); err != nil {
		return false
	}
	return len(catalog.Models) > 0
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
			if err := removeCodexAppRestoreState(); err != nil {
				return codexAppRestoreFailure(configPath, err)
			}
			return codexAppLaunchOrRestart("Restart Codex to use your usual profile?")
		}
		return codexAppRestoreFailure(configPath, err)
	}
	text := string(data)
	if err := codexValidateConfigText(text); err != nil {
		return codexAppRestoreFailure(configPath, err)
	}

	state, stateErr := loadCodexAppRestoreState()
	if stateErr == nil {
		text = codexAppRestoreRootValues(text, state)
	} else if os.IsNotExist(stateErr) {
		text = codexAppRemoveOwnedRootValues(text)
	} else {
		return codexAppRestoreFailure(configPath, stateErr)
	}
	if !codexAppRootReferencesOwnedConfig(text) {
		text = codexAppRemoveOwnedSections(text)
	}

	if err := codexValidateConfigText(text); err != nil {
		return codexAppRestoreFailure(configPath, err)
	}
	if err := fileutil.WriteWithBackup(configPath, []byte(text), codexAppIntegrationName); err != nil {
		return codexAppRestoreFailure(configPath, err)
	}
	if err := codexAppRemoveOwnedCatalogIfUnused(text); err != nil {
		return codexAppRestoreFailure(configPath, err)
	}
	if err := removeCodexAppRestoreState(); err != nil {
		return codexAppRestoreFailure(configPath, err)
	}
	return codexAppLaunchOrRestart("Restart Codex to use your usual profile?")
}

func codexAppRestoreFailure(configPath string, err error) error {
	return fmt.Errorf("restore Codex App config: %w\n\nRestore did not complete. Check these files before retrying:\n  Codex config: %s\n  Restore state: %s\n  Model catalog: %s\n  Backups: %s",
		err,
		configPath,
		codexAppRestoreStatePath(),
		codexAppModelCatalogPathForConfig(configPath),
		filepath.Join(fileutil.BackupDir(), codexAppIntegrationName),
	)
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
	switch codexAppGOOS {
	case "darwin":
		return codexAppCanOpenID()
	case "windows":
		return codexAppIsRunning() || codexAppStartID() != ""
	default:
		return false
	}
}

func codexAppModelCatalogPath() (string, error) {
	configPath, err := codexConfigPath()
	if err != nil {
		return "", err
	}
	return codexAppModelCatalogPathForConfig(configPath), nil
}

func codexAppModelCatalogPathForConfig(configPath string) string {
	return filepath.Join(filepath.Dir(configPath), codexAppModelCatalogFilename)
}

func writeCodexAppModelCatalog(path, primary string, models []string) error {
	if len(models) == 0 {
		return fmt.Errorf("codex-app model catalog cannot be empty")
	}
	client := api.NewClient(envconfig.ConnectableHost(), http.DefaultClient)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	baseInstructions := codexAppBaseInstructions()
	primaryMetadata := codexAppSelectedModelMetadata(ctx, client, primary)
	entries := make([]map[string]any, 0, len(models))
	for i, model := range models {
		metadata := codexAppDefaultModelMetadata()
		if model == primary {
			metadata = primaryMetadata
		}
		entries = append(entries, codexAppCatalogEntry(model, metadata, i, baseInstructions))
	}

	data, err := json.MarshalIndent(map[string]any{"models": entries}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return fileutil.WriteWithBackup(path, append(data, '\n'), codexAppIntegrationName)
}

func codexAppCatalogModelNames(primary string, fallback []string) []string {
	models := codexAppTagModelNames()
	if len(models) == 0 {
		models = fallback
	}
	return dedupeModelList(append([]string{primary}, models...))
}

func codexAppTagModelNames() []string {
	client := api.NewClient(envconfig.ConnectableHost(), http.DefaultClient)
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

type codexAppModelMetadata struct {
	contextWindow   int
	inputModalities []string
}

func codexAppDefaultModelMetadata() codexAppModelMetadata {
	return codexAppModelMetadata{
		contextWindow:   128_000,
		inputModalities: []string{"text"},
	}
}

func codexAppSelectedModelMetadata(ctx context.Context, client *api.Client, model string) codexAppModelMetadata {
	metadata := codexAppDefaultModelMetadata()
	resp, err := client.Show(ctx, &api.ShowRequest{Model: model})
	if err != nil {
		return metadata
	}
	if n, ok := modelInfoContextLength(resp.ModelInfo); ok {
		metadata.contextWindow = n
	}
	if slices.Contains(resp.Capabilities, modelpkg.CapabilityVision) {
		metadata.inputModalities = []string{"text", "image"}
	}
	return metadata
}

func codexAppCatalogEntry(model string, metadata codexAppModelMetadata, priority int, baseInstructions string) map[string]any {
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
		"context_window":                   metadata.contextWindow,
		"max_context_window":               metadata.contextWindow,
		"auto_compact_token_limit":         nil,
		"effective_context_window_percent": 95,
		"experimental_supported_tools":     []any{},
		"input_modalities":                 metadata.inputModalities,
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
	gracefulErr := waitForCodexAppGracefulExit(codexAppExitTimeout)
	if gracefulErr != nil && !codexAppForceQuitSupported() {
		return gracefulErr
	}
	if codexAppForceQuitSupported() && codexAppIsRunning() {
		if forceErr := codexAppForceQuit(); forceErr != nil {
			return fmt.Errorf("force stop Codex: %w", forceErr)
		}
		if err := waitForCodexAppExit(codexAppForceExitTimeout); err != nil {
			return err
		}
	} else if gracefulErr != nil {
		if codexAppIsRunning() {
			return gracefulErr
		}
	}
	if restartAppID != "" {
		return codexAppOpenStart(restartAppID)
	}
	if restartAppPath != "" {
		return codexAppOpenPath(restartAppPath)
	}
	return codexAppOpenApp()
}

func codexAppForceQuitSupported() bool {
	return codexAppGOOS == "darwin" || codexAppGOOS == "windows"
}

func waitForCodexAppGracefulExit(timeout time.Duration) error {
	return waitForCodexAppCondition(timeout, func() bool {
		if codexAppGOOS == "windows" {
			return !codexAppHasWindow()
		}
		return !codexAppIsRunning()
	})
}

func waitForCodexAppExit(timeout time.Duration) error {
	return waitForCodexAppCondition(timeout, func() bool {
		return !codexAppIsRunning()
	})
}

func waitForCodexAppCondition(timeout time.Duration, done func() bool) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if done() {
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
		return exec.Command("powershell.exe", "-NoProfile", "-Command", script).Run()
	}

	scriptErr := exec.Command("osascript", "-e", `tell application "Codex" to quit`).Run()
	if scriptErr != nil {
		scriptErr = exec.Command("osascript", "-e", `tell application id "`+codexAppBundleID+`" to quit`).Run()
	}
	return scriptErr
}

func defaultCodexAppForceQuitApp() error {
	if !codexAppForceQuitSupported() {
		return nil
	}
	pids := codexAppMatchingProcessIDs()
	if len(pids) == 0 {
		return nil
	}
	pidArgs := make([]string, 0, len(pids))
	for _, pid := range pids {
		pidArgs = append(pidArgs, strconv.Itoa(pid))
	}
	switch codexAppGOOS {
	case "windows":
		script := "Stop-Process -Id " + strings.Join(pidArgs, ",") + " -Force -ErrorAction SilentlyContinue"
		return runCodexAppForceQuitCommand(exec.Command("powershell.exe", "-NoProfile", "-Command", script))
	case "darwin":
		return runCodexAppForceQuitCommand(exec.Command("kill", append([]string{"-TERM"}, pidArgs...)...))
	default:
		return nil
	}
}

func runCodexAppForceQuitCommand(cmd *exec.Cmd) error {
	err := cmd.Run()
	if err != nil && !codexAppIsRunning() {
		return nil
	}
	return err
}

func defaultCodexAppHasOpenWindow() bool {
	if codexAppGOOS != "windows" {
		return codexAppIsRunning()
	}
	script := `(Get-Process Codex -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1).Id`
	out, err := exec.Command("powershell.exe", "-NoProfile", "-Command", script).Output()
	return err == nil && strings.TrimSpace(string(out)) != ""
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

func defaultCodexAppCanOpenBundleID() bool {
	if codexAppGOOS != "darwin" {
		return false
	}
	query := fmt.Sprintf("kMDItemCFBundleIdentifier == %q", codexAppBundleID)
	out, err := exec.Command("mdfind", query).Output()
	return err == nil && strings.TrimSpace(string(out)) != ""
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
	config, err := codexParseConfig(text)
	if err != nil {
		return false
	}
	return codexAppIsOwnedProfileName(config.RootString(codexRootProfileKey)) ||
		codexAppIsOwnedProfileName(config.RootString(codexRootModelProviderKey))
}

func codexAppRootReferencesOwnedConfig(text string) bool {
	config, err := codexParseConfig(text)
	if err != nil {
		return false
	}
	return config.RootString(codexRootProfileKey) == codexAppProfileName ||
		config.RootString(codexRootModelProviderKey) == codexAppProfileName
}

func codexAppRootReferencesCatalog(text string) bool {
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		return false
	}
	config, err := codexParseConfig(text)
	if err != nil {
		return false
	}
	return config.RootString(codexRootModelCatalogJSONKey) == catalogPath
}

func codexAppRemoveOwnedSections(text string) string {
	text = codexRemoveSection(text, codexProfileHeaderFor(codexAppProfileName))
	text = codexRemoveSection(text, codexProviderHeaderFor(codexAppProfileName))
	return text
}

func codexAppRemoveOwnedCatalogIfUnused(text string) error {
	if codexAppRootReferencesCatalog(text) {
		return nil
	}
	if catalogPath, err := codexAppModelCatalogPath(); err == nil {
		if err := os.Remove(catalogPath); err != nil && !os.IsNotExist(err) {
			return err
		}
	} else {
		return err
	}
	return nil
}

func codexAppRemoveOwnedRootValues(text string) string {
	config, err := codexParseConfig(text)
	if err != nil {
		return text
	}
	modelProvider := config.RootString(codexRootModelProviderKey)
	modelCatalogJSON := config.RootString(codexRootModelCatalogJSONKey)
	if !codexAppIsOwnedProfileName(config.RootString(codexRootProfileKey)) && !codexAppIsOwnedProfileName(modelProvider) {
		return text
	}
	text = codexRemoveRootValue(text, codexRootProfileKey)
	text = codexRemoveRootValue(text, codexRootModelKey)
	if codexAppIsOwnedProfileName(modelProvider) {
		text = codexRemoveRootValue(text, codexRootModelProviderKey)
	}
	if catalogPath, err := codexAppModelCatalogPath(); err == nil && modelCatalogJSON == catalogPath {
		text = codexRemoveRootValue(text, codexRootModelCatalogJSONKey)
	}
	return text
}

func codexAppRestoreRootValues(text string, state codexAppRestoreState) string {
	if !codexAppRootStillManaged(text) {
		return text
	}
	text = codexRestoreRootStringValue(text, codexRootProfileKey, state.HadProfile, state.Profile)
	text = codexRestoreRootStringValue(text, codexRootModelKey, state.HadModel, state.Model)
	text = codexRestoreRootStringValue(text, codexRootModelProviderKey, state.HadModelProvider, state.ModelProvider)
	text = codexRestoreRootStringValue(text, codexRootModelCatalogJSONKey, state.HadModelCatalogJSON, state.ModelCatalogJSON)
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

	if !configExists {
		return writeCodexAppRestoreState(codexAppRestoreState{})
	}

	statePath := codexAppRestoreStatePath()
	if stateData, err := os.ReadFile(statePath); err == nil {
		hasRootConfig, err := codexAppRestoreStateHasRootConfig(stateData)
		if err != nil {
			return err
		}
		if hasRootConfig {
			if configExists && !codexAppRootStillManaged(configText) {
				return writeCodexAppRestoreState(codexAppRestoreStateFromText(configText))
			}
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
	config, err := codexParseConfig(text)
	if err != nil {
		return codexAppRestoreState{}
	}
	profile, hadProfile := config.RootStringOK(codexRootProfileKey)
	model, hadModel := config.RootStringOK(codexRootModelKey)
	modelProvider, hadModelProvider := config.RootStringOK(codexRootModelProviderKey)
	modelCatalogJSON, hadModelCatalogJSON := config.RootStringOK(codexRootModelCatalogJSONKey)
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
	return fileutil.WriteWithBackup(path, data, codexAppIntegrationName)
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

func removeCodexAppRestoreState() error {
	if err := os.Remove(codexAppRestoreStatePath()); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func codexAppRestoreStatePath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "ollama-codex-app-restore.json")
	}
	return filepath.Join(home, ".ollama", "launch", "codex-app-restore.json")
}
