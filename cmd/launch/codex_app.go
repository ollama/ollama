package launch

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
)

const (
	chatGPTIntegrationName       = "chatgpt"
	codexAppIntegrationName      = "codex-app"
	codexAppProfileName          = "ollama-launch-codex-app"
	codexAppBundleID             = "com.openai.codex"
	codexAppModelCatalogFilename = "ollama-launch-models.json"
	codexAppRestoreHint          = "To restore your usual ChatGPT profile, run: ollama launch chatgpt --restore"
	codexAppConfigurationSuccess = "ChatGPT profile changed to Ollama."
	codexAppRestoreSuccess       = "ChatGPT restored to your usual profile."
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

func (c *CodexApp) String() string { return "ChatGPT" }

func (c *CodexApp) Supported() error { return codexAppSupported() }

func (c *CodexApp) Paths() []string {
	configPath, err := codexConfigPath()
	if err != nil {
		return nil
	}
	return []string{configPath}
}

func (c *CodexApp) Configure(model string) error {
	return c.ConfigureWithModels(model, launchModelsFromNames([]string{model}))
}

func (c *CodexApp) ConfigureWithModels(primary string, models []LaunchModel) error {
	primary = strings.TrimSpace(primary)
	if primary == "" {
		return fmt.Errorf("chatgpt requires a model")
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
	if err := writeCodexAppModelCatalog(catalogPath, primary, codexAppCatalogModels(primary, models)); err != nil {
		return err
	}
	return writeCodexAppConfig(configPath, primary, catalogPath)
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
				model := strings.TrimSpace(parsed.RootString(codexRootModelKey))
				if codexAppCatalogContainsModel(model) {
					return model
				}
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
	model := strings.TrimSpace(parsed.ProfileString(profileName, codexRootModelKey))
	if !codexAppCatalogContainsModel(model) {
		return ""
	}
	return model
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
	if config.Exists("profiles", profileName) && config.ProfileString(profileName, codexRootModelCatalogJSONKey) != catalogPath {
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

// codexAppCatalogContainsModel reports whether model appears as a slug in the
// Ollama-managed model catalog. When the configured model is not in the catalog
// the user has drifted away from the launch-managed model (e.g. by selecting a
// built-in OpenAI model in the Codex App UI), and the launch config should be
// treated as inactive.
func codexAppCatalogContainsModel(model string) bool {
	if strings.TrimSpace(model) == "" {
		return false
	}
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		return false
	}
	data, err := os.ReadFile(catalogPath)
	if err != nil {
		return false
	}
	var catalog struct {
		Models []struct {
			Slug string `json:"slug"`
		} `json:"models"`
	}
	if err := json.Unmarshal(data, &catalog); err != nil {
		return false
	}
	target := codexAppCatalogModelKey(model)
	for _, m := range catalog.Models {
		if codexAppCatalogModelKey(m.Slug) == target {
			return true
		}
	}
	return false
}

func writeCodexAppConfig(configPath, model, modelCatalogPath string) error {
	baseURL := codexBaseURL()

	content, readErr := os.ReadFile(configPath)
	text := ""
	if readErr == nil {
		text = string(content)
	} else if !os.IsNotExist(readErr) {
		return readErr
	}
	if _, err := codexParseConfig(text); err != nil {
		return err
	}

	text = codexRemoveRootValue(text, codexRootProfileKey)
	text = codexRemoveSection(text, codexProfileHeaderFor(codexAppProfileName))
	text = codexSetRootStringValue(text, codexRootModelKey, model)
	text = codexSetRootStringValue(text, codexRootModelProviderKey, codexAppProfileName)
	text = codexSetRootStringValue(text, codexRootModelCatalogJSONKey, modelCatalogPath)
	text = codexUpsertSection(text, codexProviderHeaderFor(codexAppProfileName), []string{
		fmt.Sprintf("name = %q", codexProviderName),
		fmt.Sprintf("base_url = %q", baseURL),
		`wire_api = "responses"`,
	})

	parsed, err := codexParseConfig(text)
	if err != nil {
		return err
	}
	if err := codexValidateAppConfigText(parsed, model, modelCatalogPath, baseURL); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}
	return fileutil.WriteWithBackup(configPath, []byte(text), codexAppIntegrationName)
}

func codexValidateAppConfigText(config codexParsedConfig, model, modelCatalogPath, baseURL string) error {
	if got, ok := config.RootStringOK(codexRootProfileKey); ok {
		return fmt.Errorf("generated ChatGPT config still contains legacy profile = %q", got)
	}
	if config.Exists("profiles", codexAppProfileName) {
		return fmt.Errorf("generated ChatGPT config still contains legacy profiles.%s table", codexAppProfileName)
	}
	for _, check := range []struct {
		path []string
		want string
	}{
		{[]string{codexRootModelKey}, model},
		{[]string{codexRootModelProviderKey}, codexAppProfileName},
		{[]string{codexRootModelCatalogJSONKey}, modelCatalogPath},
		{[]string{"model_providers", codexAppProfileName, "name"}, codexProviderName},
		{[]string{"model_providers", codexAppProfileName, "base_url"}, baseURL},
		{[]string{"model_providers", codexAppProfileName, "wire_api"}, "responses"},
	} {
		if got, ok := config.String(check.path...); !ok || got != check.want {
			return fmt.Errorf("generated ChatGPT config missing %s = %q", strings.Join(check.path, "."), check.want)
		}
	}
	return nil
}

func (c *CodexApp) Onboard() error {
	return config.MarkIntegrationOnboarded(chatGPTIntegrationName)
}

func (c *CodexApp) RequiresInteractiveOnboarding() bool {
	return false
}

func (c *CodexApp) EnrichModelMetadataFromShow() bool {
	return true
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

func (c *CodexApp) Run(_ string, _ []LaunchModel, args []string) error {
	if err := codexAppSupported(); err != nil {
		return err
	}
	if len(args) > 0 {
		return fmt.Errorf("chatgpt does not accept extra arguments")
	}
	return codexAppLaunchOrRestart("Restart ChatGPT to use Ollama?", nil)
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
			if err := removeCodexAppProfileConfig(); err != nil {
				return codexAppRestoreFailure(configPath, err)
			}
			if err := codexAppRemoveOwnedCatalog(); err != nil {
				return codexAppRestoreFailure(configPath, err)
			}
			return codexAppLaunchOrRestart("Restart ChatGPT to use your usual profile?", nil)
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
	if err := removeCodexAppProfileConfig(); err != nil {
		return codexAppRestoreFailure(configPath, err)
	}
	if err := codexAppRemoveOwnedCatalogIfUnused(text); err != nil {
		return codexAppRestoreFailure(configPath, err)
	}
	if err := removeCodexAppRestoreState(); err != nil {
		return codexAppRestoreFailure(configPath, err)
	}
	return codexAppLaunchOrRestart("Restart ChatGPT to use your usual profile?", nil)
}

func codexAppRestoreFailure(configPath string, err error) error {
	return fmt.Errorf("restore ChatGPT config: %w\n\nRestore did not complete. Check these files before retrying:\n  Codex config: %s\n  Restore state: %s\n  Model catalog: %s\n  Backups: %s",
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
		return fmt.Errorf("ChatGPT launch is only supported on macOS and Windows")
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

func codexAppProfileConfigPath() (string, error) {
	configPath, err := codexConfigPath()
	if err != nil {
		return "", err
	}
	return codexAppProfileConfigPathForConfig(configPath), nil
}

func codexAppProfileConfigPathForConfig(configPath string) string {
	return codexNamedProfileConfigPathForConfig(configPath, codexAppProfileName)
}

func codexAppModelCatalogPathForConfig(configPath string) string {
	return filepath.Join(filepath.Dir(configPath), codexAppModelCatalogFilename)
}

func writeCodexAppModelCatalog(path, primary string, models []LaunchModel) error {
	if len(models) == 0 {
		return fmt.Errorf("chatgpt model catalog cannot be empty")
	}

	baseInstructions := codexAppBaseInstructions()
	entries := make([]map[string]any, 0, len(models))
	for i, model := range models {
		entries = append(entries, codexAppCatalogEntry(model.Name, codexAppModelMetadataFromLaunchModel(model), i, baseInstructions))
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

func codexAppCatalogModels(primary string, models []LaunchModel) []LaunchModel {
	seen := make(map[string]bool, len(models)+1)
	out := make([]LaunchModel, 0, len(models)+1)
	add := func(model LaunchModel) {
		model.Name = strings.TrimSpace(model.Name)
		if model.Name == "" {
			return
		}
		key := codexAppCatalogModelKey(model.Name)
		if seen[key] {
			return
		}
		seen[key] = true
		out = append(out, model)
	}

	if model, ok := findLaunchModel(models, primary); ok {
		model.Name = primary
		add(model)
	} else {
		add(fallbackLaunchModel(primary))
	}
	for _, model := range models {
		add(model)
	}
	return out
}

func codexAppCatalogModelKey(name string) string {
	return strings.TrimSuffix(name, ":latest")
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

func codexAppModelMetadataFromLaunchModel(model LaunchModel) codexAppModelMetadata {
	metadata := codexAppDefaultModelMetadata()
	if numCtx, ok := modelfileNumCtx(model.Parameters); ok {
		metadata.contextWindow = numCtx
	} else if model.ContextLength > 0 {
		metadata.contextWindow = model.ContextLength
	}
	if model.HasCapability("vision") {
		metadata.inputModalities = []string{"text", "image"}
	}
	return metadata
}

func modelfileNumCtx(parameters string) (int, bool) {
	for _, line := range strings.Split(parameters, "\n") {
		fields := strings.Fields(strings.TrimSpace(line))
		if len(fields) < 2 || strings.ToLower(fields[0]) != "num_ctx" {
			continue
		}
		value := strings.Trim(fields[1], `"'`)
		numCtx, err := strconv.Atoi(value)
		if err == nil && numCtx > 0 {
			return numCtx, true
		}
	}
	return 0, false
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
	candidates := []string{"/Applications/ChatGPT.app", "/Applications/Codex.app"}
	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates,
			filepath.Join(home, "Applications", "ChatGPT.app"),
			filepath.Join(home, "Applications", "Codex.app"),
		)
	}
	return candidates
}

func codexAppWindowsAppCandidates() []string {
	local, err := codexAppLocalAppData()
	if err != nil {
		return nil
	}

	candidates := []string{
		filepath.Join(local, "Programs", "ChatGPT", "ChatGPT.exe"),
		filepath.Join(local, "Programs", "OpenAI ChatGPT", "ChatGPT.exe"),
		filepath.Join(local, "ChatGPT", "ChatGPT.exe"),
		filepath.Join(local, "OpenAI ChatGPT", "ChatGPT.exe"),
		filepath.Join(local, "OpenAI", "ChatGPT", "ChatGPT.exe"),
		filepath.Join(local, "Programs", "Codex", "Codex.exe"),
		filepath.Join(local, "Programs", "OpenAI Codex", "Codex.exe"),
		filepath.Join(local, "Codex", "Codex.exe"),
		filepath.Join(local, "OpenAI Codex", "Codex.exe"),
		filepath.Join(local, "OpenAI", "Codex", "Codex.exe"),
		filepath.Join(local, "openai-codex-electron", "Codex.exe"),
	}
	for _, pattern := range []string{
		filepath.Join(local, "Programs", "ChatGPT", "app-*", "ChatGPT.exe"),
		filepath.Join(local, "Programs", "OpenAI ChatGPT", "app-*", "ChatGPT.exe"),
		filepath.Join(local, "ChatGPT", "app-*", "ChatGPT.exe"),
		filepath.Join(local, "OpenAI ChatGPT", "app-*", "ChatGPT.exe"),
		filepath.Join(local, "OpenAI", "ChatGPT", "app-*", "ChatGPT.exe"),
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

func codexAppLaunchOrRestart(prompt string, launchArgs []string) error {
	if !codexAppIsRunning() {
		return codexAppOpenApp(launchArgs)
	}
	restartAppID := ""
	restartAppPath := ""
	if len(launchArgs) == 0 && codexAppGOOS == "windows" {
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
		fmt.Fprintln(os.Stderr, "\nQuit and reopen ChatGPT when you're ready for the profile change to take effect.")
		return nil
	}

	// A single spinner and cancellation channel span the entire restart flow
	// (quit, wait, force-quit, wait, reopen) so that one Ctrl+C aborts the
	// whole sequence rather than just the currently-active wait. The bubbletea
	// spinner closes Cancelled() from its raw-mode Ctrl+C handler; the ANSI
	// fallback relies on SIGINT terminating the process directly.
	sp := StartSpinner(codexAppRestartMessage)
	defer sp.Stop()
	cancelled := sp.Cancelled()
	isCancelled := func() bool {
		if cancelled == nil {
			return false
		}
		select {
		case <-cancelled:
			return true
		default:
			return false
		}
	}

	if err := codexAppQuitApp(); err != nil {
		return fmt.Errorf("quit ChatGPT: %w", err)
	}
	if isCancelled() {
		return ErrCancelled
	}
	gracefulErr := waitForCodexAppGracefulExit(codexAppExitTimeout, cancelled)
	if isCancelled() {
		return ErrCancelled
	}
	if errors.Is(gracefulErr, ErrCancelled) {
		return gracefulErr
	}
	if gracefulErr != nil && !codexAppForceQuitSupported() {
		return gracefulErr
	}
	if codexAppForceQuitSupported() && codexAppIsRunning() {
		if isCancelled() {
			return ErrCancelled
		}
		if forceErr := codexAppForceQuit(); forceErr != nil {
			return fmt.Errorf("force stop ChatGPT: %w", forceErr)
		}
		if err := waitForCodexAppExit(codexAppForceExitTimeout, cancelled); err != nil {
			return err
		}
	} else if gracefulErr != nil {
		if codexAppIsRunning() {
			return gracefulErr
		}
	}
	if isCancelled() {
		return ErrCancelled
	}
	sp.Stop()
	if restartAppID != "" {
		return codexAppOpenStart(restartAppID)
	}
	if restartAppPath != "" {
		return codexAppOpenPath(restartAppPath)
	}
	return codexAppOpenApp(launchArgs)
}

func codexAppForceQuitSupported() bool {
	return codexAppGOOS == "darwin" || codexAppGOOS == "windows"
}

func waitForCodexAppGracefulExit(timeout time.Duration, cancel <-chan struct{}) error {
	return waitForCodexAppCondition(timeout, cancel, func() bool {
		if codexAppGOOS == "windows" {
			return !codexAppHasWindow()
		}
		return !codexAppIsRunning()
	})
}

func waitForCodexAppExit(timeout time.Duration, cancel <-chan struct{}) error {
	return waitForCodexAppCondition(timeout, cancel, func() bool {
		return !codexAppIsRunning()
	})
}

// codexAppRestartMessage is the label shown next to the animated spinner while
// the ChatGPT desktop app is quitting before being reopened.
const codexAppRestartMessage = "Restarting ChatGPT..."

// waitForCodexAppCondition polls done at a 200ms cadence until it reports the
// app has exited or timeout elapses. It watches cancel (closed by the spinner
// when the user hits Ctrl+C) and returns ErrCancelled if the flow is aborted.
// The spinner itself is owned by the caller so a single spinner spans the
// whole restart sequence. When timeout is zero the loop never runs, so
// force-quit paths that short-circuit the graceful wait return immediately.
func waitForCodexAppCondition(timeout time.Duration, cancel <-chan struct{}, done func() bool) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cancel != nil {
			select {
			case <-cancel:
				return ErrCancelled
			default:
			}
		}
		if done() {
			return nil
		}
		codexAppSleep(200 * time.Millisecond)
	}
	if done() {
		return nil
	}
	return fmt.Errorf("ChatGPT did not quit; quit it manually and re-run the command")
}

func defaultCodexAppOpenApp(args []string) error {
	if len(args) > 0 {
		cmd := exec.Command("codex", args...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cmd.Env = append(os.Environ(), "OPENAI_API_KEY=ollama")
		return cmd.Run()
	}

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
		return fmt.Errorf("ChatGPT was not found; install it from https://chatgpt.com/download, then re-run 'ollama launch chatgpt'")
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
		script := `Get-Process ChatGPT,Codex -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | ForEach-Object { [void]$_.CloseMainWindow() }`
		return exec.Command("powershell.exe", "-NoProfile", "-Command", script).Run()
	}

	scriptErr := exec.Command("osascript", "-e", `tell application "ChatGPT" to quit`).Run()
	if scriptErr != nil {
		scriptErr = exec.Command("osascript", "-e", `tell application id "`+codexAppBundleID+`" to quit`).Run()
	}
	if scriptErr != nil {
		scriptErr = exec.Command("osascript", "-e", `tell application "Codex" to quit`).Run()
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
	script := `(Get-Process ChatGPT,Codex -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1).Id`
	out, err := exec.Command("powershell.exe", "-NoProfile", "-Command", script).Output()
	return err == nil && strings.TrimSpace(string(out)) != ""
}

func defaultCodexAppIsRunning() bool {
	switch codexAppGOOS {
	case "windows":
		return len(codexAppMatchingProcessIDs()) > 0
	case "darwin":
		out, err := exec.Command("osascript", "-e", `tell application "System Events" to exists process "ChatGPT"`).Output()
		if err == nil && strings.TrimSpace(string(out)) == "true" {
			return true
		}
		out, err = exec.Command("osascript", "-e", `tell application "System Events" to exists process "Codex"`).Output()
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
	script := fmt.Sprintf(`$current = %d; Get-CimInstance Win32_Process -Filter "Name = 'Codex.exe' OR Name = 'codex.exe' OR Name = 'ChatGPT.exe' OR Name = 'chatgpt.exe'" | Where-Object { $_.ProcessId -ne $current -and ((($_.Name -ieq 'Codex.exe' -or $_.Name -ieq 'ChatGPT.exe') -and (($null -eq $_.CommandLine) -or ($_.CommandLine -notlike '* --type=*'))) -or ((($_.Name -ieq 'codex.exe') -or ($_.Name -ieq 'chatgpt.exe')) -and ($_.CommandLine -like '*app-server*'))) } | Select-Object -ExpandProperty ProcessId`, os.Getpid())
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
	script := `(Get-Process ChatGPT,Codex -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 -and $_.Path } | Select-Object -First 1 -ExpandProperty Path)`
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
	script := `(Get-StartApps | Where-Object { $_.Name -eq 'ChatGPT' -or $_.Name -like 'ChatGPT*' -or $_.Name -eq 'Codex' -or $_.Name -like 'Codex*' } | Select-Object -First 1 -ExpandProperty AppID)`
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
	if (strings.Contains(command, `\Codex.exe`) || strings.Contains(command, `\ChatGPT.exe`)) && strings.Contains(command, " --type=") {
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
		"ChatGPT.app/Contents/MacOS/ChatGPT",
		"ChatGPT.app/Contents/Resources/codex app-server",
		"Codex.app/Contents/MacOS/Codex",
		"Codex.app/Contents/Resources/codex app-server",
		`\ChatGPT.exe`,
		`resources\chatgpt.exe app-server`,
		`resources\chatgpt.exe" app-server`,
		`resources\chatgpt.exe" "app-server`,
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
	return codexAppRemoveOwnedCatalog()
}

func codexAppRemoveOwnedCatalog() error {
	if catalogPath, err := codexAppModelCatalogPath(); err == nil {
		if err := os.Remove(catalogPath); err != nil && !os.IsNotExist(err) {
			return err
		}
	} else {
		return err
	}
	return nil
}

func removeCodexAppProfileConfig() error {
	profilePath, err := codexAppProfileConfigPath()
	if err != nil {
		return err
	}
	if err := os.Remove(profilePath); err != nil && !os.IsNotExist(err) {
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
		if codexAppRootStillManaged(configText) {
			// Legacy restore state did not record root model settings. If the
			// current config is still ours, do not save our generated root
			// values as the user's restore target.
			upgraded = codexAppRestoreState{}
		}
		upgraded.HadProfile = existing.HadProfile
		upgraded.Profile = existing.Profile
		return writeCodexAppRestoreState(upgraded)
	} else if !os.IsNotExist(err) {
		return err
	}

	state := codexAppRestoreStateFromText(configText)
	if codexAppRootStillManaged(configText) {
		state = codexAppRestoreState{}
	}
	return writeCodexAppRestoreState(state)
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
