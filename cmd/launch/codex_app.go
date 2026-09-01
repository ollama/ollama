package launch

import (
	"context"
	"encoding/json"
	"errors"
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

	"github.com/ollama/ollama/app/codexproxy"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
	modelpkg "github.com/ollama/ollama/types/model"
)

const (
	chatGPTIntegrationName         = "chatgpt"
	codexAppIntegrationName        = "codex-app"
	codexAppProfileName            = "ollama-launch-codex-app"
	codexAppBundleID               = "com.openai.codex"
	codexAppModelCatalogFilename   = codexproxy.ModelCatalogFilename
	codexAppRoutingCatalogFilename = codexproxy.RoutingCatalogFilename
	codexAppOllamaProfileDirName   = "chatgpt-ollama"
	codexAppOllamaUserDataName     = "electron-data"
	codexAppOllamaPIDFilename      = "chatgpt.pid"
	codexAppSingletonLockName      = "SingletonLock"
	codexAppSingletonSocketName    = "SingletonSocket"
	codexAppSingletonCookieName    = "SingletonCookie"
	codexAppRestoreHint            = "To remove Ollama models from ChatGPT, run: ollama launch chatgpt --restore"
	codexAppConfigurationSuccess   = "Ollama models added to ChatGPT."
	codexAppRestoreSuccess         = "Ollama models removed from ChatGPT."
)

var (
	codexAppGOOS            = runtime.GOOS
	codexAppStat            = os.Stat
	codexAppGlob            = filepath.Glob
	codexAppOpenApp         = defaultCodexAppOpenApp
	codexAppOpenPath        = defaultCodexAppOpenAppPath
	codexAppOpenStart       = defaultCodexAppOpenStartAppID
	codexAppQuitApp         = defaultCodexAppQuitApp
	codexAppForceQuit       = defaultCodexAppForceQuitApp
	codexAppHasWindow       = defaultCodexAppHasOpenWindow
	codexAppIsRunning       = defaultCodexAppIsRunning
	codexAppRunPath         = defaultCodexAppRunningAppPath
	codexAppStartID         = defaultCodexAppStartAppID
	codexAppCanOpenID       = defaultCodexAppCanOpenBundleID
	codexAppSleep           = time.Sleep
	codexAppNativeCatalog   = defaultCodexAppNativeModelCatalog
	codexAppCodexExecutable = defaultCodexAppCodexExecutable
	codexAppRunDebugModels  = defaultCodexAppRunDebugModels
	codexAppRouterHealth    = defaultCodexAppRouterHealth

	codexAppProfileApplication = defaultCodexAppOllamaProfileApplication
	codexAppProfileExecutable  = defaultCodexAppOllamaProfileExecutable
	codexAppStopProfile        = defaultCodexAppStopOllamaProfile
	codexAppProfileIsRunning   = defaultCodexAppOllamaProfileIsRunning
	codexAppProcessCommand     = defaultCodexAppProcessCommand

	codexAppExitTimeout      = 5 * time.Second
	codexAppForceExitTimeout = 5 * time.Second
)

// CodexApp keeps ChatGPT's built-in OpenAI provider and adds selected Ollama
// models to its catalog. A loopback router chooses the upstream per request.
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
	nativeCatalog, err := codexAppNativeCatalog(configPath)
	if err != nil {
		return fmt.Errorf("read native Codex model catalog: %w", err)
	}
	models = codexAppCatalogModels(primary, models)
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		return err
	}
	routingCatalogPath := codexAppRoutingCatalogPathForConfig(configPath)
	if err := writeCodexAppRoutingCatalog(routingCatalogPath, models); err != nil {
		return err
	}
	if err := writeCodexAppCombinedModelCatalog(catalogPath, models, nativeCatalog); err != nil {
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
	if codexAppRootUsesProxy(parsed) && codexAppCatalogHealthy(parsed, "") {
		model := strings.TrimSpace(parsed.RootString(codexRootModelKey))
		if codexAppCatalogContainsModel(model) {
			return model
		}
		return codexAppFirstRoutingModel()
	}

	// Recognize older custom-provider layouts so an existing prototype can be
	// migrated to the built-in OpenAI provider without losing restore state.
	for _, profileName := range codexAppManagedProfileNames() {
		if parsed.RootString(codexRootModelProviderKey) == profileName {
			baseURL := parsed.ProviderString(profileName, "base_url")
			if codexAppManagedProviderURL(profileName, baseURL) && codexAppCatalogHealthy(parsed, profileName) {
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

func codexAppManagedProviderURL(profileName, baseURL string) bool {
	if profileName == codexAppProfileName &&
		codexNormalizeURL(baseURL) == codexNormalizeURL(codexAppProxyBaseURL()) {
		return true
	}
	return codexNormalizeURL(baseURL) == codexNormalizeURL(codexBaseURL())
}

func codexAppCatalogHealthy(config codexParsedConfig, profileName string) bool {
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		return false
	}
	if config.RootString(codexRootModelCatalogJSONKey) != catalogPath {
		return false
	}
	if profileName != "" && config.Exists("profiles", profileName) && config.ProfileString(profileName, codexRootModelCatalogJSONKey) != catalogPath {
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

// codexAppCatalogContainsModel reports whether model appears in the Ollama-only
// routing catalog. Native ChatGPT models deliberately do not appear there.
func codexAppCatalogContainsModel(model string) bool {
	if strings.TrimSpace(model) == "" {
		return false
	}
	models, err := codexAppRoutingModels()
	if err != nil {
		return false
	}
	target := codexAppCatalogModelKey(model)
	for _, candidate := range models {
		if codexAppCatalogModelKey(candidate) == target {
			return true
		}
	}
	return false
}

func codexAppRoutingModels() ([]string, error) {
	configPath, err := codexConfigPath()
	if err != nil {
		return nil, err
	}
	catalogPath := codexAppRoutingCatalogPathForConfig(configPath)
	data, err := os.ReadFile(catalogPath)
	if err != nil {
		return nil, err
	}
	var catalog struct {
		Models []struct {
			Slug string `json:"slug"`
		} `json:"models"`
	}
	if err := json.Unmarshal(data, &catalog); err != nil {
		return nil, err
	}
	models := make([]string, 0, len(catalog.Models))
	for _, m := range catalog.Models {
		if slug := strings.TrimSpace(m.Slug); slug != "" {
			models = append(models, slug)
		}
	}
	if len(models) == 0 {
		return nil, errors.New("ChatGPT Ollama routing catalog is empty")
	}
	return models, nil
}

func codexAppFirstRoutingModel() string {
	models, err := codexAppRoutingModels()
	if err != nil {
		return ""
	}
	return models[0]
}

func writeCodexAppConfig(configPath, model, modelCatalogPath string) error {
	baseURL := codexAppProxyBaseURL()

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
	text = codexAppRemoveOwnedSections(text)
	text = codexSetRootStringValue(text, codexRootModelKey, model)
	// Keep the built-in provider identity so native and Ollama tasks remain in
	// one ChatGPT task list. The loopback base URL routes each request by model.
	text = codexSetRootStringValue(text, codexRootModelProviderKey, "openai")
	text = codexSetRootStringValue(text, codexRootModelCatalogJSONKey, modelCatalogPath)
	text = codexSetRootStringValue(text, codexRootOpenAIBaseURLKey, baseURL)

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
	if config.Exists("model_providers", codexAppProfileName) {
		return fmt.Errorf("generated ChatGPT config still contains legacy model_providers.%s table", codexAppProfileName)
	}
	for _, check := range []struct {
		path []string
		want string
	}{
		{[]string{codexRootModelKey}, model},
		{[]string{codexRootModelProviderKey}, "openai"},
		{[]string{codexRootModelCatalogJSONKey}, modelCatalogPath},
		{[]string{codexRootOpenAIBaseURLKey}, baseURL},
	} {
		if got, ok := config.String(check.path...); !ok || got != check.want {
			return fmt.Errorf("generated ChatGPT config missing %s = %q", strings.Join(check.path, "."), check.want)
		}
	}
	return nil
}

func codexAppProxyBaseURL() string {
	return strings.TrimRight(envconfig.ConnectableHost().String(), "/") + codexproxy.PathPrefix + "/v1"
}

func (c *CodexApp) Onboard() error {
	return config.MarkIntegrationOnboarded(chatGPTIntegrationName)
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

func (c *CodexApp) Run(_ string, _ []LaunchModel, args []string) error {
	if err := codexAppSupported(); err != nil {
		return err
	}
	if len(args) > 0 {
		return fmt.Errorf("chatgpt does not accept extra arguments")
	}
	return codexAppLaunchOrRestart("Restart ChatGPT to add Ollama models?", nil)
}

// Installed reports whether ChatGPT can be opened on this host.
func (c *CodexApp) Installed() bool {
	return codexAppInstalled()
}

// OllamaConfigured reports whether the regular ChatGPT profile has the
// additive Ollama catalog and loopback router enabled.
func (c *CodexApp) OllamaConfigured() bool {
	configPath, err := codexConfigPath()
	if err == nil {
		if data, readErr := os.ReadFile(configPath); readErr == nil {
			if parsed, parseErr := codexParseConfig(string(data)); parseErr == nil && codexAppRootUsesProxy(parsed) {
				// Report the managed root independently of catalog health so a
				// damaged or deleted catalog never hides the off-switch.
				return true
			}
		}
	}
	return c.CurrentModel() != ""
}

// Running reports whether the user's regular ChatGPT app is open.
func (c *CodexApp) Running() bool {
	return codexAppIsRunning()
}

// UseOllamaFromDesktop adds Ollama models to the regular ChatGPT profile.
// ChatGPT is stopped before its startup-only catalog changes, then reopened
// with the same account, native models, chats, plugins, and skills.
func (c *CodexApp) UseOllamaFromDesktop(primary string, models []LaunchModel) error {
	if err := codexAppSupported(); err != nil {
		return err
	}
	if !codexAppInstalled() {
		return fmt.Errorf("ChatGPT is not installed")
	}
	if err := codexAppRouterHealth(); err != nil {
		return err
	}
	if err := stopLegacyCodexAppOllamaProfile(); err != nil {
		return err
	}
	return codexAppApplyProfileFromDesktop(func() error {
		if err := resetCodexAppRegularProfileRequestCount(); err != nil {
			return err
		}
		return c.ConfigureWithModels(primary, models)
	}, true)
}

func defaultCodexAppRouterHealth() error {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	endpoint := strings.TrimRight(envconfig.ConnectableHost().String(), "/") + codexproxy.PathPrefix + "/_health"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return fmt.Errorf("create ChatGPT router health check: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("ChatGPT routing is unavailable at %s; restart Ollama and try again: %w", endpoint, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("the running Ollama server does not include ChatGPT routing; restart Ollama using this build and try again")
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ChatGPT router health check returned %s", resp.Status)
	}

	var status struct {
		OK bool `json:"ok"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return fmt.Errorf("read ChatGPT router health check: %w", err)
	}
	if !status.OK {
		return errors.New("ChatGPT router is not ready; restart Ollama and try again")
	}
	return nil
}

// RestoreFromDesktop removes the additive Ollama catalog and restores the
// prior base URL and model settings. A stopped ChatGPT app remains stopped.
func (c *CodexApp) RestoreFromDesktop() error {
	if err := codexAppSupported(); err != nil {
		return err
	}
	if err := stopLegacyCodexAppOllamaProfile(); err != nil {
		return err
	}
	return codexAppApplyProfileFromDesktop(restoreCodexAppProfile, false)
}

// RestoreForShutdown removes the additive catalog and router without reopening
// ChatGPT while Ollama itself is shutting down.
func (c *CodexApp) RestoreForShutdown(ctx context.Context) error {
	if err := codexAppSupported(); err != nil {
		return err
	}
	if !c.OllamaConfigured() {
		return nil
	}
	if codexAppIsRunning() {
		if err := codexAppQuitApp(); err != nil {
			return fmt.Errorf("quit ChatGPT: %w", err)
		}
		if err := waitForCodexAppExitContext(ctx); err != nil {
			return err
		}
	}
	return restoreCodexAppProfile()
}

func (c *CodexApp) OllamaRequestCount() uint64 {
	return codexAppRegularProfileRequestCount()
}

func stopLegacyCodexAppOllamaProfile() error {
	if codexAppGOOS != "darwin" || !codexAppProfileIsRunning() {
		return nil
	}
	if err := codexAppStopProfile(); err != nil {
		return fmt.Errorf("close the previous ChatGPT · Ollama profile: %w", err)
	}
	return nil
}

func (c *CodexApp) Restore() error {
	if err := codexAppSupported(); err != nil {
		return err
	}
	if err := restoreCodexAppProfile(); err != nil {
		return err
	}
	return codexAppLaunchOrRestart("Restart ChatGPT to use your usual profile?", nil)
}

func restoreCodexAppProfile() error {
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
			return nil
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
	return nil
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

func codexAppOllamaProfileRoot() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", codexAppOllamaProfileDirName), nil
}

func codexAppOllamaProfileUserDataDir() (string, error) {
	root, err := codexAppOllamaProfileRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, codexAppOllamaUserDataName), nil
}

func codexAppOllamaProfilePIDPath() (string, error) {
	root, err := codexAppOllamaProfileRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, codexAppOllamaPIDFilename), nil
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

func codexAppRoutingCatalogPathForConfig(configPath string) string {
	return filepath.Join(filepath.Dir(configPath), codexAppRoutingCatalogFilename)
}

type codexAppRawModelCatalog struct {
	Models []json.RawMessage `json:"models"`
}

// writeCodexAppCombinedModelCatalog prepends selected Ollama models to the
// native catalog. The remote signed-in catalog can still merge over matching
// native slugs, while the separately written routing catalog remains the only
// authority for deciding which requests may be sent to Ollama.
func writeCodexAppCombinedModelCatalog(path string, models []LaunchModel, nativeCatalogData []byte) error {
	if len(models) == 0 {
		return fmt.Errorf("chatgpt model catalog cannot be empty")
	}

	nativeCatalog, err := parseCodexAppModelCatalog(nativeCatalogData)
	if err != nil {
		return fmt.Errorf("parse native Codex model catalog: %w", err)
	}
	baseInstructions := codexAppBaseInstructionsFromCatalog(nativeCatalog)
	ollamaPriorityStart := codexAppOllamaPriorityStart(nativeCatalog, len(models))
	entries := make([]json.RawMessage, 0, len(models)+len(nativeCatalog.Models))
	seen := make(map[string]bool, len(models)+len(nativeCatalog.Models))
	for i, model := range models {
		entry, err := json.Marshal(codexAppCatalogEntry(model.Name, codexAppModelMetadataFromLaunchModel(model), ollamaPriorityStart+i, baseInstructions))
		if err != nil {
			return err
		}
		entries = append(entries, entry)
		seen[codexAppCatalogModelKey(model.Name)] = true
	}
	for _, entry := range nativeCatalog.Models {
		slug, err := codexAppRawCatalogSlug(entry)
		if err != nil {
			return fmt.Errorf("parse native Codex model: %w", err)
		}
		if seen[codexAppCatalogModelKey(slug)] {
			continue
		}
		seen[codexAppCatalogModelKey(slug)] = true
		entries = append(entries, entry)
	}

	data, err := json.MarshalIndent(codexAppRawModelCatalog{Models: entries}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return fileutil.WriteWithBackup(path, append(data, '\n'), codexAppIntegrationName)
}

func codexAppOllamaPriorityStart(nativeCatalog codexAppRawModelCatalog, ollamaModelCount int) int {
	lowestNativePriority := 0
	found := false
	for _, entry := range nativeCatalog.Models {
		var model struct {
			Priority *int `json:"priority"`
		}
		if json.Unmarshal(entry, &model) != nil || model.Priority == nil {
			continue
		}
		if !found || *model.Priority < lowestNativePriority {
			lowestNativePriority = *model.Priority
			found = true
		}
	}
	if !found {
		return -ollamaModelCount
	}
	return lowestNativePriority - ollamaModelCount
}

func writeCodexAppRoutingCatalog(path string, models []LaunchModel) error {
	if len(models) == 0 {
		return fmt.Errorf("chatgpt routing catalog cannot be empty")
	}
	entries := make([]map[string]string, 0, len(models))
	for _, model := range models {
		entries = append(entries, map[string]string{"slug": model.Name})
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

func parseCodexAppModelCatalog(data []byte) (codexAppRawModelCatalog, error) {
	var catalog codexAppRawModelCatalog
	if err := json.Unmarshal(data, &catalog); err != nil {
		return catalog, err
	}
	if len(catalog.Models) == 0 {
		return catalog, fmt.Errorf("model catalog is empty")
	}
	for _, entry := range catalog.Models {
		if _, err := codexAppRawCatalogSlug(entry); err != nil {
			return catalog, err
		}
	}
	return catalog, nil
}

func codexAppRawCatalogSlug(entry json.RawMessage) (string, error) {
	var model struct {
		Slug string `json:"slug"`
	}
	if err := json.Unmarshal(entry, &model); err != nil {
		return "", err
	}
	model.Slug = strings.TrimSpace(model.Slug)
	if model.Slug == "" {
		return "", fmt.Errorf("model catalog entry has no slug")
	}
	return model.Slug, nil
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
	contextWindow        int
	inputModalities      []string
	defaultThinkingLevel string
	thinkingLevels       []string
	toolCapable          bool
}

func codexAppDefaultModelMetadata() codexAppModelMetadata {
	return codexAppModelMetadata{
		contextWindow:   128_000,
		inputModalities: []string{"text"},
	}
}

func codexAppModelMetadataFromLaunchModel(model LaunchModel) codexAppModelMetadata {
	metadata := codexAppDefaultModelMetadata()
	if model.ContextLength > 0 {
		metadata.contextWindow = model.ContextLength
	}
	if model.HasCapability("vision") {
		metadata.inputModalities = []string{"text", "image"}
	}
	metadata.defaultThinkingLevel, metadata.thinkingLevels = codexAppThinkingLevels(model)
	metadata.toolCapable = model.ToolCapable || model.HasCapability(modelpkg.CapabilityTools)
	return metadata
}

// codexAppThinkingLevels translates Ollama's model-family contract into the
// exact effort ladder ChatGPT may send. A binary Thinking capability alone is
// not evidence that every effort is supported, so unknown families get one
// compatible enabled value instead of a misleading four-level picker.
//
// TODO: Move these exact ladders into server-owned /api/show model metadata.
func codexAppThinkingLevels(model LaunchModel) (string, []string) {
	if !model.HasCapability(modelpkg.CapabilityThinking) {
		return "", nil
	}

	families := append([]string{model.Details.Family}, model.Details.Families...)
	for _, family := range families {
		normalized := strings.NewReplacer("-", "", "_", "", ".", "").Replace(strings.ToLower(strings.TrimSpace(family)))
		switch normalized {
		case "glm5next":
			// GLM-5.3 reasoning is always enabled and accepts low, high, or max.
			return "high", []string{"low", "high", "max"}
		case "gptoss":
			// GPT-OSS reasoning is always enabled and accepts low, medium, or high.
			return "medium", []string{"low", "medium", "high"}
		}
	}

	return "medium", []string{"medium"}
}

func codexAppThinkingLevelDescription(level string) string {
	switch level {
	case "low":
		return "Fast responses with lighter thinking"
	case "medium":
		return "Balances speed and thinking depth for everyday tasks"
	case "high":
		return "Greater thinking depth for complex tasks"
	case "max":
		return "Maximum thinking depth for the hardest tasks"
	default:
		return "Thinking effort"
	}
}

func codexAppCatalogEntry(model string, metadata codexAppModelMetadata, priority int, baseInstructions string) map[string]any {
	var defaultReasoningLevel any
	if metadata.defaultThinkingLevel != "" {
		defaultReasoningLevel = metadata.defaultThinkingLevel
	}
	supportedReasoningLevels := make([]any, 0, len(metadata.thinkingLevels))
	for _, level := range metadata.thinkingLevels {
		supportedReasoningLevels = append(supportedReasoningLevels, map[string]any{
			"effort":      level,
			"description": codexAppThinkingLevelDescription(level),
		})
	}

	return map[string]any{
		"slug":                                 model,
		"display_name":                         model,
		"description":                          "Ollama model",
		"default_reasoning_level":              defaultReasoningLevel,
		"supported_reasoning_levels":           supportedReasoningLevels,
		"shell_type":                           "unified_exec",
		"visibility":                           "list",
		"supported_in_api":                     false,
		"priority":                             priority,
		"additional_speed_tiers":               []any{},
		"service_tiers":                        []any{},
		"default_service_tier":                 nil,
		"availability_nux":                     nil,
		"upgrade":                              nil,
		"base_instructions":                    codexAppInstructionsForModel(baseInstructions, model),
		"model_messages":                       nil,
		"include_skills_usage_instructions":    true,
		"include_plugin_usage_instructions":    true,
		"include_apps_usage_instructions":      true,
		"supports_reasoning_summary_parameter": false,
		"supports_reasoning_summaries":         false,
		"default_reasoning_summary":            "auto",
		"support_verbosity":                    false,
		"default_verbosity":                    nil,
		"apply_patch_tool_type":                nil,
		"web_search_tool_type":                 "text",
		"truncation_policy":                    map[string]any{"mode": "tokens", "limit": 10_000},
		"supports_parallel_tool_calls":         metadata.toolCapable,
		"supports_image_detail_original":       false,
		"context_window":                       metadata.contextWindow,
		"max_context_window":                   metadata.contextWindow,
		"auto_compact_token_limit":             nil,
		"effective_context_window_percent":     95,
		"experimental_supported_tools":         []any{},
		"input_modalities":                     metadata.inputModalities,
		"supports_search_tool":                 metadata.toolCapable,
	}
}

func codexAppInstructionsForModel(baseInstructions, model string) string {
	replacement := fmt.Sprintf("You are Codex, a coding agent powered by %s through Ollama.", model)
	for _, identity := range []string{
		"You are Codex, an agent based on GPT-5.",
		"You are Codex, a coding agent based on GPT-5.",
	} {
		if strings.Contains(baseInstructions, identity) {
			return strings.Replace(baseInstructions, identity, replacement, 1)
		}
	}
	return baseInstructions
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

func codexAppBaseInstructionsFromCatalog(catalog codexAppRawModelCatalog) string {
	for _, entry := range catalog.Models {
		var model struct {
			BaseInstructions string `json:"base_instructions"`
		}
		if json.Unmarshal(entry, &model) == nil && strings.TrimSpace(model.BaseInstructions) != "" {
			return model.BaseInstructions
		}
	}
	return codexAppBaseInstructions()
}

// defaultCodexAppNativeModelCatalog asks the desktop app's Codex binary for a
// clean native catalog using a scratch CODEX_HOME. The user's auth and cache
// are copied read-only so signed-in model metadata remains available without
// loading this integration's generated catalog.
func defaultCodexAppNativeModelCatalog(configPath string) ([]byte, error) {
	var attempts []error
	executable, executableErr := codexAppCodexExecutable()
	if executableErr == nil {
		scratchHome, err := os.MkdirTemp("", "ollama-codex-models-")
		if err != nil {
			attempts = append(attempts, fmt.Errorf("create scratch CODEX_HOME: %w", err))
		} else {
			defer os.RemoveAll(scratchHome)
			for _, name := range []string{"auth.json", "models_cache.json"} {
				if err := copyCodexAppScratchFile(filepath.Join(filepath.Dir(configPath), name), filepath.Join(scratchHome, name)); err != nil && !os.IsNotExist(err) {
					attempts = append(attempts, fmt.Errorf("copy %s to scratch CODEX_HOME: %w", name, err))
				}
			}

			data, err := codexAppRunDebugModels(executable, scratchHome, false)
			if err == nil {
				if normalized, normalizeErr := normalizeCodexAppModelCatalog(data); normalizeErr == nil {
					return normalized, nil
				} else {
					attempts = append(attempts, fmt.Errorf("validate codex debug models output: %w", normalizeErr))
				}
			} else {
				attempts = append(attempts, fmt.Errorf("run codex debug models: %w", err))
			}
		}
	} else {
		attempts = append(attempts, executableErr)
	}

	cachePath := filepath.Join(filepath.Dir(configPath), "models_cache.json")
	if data, err := os.ReadFile(cachePath); err == nil {
		if normalized, normalizeErr := normalizeCodexAppModelCatalog(data); normalizeErr == nil {
			return normalized, nil
		} else {
			attempts = append(attempts, fmt.Errorf("validate cached Codex models: %w", normalizeErr))
		}
	} else {
		attempts = append(attempts, fmt.Errorf("read cached Codex models: %w", err))
	}

	if executableErr == nil {
		data, err := codexAppRunDebugModels(executable, "", true)
		if err == nil {
			if normalized, normalizeErr := normalizeCodexAppModelCatalog(data); normalizeErr == nil {
				return normalized, nil
			} else {
				attempts = append(attempts, fmt.Errorf("validate codex debug models --bundled output: %w", normalizeErr))
			}
		} else {
			attempts = append(attempts, fmt.Errorf("run codex debug models --bundled: %w", err))
		}
	}

	return nil, errors.Join(attempts...)
}

func normalizeCodexAppModelCatalog(data []byte) ([]byte, error) {
	catalog, err := parseCodexAppModelCatalog(data)
	if err != nil {
		return nil, err
	}
	return json.Marshal(catalog)
}

func copyCodexAppScratchFile(source, destination string) error {
	data, err := os.ReadFile(source)
	if err != nil {
		return err
	}
	return os.WriteFile(destination, data, 0o600)
}

func defaultCodexAppCodexExecutable() (string, error) {
	appPath := codexAppAppPath()
	if appPath == "" && codexAppGOOS == "windows" {
		appPath = codexAppRunPath()
	}
	var candidates []string
	if appPath != "" {
		switch codexAppGOOS {
		case "darwin":
			candidates = append(candidates, filepath.Join(appPath, "Contents", "Resources", "codex"))
		case "windows":
			appDir := filepath.Dir(appPath)
			candidates = append(candidates,
				filepath.Join(appDir, "resources", "codex.exe"),
				filepath.Join(appDir, "Resources", "codex.exe"),
			)
		}
	}
	for _, candidate := range candidates {
		if info, err := codexAppStat(candidate); err == nil && !info.IsDir() {
			return candidate, nil
		}
	}
	if path, err := exec.LookPath("codex"); err == nil {
		return path, nil
	}
	return "", fmt.Errorf("could not find the Codex executable used by the desktop app")
}

func defaultCodexAppRunDebugModels(executable, codexHome string, bundled bool) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	args := []string{"debug", "models"}
	if bundled {
		args = append(args, "--bundled")
	}
	cmd := exec.CommandContext(ctx, executable, args...)
	cmd.Env = codexAppDebugModelsEnvironment(codexHome)
	// Never inherit Ollama.app's launch directory. On macOS that directory can
	// be Documents, which makes this read-only catalog probe trigger an unrelated
	// Files & Folders permission prompt.
	if codexHome != "" {
		cmd.Dir = codexHome
	} else {
		cmd.Dir = os.TempDir()
	}
	data, err := cmd.Output()
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if err != nil {
		return nil, err
	}
	return data, nil
}

func codexAppDebugModelsEnvironment(codexHome string) []string {
	env := make([]string, 0, len(os.Environ())+1)
	for _, item := range os.Environ() {
		name, _, _ := strings.Cut(item, "=")
		switch name {
		case "CODEX_HOME", "OPENAI_API_KEY", "CODEX_API_KEY":
			continue
		}
		env = append(env, item)
	}
	if codexHome != "" {
		env = append(env, "CODEX_HOME="+codexHome)
	}
	return env
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
	if sp != nil {
		sp.Stop()
	}
	return codexAppOpenAfterRestart(restartAppID, restartAppPath, launchArgs)
}

func codexAppApplyProfileFromDesktop(change func() error, openWhenStopped bool) error {
	if !codexAppIsRunning() {
		if err := change(); err != nil {
			return err
		}
		if openWhenStopped {
			return codexAppOpenApp(nil)
		}
		return nil
	}

	restartAppID, restartAppPath := codexAppRestartTarget()
	if err := codexAppStopForRestart(nil); err != nil {
		return err
	}
	if err := change(); err != nil {
		changeErr := fmt.Errorf("change ChatGPT profile: %w", err)
		if openErr := codexAppOpenAfterRestart(restartAppID, restartAppPath, nil); openErr != nil {
			return errors.Join(changeErr, fmt.Errorf("reopen ChatGPT after profile failure: %w", openErr))
		}
		return changeErr
	}
	return codexAppOpenAfterRestart(restartAppID, restartAppPath, nil)
}

func codexAppRestartTarget() (appID, appPath string) {
	if codexAppGOOS != "windows" {
		return "", ""
	}
	if appID = codexAppStartID(); appID == "" {
		appPath = codexAppRunPath()
	}
	return appID, appPath
}

func codexAppStopForRestart(cancel <-chan struct{}) error {
	if err := codexAppQuitApp(); err != nil {
		return fmt.Errorf("quit ChatGPT: %w", err)
	}
	gracefulErr := waitForCodexAppGracefulExit(codexAppExitTimeout, cancel)
	if errors.Is(gracefulErr, ErrCancelled) {
		return gracefulErr
	}
	if gracefulErr != nil && !codexAppForceQuitSupported() {
		return gracefulErr
	}
	if codexAppForceQuitSupported() && codexAppIsRunning() {
		if forceErr := codexAppForceQuit(); forceErr != nil {
			return fmt.Errorf("force stop ChatGPT: %w", forceErr)
		}
		return waitForCodexAppExit(codexAppForceExitTimeout, cancel)
	}
	if gracefulErr != nil && codexAppIsRunning() {
		return gracefulErr
	}
	return nil
}

func codexAppOpenAfterRestart(appID, appPath string, launchArgs []string) error {
	if appID != "" {
		return codexAppOpenStart(appID)
	}
	if appPath != "" {
		return codexAppOpenPath(appPath)
	}
	return codexAppOpenApp(launchArgs)
}

func waitForCodexAppExitContext(ctx context.Context) error {
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for codexAppIsRunning() {
		select {
		case <-ctx.Done():
			return fmt.Errorf("wait for ChatGPT to quit: %w", ctx.Err())
		case <-ticker.C:
		}
	}
	return nil
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

func defaultCodexAppOllamaProfileApplication() (string, error) {
	appPath := codexAppAppPath()
	if appPath == "" {
		return "", fmt.Errorf("ChatGPT was not found; install it from https://chatgpt.com/download")
	}
	return appPath, nil
}

func defaultCodexAppOllamaProfileExecutable() (string, error) {
	appPath, err := codexAppProfileApplication()
	if err != nil {
		return "", err
	}
	executable := filepath.Join(appPath, "Contents", "MacOS", "ChatGPT")
	if info, err := codexAppStat(executable); err != nil || info.IsDir() {
		return "", fmt.Errorf("could not find the ChatGPT desktop executable at %s", executable)
	}
	return executable, nil
}

func prepareCodexAppOllamaProfileUserData(userDataDir string) (bool, error) {
	if err := os.MkdirAll(userDataDir, 0o700); err != nil {
		return false, err
	}
	if adoptCodexAppOllamaProfileSingleton(userDataDir) {
		return true, nil
	}
	for _, name := range []string{
		codexAppSingletonLockName,
		codexAppSingletonSocketName,
		codexAppSingletonCookieName,
	} {
		if err := os.Remove(filepath.Join(userDataDir, name)); err != nil && !os.IsNotExist(err) {
			return false, fmt.Errorf("clear stale ChatGPT · Ollama %s: %w", name, err)
		}
	}
	return false, nil
}

func adoptCodexAppOllamaProfileSingleton(userDataDir string) bool {
	pid, ok := codexAppOllamaProfileSingletonPID(userDataDir)
	if !ok || !codexAppOllamaProfileProcessMatches(pid) {
		return false
	}
	return writeCodexAppOllamaProfilePID(pid) == nil
}

func codexAppOllamaProfileSingletonPID(userDataDir string) (int, bool) {
	target, err := os.Readlink(filepath.Join(userDataDir, codexAppSingletonLockName))
	if err != nil {
		return 0, false
	}
	base := filepath.Base(strings.TrimSpace(target))
	separator := strings.LastIndexByte(base, '-')
	if separator < 0 || separator == len(base)-1 {
		return 0, false
	}
	pid, err := strconv.Atoi(base[separator+1:])
	if err != nil || pid <= 1 || pid == os.Getpid() {
		return 0, false
	}
	return pid, true
}

// The current integration never starts a second ChatGPT instance. This legacy
// detector remains only to close an isolated prototype process before the
// single-profile integration is enabled or restored.
func defaultCodexAppOllamaProfileIsRunning() bool {
	pid, ok := codexAppOllamaProfilePID()
	if !ok {
		return reconcileCodexAppOllamaProfileSingleton()
	}
	running, matches := codexAppOllamaProfileProcessIdentity(pid)
	if !running || !matches {
		_ = removeCodexAppOllamaProfilePID()
		return reconcileCodexAppOllamaProfileSingleton()
	}
	return true
}

func reconcileCodexAppOllamaProfileSingleton() bool {
	userDataDir, err := codexAppOllamaProfileUserDataDir()
	if err != nil {
		return false
	}
	running, err := prepareCodexAppOllamaProfileUserData(userDataDir)
	return err == nil && running
}

func writeCodexAppOllamaProfilePID(pid int) error {
	if pid <= 1 || pid == os.Getpid() {
		return fmt.Errorf("invalid ChatGPT · Ollama process ID %d", pid)
	}
	pidPath, err := codexAppOllamaProfilePIDPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(pidPath), 0o700); err != nil {
		return err
	}
	return os.WriteFile(pidPath, []byte(strconv.Itoa(pid)+"\n"), 0o600)
}

func codexAppOllamaProfilePID() (int, bool) {
	pidPath, err := codexAppOllamaProfilePIDPath()
	if err != nil {
		return 0, false
	}
	data, err := os.ReadFile(pidPath)
	if err != nil {
		return 0, false
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil || pid <= 1 || pid == os.Getpid() {
		_ = os.Remove(pidPath)
		return 0, false
	}
	return pid, true
}

func codexAppOllamaProfileProcessMatches(pid int) bool {
	_, matches := codexAppOllamaProfileProcessIdentity(pid)
	return matches
}

func codexAppOllamaProfileProcessIdentity(pid int) (bool, bool) {
	if pid <= 1 || pid == os.Getpid() {
		return false, false
	}
	executable, err := codexAppProfileExecutable()
	if err != nil {
		return false, false
	}
	userDataDir, err := codexAppOllamaProfileUserDataDir()
	if err != nil {
		return false, false
	}
	command, err := codexAppProcessCommand(pid)
	if err != nil {
		return false, false
	}
	command = strings.TrimSpace(command)
	return true, strings.HasPrefix(command, executable+" ") &&
		strings.Contains(command, "--user-data-dir="+userDataDir)
}

func defaultCodexAppProcessCommand(pid int) (string, error) {
	out, err := exec.Command("ps", "-p", strconv.Itoa(pid), "-o", "command=").Output()
	return string(out), err
}

func removeCodexAppOllamaProfilePID() error {
	pidPath, err := codexAppOllamaProfilePIDPath()
	if err != nil {
		return err
	}
	if err := os.Remove(pidPath); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func removeCodexAppOllamaProfilePIDIf(pid int) error {
	current, ok := codexAppOllamaProfilePID()
	if !ok || current != pid {
		return nil
	}
	return removeCodexAppOllamaProfilePID()
}

func defaultCodexAppStopOllamaProfile() error {
	pid, ok := codexAppOllamaProfilePID()
	if !ok {
		return removeCodexAppOllamaProfilePID()
	}
	running, matches := codexAppOllamaProfileProcessIdentity(pid)
	if !running {
		return removeCodexAppOllamaProfilePID()
	}
	if !matches {
		_ = removeCodexAppOllamaProfilePID()
		return fmt.Errorf("ChatGPT · Ollama process identity could not be verified")
	}
	process, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	if err := process.Signal(syscall.SIGTERM); err != nil {
		return fmt.Errorf("stop ChatGPT · Ollama: %w", err)
	}
	deadline := time.Now().Add(codexAppExitTimeout)
	for time.Now().Before(deadline) {
		if !codexAppOllamaProfileProcessMatches(pid) {
			return removeCodexAppOllamaProfilePID()
		}
		codexAppSleep(100 * time.Millisecond)
	}
	return fmt.Errorf("ChatGPT · Ollama did not close; quit that window and try again")
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
		codexAppIsOwnedProfileName(config.RootString(codexRootModelProviderKey)) ||
		codexAppRootUsesProxy(config)
}

func codexAppRootReferencesOwnedConfig(text string) bool {
	config, err := codexParseConfig(text)
	if err != nil {
		return false
	}
	return config.RootString(codexRootProfileKey) == codexAppProfileName ||
		config.RootString(codexRootModelProviderKey) == codexAppProfileName ||
		codexAppRootUsesProxy(config)
}

func codexAppRootUsesProxy(config codexParsedConfig) bool {
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		return false
	}
	return config.RootString(codexRootModelProviderKey) == "openai" &&
		codexNormalizeURL(config.RootString(codexRootOpenAIBaseURLKey)) == codexNormalizeURL(codexAppProxyBaseURL()) &&
		config.RootString(codexRootModelCatalogJSONKey) == catalogPath
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
	text = codexAppRemoveOwnedSection(text, codexProfileHeaderFor(codexAppProfileName))
	text = codexAppRemoveOwnedSection(text, codexProviderHeaderFor(codexAppProfileName))
	return text
}

func codexAppRemoveOwnedSection(text, header string) string {
	targetPath, ok := codexTableHeaderPath(header)
	if !ok {
		return text
	}
	start, end, found := codexSectionRange(text, targetPath)
	if !found {
		return text
	}
	if start > 0 && strings.HasSuffix(text[:start], "\n\n") {
		start--
	}
	return text[:start] + text[end:]
}

func codexAppRemoveOwnedCatalogIfUnused(text string) error {
	if codexAppRootReferencesCatalog(text) {
		return nil
	}
	return codexAppRemoveOwnedCatalog()
}

func codexAppRemoveOwnedCatalog() error {
	configPath, err := codexConfigPath()
	if err != nil {
		return err
	}
	for _, path := range []string{
		codexAppModelCatalogPathForConfig(configPath),
		codexAppRoutingCatalogPathForConfig(configPath),
	} {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return err
		}
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
	usesProxy := codexAppRootUsesProxy(config)
	if !codexAppIsOwnedProfileName(config.RootString(codexRootProfileKey)) && !codexAppIsOwnedProfileName(modelProvider) && !usesProxy {
		return text
	}
	text = codexRemoveRootValue(text, codexRootProfileKey)
	text = codexRemoveRootValue(text, codexRootModelKey)
	if codexAppIsOwnedProfileName(modelProvider) || usesProxy {
		text = codexRemoveRootValue(text, codexRootModelProviderKey)
	}
	if catalogPath, err := codexAppModelCatalogPath(); err == nil && modelCatalogJSON == catalogPath {
		text = codexRemoveRootValue(text, codexRootModelCatalogJSONKey)
	}
	if usesProxy {
		text = codexRemoveRootValue(text, codexRootOpenAIBaseURLKey)
	}
	return text
}

func codexAppRestoreRootValues(text string, state codexAppRestoreState) string {
	if !codexAppRootStillManaged(text) {
		return text
	}
	preserveCurrentModel := codexAppShouldPreserveCurrentModel(text)
	text = codexRestoreRootStringValue(text, codexRootProfileKey, state.HadProfile, state.Profile)
	if !preserveCurrentModel {
		text = codexRestoreRootStringValue(text, codexRootModelKey, state.HadModel, state.Model)
	}
	text = codexRestoreRootStringValue(text, codexRootModelProviderKey, state.HadModelProvider, state.ModelProvider)
	text = codexRestoreRootStringValue(text, codexRootModelCatalogJSONKey, state.HadModelCatalogJSON, state.ModelCatalogJSON)
	text = codexRestoreRootStringValue(text, codexRootOpenAIBaseURLKey, state.HadOpenAIBaseURL, state.OpenAIBaseURL)
	return text
}

// codexAppShouldPreserveCurrentModel keeps a native model the user selected
// while Ollama routing was active. Selected Ollama slugs are launch-owned and
// must still be replaced with the pre-integration model during restore.
func codexAppShouldPreserveCurrentModel(text string) bool {
	config, err := codexParseConfig(text)
	if err != nil {
		return false
	}
	current, ok := config.RootStringOK(codexRootModelKey)
	if !ok || strings.TrimSpace(current) == "" {
		return false
	}
	models, err := codexAppRoutingModels()
	if err != nil {
		return false
	}
	target := codexAppCatalogModelKey(current)
	for _, model := range models {
		if codexAppCatalogModelKey(model) == target {
			return false
		}
	}
	return true
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
	HadOpenAIBaseURL    bool   `json:"had_openai_base_url"`
	OpenAIBaseURL       string `json:"openai_base_url,omitempty"`
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
		var existingFields map[string]json.RawMessage
		if err := json.Unmarshal(stateData, &existingFields); err != nil {
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
		if _, ok := existingFields["had_model"]; ok {
			upgraded.HadModel = existing.HadModel
			upgraded.Model = existing.Model
		}
		if _, ok := existingFields["had_model_provider"]; ok {
			upgraded.HadModelProvider = existing.HadModelProvider
			upgraded.ModelProvider = existing.ModelProvider
		}
		if _, ok := existingFields["had_model_catalog_json"]; ok {
			upgraded.HadModelCatalogJSON = existing.HadModelCatalogJSON
			upgraded.ModelCatalogJSON = existing.ModelCatalogJSON
		}
		if _, ok := existingFields["had_openai_base_url"]; ok {
			upgraded.HadOpenAIBaseURL = existing.HadOpenAIBaseURL
			upgraded.OpenAIBaseURL = existing.OpenAIBaseURL
		}
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
	_, hasOpenAIBaseURL := raw["had_openai_base_url"]
	return hasModel && hasModelProvider && hasModelCatalogJSON && hasOpenAIBaseURL, nil
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
	openAIBaseURL, hadOpenAIBaseURL := config.RootStringOK(codexRootOpenAIBaseURLKey)
	return codexAppRestoreState{
		HadProfile:          hadProfile,
		Profile:             profile,
		HadModel:            hadModel,
		Model:               model,
		HadModelProvider:    hadModelProvider,
		ModelProvider:       modelProvider,
		HadModelCatalogJSON: hadModelCatalogJSON,
		ModelCatalogJSON:    modelCatalogJSON,
		HadOpenAIBaseURL:    hadOpenAIBaseURL,
		OpenAIBaseURL:       openAIBaseURL,
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
