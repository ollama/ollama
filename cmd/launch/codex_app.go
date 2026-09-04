package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/proxy"
	modelpkg "github.com/ollama/ollama/types/model"
)

const (
	chatGPTIntegrationName         = "chatgpt"
	codexAppIntegrationName        = "codex-app"
	codexAppProfileName            = "ollama-launch-codex-app"
	codexAppBundleID               = "com.openai.codex"
	codexAppModelCatalogFilename   = proxy.CodexDesktopModelCatalogFilename
	codexAppRoutingCatalogFilename = proxy.CodexDesktopRoutingCatalogFilename
	codexAppAutoReviewModelEnv     = "OLLAMA_CODEX_AUTO_REVIEW_MODEL"
	codexAppOllamaProfileDirName   = "chatgpt-ollama"
	codexAppOllamaUserDataName     = "electron-data"
	codexAppOllamaPIDFilename      = "chatgpt.pid"
	codexAppSingletonLockName      = "SingletonLock"
	codexAppSingletonSocketName    = "SingletonSocket"
	codexAppSingletonCookieName    = "SingletonCookie"
	codexAppDesktopTableName       = "desktop"
	codexAppReasoningEffortsKey    = "enabled-reasoning-efforts"
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

	codexAppExitTimeout = 5 * time.Second
)

// CodexApp adds Ollama models to ChatGPT's native catalog using a loopback router.
type CodexApp struct{}

// ErrCodexAppRestartConfirmationRequired reports that changing the regular
// ChatGPT profile would interrupt a running task.
var ErrCodexAppRestartConfirmationRequired = errors.New("ChatGPT restart confirmation is required before changing its profile")

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
	models = codexAppCatalogModels(primary, models)
	autoReviewModel, err := codexAppConfiguredAutoReviewModel(primary, models)
	if err != nil {
		return err
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
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		return err
	}
	routingCatalogPath := codexAppRoutingCatalogPathForConfig(configPath)
	if err := writeCodexAppRoutingCatalog(routingCatalogPath, models, autoReviewModel); err != nil {
		return err
	}
	if err := writeCodexAppCombinedModelCatalog(catalogPath, models, nativeCatalog); err != nil {
		return err
	}
	createdAuth, err := ensureCodexAppManagedAuth(configPath)
	if err != nil {
		return err
	}
	if err := writeCodexAppConfig(configPath, primary, catalogPath); err != nil {
		if createdAuth {
			if removeErr := removeCodexAppManagedAuth(configPath); removeErr != nil {
				return errors.Join(err, fmt.Errorf("remove ChatGPT local auth after failed configuration: %w", removeErr))
			}
		}
		return err
	}
	return nil
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

	// Recognize legacy provider layouts without losing restore state.
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

// Check the routing allow-list, not the combined picker catalog.
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
	// Leave model_provider unset to preserve ChatGPT's native account controls.
	text = codexRemoveRootValue(text, codexRootModelProviderKey)
	text = codexSetRootStringValue(text, codexRootModelCatalogJSONKey, modelCatalogPath)
	text = codexSetRootStringValue(text, codexRootOpenAIBaseURLKey, baseURL)
	text = codexAppSetReasoningEfforts(text, codexAppReasoningEffortsForConfig(text))

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
	if got, ok := config.RootStringOK(codexRootModelProviderKey); ok {
		return fmt.Errorf("generated ChatGPT config still contains explicit model_provider = %q", got)
	}
	efforts, ok := codexAppConfigReasoningEfforts(config)
	if !ok || !slices.Contains(efforts, "none") || !slices.Contains(efforts, "max") {
		return fmt.Errorf("generated ChatGPT config does not enable none and max thinking controls")
	}
	for _, check := range []struct {
		path []string
		want string
	}{
		{[]string{codexRootModelKey}, model},
		{[]string{codexRootModelCatalogJSONKey}, modelCatalogPath},
		{[]string{codexRootOpenAIBaseURLKey}, baseURL},
	} {
		if got, ok := config.String(check.path...); !ok || got != check.want {
			return fmt.Errorf("generated ChatGPT config missing %s = %q", strings.Join(check.path, "."), check.want)
		}
	}
	return nil
}

func codexAppReasoningEffortsForConfig(text string) []string {
	config, err := codexParseConfig(text)
	if err == nil {
		if existing, ok := codexAppConfigReasoningEfforts(config); ok {
			return codexAppMergeReasoningEfforts(existing)
		}
	}
	// Add missing thinking levels without replacing native choices.
	return []string{"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
}

func codexAppConfigReasoningEfforts(config codexParsedConfig) ([]string, bool) {
	desktop, ok := config.values[codexAppDesktopTableName].(map[string]any)
	if !ok {
		return nil, false
	}
	value, ok := desktop[codexAppReasoningEffortsKey]
	if !ok {
		return nil, false
	}
	items, ok := value.([]any)
	if !ok {
		if stringItems, stringsOK := value.([]string); stringsOK {
			return append([]string(nil), stringItems...), true
		}
		return nil, false
	}
	efforts := make([]string, 0, len(items))
	for _, item := range items {
		effort, ok := item.(string)
		if !ok {
			return nil, false
		}
		efforts = append(efforts, effort)
	}
	return efforts, true
}

func codexAppMergeReasoningEfforts(existing []string) []string {
	efforts := append([]string(nil), existing...)
	if !slices.Contains(efforts, "none") {
		efforts = append([]string{"none"}, efforts...)
	}
	if !slices.Contains(efforts, "max") {
		insertAt := len(efforts)
		for i, effort := range efforts {
			if effort == "ultra" || effort == "persistent" {
				insertAt = i
				break
			}
		}
		efforts = append(efforts, "")
		copy(efforts[insertAt+1:], efforts[insertAt:])
		efforts[insertAt] = "max"
	}
	return efforts
}

func codexAppSetReasoningEfforts(text string, efforts []string) string {
	assignment := codexAppStringArrayAssignment(codexAppReasoningEffortsKey, efforts)
	if start, end, found := codexSectionRange(text, []string{codexAppDesktopTableName}); found {
		if assignmentStart, assignmentEnd, assignmentFound := codexAppSectionAssignmentRange(text, start, end, codexAppReasoningEffortsKey); assignmentFound {
			return text[:assignmentStart] + assignment + codexAppReplacementLineEnding(text[assignmentStart:assignmentEnd]) + text[assignmentEnd:]
		}
		insert := assignment + "\n"
		if end > start && !strings.HasSuffix(text[:end], "\n") {
			insert = "\n" + insert
		}
		return text[:end] + insert + text[end:]
	}

	dottedKey := codexAppDesktopTableName + "." + codexAppReasoningEffortsKey
	if start, end, found := codexAppRootAssignmentRange(text, dottedKey); found {
		dottedAssignment := codexAppStringArrayAssignment(dottedKey, efforts)
		return text[:start] + dottedAssignment + codexAppReplacementLineEnding(text[start:end]) + text[end:]
	}

	// A dotted root assignment is equivalent to a [desktop] table entry and can
	// be removed cleanly when the user did not already have this setting.
	dottedAssignment := codexAppStringArrayAssignment(dottedKey, efforts)
	return codexAppInsertRootAssignment(text, dottedAssignment)
}

func codexAppRemoveReasoningEfforts(text string) string {
	if start, end, found := codexSectionRange(text, []string{codexAppDesktopTableName}); found {
		if assignmentStart, assignmentEnd, assignmentFound := codexAppSectionAssignmentRange(text, start, end, codexAppReasoningEffortsKey); assignmentFound {
			return text[:assignmentStart] + text[assignmentEnd:]
		}
	}
	dottedKey := codexAppDesktopTableName + "." + codexAppReasoningEffortsKey
	if start, end, found := codexAppRootAssignmentRange(text, dottedKey); found {
		return text[:start] + text[end:]
	}
	return text
}

func codexAppStringArrayAssignment(key string, values []string) string {
	data, _ := json.Marshal(values)
	return key + " = " + string(data)
}

func codexAppSectionAssignmentRange(text string, sectionStart, sectionEnd int, key string) (int, int, bool) {
	return codexAppAssignmentRange(text, sectionStart, sectionEnd, key, "[desktop]\n")
}

func codexAppRootAssignmentRange(text, key string) (int, int, bool) {
	rootEnd := len(text)
	if index := strings.Index(text, "\n["); index >= 0 {
		rootEnd = index + 1
	} else if strings.HasPrefix(strings.TrimSpace(text), "[") {
		rootEnd = 0
	}
	return codexAppAssignmentRange(text, 0, rootEnd, key, "")
}

func codexAppInsertRootAssignment(text, assignment string) string {
	offset := 0
	for _, line := range strings.SplitAfter(text, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "[") {
			return text[:offset] + assignment + "\n" + text[offset:]
		}
		offset += len(line)
	}
	if text != "" && !strings.HasSuffix(text, "\n") {
		text += "\n"
	}
	return text + assignment + "\n"
}

func codexAppAssignmentRange(text string, start, end int, key, parsePrefix string) (int, int, bool) {
	section := text[start:end]
	lines := strings.SplitAfter(section, "\n")
	offset := 0
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if !codexAppAssignmentStartsWithKey(trimmed, key) {
			offset += len(line)
			continue
		}
		assignmentStart := start + offset
		candidate := ""
		candidateEnd := assignmentStart
		for _, candidateLine := range lines[i:] {
			candidate += candidateLine
			candidateEnd += len(candidateLine)
			if _, err := codexParseConfigText(parsePrefix + candidate); err == nil {
				return assignmentStart, candidateEnd, true
			}
		}
		return 0, 0, false
	}
	return 0, 0, false
}

func codexAppAssignmentStartsWithKey(line, key string) bool {
	if !strings.HasPrefix(line, key) {
		return false
	}
	rest := strings.TrimSpace(strings.TrimPrefix(line, key))
	return strings.HasPrefix(rest, "=")
}

func codexAppReplacementLineEnding(replaced string) string {
	if strings.HasSuffix(replaced, "\n") {
		return "\n"
	}
	return ""
}

func codexAppProxyBaseURL() string {
	return strings.TrimRight(envconfig.ConnectableHost().String(), "/") + proxy.CodexDesktopPathPrefix + "/v1"
}

type codexAppManagedAuth struct {
	OpenAIAPIKey string `json:"OPENAI_API_KEY"`
	AuthMode     string `json:"auth_mode"`
}

// Create local auth only when absent; never read or replace an existing auth file.
func ensureCodexAppManagedAuth(configPath string) (bool, error) {
	authPath := filepath.Join(filepath.Dir(configPath), "auth.json")
	if err := os.MkdirAll(filepath.Dir(authPath), 0o700); err != nil {
		return false, err
	}

	data, err := json.MarshalIndent(codexAppManagedAuth{
		OpenAIAPIKey: proxy.CodexDesktopManagedAPIKey,
		AuthMode:     "apikey",
	}, "", "  ")
	if err != nil {
		return false, err
	}
	data = append(data, '\n')

	file, err := os.OpenFile(authPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if errors.Is(err, os.ErrExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	cleanup := true
	defer func() {
		if cleanup {
			_ = file.Close()
			_ = os.Remove(authPath)
		}
	}()
	if _, err := file.Write(data); err != nil {
		return false, err
	}
	if err := file.Close(); err != nil {
		return false, err
	}
	cleanup = false
	return true, nil
}

// Remove only Ollama's exact sentinel, never a user login or API key.
func removeCodexAppManagedAuth(configPath string) error {
	authPath := filepath.Join(filepath.Dir(configPath), "auth.json")
	data, err := os.ReadFile(authPath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if !isCodexAppManagedAuth(data) {
		return nil
	}
	if err := os.Remove(authPath); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func isCodexAppManagedAuth(data []byte) bool {
	var auth codexAppManagedAuth
	if json.Unmarshal(data, &auth) != nil {
		return false
	}
	return auth.AuthMode == "apikey" && auth.OpenAIAPIKey == proxy.CodexDesktopManagedAPIKey
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
				// A damaged catalog must not hide the off-switch.
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
// Its startup-only catalog requires a confirmed restart when running.
func (c *CodexApp) UseOllamaFromDesktop(primary string, models []LaunchModel, restartConfirmed bool) error {
	return c.updateOllamaModelsFromDesktop(primary, models, true, restartConfirmed)
}

// UpdateOllamaModelsFromDesktop changes the Ollama catalog without opening a
// stopped ChatGPT app. A running app still restarts after confirmation.
func (c *CodexApp) UpdateOllamaModelsFromDesktop(primary string, models []LaunchModel, restartConfirmed bool) error {
	return c.updateOllamaModelsFromDesktop(primary, models, false, restartConfirmed)
}

func (c *CodexApp) updateOllamaModelsFromDesktop(primary string, models []LaunchModel, openWhenStopped, restartConfirmed bool) error {
	if err := codexAppSupported(); err != nil {
		return err
	}
	if !codexAppInstalled() {
		return fmt.Errorf("ChatGPT is not installed")
	}
	if err := codexAppRouterHealth(); err != nil {
		return err
	}
	// Check before cleanup so cancellation is a no-op; check again when applying the profile.
	if (codexAppIsRunning() || codexAppProfileIsRunning()) && !restartConfirmed {
		return ErrCodexAppRestartConfirmationRequired
	}
	if err := stopLegacyCodexAppOllamaProfile(); err != nil {
		return err
	}
	return codexAppApplyProfileFromDesktop(func() error {
		if err := resetCodexAppRegularProfileRequestCount(); err != nil {
			return err
		}
		return c.ConfigureWithModels(primary, models)
	}, openWhenStopped, restartConfirmed)
}

func defaultCodexAppRouterHealth() error {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	endpoint := strings.TrimRight(envconfig.ConnectableHost().String(), "/") + proxy.CodexDesktopPathPrefix + "/_health"
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
func (c *CodexApp) RestoreFromDesktop(restartConfirmed bool) error {
	if err := codexAppSupported(); err != nil {
		return err
	}
	if (codexAppIsRunning() || codexAppProfileIsRunning()) && !restartConfirmed {
		return ErrCodexAppRestartConfirmationRequired
	}
	if err := stopLegacyCodexAppOllamaProfile(); err != nil {
		return err
	}
	return codexAppApplyProfileFromDesktop(restoreCodexAppProfile, false, restartConfirmed)
}

// RestartFromDesktop repairs and restarts the saved profile without requiring live inventory.
func (c *CodexApp) RestartFromDesktop(restartConfirmed bool) error {
	if err := codexAppSupported(); err != nil {
		return err
	}
	if !codexAppInstalled() {
		return errors.New("ChatGPT is not installed")
	}
	if !c.OllamaConfigured() {
		return errors.New("ChatGPT is not configured to use Ollama")
	}
	return codexAppApplyProfileFromDesktop(repairCodexAppCatalogAuthVisibility, true, restartConfirmed)
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
			if err := removeCodexAppManagedAuth(configPath); err != nil {
				return codexAppRestoreFailure(configPath, err)
			}
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
	if err := removeCodexAppManagedAuth(configPath); err != nil {
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

// supported_in_api controls visibility outside ChatGPT-account sessions.
// The separate routing catalog, not this combined picker, determines upstreams.
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
		entry, err = codexAppCatalogEntryWithAPISupport(entry, false)
		if err != nil {
			return fmt.Errorf("mark native Codex model %q as ChatGPT-only: %w", slug, err)
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

func codexAppCatalogEntryWithAPISupport(entry json.RawMessage, supported bool) (json.RawMessage, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(entry, &fields); err != nil {
		return nil, err
	}
	if supported {
		fields["supported_in_api"] = json.RawMessage("true")
	} else {
		fields["supported_in_api"] = json.RawMessage("false")
	}
	return json.Marshal(fields)
}

// Repair legacy visibility using the routing catalog without contacting Ollama.
func repairCodexAppCatalogAuthVisibility() error {
	configPath, err := codexConfigPath()
	if err != nil {
		return fmt.Errorf("find ChatGPT configuration: %w", err)
	}
	routingModels, err := codexAppRoutingModels()
	if err != nil {
		return fmt.Errorf("read ChatGPT Ollama routing catalog: %w", err)
	}
	routed := make(map[string]bool, len(routingModels))
	for _, model := range routingModels {
		routed[codexAppCatalogModelKey(model)] = true
	}

	catalogPath := codexAppModelCatalogPathForConfig(configPath)
	data, err := os.ReadFile(catalogPath)
	if err != nil {
		return fmt.Errorf("read ChatGPT model catalog: %w", err)
	}
	catalog, err := parseCodexAppModelCatalog(data)
	if err != nil {
		return fmt.Errorf("parse ChatGPT model catalog: %w", err)
	}
	changed := false
	for i, entry := range catalog.Models {
		slug, err := codexAppRawCatalogSlug(entry)
		if err != nil {
			return fmt.Errorf("parse ChatGPT model: %w", err)
		}
		want := routed[codexAppCatalogModelKey(slug)]
		var metadata struct {
			SupportedInAPI *bool `json:"supported_in_api"`
		}
		if err := json.Unmarshal(entry, &metadata); err != nil {
			return fmt.Errorf("parse ChatGPT model %q: %w", slug, err)
		}
		if metadata.SupportedInAPI != nil && *metadata.SupportedInAPI == want {
			continue
		}
		catalog.Models[i], err = codexAppCatalogEntryWithAPISupport(entry, want)
		if err != nil {
			return fmt.Errorf("update ChatGPT model %q: %w", slug, err)
		}
		changed = true
	}
	if !changed {
		return nil
	}
	data, err = json.MarshalIndent(catalog, "", "  ")
	if err != nil {
		return fmt.Errorf("encode ChatGPT model catalog: %w", err)
	}
	if err := fileutil.WriteWithBackup(catalogPath, append(data, '\n'), codexAppIntegrationName); err != nil {
		return fmt.Errorf("update ChatGPT model catalog: %w", err)
	}
	return nil
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

func writeCodexAppRoutingCatalog(path string, models []LaunchModel, autoReviewModel string) error {
	if len(models) == 0 {
		return fmt.Errorf("chatgpt routing catalog cannot be empty")
	}
	type thinkingMetadata struct {
		Supported bool           `json:"supported"`
		Levels    []string       `json:"levels,omitempty"`
		Values    map[string]any `json:"values,omitempty"`
	}
	type routingEntry struct {
		Slug     string           `json:"slug"`
		Thinking thinkingMetadata `json:"thinking"`
	}
	entries := make([]routingEntry, 0, len(models))
	for _, model := range models {
		metadata := codexAppModelMetadataFromLaunchModel(model)
		entries = append(entries, routingEntry{
			Slug: model.Name,
			Thinking: thinkingMetadata{
				Supported: len(metadata.thinking.levels) > 0,
				Levels:    metadata.thinking.levels,
				Values:    metadata.thinking.values,
			},
		})
	}
	catalog := struct {
		Models          []routingEntry `json:"models"`
		AutoReviewModel string         `json:"auto_review_model,omitempty"`
	}{
		Models:          entries,
		AutoReviewModel: autoReviewModel,
	}
	data, err := json.MarshalIndent(catalog, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return fileutil.WriteWithBackup(path, append(data, '\n'), codexAppIntegrationName)
}

func codexAppConfiguredAutoReviewModel(primary string, models []LaunchModel) (string, error) {
	configured := strings.TrimSpace(os.Getenv(codexAppAutoReviewModelEnv))
	switch strings.ToLower(configured) {
	case "", "selected", "ollama":
		return primary, nil
	case "native", "chatgpt":
		return "", nil
	}

	target := codexAppCatalogModelKey(configured)
	for _, model := range models {
		if codexAppCatalogModelKey(model.Name) == target {
			return model.Name, nil
		}
	}
	return "", fmt.Errorf("%s=%q is not one of the configured Ollama models", codexAppAutoReviewModelEnv, configured)
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

	if _, ok := findLaunchModel(models, primary); !ok {
		add(fallbackLaunchModel(primary))
	}
	for _, model := range models {
		if launchModelMatches(model.Name, primary) {
			model.Name = primary
		}
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
	thinking        codexAppThinkingContract
	toolCapable     bool
}

type codexAppThinkingContract struct {
	defaultLevel string
	levels       []string
	values       map[string]any
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
	metadata.thinking = codexAppThinkingContractForModel(model)
	metadata.toolCapable = model.ToolCapable || model.HasCapability(modelpkg.CapabilityTools)
	return metadata
}

// Retain raw thinking values so the proxy can round-trip Codex levels, including booleans.
func codexAppThinkingContractForModel(model LaunchModel) codexAppThinkingContract {
	if contract, ok := codexAppThinkingContractFromRecommendation(model.Thinking); ok {
		return contract
	}

	if !model.HasCapability(modelpkg.CapabilityThinking) {
		return codexAppThinkingContract{}
	}

	families := append([]string{model.Details.Family}, model.Details.Families...)
	for _, family := range families {
		normalized := strings.NewReplacer("-", "", "_", "", ".", "").Replace(strings.ToLower(strings.TrimSpace(family)))
		switch normalized {
		case "glm5next", "glmdsamoe":
			// These family identifiers share a thinking contract.
			return codexAppStringThinkingContract("max", "low", "high", "max")
		case "gptoss":
			return codexAppStringThinkingContract("medium", "low", "medium", "high")
		}
	}

	// Binary thinking maps "none" to off and "medium" to on.
	return codexAppThinkingContract{
		defaultLevel: "medium",
		levels:       []string{"none", "medium"},
		values:       map[string]any{"none": false, "medium": true},
	}
}

func codexAppThinkingContractFromRecommendation(thinking *api.ModelRecommendationThinking) (codexAppThinkingContract, bool) {
	if thinking == nil || len(thinking.Values) == 0 || thinking.Default == nil {
		return codexAppThinkingContract{}, false
	}

	contract := codexAppThinkingContract{values: make(map[string]any, len(thinking.Values))}
	for _, value := range thinking.Values {
		level, ok := codexAppThinkingLevelForOllamaValue(value)
		if !ok {
			return codexAppThinkingContract{}, false
		}
		if _, duplicate := contract.values[level]; duplicate {
			return codexAppThinkingContract{}, false
		}
		contract.levels = append(contract.levels, level)
		contract.values[level] = value
	}

	defaultLevel, ok := codexAppThinkingLevelForOllamaValue(thinking.Default)
	advertisedDefault, advertised := contract.values[defaultLevel]
	if !ok || !advertised || advertisedDefault != thinking.Default {
		return codexAppThinkingContract{}, false
	}
	if len(contract.levels) == 1 && contract.levels[0] == "none" {
		return codexAppThinkingContract{}, true
	}
	contract.defaultLevel = defaultLevel
	return contract, true
}

func codexAppThinkingLevelForOllamaValue(value any) (string, bool) {
	switch value := value.(type) {
	case bool:
		if value {
			return "medium", true
		}
		return "none", true
	case string:
		think := api.ThinkValue{Value: value}
		return value, think.IsValid()
	}
	return "", false
}

func codexAppStringThinkingContract(defaultLevel string, levels ...string) codexAppThinkingContract {
	values := make(map[string]any, len(levels))
	for _, level := range levels {
		values[level] = level
	}
	return codexAppThinkingContract{defaultLevel: defaultLevel, levels: levels, values: values}
}

func codexAppThinkingLevelDescription(level string) string {
	switch level {
	case "none":
		return "Turn thinking off"
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
	if metadata.thinking.defaultLevel != "" {
		defaultReasoningLevel = metadata.thinking.defaultLevel
	}
	supportedReasoningLevels := make([]any, 0, len(metadata.thinking.levels))
	for _, level := range metadata.thinking.levels {
		supportedReasoningLevels = append(supportedReasoningLevels, map[string]any{
			"effort":      level,
			"description": codexAppThinkingLevelDescription(level),
		})
	}

	// Keep Ollama models visible in API-key sessions.
	return map[string]any{
		"slug":                                 model,
		"display_name":                         model,
		"description":                          "Ollama model",
		"default_reasoning_level":              defaultReasoningLevel,
		"supported_reasoning_levels":           supportedReasoningLevels,
		"shell_type":                           "unified_exec",
		"visibility":                           "list",
		"supported_in_api":                     true,
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

// Probe the native catalog in a scratch CODEX_HOME to avoid loading the generated catalog.
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
	// Avoid inheriting Documents as cwd and triggering a Files & Folders prompt.
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
	// (quit, wait, reopen) so that one Ctrl+C aborts the whole sequence rather
	// than just the currently-active wait. The bubbletea
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
	if err := waitForCodexAppGracefulExit(codexAppExitTimeout, cancelled); err != nil {
		return err
	}
	if isCancelled() {
		return ErrCancelled
	}
	if sp != nil {
		sp.Stop()
	}
	return codexAppOpenAfterRestart(restartAppID, restartAppPath, launchArgs)
}

func codexAppApplyProfileFromDesktop(change func() error, openWhenStopped, restartConfirmed bool) error {
	if !codexAppIsRunning() {
		if err := change(); err != nil {
			return err
		}
		if openWhenStopped {
			return codexAppOpenApp(nil)
		}
		return nil
	}
	if !restartConfirmed {
		return ErrCodexAppRestartConfirmationRequired
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
	return waitForCodexAppGracefulExit(codexAppExitTimeout, cancel)
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

func waitForCodexAppGracefulExit(timeout time.Duration, cancel <-chan struct{}) error {
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
// whole restart sequence. Timing out never escalates to a forced termination;
// the profile remains unchanged and the user can retry after closing ChatGPT.
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
	return fmt.Errorf("ChatGPT did not quit. Close it manually, then try again")
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

// Detect isolated legacy processes for cleanup; this integration uses one profile.
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
	// Older managed configs explicitly selected the built-in OpenAI provider.
	// Accept both forms so they remain detectable and restorable after upgrade.
	if modelProvider, ok := config.RootStringOK(codexRootModelProviderKey); ok && modelProvider != "openai" {
		return false
	}
	return codexAppManagedProxyURL(config.RootString(codexRootOpenAIBaseURLKey)) &&
		config.RootString(codexRootModelCatalogJSONKey) == catalogPath
}

func codexAppManagedProxyURL(raw string) bool {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Scheme == "" || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return false
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return false
	}
	host := u.Hostname()
	if !strings.EqualFold(host, "localhost") {
		ip := net.ParseIP(host)
		if ip == nil || !ip.IsLoopback() {
			return false
		}
	}
	return strings.TrimSuffix(u.Path, "/") == proxy.CodexDesktopPathPrefix+"/v1"
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
	if state.HadDesktopReasoningEfforts {
		text = codexAppSetReasoningEfforts(text, state.DesktopReasoningEfforts)
	} else {
		text = codexAppRemoveReasoningEfforts(text)
	}
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
	HadProfile                 bool     `json:"had_profile"`
	Profile                    string   `json:"profile,omitempty"`
	HadModel                   bool     `json:"had_model"`
	Model                      string   `json:"model,omitempty"`
	HadModelProvider           bool     `json:"had_model_provider"`
	ModelProvider              string   `json:"model_provider,omitempty"`
	HadModelCatalogJSON        bool     `json:"had_model_catalog_json"`
	ModelCatalogJSON           string   `json:"model_catalog_json,omitempty"`
	HadOpenAIBaseURL           bool     `json:"had_openai_base_url"`
	OpenAIBaseURL              string   `json:"openai_base_url,omitempty"`
	HadDesktopReasoningEfforts bool     `json:"had_desktop_reasoning_efforts"`
	DesktopReasoningEfforts    []string `json:"desktop_reasoning_efforts,omitempty"`
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
			// values as the user's restore target. The old integration did not
			// manage desktop reasoning efforts, so their current value is still
			// the user's original value and must be retained during the upgrade.
			upgraded = codexAppRestoreState{
				HadDesktopReasoningEfforts: upgraded.HadDesktopReasoningEfforts,
				DesktopReasoningEfforts:    upgraded.DesktopReasoningEfforts,
			}
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
	_, hasDesktopReasoningEfforts := raw["had_desktop_reasoning_efforts"]
	return hasModel && hasModelProvider && hasModelCatalogJSON && hasOpenAIBaseURL && hasDesktopReasoningEfforts, nil
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
	desktopReasoningEfforts, hadDesktopReasoningEfforts := codexAppConfigReasoningEfforts(config)
	return codexAppRestoreState{
		HadProfile:                 hadProfile,
		Profile:                    profile,
		HadModel:                   hadModel,
		Model:                      model,
		HadModelProvider:           hadModelProvider,
		ModelProvider:              modelProvider,
		HadModelCatalogJSON:        hadModelCatalogJSON,
		ModelCatalogJSON:           modelCatalogJSON,
		HadOpenAIBaseURL:           hadOpenAIBaseURL,
		OpenAIBaseURL:              openAIBaseURL,
		HadDesktopReasoningEfforts: hadDesktopReasoningEfforts,
		DesktopReasoningEfforts:    desktopReasoningEfforts,
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
