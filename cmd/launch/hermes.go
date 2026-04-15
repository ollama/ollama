package launch

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	pathpkg "path"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
)

const (
	hermesInstallScript     = "curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --skip-setup"
	hermesProviderName      = "Ollama"
	hermesProviderKey       = "ollama-launch"
	hermesLegacyKey         = "ollama"
	hermesPlaceholderKey    = "ollama"
	hermesGatewaySetupHint  = "hermes gateway setup"
	hermesGatewaySetupTitle = "Connect a messaging app now?"
)

var (
	hermesGOOS      = runtime.GOOS
	hermesLookPath  = exec.LookPath
	hermesCommand   = exec.Command
	hermesUserHome  = os.UserHomeDir
	hermesOllamaURL = envconfig.ConnectableHost
)

var hermesMessagingEnvGroups = [][]string{
	{"TELEGRAM_BOT_TOKEN"},
	{"DISCORD_BOT_TOKEN"},
	{"SLACK_BOT_TOKEN"},
	{"SIGNAL_ACCOUNT"},
	{"EMAIL_ADDRESS"},
	{"TWILIO_ACCOUNT_SID"},
	{"MATRIX_ACCESS_TOKEN", "MATRIX_PASSWORD"},
	{"MATTERMOST_TOKEN"},
	{"WHATSAPP_PHONE_NUMBER_ID"},
	{"DINGTALK_CLIENT_ID"},
	{"FEISHU_APP_ID"},
	{"WECOM_BOT_ID"},
	{"WEIXIN_ACCOUNT_ID"},
	{"BLUEBUBBLES_SERVER_URL"},
	{"WEBHOOK_ENABLED"},
}

// Hermes is intentionally not an Editor integration: launch owns one primary
// model and the local Ollama endpoint, while Hermes keeps its own discovery and
// switching UX after startup.
type Hermes struct{}

type hermesConfigBackend struct {
	displayPath string
	read        func() ([]byte, error)
	write       func([]byte) error
}

func (h *Hermes) String() string { return "Hermes Agent" }

func (h *Hermes) Run(_ string, args []string) error {
	// Hermes reads its primary model from config.yaml. launch configures that
	// default model ahead of time so we can keep runtime invocation simple and
	// still let Hermes discover additional models later via its own UX.
	if hermesGOOS == "windows" {
		return h.runWindows(args)
	}

	bin, err := h.findUnixBinary()
	if err != nil {
		return err
	}
	if err := h.runGatewaySetupPreflight(args, func() error {
		return hermesAttachedCommand(bin, "gateway", "setup").Run()
	}); err != nil {
		return err
	}
	return hermesAttachedCommand(bin, args...).Run()
}

func (h *Hermes) Paths() []string {
	backend, err := h.configBackend()
	if err != nil {
		return nil
	}
	return []string{backend.displayPath}
}

func (h *Hermes) Configure(model string) error {
	backend, err := h.configBackend()
	if err != nil {
		return err
	}

	cfg := map[string]any{}
	if data, err := backend.read(); err == nil {
		if err := yaml.Unmarshal(data, &cfg); err != nil {
			return fmt.Errorf("parse hermes config: %w", err)
		}
	} else if !os.IsNotExist(err) {
		return err
	}

	modelSection, _ := cfg["model"].(map[string]any)
	if modelSection == nil {
		modelSection = make(map[string]any)
	}
	models := h.listModels(model)
	applyHermesManagedProviders(cfg, hermesBaseURL(), model, models)

	// launch writes the minimum provider/default-model settings needed to
	// bootstrap Hermes against Ollama. The active provider stays on a
	// launch-owned key so /model stays aligned with the launcher-managed entry,
	// and the Ollama endpoint lives in providers: so the picker shows one row.
	modelSection["provider"] = hermesProviderKey
	modelSection["default"] = model
	modelSection["base_url"] = hermesBaseURL()
	modelSection["api_key"] = hermesPlaceholderKey
	cfg["model"] = modelSection

	// use Hermes' built-in web toolset for now.
	// TODO(parthsareen): move this to using Ollama web search
	cfg["toolsets"] = mergeHermesToolsets(cfg["toolsets"])

	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	return backend.write(data)
}

func (h *Hermes) CurrentModel() string {
	backend, err := h.configBackend()
	if err != nil {
		return ""
	}
	data, err := backend.read()
	if err != nil {
		return ""
	}

	cfg := map[string]any{}
	if yaml.Unmarshal(data, &cfg) != nil {
		return ""
	}
	return hermesManagedCurrentModel(cfg, hermesBaseURL())
}

func (h *Hermes) Onboard() error {
	return config.MarkIntegrationOnboarded("hermes")
}

func (h *Hermes) RequiresInteractiveOnboarding() bool {
	return false
}

func (h *Hermes) RefreshRuntimeAfterConfigure() error {
	running, err := h.gatewayRunning()
	if err != nil {
		return fmt.Errorf("check Hermes gateway status: %w", err)
	}
	if !running {
		return nil
	}

	fmt.Fprintf(os.Stderr, "%sRefreshing Hermes messaging gateway...%s\n", ansiGray, ansiReset)
	if err := h.restartGateway(); err != nil {
		return fmt.Errorf("restart Hermes gateway: %w", err)
	}
	fmt.Fprintln(os.Stderr)
	return nil
}

func (h *Hermes) installed() bool {
	if hermesGOOS == "windows" {
		if _, err := hermesLookPath("hermes"); err == nil {
			return true
		}
		return h.wslHasHermes()
	}

	_, err := h.findUnixBinary()
	return err == nil
}

func (h *Hermes) ensureInstalled() error {
	if h.installed() {
		return nil
	}

	if hermesGOOS == "windows" {
		return h.ensureInstalledWindows()
	}

	var missing []string
	for _, dep := range []string{"bash", "curl", "git"} {
		if _, err := hermesLookPath(dep); err != nil {
			missing = append(missing, dep)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("Hermes is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch hermes", strings.Join(missing, "\n  "))
	}

	ok, err := ConfirmPrompt("Hermes is not installed. Install now?")
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("hermes installation cancelled")
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Hermes...\n")
	if err := hermesAttachedCommand("bash", "-lc", hermesInstallScript).Run(); err != nil {
		return fmt.Errorf("failed to install hermes: %w", err)
	}

	if !h.installed() {
		return fmt.Errorf("hermes was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sHermes installed successfully%s\n\n", ansiGreen, ansiReset)
	return nil
}

func (h *Hermes) ensureInstalledWindows() error {
	// Hermes upstream support is WSL-oriented, so Windows launch uses a hybrid
	// WSL handoff that stays on the same install path as upstream Hermes.
	if _, err := hermesLookPath("hermes"); err == nil {
		return nil
	}
	if !h.wslAvailable() {
		return hermesWindowsHint(fmt.Errorf("hermes is not installed"))
	}
	if h.wslHasHermes() {
		return nil
	}

	ok, err := ConfirmPromptWithOptions("Hermes runs through WSL2 on Windows. Install it in WSL now?", ConfirmOptions{
		YesLabel: "Use WSL",
		NoLabel:  "Show manual steps",
	})
	if err != nil {
		return err
	}
	if !ok {
		return hermesWindowsHint(fmt.Errorf("hermes is not installed"))
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Hermes in WSL...\n")
	if err := h.runWSL("bash", "-lc", hermesInstallScript); err != nil {
		return hermesWindowsHint(fmt.Errorf("failed to install hermes in WSL: %w", err))
	}
	if !h.wslHasHermes() {
		return hermesWindowsHint(fmt.Errorf("hermes install finished but the WSL binary was not found"))
	}

	fmt.Fprintf(os.Stderr, "%sHermes installed successfully in WSL%s\n\n", ansiGreen, ansiReset)
	return nil
}

func (h *Hermes) listModels(defaultModel string) []string {
	client := hermesOllamaClient()
	resp, err := client.List(context.Background())
	if err != nil {
		return []string{defaultModel}
	}

	models := make([]string, 0, len(resp.Models)+1)
	seen := make(map[string]struct{}, len(resp.Models)+1)
	add := func(name string) {
		name = strings.TrimSpace(name)
		if name == "" {
			return
		}
		if _, ok := seen[name]; ok {
			return
		}
		seen[name] = struct{}{}
		models = append(models, name)
	}

	add(defaultModel)
	for _, entry := range resp.Models {
		add(entry.Name)
	}
	if len(models) == 0 {
		return []string{defaultModel}
	}
	return models
}

func (h *Hermes) findUnixBinary() (string, error) {
	if path, err := hermesLookPath("hermes"); err == nil {
		return path, nil
	}

	home, err := hermesUserHome()
	if err != nil {
		return "", err
	}
	fallback := filepath.Join(home, ".local", "bin", "hermes")
	if _, err := os.Stat(fallback); err == nil {
		return fallback, nil
	}

	return "", fmt.Errorf("hermes is not installed")
}

func (h *Hermes) runWindows(args []string) error {
	if path, err := hermesLookPath("hermes"); err == nil {
		if err := h.runGatewaySetupPreflight(args, func() error {
			return hermesAttachedCommand(path, "gateway", "setup").Run()
		}); err != nil {
			return err
		}
		return hermesAttachedCommand(path, args...).Run()
	}
	if !h.wslAvailable() {
		return hermesWindowsHint(fmt.Errorf("hermes is not installed"))
	}
	if err := h.runGatewaySetupPreflight(args, func() error {
		return h.runWSL("hermes", "gateway", "setup")
	}); err != nil {
		return err
	}
	if err := h.runWSL(append([]string{"hermes"}, args...)...); err != nil {
		return hermesWindowsHint(err)
	}
	return nil
}

func (h *Hermes) runWSL(args ...string) error {
	if !h.wslAvailable() {
		return fmt.Errorf("wsl.exe is not available")
	}

	return hermesAttachedCommand("wsl.exe", "bash", "-lc", shellQuoteArgs(args)).Run()
}

func (h *Hermes) runWSLCombinedOutput(args ...string) ([]byte, error) {
	if !h.wslAvailable() {
		return nil, fmt.Errorf("wsl.exe is not available")
	}

	return hermesCommand("wsl.exe", "bash", "-lc", shellQuoteArgs(args)).CombinedOutput()
}

func (h *Hermes) wslAvailable() bool {
	_, err := hermesLookPath("wsl.exe")
	return err == nil
}

func (h *Hermes) wslHasHermes() bool {
	if !h.wslAvailable() {
		return false
	}
	cmd := hermesCommand("wsl.exe", "bash", "-lc", "command -v hermes >/dev/null 2>&1")
	return cmd.Run() == nil
}

func (h *Hermes) configBackend() (*hermesConfigBackend, error) {
	if hermesGOOS == "windows" {
		if _, err := hermesLookPath("hermes"); err == nil {
			return hermesLocalConfigBackend()
		}
		if h.wslAvailable() {
			return h.wslConfigBackend()
		}
	}
	return hermesLocalConfigBackend()
}

func hermesConfigPath() (string, error) {
	home, err := hermesUserHome()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".hermes", "config.yaml"), nil
}

func hermesLocalConfigBackend() (*hermesConfigBackend, error) {
	configPath, err := hermesConfigPath()
	if err != nil {
		return nil, err
	}
	return &hermesConfigBackend{
		displayPath: configPath,
		read: func() ([]byte, error) {
			return os.ReadFile(configPath)
		},
		write: func(data []byte) error {
			if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
				return err
			}
			return fileutil.WriteWithBackup(configPath, data)
		},
	}, nil
}

func (h *Hermes) wslConfigBackend() (*hermesConfigBackend, error) {
	home, err := h.wslHome()
	if err != nil {
		return nil, err
	}
	configPath := pathpkg.Join(home, ".hermes", "config.yaml")
	return &hermesConfigBackend{
		displayPath: configPath,
		read: func() ([]byte, error) {
			return h.readWSLFile(configPath)
		},
		write: func(data []byte) error {
			return h.writeWSLConfig(configPath, data)
		},
	}, nil
}

func (h *Hermes) wslHome() (string, error) {
	if !h.wslAvailable() {
		return "", fmt.Errorf("wsl.exe is not available")
	}
	cmd := hermesCommand("wsl.exe", "bash", "-lc", `printf %s "$HOME"`)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	home := strings.TrimSpace(string(out))
	if home == "" {
		return "", fmt.Errorf("could not resolve WSL home directory")
	}
	return home, nil
}

func (h *Hermes) readWSLFile(path string) ([]byte, error) {
	pathArg := shellQuoteArgs([]string{path})
	cmd := hermesCommand("wsl.exe", "bash", "-lc", fmt.Sprintf("if [ -f %s ]; then cat %s; else exit 42; fi", pathArg, pathArg))
	out, err := cmd.Output()
	if err == nil {
		return out, nil
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) && exitErr.ExitCode() == 42 {
		return nil, os.ErrNotExist
	}
	return nil, err
}

func (h *Hermes) writeWSLConfig(path string, data []byte) error {
	if existing, err := h.readWSLFile(path); err == nil {
		if !bytes.Equal(existing, data) {
			if err := hermesBackupData(path, existing); err != nil {
				return fmt.Errorf("backup failed: %w", err)
			}
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("read existing file: %w", err)
	}

	dir := pathpkg.Dir(path)
	dirArg := shellQuoteArgs([]string{dir})
	pathArg := shellQuoteArgs([]string{path})
	script := fmt.Sprintf(
		"dir=%s; path=%s; mkdir -p \"$dir\" && tmp=$(mktemp \"$dir/.tmp-XXXXXX\") && cat > \"$tmp\" && mv \"$tmp\" \"$path\"",
		dirArg,
		pathArg,
	)
	cmd := hermesCommand("wsl.exe", "bash", "-lc", script)
	cmd.Stdin = bytes.NewReader(data)
	if out, err := cmd.CombinedOutput(); err != nil {
		if msg := strings.TrimSpace(string(out)); msg != "" {
			return fmt.Errorf("%w: %s", err, msg)
		}
		return err
	}
	return nil
}

func hermesBackupData(path string, data []byte) error {
	if err := os.MkdirAll(fileutil.BackupDir(), 0o755); err != nil {
		return err
	}
	backupPath := filepath.Join(fileutil.BackupDir(), fmt.Sprintf("%s.%d", filepath.Base(path), time.Now().Unix()))
	return os.WriteFile(backupPath, data, 0o644)
}

func hermesBaseURL() string {
	return strings.TrimRight(hermesOllamaURL().String(), "/") + "/v1"
}

func hermesEnvPath() (string, error) {
	home, err := hermesUserHome()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".hermes", ".env"), nil
}

func (h *Hermes) runGatewaySetupPreflight(args []string, runSetup func() error) error {
	if len(args) > 0 || !isInteractiveSession() || currentLaunchConfirmPolicy.yes || currentLaunchConfirmPolicy.requireYesMessage {
		return nil
	}
	if h.messagingConfigured() {
		return nil
	}

	fmt.Fprintf(os.Stderr, "\nHermes can message you on Telegram, Discord, Slack, and more.\n\n")
	ok, err := ConfirmPromptWithOptions(hermesGatewaySetupTitle, ConfirmOptions{
		YesLabel: "Yes",
		NoLabel:  "Set up later",
	})
	if err != nil {
		return err
	}
	if !ok {
		return nil
	}
	if err := runSetup(); err != nil {
		return fmt.Errorf("hermes messaging setup failed: %w\n\nTry running: %s", err, hermesGatewaySetupHint)
	}
	return nil
}

func (h *Hermes) messagingConfigured() bool {
	envVars, err := h.gatewayEnvVars()
	if err != nil {
		return false
	}
	for _, group := range hermesMessagingEnvGroups {
		for _, key := range group {
			if strings.TrimSpace(envVars[key]) != "" {
				return true
			}
		}
	}
	return false
}

func (h *Hermes) gatewayEnvVars() (map[string]string, error) {
	envVars := make(map[string]string)

	data, err := h.readGatewayEnvFile()
	switch {
	case err == nil:
		for key, value := range hermesParseEnvFile(data) {
			envVars[key] = value
		}
	case os.IsNotExist(err):
		// nothing persisted yet
	default:
		return nil, err
	}

	if h.usesLocalRuntimeEnv() {
		for _, group := range hermesMessagingEnvGroups {
			for _, key := range group {
				if value, ok := os.LookupEnv(key); ok {
					envVars[key] = value
				}
			}
		}
	}

	return envVars, nil
}

func (h *Hermes) readGatewayEnvFile() ([]byte, error) {
	if hermesGOOS == "windows" {
		if _, err := hermesLookPath("hermes"); err == nil {
			path, err := hermesEnvPath()
			if err != nil {
				return nil, err
			}
			return os.ReadFile(path)
		}
		if h.wslAvailable() {
			home, err := h.wslHome()
			if err != nil {
				return nil, err
			}
			return h.readWSLFile(pathpkg.Join(home, ".hermes", ".env"))
		}
	}

	path, err := hermesEnvPath()
	if err != nil {
		return nil, err
	}
	return os.ReadFile(path)
}

func (h *Hermes) usesLocalRuntimeEnv() bool {
	if hermesGOOS != "windows" {
		return true
	}
	_, err := hermesLookPath("hermes")
	return err == nil
}

func (h *Hermes) gatewayRunning() (bool, error) {
	status, err := h.gatewayStatusOutput()
	if err != nil {
		return false, err
	}
	return hermesGatewayStatusRunning(status), nil
}

func (h *Hermes) gatewayStatusOutput() (string, error) {
	if hermesGOOS == "windows" {
		if path, err := hermesLookPath("hermes"); err == nil {
			out, err := hermesCommand(path, "gateway", "status").CombinedOutput()
			return string(out), err
		}
		if !h.wslAvailable() {
			return "", hermesWindowsHint(fmt.Errorf("hermes is not installed"))
		}
		out, err := h.runWSLCombinedOutput("hermes", "gateway", "status")
		return string(out), err
	}

	bin, err := h.findUnixBinary()
	if err != nil {
		return "", err
	}
	out, err := hermesCommand(bin, "gateway", "status").CombinedOutput()
	return string(out), err
}

func (h *Hermes) restartGateway() error {
	if hermesGOOS == "windows" {
		if path, err := hermesLookPath("hermes"); err == nil {
			return hermesAttachedCommand(path, "gateway", "restart").Run()
		}
		if !h.wslAvailable() {
			return hermesWindowsHint(fmt.Errorf("hermes is not installed"))
		}
		if err := h.runWSL("hermes", "gateway", "restart"); err != nil {
			return hermesWindowsHint(err)
		}
		return nil
	}

	bin, err := h.findUnixBinary()
	if err != nil {
		return err
	}
	return hermesAttachedCommand(bin, "gateway", "restart").Run()
}

func hermesGatewayStatusRunning(output string) bool {
	status := strings.ToLower(output)
	switch {
	case strings.Contains(status, "gateway is not running"):
		return false
	case strings.Contains(status, "gateway service is stopped"):
		return false
	case strings.Contains(status, "gateway service is not loaded"):
		return false
	case strings.Contains(status, "gateway is running"):
		return true
	case strings.Contains(status, "gateway service is running"):
		return true
	case strings.Contains(status, "gateway service is loaded"):
		return true
	default:
		return false
	}
}

func hermesParseEnvFile(data []byte) map[string]string {
	out := make(map[string]string)
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := strings.TrimSpace(strings.TrimPrefix(scanner.Text(), "\ufeff"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "export ") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "export "))
		}

		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}

		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}

		value = strings.TrimSpace(value)
		if len(value) >= 2 {
			switch {
			case value[0] == '"' && value[len(value)-1] == '"':
				if unquoted, err := strconv.Unquote(value); err == nil {
					value = unquoted
				}
			case value[0] == '\'' && value[len(value)-1] == '\'':
				value = value[1 : len(value)-1]
			}
		}

		out[key] = value
	}
	return out
}

func hermesOllamaClient() *api.Client {
	// Hermes queries the same launch-resolved Ollama host that launch writes
	// into config, so model discovery follows the configured endpoint.
	return api.NewClient(hermesOllamaURL(), http.DefaultClient)
}

func applyHermesManagedProviders(cfg map[string]any, baseURL string, model string, models []string) {
	providers := hermesUserProviders(cfg["providers"])
	entry := hermesManagedProviderEntry(providers)
	if entry == nil {
		entry = make(map[string]any)
	}
	entry["name"] = hermesProviderName
	entry["api"] = baseURL
	entry["default_model"] = model
	entry["models"] = hermesStringListAny(models)
	providers[hermesProviderKey] = entry
	delete(providers, hermesLegacyKey)
	cfg["providers"] = providers

	customProviders := hermesWithoutManagedCustomProviders(cfg["custom_providers"])
	if len(customProviders) == 0 {
		delete(cfg, "custom_providers")
		return
	}
	cfg["custom_providers"] = customProviders
}

func hermesManagedCurrentModel(cfg map[string]any, baseURL string) string {
	modelCfg, _ := cfg["model"].(map[string]any)
	if modelCfg == nil {
		return ""
	}

	provider, _ := modelCfg["provider"].(string)
	if strings.TrimSpace(strings.ToLower(provider)) != hermesProviderKey {
		return ""
	}

	configBaseURL, _ := modelCfg["base_url"].(string)
	if hermesNormalizeURL(configBaseURL) != hermesNormalizeURL(baseURL) {
		return ""
	}

	current, _ := modelCfg["default"].(string)
	current = strings.TrimSpace(current)
	if current == "" {
		return ""
	}

	providers := hermesUserProviders(cfg["providers"])
	entry, _ := providers[hermesProviderKey].(map[string]any)
	if entry == nil {
		return ""
	}
	if hermesHasManagedCustomProvider(cfg["custom_providers"]) {
		return ""
	}

	apiURL, _ := entry["api"].(string)
	if hermesNormalizeURL(apiURL) != hermesNormalizeURL(baseURL) {
		return ""
	}

	defaultModel, _ := entry["default_model"].(string)
	if strings.TrimSpace(defaultModel) != current {
		return ""
	}

	return current
}

func hermesUserProviders(current any) map[string]any {
	switch existing := current.(type) {
	case map[string]any:
		out := make(map[string]any, len(existing))
		for key, value := range existing {
			out[key] = value
		}
		return out
	case map[any]any:
		out := make(map[string]any, len(existing))
		for key, value := range existing {
			if s, ok := key.(string); ok {
				out[s] = value
			}
		}
		return out
	default:
		return make(map[string]any)
	}
}

func hermesCustomProviders(current any) []any {
	switch existing := current.(type) {
	case []any:
		return append([]any(nil), existing...)
	case []map[string]any:
		out := make([]any, 0, len(existing))
		for _, entry := range existing {
			out = append(out, entry)
		}
		return out
	default:
		return nil
	}
}

func hermesManagedProviderEntry(providers map[string]any) map[string]any {
	for _, key := range []string{hermesProviderKey, hermesLegacyKey} {
		if entry, _ := providers[key].(map[string]any); entry != nil {
			return entry
		}
	}
	return nil
}

func hermesWithoutManagedCustomProviders(current any) []any {
	customProviders := hermesCustomProviders(current)
	preserved := make([]any, 0, len(customProviders))

	for _, item := range customProviders {
		entry, _ := item.(map[string]any)
		if entry == nil {
			preserved = append(preserved, item)
			continue
		}
		if hermesManagedCustomProvider(entry) {
			continue
		}
		preserved = append(preserved, entry)
	}

	return preserved
}

func hermesHasManagedCustomProvider(current any) bool {
	for _, item := range hermesCustomProviders(current) {
		entry, _ := item.(map[string]any)
		if entry != nil && hermesManagedCustomProvider(entry) {
			return true
		}
	}
	return false
}

func hermesManagedCustomProvider(entry map[string]any) bool {
	name, _ := entry["name"].(string)
	return strings.EqualFold(strings.TrimSpace(name), hermesProviderName)
}

func hermesNormalizeURL(raw string) string {
	return strings.TrimRight(strings.TrimSpace(raw), "/")
}

func hermesStringListAny(models []string) []any {
	out := make([]any, 0, len(models))
	for _, model := range dedupeModelList(models) {
		model = strings.TrimSpace(model)
		if model == "" {
			continue
		}
		out = append(out, model)
	}
	return out
}

func mergeHermesToolsets(current any) any {
	added := false
	switch existing := current.(type) {
	case []any:
		out := make([]any, 0, len(existing)+1)
		for _, item := range existing {
			out = append(out, item)
			if s, _ := item.(string); s == "web" {
				added = true
			}
		}
		if !added {
			out = append(out, "web")
		}
		return out
	case []string:
		out := append([]string(nil), existing...)
		if !slices.Contains(out, "web") {
			out = append(out, "web")
		}
		asAny := make([]any, 0, len(out))
		for _, item := range out {
			asAny = append(asAny, item)
		}
		return asAny
	case string:
		if strings.TrimSpace(existing) == "" {
			return []any{"hermes-cli", "web"}
		}
		parts := strings.Split(existing, ",")
		out := make([]any, 0, len(parts)+1)
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			if part == "web" {
				added = true
			}
			out = append(out, part)
		}
		if !added {
			out = append(out, "web")
		}
		return out
	default:
		return []any{"hermes-cli", "web"}
	}
}

func shellQuoteArgs(args []string) string {
	quoted := make([]string, 0, len(args))
	for _, arg := range args {
		quoted = append(quoted, "'"+strings.ReplaceAll(arg, "'", `'\''`)+"'")
	}
	return strings.Join(quoted, " ")
}

func hermesAttachedCommand(name string, args ...string) *exec.Cmd {
	cmd := hermesCommand(name, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd
}

func hermesWindowsHint(err error) error {
	if hermesGOOS != "windows" {
		return err
	}
	return fmt.Errorf("%w\n\nHermes runs on Windows through WSL2.\nQuick setup: wsl --install\nInstaller docs: https://hermes-agent.nousresearch.com/docs/getting-started/installation/", err)
}
