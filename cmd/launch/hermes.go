package launch

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"

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

func (h *Hermes) String() string { return "Hermes Agent" }

func (h *Hermes) Run(_ string, args []string) error {
	// Hermes reads its primary model from config.yaml. launch configures that
	// default model ahead of time so we can keep runtime invocation simple and
	// still let Hermes discover additional models later via its own UX.
	bin, err := h.binary()
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
	configPath, err := hermesConfigPath()
	if err != nil {
		return nil
	}
	return []string{configPath}
}

func (h *Hermes) Configure(model string) error {
	configPath, err := hermesConfigPath()
	if err != nil {
		return err
	}

	cfg := map[string]any{}
	if data, err := os.ReadFile(configPath); err == nil {
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
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}
	return fileutil.WriteWithBackup(configPath, data, "hermes")
}

func (h *Hermes) CurrentModel() string {
	configPath, err := hermesConfigPath()
	if err != nil {
		return ""
	}
	data, err := os.ReadFile(configPath)
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
	_, err := h.binary()
	return err == nil
}

func (h *Hermes) ensureInstalled() error {
	if h.installed() {
		return nil
	}

	if hermesGOOS == "windows" {
		return hermesWindowsHint()
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

func (h *Hermes) binary() (string, error) {
	if path, err := hermesLookPath("hermes"); err == nil {
		return path, nil
	}

	if hermesGOOS == "windows" {
		return "", hermesWindowsHint()
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

func hermesConfigPath() (string, error) {
	home, err := hermesUserHome()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".hermes", "config.yaml"), nil
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

	envFilePath, err := hermesEnvPath()
	if err != nil {
		return nil, err
	}
	switch data, err := os.ReadFile(envFilePath); {
	case err == nil:
		for key, value := range hermesParseEnvFile(data) {
			envVars[key] = value
		}
	case os.IsNotExist(err):
		// nothing persisted yet
	default:
		return nil, err
	}

	for _, group := range hermesMessagingEnvGroups {
		for _, key := range group {
			if value, ok := os.LookupEnv(key); ok {
				envVars[key] = value
			}
		}
	}

	return envVars, nil
}

func (h *Hermes) gatewayRunning() (bool, error) {
	status, err := h.gatewayStatusOutput()
	if err != nil {
		return false, err
	}
	return hermesGatewayStatusRunning(status), nil
}

func (h *Hermes) gatewayStatusOutput() (string, error) {
	bin, err := h.binary()
	if err != nil {
		return "", err
	}
	out, err := hermesCommand(bin, "gateway", "status").CombinedOutput()
	return string(out), err
}

func (h *Hermes) restartGateway() error {
	bin, err := h.binary()
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

func hermesAttachedCommand(name string, args ...string) *exec.Cmd {
	cmd := hermesCommand(name, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd
}

func hermesWindowsHint() error {
	return fmt.Errorf("Hermes on Windows requires WSL2. Install WSL with: wsl --install\n" +
		"Then run 'ollama launch hermes' from inside your WSL shell.\n" +
		"Docs: https://hermes-agent.nousresearch.com/docs/getting-started/installation/")
}
