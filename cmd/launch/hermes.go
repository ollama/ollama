package launch

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
)

const (
	hermesInstallScript  = "curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash"
	hermesProviderName   = "Ollama"
	hermesProviderKey    = "ollama-launch"
	hermesLegacyKey      = "ollama"
	hermesPlaceholderKey = "ollama"
)

var (
	hermesGOOS               = runtime.GOOS
	hermesLookPath           = exec.LookPath
	hermesCommand            = exec.Command
	hermesUserHome           = os.UserHomeDir
	hermesOllamaURL          = envconfig.ConnectableHost
	hermesGatewayAddr        = "127.0.0.1:18789"
	hermesGatewayStartWait   = 15 * time.Second
	hermesGatewayServiceWait = 5 * time.Second
)

// Hermes is intentionally not an Editor integration: launch owns one primary
// model and the local Ollama endpoint, while Hermes keeps its own discovery and
// switching UX after startup.
type Hermes struct{}

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
	if err := h.ensureGatewayRunning(bin, args); err != nil {
		return err
	}
	return hermesAttachedCommand(bin, args...).Run()
}

func (h *Hermes) Paths() []string {
	path, err := hermesConfigPath()
	if err != nil {
		return nil
	}
	return []string{path}
}

func (h *Hermes) Configure(model string) error {
	configPath, err := hermesConfigPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
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

	// launch intentionally writes only the minimum provider/default-model
	// settings needed to bootstrap Hermes against Ollama. We mirror Ollama's
	// catalog into both providers: and custom_providers:, but keep the active
	// provider on a launch-owned key so Hermes' built-in ollama aliases do not
	// hijack /model.
	modelSection["provider"] = hermesProviderKey
	modelSection["default"] = model
	modelSection["base_url"] = hermesBaseURL()
	modelSection["api_key"] = hermesPlaceholderKey
	cfg["model"] = modelSection

	// v1 uses Hermes' built-in web toolset intentionally. Matching Ollama's
	// OpenClaw-style web-search plugin flow is a separate enhancement.
	cfg["toolsets"] = mergeHermesToolsets(cfg["toolsets"])

	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	return fileutil.WriteWithBackup(configPath, data)
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

	switch modelCfg := cfg["model"].(type) {
	case string:
		return strings.TrimSpace(modelCfg)
	case map[string]any:
		if current, _ := modelCfg["default"].(string); strings.TrimSpace(current) != "" {
			return strings.TrimSpace(current)
		}
		if current, _ := modelCfg["model"].(string); strings.TrimSpace(current) != "" {
			return strings.TrimSpace(current)
		}
		if provider, _ := modelCfg["provider"].(string); strings.TrimSpace(provider) != "" {
			if current := hermesNamedProviderModel(cfg, provider); current != "" {
				return current
			}
		}
	}

	return ""
}

func (h *Hermes) Onboard() error {
	cfg, err := loadStoredIntegrationConfig("hermes")
	if err == nil && cfg.Onboarded {
		return nil
	}

	if hermesGOOS == "windows" {
		if _, err := hermesLookPath("hermes"); err == nil {
			if err := hermesAttachedCommand("hermes", "setup", "gateway").Run(); err != nil {
				return hermesWindowsHint(fmt.Errorf("hermes gateway setup failed: %w", err))
			}
		} else if err := h.runWSL("hermes", "setup", "gateway"); err != nil {
			return hermesWindowsHint(fmt.Errorf("hermes gateway setup failed: %w", err))
		}
	} else {
		bin, err := h.findUnixBinary()
		if err != nil {
			return err
		}
		if err := hermesAttachedCommand(bin, "setup", "gateway").Run(); err != nil {
			return fmt.Errorf("hermes gateway setup failed: %w", err)
		}
	}

	return config.MarkIntegrationOnboarded("hermes")
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
	// Hermes upstream support is WSL-oriented, so native Windows launch uses a
	// hybrid WSL handoff instead of trying to maintain a separate native
	// installation path in Ollama launch.
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
		if err := h.ensureGatewayRunning(path, args); err != nil {
			return err
		}
		return hermesAttachedCommand(path, args...).Run()
	}
	if !h.wslAvailable() {
		return hermesWindowsHint(fmt.Errorf("hermes is not installed"))
	}
	if err := h.ensureGatewayRunning("hermes", args); err != nil {
		return hermesWindowsHint(err)
	}
	return hermesWindowsHint(h.runWSL(append([]string{"hermes"}, args...)...))
}

func (h *Hermes) runWSL(args ...string) error {
	if !h.wslAvailable() {
		return fmt.Errorf("wsl.exe is not available")
	}

	return hermesAttachedCommand("wsl.exe", "bash", "-lc", shellQuoteArgs(args)).Run()
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

func hermesOllamaClient() *api.Client {
	// Hermes should query the same launch-resolved Ollama host that launch
	// writes into config, rather than depending on a potentially different
	// OLLAMA_HOST in the user's shell.
	return api.NewClient(hermesOllamaURL(), http.DefaultClient)
}

func hermesCustomProviderSlug(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		name = hermesProviderName
	}
	return "custom:" + strings.ToLower(strings.ReplaceAll(name, " ", "-"))
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

	customProviders, customEntry := hermesManagedCustomProviders(cfg["custom_providers"], baseURL)
	customEntry["base_url"] = baseURL
	customEntry["model"] = model
	customEntry["api_key"] = hermesPlaceholderKey
	customEntry["api_mode"] = "chat_completions"
	customEntry["models"] = hermesCustomProviderModels(customEntry["models"], models)
	cfg["custom_providers"] = customProviders
}

func hermesNamedProviderModel(cfg map[string]any, provider string) string {
	provider = strings.TrimSpace(strings.ToLower(provider))
	if provider == hermesProviderKey || provider == hermesLegacyKey || provider == hermesCustomProviderSlug(hermesProviderName) {
		providers := hermesUserProviders(cfg["providers"])
		for _, key := range []string{hermesProviderKey, hermesLegacyKey} {
			if entry, _ := providers[key].(map[string]any); entry != nil {
				current, _ := entry["default_model"].(string)
				if current = strings.TrimSpace(current); current != "" {
					return current
				}
			}
		}
	}
	if !strings.HasPrefix(provider, "custom:") {
		return ""
	}
	for _, item := range hermesCustomProviders(cfg["custom_providers"]) {
		entry, _ := item.(map[string]any)
		if entry == nil {
			continue
		}
		name, _ := entry["name"].(string)
		if hermesCustomProviderSlug(name) != provider {
			continue
		}
		current, _ := entry["model"].(string)
		return strings.TrimSpace(current)
	}
	return ""
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

func hermesManagedCustomProviders(current any, baseURL string) ([]any, map[string]any) {
	customProviders := hermesCustomProviders(current)
	normalizedURL := hermesNormalizeURL(baseURL)
	preserved := make([]any, 0, len(customProviders)+1)
	var managed map[string]any

	for _, item := range customProviders {
		entry, _ := item.(map[string]any)
		if entry == nil {
			preserved = append(preserved, item)
			continue
		}
		if hermesManagedCustomProvider(entry, normalizedURL) {
			if managed == nil {
				managed = entry
			}
			continue
		}
		preserved = append(preserved, entry)
	}

	if managed == nil {
		managed = make(map[string]any)
	}
	managed["name"] = hermesProviderName
	preserved = append([]any{managed}, preserved...)
	return preserved, managed
}

func hermesManagedCustomProvider(entry map[string]any, baseURL string) bool {
	name, _ := entry["name"].(string)
	if strings.EqualFold(strings.TrimSpace(name), hermesProviderName) {
		return true
	}
	for _, key := range []string{"base_url", "url", "api"} {
		if value, _ := entry[key].(string); baseURL != "" && hermesNormalizeURL(value) == baseURL {
			return true
		}
	}
	return false
}

func hermesNormalizeURL(raw string) string {
	return strings.TrimRight(strings.TrimSpace(raw), "/")
}

func hermesCustomProviderModels(current any, models []string) map[string]any {
	existing := make(map[string]any)
	if currentMap, _ := current.(map[string]any); currentMap != nil {
		for key, value := range currentMap {
			existing[key] = value
		}
	}

	out := make(map[string]any)
	for _, model := range dedupeModelList(models) {
		model = strings.TrimSpace(model)
		if model == "" {
			continue
		}
		if cfg, _ := existing[model].(map[string]any); cfg != nil {
			out[model] = cfg
			continue
		}
		out[model] = map[string]any{}
	}
	return out
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

func hermesGatewaySetupHint(err error) error {
	return fmt.Errorf("%w\n\nIf Hermes gateway prerequisites are missing, run 'hermes setup gateway' and retry.\nInstall docs: https://hermes-agent.nousresearch.com/docs/getting-started/installation/", err)
}

func (h *Hermes) ensureGatewayRunning(bin string, args []string) error {
	if len(args) > 0 {
		return nil
	}
	if portOpen(hermesGatewayAddr) {
		return nil
	}

	fmt.Fprintf(os.Stderr, "\n%sStarting Hermes gateway...%s\n", ansiGray, ansiReset)
	if err := h.startGateway(bin); err != nil {
		return hermesGatewaySetupHint(fmt.Errorf("failed to start hermes gateway: %w", err))
	}
	if !waitForPort(hermesGatewayAddr, hermesGatewayStartWait) {
		return hermesGatewaySetupHint(fmt.Errorf("hermes gateway did not start on %s", hermesGatewayAddr))
	}
	fmt.Fprintf(os.Stderr, "%sHermes gateway is running%s\n\n", ansiGreen, ansiReset)
	return nil
}

func (h *Hermes) startGateway(bin string) error {
	if h.tryStartGatewayService(bin) {
		return nil
	}
	return h.spawnGatewayRun(bin)
}

func (h *Hermes) tryStartGatewayService(bin string) bool {
	cmd, closeFiles, err := h.gatewayCommand(bin, []string{"gateway", "start"}, false)
	if err != nil {
		return false
	}
	defer closeFiles()
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if cmd.Run() != nil {
		return false
	}
	return waitForPort(hermesGatewayAddr, hermesGatewayServiceWait)
}

func (h *Hermes) spawnGatewayRun(bin string) error {
	cmd, closeFiles, err := h.gatewayCommand(bin, []string{"gateway", "run"}, true)
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		closeFiles()
		return err
	}
	closeFiles()
	return nil
}

func (h *Hermes) gatewayCommand(bin string, args []string, background bool) (*exec.Cmd, func(), error) {
	if hermesGOOS == "windows" {
		if _, err := hermesLookPath("hermes"); err == nil {
			cmd := hermesCommand(bin, args...)
			return cmd, func() {}, nil
		}
		if !h.wslAvailable() {
			return nil, nil, fmt.Errorf("wsl.exe is not available")
		}
		if background {
			cmd := hermesCommand("wsl.exe", "bash", "-lc", shellQuoteArgs(append([]string{"nohup", "hermes"}, args...))+" >/dev/null 2>&1 </dev/null &")
			return cmd, func() {}, nil
		}
		cmd := hermesCommand("wsl.exe", "bash", "-lc", shellQuoteArgs(append([]string{"hermes"}, args...)))
		return cmd, func() {}, nil
	}

	cmd := hermesCommand(bin, args...)
	if !background {
		return cmd, func() {}, nil
	}

	devnull, err := os.OpenFile(os.DevNull, os.O_RDWR, 0)
	if err != nil {
		return nil, nil, err
	}
	cmd.Stdin = devnull
	cmd.Stdout = devnull
	cmd.Stderr = devnull
	return cmd, func() { _ = devnull.Close() }, nil
}
