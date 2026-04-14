package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"

	"golang.org/x/mod/semver"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

const defaultGatewayPort = 18789

// Bound model capability probing so launch/config cannot hang on slow/unreachable API calls.
var openclawModelShowTimeout = 5 * time.Second

// openclawFreshInstall is set to true when ensureOpenclawInstalled performs an install
var openclawFreshInstall bool

type Openclaw struct{}

func (c *Openclaw) String() string { return "OpenClaw" }

func (c *Openclaw) Run(model string, args []string) error {
	bin, err := ensureOpenclawInstalled()
	if err != nil {
		return err
	}

	firstLaunch := !c.onboarded()

	if firstLaunch {
		fmt.Fprintf(os.Stderr, "\n%sSecurity%s\n\n", ansiBold, ansiReset)
		fmt.Fprintf(os.Stderr, "  OpenClaw can read files and run actions when tools are enabled.\n")
		fmt.Fprintf(os.Stderr, "  A bad prompt can trick it into doing unsafe things.\n\n")
		fmt.Fprintf(os.Stderr, "%s  Learn more: https://docs.openclaw.ai/gateway/security%s\n\n", ansiGray, ansiReset)

		ok, err := ConfirmPrompt("I understand the risks. Continue?")
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}

		// Ensure the latest version is installed before onboarding so we get
		// the newest wizard flags (e.g. --auth-choice ollama).
		if !openclawFreshInstall {
			update := exec.Command(bin, "update")
			update.Stdout = os.Stdout
			update.Stderr = os.Stderr
			_ = update.Run() // best-effort; continue even if update fails
		}

		fmt.Fprintf(os.Stderr, "\n%sSetting up OpenClaw with Ollama...%s\n", ansiGreen, ansiReset)
		fmt.Fprintf(os.Stderr, "%s  Model: %s%s\n\n", ansiGray, model, ansiReset)

		onboardArgs := []string{
			"onboard",
			"--non-interactive",
			"--accept-risk",
			"--auth-choice", "ollama",
			"--custom-base-url", envconfig.Host().String(),
			"--custom-model-id", model,
			"--skip-channels",
			"--skip-skills",
		}
		if canInstallDaemon() {
			onboardArgs = append(onboardArgs, "--install-daemon")
		} else {
			// When we can't install a daemon (e.g. no systemd, sudo dropped
			// XDG_RUNTIME_DIR, or container environment), skip the gateway
			// health check so non-interactive onboarding completes. The
			// gateway is started as a foreground child process after onboarding.
			onboardArgs = append(onboardArgs, "--skip-health")
		}
		cmd := exec.Command(bin, onboardArgs...)
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return windowsHint(fmt.Errorf("openclaw onboarding failed: %w\n\nTry running: openclaw onboard", err))
		}

		patchDeviceScopes()
	}

	if ensureWebSearchPlugin() {
		registerWebSearchPlugin()
	}

	// When extra args are passed through, run exactly what the user asked for
	// after setup and skip the built-in gateway+TUI convenience flow.
	if len(args) > 0 {
		cmd := exec.Command(bin, args...)
		cmd.Env = openclawEnv()
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return windowsHint(err)
		}
		return nil
	}

	if err := c.runChannelSetupPreflight(bin); err != nil {
		return err
	}
	// Keep local pairing scopes up to date before the gateway lifecycle
	// (restart/start) regardless of channel preflight branch behavior.
	patchDeviceScopes()

	fmt.Fprintf(os.Stderr, "\n%sStarting your assistant — this may take a moment...%s\n\n", ansiGray, ansiReset)

	token, port := c.gatewayInfo()
	addr := fmt.Sprintf("localhost:%d", port)

	// If the gateway is already running (e.g. via the daemon), restart it
	// so it picks up any config changes (model, provider, etc.).
	if portOpen(addr) {
		restart := exec.Command(bin, "daemon", "restart")
		restart.Env = openclawEnv()
		if err := restart.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "%s  Warning: daemon restart failed: %v%s\n", ansiYellow, err, ansiReset)
		}
		if !waitForPort(addr, 10*time.Second) {
			fmt.Fprintf(os.Stderr, "%s  Warning: gateway did not come back after restart%s\n", ansiYellow, ansiReset)
		}
	}

	// If the gateway isn't running, start it as a background child process.
	if !portOpen(addr) {
		gw := exec.Command(bin, "gateway", "run", "--force")
		gw.Env = openclawEnv()
		if err := gw.Start(); err != nil {
			return windowsHint(fmt.Errorf("failed to start gateway: %w", err))
		}
		defer func() {
			if gw.Process != nil {
				_ = gw.Process.Kill()
				_ = gw.Wait()
			}
		}()
	}

	fmt.Fprintf(os.Stderr, "%sStarting gateway...%s\n", ansiGray, ansiReset)
	if !waitForPort(addr, 30*time.Second) {
		return windowsHint(fmt.Errorf("gateway did not start on %s", addr))
	}

	printOpenclawReady(bin, token, port, firstLaunch)

	tuiArgs := []string{"tui"}
	if firstLaunch {
		tuiArgs = append(tuiArgs, "--message", "Wake up, my friend!")
	}
	tui := exec.Command(bin, tuiArgs...)
	tui.Env = openclawEnv()
	tui.Stdin = os.Stdin
	tui.Stdout = os.Stdout
	tui.Stderr = os.Stderr
	if err := tui.Run(); err != nil {
		return windowsHint(err)
	}

	return nil
}

// runChannelSetupPreflight prompts users to connect a messaging channel before
// starting the built-in gateway+TUI flow. In interactive sessions, it loops
// until a channel is configured, unless the user chooses "Set up later".
func (c *Openclaw) runChannelSetupPreflight(bin string) error {
	if !isInteractiveSession() {
		return nil
	}
	// --yes is headless; channel setup spawns an interactive picker we can't
	// auto-answer, so skip it. Users can run `openclaw channels add` later.
	if currentLaunchConfirmPolicy.yes {
		return nil
	}

	for {
		if c.channelsConfigured() {
			return nil
		}

		fmt.Fprintf(os.Stderr, "\nYour assistant can message you on WhatsApp, Telegram, Discord, and more.\n\n")
		ok, err := ConfirmPromptWithOptions("Connect a channel (messaging app) now?", ConfirmOptions{
			YesLabel: "Yes",
			NoLabel:  "Set up later",
		})
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}

		cmd := exec.Command(bin, "channels", "add")
		cmd.Env = openclawEnv()
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return windowsHint(fmt.Errorf("openclaw channel setup failed: %w\n\nTry running: %s channels add", err, bin))
		}
	}
}

// channelsConfigured reports whether local OpenClaw config contains at least
// one meaningfully configured channel entry.
func (c *Openclaw) channelsConfigured() bool {
	home, err := os.UserHomeDir()
	if err != nil {
		return false
	}

	for _, path := range []string{
		filepath.Join(home, ".openclaw", "openclaw.json"),
		filepath.Join(home, ".clawdbot", "clawdbot.json"),
	} {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}

		var cfg map[string]any
		if json.Unmarshal(data, &cfg) != nil {
			continue
		}

		channels, _ := cfg["channels"].(map[string]any)
		if channels == nil {
			return false
		}

		for key, value := range channels {
			if key == "defaults" || key == "modelByChannel" {
				continue
			}
			entry, ok := value.(map[string]any)
			if !ok {
				continue
			}
			for entryKey := range entry {
				if entryKey != "enabled" {
					return true
				}
			}
		}
		return false
	}

	return false
}

// gatewayInfo reads the gateway auth token and port from the OpenClaw config.
func (c *Openclaw) gatewayInfo() (token string, port int) {
	port = defaultGatewayPort
	home, err := os.UserHomeDir()
	if err != nil {
		return "", port
	}

	for _, path := range []string{
		filepath.Join(home, ".openclaw", "openclaw.json"),
		filepath.Join(home, ".clawdbot", "clawdbot.json"),
	} {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var config map[string]any
		if json.Unmarshal(data, &config) != nil {
			continue
		}
		gw, _ := config["gateway"].(map[string]any)
		if p, ok := gw["port"].(float64); ok && p > 0 {
			port = int(p)
		}
		auth, _ := gw["auth"].(map[string]any)
		if t, _ := auth["token"].(string); t != "" {
			token = t
		}
		return token, port
	}
	return "", port
}

func printOpenclawReady(bin, token string, port int, firstLaunch bool) {
	u := fmt.Sprintf("http://localhost:%d", port)
	if token != "" {
		u += "/#token=" + url.QueryEscape(token)
	}

	fmt.Fprintf(os.Stderr, "\n%s✓ OpenClaw is running%s\n\n", ansiGreen, ansiReset)
	fmt.Fprintf(os.Stderr, "  Open the Web UI:\n")
	fmt.Fprintf(os.Stderr, "    %s\n\n", hyperlink(u, u))

	if firstLaunch {
		fmt.Fprintf(os.Stderr, "%s  Quick start:%s\n", ansiBold, ansiReset)
		fmt.Fprintf(os.Stderr, "%s    /help             see all commands%s\n", ansiGray, ansiReset)
		fmt.Fprintf(os.Stderr, "%s    %s skills                         browse and install skills%s\n\n", ansiGray, bin, ansiReset)
		fmt.Fprintf(os.Stderr, "%s  The OpenClaw gateway is running in the background.%s\n", ansiYellow, ansiReset)
		fmt.Fprintf(os.Stderr, "%s  Stop it with: %s gateway stop%s\n\n", ansiYellow, bin, ansiReset)
	}
}

// openclawEnv returns the current environment with provider API keys cleared
// so openclaw only uses the Ollama gateway, not keys from the user's shell.
func openclawEnv() []string {
	clear := map[string]bool{
		"ANTHROPIC_API_KEY":     true,
		"ANTHROPIC_OAUTH_TOKEN": true,
		"OPENAI_API_KEY":        true,
		"GEMINI_API_KEY":        true,
		"MISTRAL_API_KEY":       true,
		"GROQ_API_KEY":          true,
		"XAI_API_KEY":           true,
		"OPENROUTER_API_KEY":    true,
	}
	var env []string
	for _, e := range os.Environ() {
		key, _, _ := strings.Cut(e, "=")
		if !clear[key] {
			env = append(env, e)
		}
	}
	return env
}

// portOpen checks if a TCP port is currently accepting connections.
func portOpen(addr string) bool {
	conn, err := net.DialTimeout("tcp", addr, 500*time.Millisecond)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

func waitForPort(addr string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 500*time.Millisecond)
		if err == nil {
			conn.Close()
			return true
		}
		time.Sleep(250 * time.Millisecond)
	}
	return false
}

func windowsHint(err error) error {
	if runtime.GOOS != "windows" {
		return err
	}
	return fmt.Errorf("%w\n\n"+
		"OpenClaw runs best on WSL2.\n"+
		"Quick setup: wsl --install\n"+
		"Guide: https://docs.openclaw.ai/windows", err)
}

// onboarded checks if OpenClaw onboarding wizard was completed
// by looking for the wizard.lastRunAt marker in the config
func (c *Openclaw) onboarded() bool {
	home, err := os.UserHomeDir()
	if err != nil {
		return false
	}

	configPath := filepath.Join(home, ".openclaw", "openclaw.json")
	legacyPath := filepath.Join(home, ".clawdbot", "clawdbot.json")

	config := make(map[string]any)
	if data, err := os.ReadFile(configPath); err == nil {
		_ = json.Unmarshal(data, &config)
	} else if data, err := os.ReadFile(legacyPath); err == nil {
		_ = json.Unmarshal(data, &config)
	} else {
		return false
	}

	// Check for wizard.lastRunAt marker (set when onboarding completes)
	wizard, _ := config["wizard"].(map[string]any)
	if wizard == nil {
		return false
	}
	lastRunAt, _ := wizard["lastRunAt"].(string)
	return lastRunAt != ""
}

// patchDeviceScopes upgrades the local CLI device's paired operator scopes so
// newer gateway auth baselines (approvedScopes) allow launch+TUI reconnects
// without forcing an interactive re-pair. Only patches the local device,
// not remote ones. Best-effort: silently returns on any error.
func patchDeviceScopes() {
	home, err := os.UserHomeDir()
	if err != nil {
		return
	}

	deviceID := readLocalDeviceID(home)
	if deviceID == "" {
		return
	}

	path := filepath.Join(home, ".openclaw", "devices", "paired.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}

	var devices map[string]map[string]any
	if err := json.Unmarshal(data, &devices); err != nil {
		return
	}

	dev, ok := devices[deviceID]
	if !ok {
		return
	}

	required := []string{
		"operator.read",
		"operator.admin",
		"operator.approvals",
		"operator.pairing",
	}

	changed := patchScopes(dev, "scopes", required)
	if patchScopes(dev, "approvedScopes", required) {
		changed = true
	}
	if tokens, ok := dev["tokens"].(map[string]any); ok {
		for role, tok := range tokens {
			if tokenMap, ok := tok.(map[string]any); ok {
				if !isOperatorToken(role, tokenMap) {
					continue
				}
				if patchScopes(tokenMap, "scopes", required) {
					changed = true
				}
			}
		}
	}

	if !changed {
		return
	}

	out, err := json.MarshalIndent(devices, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(path, out, 0o600)
}

// readLocalDeviceID reads the local device ID from openclaw's identity file.
func readLocalDeviceID(home string) string {
	data, err := os.ReadFile(filepath.Join(home, ".openclaw", "identity", "device-auth.json"))
	if err != nil {
		return ""
	}
	var auth map[string]any
	if err := json.Unmarshal(data, &auth); err != nil {
		return ""
	}
	id, _ := auth["deviceId"].(string)
	return id
}

// patchScopes ensures obj[key] contains all required scopes. Returns true if
// any scopes were added.
func patchScopes(obj map[string]any, key string, required []string) bool {
	existing, _ := obj[key].([]any)
	have := make(map[string]bool, len(existing))
	for _, s := range existing {
		if str, ok := s.(string); ok {
			have[str] = true
		}
	}
	added := false
	for _, s := range required {
		if !have[s] {
			existing = append(existing, s)
			added = true
		}
	}
	if added {
		obj[key] = existing
	}
	return added
}

func isOperatorToken(tokenRole string, token map[string]any) bool {
	if strings.EqualFold(strings.TrimSpace(tokenRole), "operator") {
		return true
	}
	role, _ := token["role"].(string)
	return strings.EqualFold(strings.TrimSpace(role), "operator")
}

// canInstallDaemon reports whether the openclaw daemon can be installed as a
// background service. Returns false on Linux when systemd is absent (e.g.
// containers) so that --install-daemon is omitted and the gateway is started
// as a foreground child process instead. Returns true in all other cases.
func canInstallDaemon() bool {
	if runtime.GOOS != "linux" {
		return true
	}
	// /run/systemd/system exists as a directory when systemd is the init system.
	// This is absent in most containers.
	fi, err := os.Stat("/run/systemd/system")
	if err != nil || !fi.IsDir() {
		return false
	}
	// Even when systemd is the init system, user services require a user
	// manager instance. XDG_RUNTIME_DIR being set is a prerequisite.
	return os.Getenv("XDG_RUNTIME_DIR") != ""
}

func ensureOpenclawInstalled() (string, error) {
	if _, err := exec.LookPath("openclaw"); err == nil {
		return "openclaw", nil
	}
	if _, err := exec.LookPath("clawdbot"); err == nil {
		return "clawdbot", nil
	}

	_, npmErr := exec.LookPath("npm")
	_, gitErr := exec.LookPath("git")
	if npmErr != nil || gitErr != nil {
		var missing []string
		if npmErr != nil {
			missing = append(missing, "npm (Node.js): https://nodejs.org/")
		}
		if gitErr != nil {
			missing = append(missing, "git: https://git-scm.com/")
		}
		return "", fmt.Errorf("OpenClaw is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch openclaw", strings.Join(missing, "\n  "))
	}

	ok, err := ConfirmPrompt("OpenClaw is not installed. Install with npm?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("openclaw installation cancelled")
	}

	fmt.Fprintf(os.Stderr, "\nInstalling OpenClaw...\n")
	cmd := exec.Command("npm", "install", "-g", "openclaw@latest")
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install openclaw: %w", err)
	}

	if _, err := exec.LookPath("openclaw"); err != nil {
		return "", fmt.Errorf("openclaw was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sOpenClaw installed successfully%s\n\n", ansiGreen, ansiReset)
	openclawFreshInstall = true
	return "openclaw", nil
}

func (c *Openclaw) Paths() []string {
	home, _ := os.UserHomeDir()
	p := filepath.Join(home, ".openclaw", "openclaw.json")
	if _, err := os.Stat(p); err == nil {
		return []string{p}
	}
	legacy := filepath.Join(home, ".clawdbot", "clawdbot.json")
	if _, err := os.Stat(legacy); err == nil {
		return []string{legacy}
	}
	return nil
}

func (c *Openclaw) Edit(models []string) error {
	if len(models) == 0 {
		return nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	configPath := filepath.Join(home, ".openclaw", "openclaw.json")
	legacyPath := filepath.Join(home, ".clawdbot", "clawdbot.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}

	// Read into map[string]any to preserve unknown fields
	config := make(map[string]any)
	if data, err := os.ReadFile(configPath); err == nil {
		_ = json.Unmarshal(data, &config)
	} else if data, err := os.ReadFile(legacyPath); err == nil {
		_ = json.Unmarshal(data, &config)
	}

	// Navigate/create: models.providers.ollama (preserving other providers)
	modelsSection, _ := config["models"].(map[string]any)
	if modelsSection == nil {
		modelsSection = make(map[string]any)
	}
	providers, _ := modelsSection["providers"].(map[string]any)
	if providers == nil {
		providers = make(map[string]any)
	}
	ollama, _ := providers["ollama"].(map[string]any)
	if ollama == nil {
		ollama = make(map[string]any)
	}

	ollama["baseUrl"] = envconfig.Host().String()
	// needed to register provider
	ollama["apiKey"] = "ollama-local"
	ollama["api"] = "ollama"

	// Build map of existing models to preserve user customizations
	existingModels, _ := ollama["models"].([]any)
	existingByID := make(map[string]map[string]any)
	for _, m := range existingModels {
		if entry, ok := m.(map[string]any); ok {
			if id, ok := entry["id"].(string); ok {
				existingByID[id] = entry
			}
		}
	}

	client, _ := api.ClientFromEnvironment()

	var newModels []any
	for _, m := range models {
		entry, _ := openclawModelConfig(context.Background(), client, m)
		// Merge existing fields (user customizations)
		if existing, ok := existingByID[m]; ok {
			for k, v := range existing {
				if _, isNew := entry[k]; !isNew {
					entry[k] = v
				}
			}
		}
		newModels = append(newModels, entry)
	}
	ollama["models"] = newModels

	providers["ollama"] = ollama
	modelsSection["providers"] = providers
	config["models"] = modelsSection

	// Update agents.defaults.model.primary (preserving other agent settings)
	agents, _ := config["agents"].(map[string]any)
	if agents == nil {
		agents = make(map[string]any)
	}
	defaults, _ := agents["defaults"].(map[string]any)
	if defaults == nil {
		defaults = make(map[string]any)
	}
	modelConfig, _ := defaults["model"].(map[string]any)
	if modelConfig == nil {
		modelConfig = make(map[string]any)
	}
	modelConfig["primary"] = "ollama/" + models[0]
	defaults["model"] = modelConfig
	agents["defaults"] = defaults
	config["agents"] = agents

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	if err := fileutil.WriteWithBackup(configPath, data); err != nil {
		return err
	}

	// Clear any per-session model overrides so the new primary takes effect
	// immediately rather than being shadowed by a cached modelOverride.
	clearSessionModelOverride(models[0])
	return nil
}

// clearSessionModelOverride removes per-session model overrides from the main
// agent session so the global primary model takes effect on the next TUI launch.
func clearSessionModelOverride(primary string) {
	home, err := os.UserHomeDir()
	if err != nil {
		return
	}
	path := filepath.Join(home, ".openclaw", "agents", "main", "sessions", "sessions.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var sessions map[string]map[string]any
	if json.Unmarshal(data, &sessions) != nil {
		return
	}
	changed := false
	for _, sess := range sessions {
		if override, _ := sess["modelOverride"].(string); override != "" && override != primary {
			delete(sess, "modelOverride")
			delete(sess, "providerOverride")
		}
		if model, _ := sess["model"].(string); model != "" && model != primary {
			sess["model"] = primary
			changed = true
		}
	}
	if !changed {
		return
	}
	out, err := json.MarshalIndent(sessions, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(path, out, 0o600)
}

const (
	webSearchNpmPackage = "@ollama/openclaw-web-search"
	webSearchMinVersion = "0.2.1"
)

// ensureWebSearchPlugin installs the openclaw-web-search extension into the
// user-level extensions directory (~/.openclaw/extensions/) if it isn't already
// present, or re-installs if the installed version is older than webSearchMinVersion.
// Returns true if the extension is available.
func ensureWebSearchPlugin() bool {
	home, err := os.UserHomeDir()
	if err != nil {
		return false
	}

	pluginDir := filepath.Join(home, ".openclaw", "extensions", "openclaw-web-search")
	if webSearchPluginUpToDate(pluginDir) {
		return true
	}

	npmBin, err := exec.LookPath("npm")
	if err != nil {
		return false
	}

	if err := os.MkdirAll(pluginDir, 0o755); err != nil {
		return false
	}

	// Download the tarball via `npm pack`, extract it flat into the plugin dir.
	pack := exec.Command(npmBin, "pack", webSearchNpmPackage, "--pack-destination", pluginDir)
	out, err := pack.Output()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s  Warning: could not download web search plugin: %v%s\n", ansiYellow, err, ansiReset)
		return false
	}

	tgzName := strings.TrimSpace(string(out))
	tgzPath := filepath.Join(pluginDir, tgzName)
	defer os.Remove(tgzPath)

	tar := exec.Command("tar", "xzf", tgzPath, "--strip-components=1", "-C", pluginDir)
	if err := tar.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "%s  Warning: could not extract web search plugin: %v%s\n", ansiYellow, err, ansiReset)
		return false
	}

	fmt.Fprintf(os.Stderr, "%s  ✓ Installed Ollama web search %s\n", ansiGreen, ansiReset)
	return true
}

// webSearchPluginUpToDate returns true if the plugin is installed and its
// package.json version is >= webSearchMinVersion.
func webSearchPluginUpToDate(pluginDir string) bool {
	data, err := os.ReadFile(filepath.Join(pluginDir, "package.json"))
	if err != nil {
		return false
	}
	var pkg struct {
		Version string `json:"version"`
	}
	if json.Unmarshal(data, &pkg) != nil || pkg.Version == "" {
		return false
	}
	return !versionLessThan(pkg.Version, webSearchMinVersion)
}

// versionLessThan compares two semver version strings (major.minor.patch).
// Inputs may omit the "v" prefix; it is added automatically for semver.Compare.
func versionLessThan(a, b string) bool {
	if !strings.HasPrefix(a, "v") {
		a = "v" + a
	}
	if !strings.HasPrefix(b, "v") {
		b = "v" + b
	}
	return semver.Compare(a, b) < 0
}

// registerWebSearchPlugin adds plugins.entries.openclaw-web-search to the OpenClaw
// config so the gateway activates it on next start. Best-effort; silently returns
// on any error.
func registerWebSearchPlugin() {
	home, err := os.UserHomeDir()
	if err != nil {
		return
	}
	configPath := filepath.Join(home, ".openclaw", "openclaw.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return
	}
	var config map[string]any
	if json.Unmarshal(data, &config) != nil {
		return
	}

	plugins, _ := config["plugins"].(map[string]any)
	if plugins == nil {
		plugins = make(map[string]any)
	}
	entries, _ := plugins["entries"].(map[string]any)
	if entries == nil {
		entries = make(map[string]any)
	}
	entries["openclaw-web-search"] = map[string]any{"enabled": true}
	plugins["entries"] = entries

	// Pin trust so the gateway doesn't warn about untracked plugins.
	allow, _ := plugins["allow"].([]any)
	hasAllow := false
	for _, v := range allow {
		if s, ok := v.(string); ok && s == "openclaw-web-search" {
			hasAllow = true
			break
		}
	}
	if !hasAllow {
		allow = append(allow, "openclaw-web-search")
	}
	plugins["allow"] = allow

	// Record install provenance so the loader can verify the plugin origin.
	installs, _ := plugins["installs"].(map[string]any)
	if installs == nil {
		installs = make(map[string]any)
	}
	pluginDir := filepath.Join(home, ".openclaw", "extensions", "openclaw-web-search")
	installs["openclaw-web-search"] = map[string]any{
		"source":      "npm",
		"spec":        webSearchNpmPackage,
		"installPath": pluginDir,
	}
	plugins["installs"] = installs

	config["plugins"] = plugins

	// Add plugin tools to tools.alsoAllow so they survive the coding profile's
	// policy pipeline (which has an explicit allow list of core tools only).
	tools, _ := config["tools"].(map[string]any)
	if tools == nil {
		tools = make(map[string]any)
	}

	alsoAllow, _ := tools["alsoAllow"].([]any)
	needed := []string{"ollama_web_search", "ollama_web_fetch"}
	have := make(map[string]bool, len(alsoAllow))
	for _, v := range alsoAllow {
		if s, ok := v.(string); ok {
			have[s] = true
		}
	}
	for _, name := range needed {
		if !have[name] {
			alsoAllow = append(alsoAllow, name)
		}
	}
	tools["alsoAllow"] = alsoAllow

	// Disable built-in web search/fetch since our plugin replaces them.
	web, _ := tools["web"].(map[string]any)
	if web == nil {
		web = make(map[string]any)
	}
	web["search"] = map[string]any{"enabled": false}
	web["fetch"] = map[string]any{"enabled": false}
	tools["web"] = web
	config["tools"] = tools

	out, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(configPath, out, 0o600)
}

// openclawModelConfig builds an OpenClaw model config entry with capability detection.
// The second return value indicates whether the model is a cloud (remote) model.
func openclawModelConfig(ctx context.Context, client *api.Client, modelID string) (map[string]any, bool) {
	entry := map[string]any{
		"id":    modelID,
		"name":  modelID,
		"input": []any{"text"},
		"cost": map[string]any{
			"input":      0,
			"output":     0,
			"cacheRead":  0,
			"cacheWrite": 0,
		},
	}

	if client == nil {
		return entry, false
	}

	showCtx := ctx
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		var cancel context.CancelFunc
		showCtx, cancel = context.WithTimeout(ctx, openclawModelShowTimeout)
		defer cancel()
	}

	resp, err := client.Show(showCtx, &api.ShowRequest{Model: modelID})
	if err != nil {
		return entry, false
	}

	// Set input types based on vision capability
	if slices.Contains(resp.Capabilities, model.CapabilityVision) {
		entry["input"] = []any{"text", "image"}
	}

	// Set reasoning based on thinking capability
	if slices.Contains(resp.Capabilities, model.CapabilityThinking) {
		entry["reasoning"] = true
	}

	// Cloud models: use hardcoded limits for context/output tokens.
	// Capability detection above still applies (vision, thinking).
	if resp.RemoteModel != "" {
		if l, ok := lookupCloudModelLimit(modelID); ok {
			entry["contextWindow"] = l.Context
			entry["maxTokens"] = l.Output
		}
		return entry, true
	}

	// Extract context window from ModelInfo (local models only)
	for key, val := range resp.ModelInfo {
		if strings.HasSuffix(key, ".context_length") {
			if ctxLen, ok := val.(float64); ok && ctxLen > 0 {
				entry["contextWindow"] = int(ctxLen)
			}
			break
		}
	}

	return entry, false
}

func (c *Openclaw) Models() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	config, err := fileutil.ReadJSON(filepath.Join(home, ".openclaw", "openclaw.json"))
	if err != nil {
		config, err = fileutil.ReadJSON(filepath.Join(home, ".clawdbot", "clawdbot.json"))
		if err != nil {
			return nil
		}
	}

	modelsSection, _ := config["models"].(map[string]any)
	providers, _ := modelsSection["providers"].(map[string]any)
	ollama, _ := providers["ollama"].(map[string]any)
	modelList, _ := ollama["models"].([]any)

	var result []string
	for _, m := range modelList {
		if entry, ok := m.(map[string]any); ok {
			if id, ok := entry["id"].(string); ok {
				result = append(result, id)
			}
		}
	}
	return result
}
