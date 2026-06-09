package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
)

// Pi implements Runner and Editor for Pi (Pi Coding Agent) integration
type Pi struct{}

const (
	piNpmPackage       = "@earendil-works/pi-coding-agent"
	piLegacyNpmPackage = "@mariozechner/pi-coding-agent"
	piWebSearchSource  = "npm:@ollama/pi-web-search"
	piWebSearchPkg     = "@ollama/pi-web-search"
)

func (p *Pi) String() string { return "Pi" }

var npmRegistryBaseURL = "https://registry.npmjs.org"

func (p *Pi) Run(_ string, _ []LaunchModel, args []string) error {
	fmt.Fprintf(os.Stderr, "\n%sPreparing Pi...%s\n", ansiGray, ansiReset)
	if err := ensureNpmInstalled(); err != nil {
		return err
	}

	fmt.Fprintf(os.Stderr, "%sChecking Pi installation...%s\n", ansiGray, ansiReset)
	bin, err := ensurePiInstalled()
	if err != nil {
		return err
	}

	ensurePiWebSearchPackage(bin)

	fmt.Fprintf(os.Stderr, "\n%sLaunching Pi...%s\n\n", ansiGray, ansiReset)

	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func ensureNpmInstalled() error {
	if _, err := exec.LookPath("npm"); err != nil {
		return fmt.Errorf("npm (Node.js) is required to launch pi\n\nInstall it first:\n  https://nodejs.org/\n\nThen re-run:\n  ollama launch pi")
	}
	return nil
}

func ensurePiInstalled() (string, error) {
	if _, err := exec.LookPath("pi"); err == nil {
		install, pkgErr := installedPiPackageInfo()
		if pkgErr != nil {
			fmt.Fprintf(os.Stderr, "%sCould not verify which Pi package is installed: %v%s\n", ansiYellow, pkgErr, ansiReset)
			fmt.Fprintf(os.Stderr, "Pi will still launch. To switch to the official package manually:\n  npm uninstall -g %s\n  npm install -g %s\n\n", piLegacyNpmPackage, piNpmPackage)
			return "pi", nil
		}

		if install.packageName == piLegacyNpmPackage {
			fmt.Fprintf(os.Stderr, "%sUpdating Pi...%s\n", ansiGray, ansiReset)
			if err := migrateLegacyPiPackage(install.npmPrefix); err != nil {
				return "", err
			}
			if err := requirePiOnPath(); err != nil {
				return "", err
			}
		}
		return "pi", nil
	}

	if _, err := exec.LookPath("npm"); err != nil {
		return "", fmt.Errorf("pi is not installed and required dependencies are missing\n\nInstall the following first:\n  npm (Node.js): https://nodejs.org/\n\nThen re-run:\n  ollama launch pi")
	}

	install, pkgErr := installedPiPackageInfo()
	if pkgErr == nil && install.packageName == piLegacyNpmPackage {
		fmt.Fprintf(os.Stderr, "%sUpdating Pi...%s\n", ansiGray, ansiReset)
		if err := migrateLegacyPiPackage(install.npmPrefix); err != nil {
			return "", err
		}
		if err := requirePiOnPath(); err != nil {
			return "", err
		}
		return "pi", nil
	}
	if pkgErr == nil && install.packageName == piNpmPackage {
		fmt.Fprintf(os.Stderr, "%sInstalling Pi...%s\n", ansiGray, ansiReset)
		if err := installPiPackageWithPrefix(install.npmPrefix); err != nil {
			return "", err
		}
		if err := requirePiOnPath(); err != nil {
			return "", err
		}
		return "pi", nil
	}

	ok, err := ConfirmPrompt("Install Pi with npm?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("pi installation cancelled")
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Pi...\n")
	if err := installPiPackage(); err != nil {
		return "", err
	}

	if err := requirePiOnPath(); err != nil {
		return "", err
	}

	fmt.Fprintf(os.Stderr, "%sPi installed successfully%s\n\n", ansiGreen, ansiReset)
	return "pi", nil
}

func requirePiOnPath() error {
	if _, err := exec.LookPath("pi"); err != nil {
		return fmt.Errorf("pi was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}
	return nil
}

func installPiPackage() error {
	return installPiPackageWithPrefix("")
}

func installPiPackageWithPrefix(prefix string) error {
	if err := runQuietCommand("npm", npmArgs(prefix, "install", "-g", piNpmPackage+"@latest")...); err != nil {
		return fmt.Errorf("failed to install pi: %w", err)
	}
	return nil
}

func migrateLegacyPiPackage(prefix string) error {
	if err := installPiPackageForced(prefix); err != nil {
		return err
	}

	installed, err := npmPackageInstalledWithPrefix(piNpmPackage, prefix)
	if err != nil {
		return fmt.Errorf("failed to verify official pi package: %w", err)
	}
	if !installed {
		return fmt.Errorf("failed to verify official pi package")
	}

	if err := uninstallLegacyPiPackageWithPrefix(prefix); err != nil {
		return err
	}
	return installPiPackageWithPrefix(prefix)
}

func installPiPackageForced(prefix string) error {
	if err := runQuietCommand("npm", npmArgs(prefix, "install", "-g", piNpmPackage+"@latest", "--force")...); err != nil {
		return fmt.Errorf("failed to install pi: %w", err)
	}
	return nil
}

func uninstallLegacyPiPackageWithPrefix(prefix string) error {
	if err := runQuietCommand("npm", npmArgs(prefix, "uninstall", "-g", piLegacyNpmPackage)...); err != nil {
		return fmt.Errorf("failed to remove legacy pi package: %w", err)
	}
	return nil
}

func runQuietCommand(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	out, err := cmd.CombinedOutput()
	if err == nil {
		return nil
	}
	msg := strings.TrimSpace(string(out))
	if msg == "" {
		return err
	}
	return fmt.Errorf("%w: %s", err, msg)
}

type piPackageInstall struct {
	packageName string
	npmPrefix   string
}

func installedPiPackageInfo() (piPackageInstall, error) {
	if _, err := exec.LookPath("npm"); err != nil {
		return piPackageInstall{}, err
	}

	if bin, err := exec.LookPath("pi"); err == nil {
		install, err := piPackageInstallFromBinary(bin)
		if err == nil && install.packageName != "" {
			return install, nil
		}
	}

	installed, err := npmPackageInstalled(piLegacyNpmPackage)
	if err != nil {
		return piPackageInstall{}, err
	}
	if installed {
		return piPackageInstall{packageName: piLegacyNpmPackage}, nil
	}

	installed, err = npmPackageInstalled(piNpmPackage)
	if err != nil {
		return piPackageInstall{}, err
	}
	if installed {
		return piPackageInstall{packageName: piNpmPackage}, nil
	}

	return piPackageInstall{}, nil
}

func piPackageInstallFromBinary(bin string) (piPackageInstall, error) {
	realPath, err := filepath.EvalSymlinks(bin)
	if err != nil {
		realPath = bin
	}

	dir := filepath.Dir(realPath)
	for {
		packageJSON := filepath.Join(dir, "package.json")
		data, err := os.ReadFile(packageJSON)
		if err == nil {
			var payload struct {
				Name string `json:"name"`
			}
			if json.Unmarshal(data, &payload) == nil && (payload.Name == piLegacyNpmPackage || payload.Name == piNpmPackage) {
				return piPackageInstall{packageName: payload.Name, npmPrefix: npmPrefixForPackageRoot(dir)}, nil
			}
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	return piPackageInstall{}, nil
}

func npmPrefixForPackageRoot(packageRoot string) string {
	return npmPrefixForPackageRootForGOOS(filepath.Clean(packageRoot), runtime.GOOS, string(filepath.Separator))
}

func npmPrefixForPackageRootForGOOS(packageRoot, goos, separator string) string {
	packageRoot = strings.TrimRight(packageRoot, separator)
	nodeModules := separator + "node_modules" + separator
	idx := strings.LastIndex(packageRoot, nodeModules)
	if idx == -1 {
		return ""
	}

	rootDir := packageRoot[:idx]
	if pathBaseForSeparator(rootDir, separator) == "lib" {
		// Unix npm global root is <prefix>/lib/node_modules.
		return pathDirForSeparator(rootDir, separator)
	}
	if goos == "windows" {
		// Windows npm global root is usually <prefix>\node_modules.
		return rootDir
	}
	return ""
}

func pathBaseForSeparator(path, separator string) string {
	path = strings.TrimRight(path, separator)
	idx := strings.LastIndex(path, separator)
	if idx == -1 {
		return path
	}
	return path[idx+len(separator):]
}

func pathDirForSeparator(path, separator string) string {
	path = strings.TrimRight(path, separator)
	idx := strings.LastIndex(path, separator)
	if idx == -1 {
		return ""
	}
	if idx == 0 {
		return separator
	}
	return path[:idx]
}

func npmPackageInstalled(pkg string) (bool, error) {
	return npmPackageInstalledWithPrefix(pkg, "")
}

func npmPackageInstalledWithPrefix(pkg, prefix string) (bool, error) {
	cmd := exec.Command("npm", npmArgs(prefix, "ls", "-g", pkg, "--depth=0", "--json")...)
	out, err := cmd.Output()

	var payload struct {
		Dependencies map[string]json.RawMessage `json:"dependencies"`
	}

	if parseErr := json.Unmarshal(out, &payload); parseErr == nil {
		_, ok := payload.Dependencies[pkg]
		if ok {
			return true, nil
		}
		return false, nil
	}

	if err == nil {
		return false, nil
	}

	if exitErr, ok := err.(*exec.ExitError); ok {
		msg := strings.TrimSpace(string(exitErr.Stderr))
		if msg == "" {
			msg = strings.TrimSpace(string(out))
		}
		if msg == "" {
			return false, err
		}
		return false, fmt.Errorf("%w: %s", err, msg)
	}

	return false, err
}

func npmArgs(prefix string, args ...string) []string {
	if prefix == "" {
		return args
	}
	return append([]string{"--prefix", prefix}, args...)
}

func ensurePiWebSearchPackage(bin string) {
	if !shouldManageOllamaWebSearch() {
		fmt.Fprintf(os.Stderr, "%sCloud is disabled; skipping %s setup.%s\n", ansiGray, piWebSearchPkg, ansiReset)
		return
	}

	fmt.Fprintf(os.Stderr, "%sChecking Pi web search package...%s\n", ansiGray, ansiReset)

	pkg, err := piPackageInfo(bin, piWebSearchSource)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s  Warning: could not check %s installation: %v%s\n", ansiYellow, piWebSearchPkg, err, ansiReset)
		return
	}

	if !pkg.installed {
		fmt.Fprintf(os.Stderr, "%sInstalling %s...%s\n", ansiGray, piWebSearchPkg, ansiReset)
		cmd := exec.Command(bin, "install", piWebSearchSource)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "%s  Warning: could not install %s: %v%s\n", ansiYellow, piWebSearchPkg, err, ansiReset)
			return
		}

		fmt.Fprintf(os.Stderr, "%s  ✓ Installed %s%s\n", ansiGreen, piWebSearchPkg, ansiReset)
		return
	}

	updateAvailable, err := piWebSearchUpdateAvailable(pkg.installedPath)
	if err != nil || !updateAvailable {
		return
	}

	fmt.Fprintf(os.Stderr, "%sUpdating %s...%s\n", ansiGray, piWebSearchPkg, ansiReset)
	cmd := exec.Command(bin, "update", piWebSearchSource)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "%s  Warning: could not update %s: %v%s\n", ansiYellow, piWebSearchPkg, err, ansiReset)
		return
	}

	fmt.Fprintf(os.Stderr, "%s  ✓ Updated %s%s\n", ansiGreen, piWebSearchPkg, ansiReset)
}

func shouldManageOllamaWebSearch() bool {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return true
	}

	disabled, known := cloudStatusDisabled(context.Background(), client)
	if known && disabled {
		return false
	}
	return true
}

type piPackageListEntry struct {
	installed     bool
	installedPath string
}

func piPackageInfo(bin, source string) (piPackageListEntry, error) {
	cmd := exec.Command(bin, "list")
	out, err := cmd.CombinedOutput()
	if err != nil {
		msg := strings.TrimSpace(string(out))
		if msg == "" {
			return piPackageListEntry{}, err
		}
		return piPackageListEntry{}, fmt.Errorf("%w: %s", err, msg)
	}

	lines := strings.Split(string(out), "\n")
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, source) {
			return piPackageListEntry{installed: true, installedPath: piPackageListInstalledPath(lines[i+1:])}, nil
		}
	}

	return piPackageListEntry{}, nil
}

func piPackageListInstalledPath(lines []string) string {
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		if strings.HasPrefix(trimmed, "npm:") || strings.HasPrefix(trimmed, "git:") || strings.HasSuffix(trimmed, ":") {
			return ""
		}
		if filepath.IsAbs(trimmed) {
			return trimmed
		}
		return ""
	}
	return ""
}

func piWebSearchUpdateAvailable(installedPath string) (bool, error) {
	if piOfflineModeEnabled() || installedPath == "" {
		return false, nil
	}

	installedVersion, err := npmInstalledPackageVersion(installedPath)
	if err != nil || installedVersion == "" {
		return false, err
	}

	latestVersion, err := npmLatestPackageVersion(piWebSearchPkg)
	if err != nil || latestVersion == "" {
		return false, err
	}

	return latestVersion != installedVersion, nil
}

func piOfflineModeEnabled() bool {
	value := os.Getenv("PI_OFFLINE")
	return value == "1" || strings.EqualFold(value, "true") || strings.EqualFold(value, "yes")
}

func npmInstalledPackageVersion(installedPath string) (string, error) {
	data, err := os.ReadFile(filepath.Join(installedPath, "package.json"))
	if err != nil {
		return "", err
	}

	var payload struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(data, &payload); err != nil {
		return "", err
	}
	return payload.Version, nil
}

func npmLatestPackageVersion(pkg string) (string, error) {
	client := http.Client{Timeout: 10 * time.Second}
	requestURL := strings.TrimRight(npmRegistryBaseURL, "/") + "/" + url.PathEscape(pkg) + "/latest"
	resp, err := client.Get(requestURL)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("npm registry returned %s", resp.Status)
	}

	var payload struct {
		Version string `json:"version"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return "", err
	}
	return payload.Version, nil
}

func (p *Pi) Paths() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	var paths []string
	modelsPath := filepath.Join(home, ".pi", "agent", "models.json")
	if _, err := os.Stat(modelsPath); err == nil {
		paths = append(paths, modelsPath)
	}
	settingsPath := filepath.Join(home, ".pi", "agent", "settings.json")
	if _, err := os.Stat(settingsPath); err == nil {
		paths = append(paths, settingsPath)
	}
	return paths
}

func (p *Pi) Edit(models []LaunchModel) error {
	if len(models) == 0 {
		return nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	configPath := filepath.Join(home, ".pi", "agent", "models.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}

	config := make(map[string]any)
	if data, err := os.ReadFile(configPath); err == nil {
		_ = json.Unmarshal(data, &config)
	}

	providers, ok := config["providers"].(map[string]any)
	if !ok {
		providers = make(map[string]any)
	}

	ollama, ok := providers["ollama"].(map[string]any)
	if !ok {
		ollama = map[string]any{
			"baseUrl": envconfig.Host().String() + "/v1",
			"api":     "openai-completions",
			"apiKey":  "ollama",
		}
	}

	existingModels, ok := ollama["models"].([]any)
	if !ok {
		existingModels = make([]any, 0)
	}

	// Build set of selected models to track which need to be added
	selectedSet := make(map[string]bool, len(models))
	for _, m := range models {
		selectedSet[m.Name] = true
	}

	// Build new models list:
	// 1. Keep user-managed models (no _launch marker) - untouched
	// 2. Keep ollama-managed models (_launch marker) that are still selected,
	//    except stale cloud entries that should be rebuilt below
	// 3. Add new ollama-managed models
	var newModels []any
	for _, m := range existingModels {
		if modelObj, ok := m.(map[string]any); ok {
			if id, ok := modelObj["id"].(string); ok {
				// User-managed model (no _launch marker) - always preserve
				if !isPiOllamaModel(modelObj) {
					newModels = append(newModels, m)
				} else if selectedSet[id] {
					// Rebuild stale managed cloud entries so createConfig refreshes
					// the whole entry instead of patching it in place.
					if !hasContextWindow(modelObj) {
						if _, ok := lookupCloudModelLimit(id); ok {
							continue
						}
					}
					newModels = append(newModels, m)
					selectedSet[id] = false
				}
			}
		}
	}

	// Add newly selected models that weren't already in the list
	for _, model := range models {
		if selectedSet[model.Name] {
			newModels = append(newModels, createConfig(model))
		}
	}

	ollama["models"] = newModels
	providers["ollama"] = ollama
	config["providers"] = providers

	configData, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	if err := fileutil.WriteWithBackup(configPath, configData, "pi"); err != nil {
		return err
	}

	// Update settings.json with default provider and model
	settingsPath := filepath.Join(home, ".pi", "agent", "settings.json")
	settings := make(map[string]any)
	if data, err := os.ReadFile(settingsPath); err == nil {
		_ = json.Unmarshal(data, &settings)
	}

	settings["defaultProvider"] = "ollama"
	settings["defaultModel"] = models[0].Name

	settingsData, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	return fileutil.WriteWithBackup(settingsPath, settingsData, "pi")
}

func (p *Pi) Models() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	configPath := filepath.Join(home, ".pi", "agent", "models.json")
	config, err := fileutil.ReadJSON(configPath)
	if err != nil {
		return nil
	}

	providers, _ := config["providers"].(map[string]any)
	ollama, _ := providers["ollama"].(map[string]any)
	models, _ := ollama["models"].([]any)

	var result []string
	for _, m := range models {
		if modelObj, ok := m.(map[string]any); ok {
			if id, ok := modelObj["id"].(string); ok {
				result = append(result, id)
			}
		}
	}
	slices.Sort(result)
	return result
}

// isPiOllamaModel reports whether a model config entry is managed by ollama launch
func isPiOllamaModel(cfg map[string]any) bool {
	if v, ok := cfg["_launch"].(bool); ok && v {
		return true
	}
	return false
}

func hasContextWindow(cfg map[string]any) bool {
	switch v := cfg["contextWindow"].(type) {
	case float64:
		return v > 0
	case int:
		return v > 0
	case int64:
		return v > 0
	default:
		return false
	}
}

// createConfig builds Pi model config with capability detection.
func createConfig(model LaunchModel) map[string]any {
	cfg := map[string]any{
		"id":      model.Name,
		"_launch": true,
	}

	// Set input types based on vision capability
	if model.HasCapability("vision") {
		cfg["input"] = []string{"text", "image"}
	} else {
		cfg["input"] = []string{"text"}
	}

	// Set reasoning based on thinking capability
	if model.HasCapability("thinking") {
		cfg["reasoning"] = true
	}

	if model.ContextLength > 0 {
		cfg["contextWindow"] = model.ContextLength
	}

	return cfg
}
