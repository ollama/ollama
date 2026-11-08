package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
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

var (
	npmRegistryBaseURL      = "https://registry.npmjs.org"
	piModelDiscoveryTimeout = 10 * time.Second
)

func piHasModelArgs(args []string) bool {
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--provider", "--model":
			return i+1 < len(args)
		case "--mode", "--api-key", "--system-prompt", "--append-system-prompt",
			"--session", "--fork", "--session-dir", "--models", "--tools", "-t",
			"--thinking", "--export", "--extension", "-e", "--skill", "--prompt-template", "--theme":
			// These Pi options consume a value, even when it looks like a flag.
			i++
		}
	}
	return false
}

// launchPiIntegration saves one starting model and synchronizes Pi's catalog
// independently of that choice. Only the starting model needs readiness checks.
func (c *launcherClient) launchPiIntegration(ctx context.Context, pi *Pi, saved *config.IntegrationConfig, req IntegrationLaunchRequest) error {
	current := primaryModelFromConfig(saved)
	if model, err := pi.currentModel(); err != nil {
		return err
	} else if model != "" {
		current = model
	}
	target := req.ModelOverride
	if target == "" && !req.ForceConfigure && !req.ConfigureOnly {
		target = current
	}
	discoveryCtx, cancelDiscovery := context.WithTimeout(ctx, piModelDiscoveryTimeout)
	defer cancelDiscovery()
	if isCloudModelName(target) && !req.ForceConfigure && !req.ConfigureOnly {
		if disabled, _ := cloudStatusDisabled(discoveryCtx, c.apiClient); disabled {
			return fmt.Errorf("cloud is disabled; enable cloud to use %s", target)
		}
	}
	// Resolve an explicit model before spending the discovery budget on the catalog.
	if req.ModelOverride == "" {
		if _, err := c.modelInventory().Load(discoveryCtx); err != nil && !errors.Is(err, context.DeadlineExceeded) {
			return err
		}
	}
	if target == "" {
		target = current
	}
	pick := req.ForceConfigure || req.ConfigureOnly || target == ""
	picked := false
	var models []LaunchModel
	for {
		if pick {
			cancelDiscovery()
			var err error
			target, err = c.selectSingleModelWithSelectorReady(ctx, "Select model for Pi:", target, DefaultSingleSelector, false, "Pi", "pi")
			if err != nil {
				return err
			}
			picked = true
			discoveryCtx, cancelDiscovery = context.WithTimeout(ctx, piModelDiscoveryTimeout)
			defer cancelDiscovery()
		}
		remaining := piModelDiscoveryTimeout
		model, err := c.piTargetModel(discoveryCtx, target, &remaining)
		if errors.Is(err, context.DeadlineExceeded) && ctx.Err() == nil {
			err = nil
		}
		var status api.StatusError
		if errors.As(err, &status) && status.StatusCode == http.StatusNotFound {
			if !picked && req.ModelOverride == "" {
				pick = true
				continue
			}
			if model.Remote {
				return err
			}
			if err := pullMissingModelWithPolicy(ctx, c.apiClient, target, c.policy.missingModelPolicy()); err != nil {
				return err
			}
			model, err = c.piTargetModel(ctx, target, &remaining)
		}
		if err != nil {
			return err
		}
		if !piSupportsChat(model) {
			if !picked && req.ModelOverride == "" {
				pick = true
				continue
			}
			return fmt.Errorf("model %q does not support chat; choose a chat model", target)
		}
		if target != current {
			paymentTarget := target
			if model.RemoteModel != "" {
				paymentTarget = model.RemoteModel
				if !isCloudModelName(paymentTarget) {
					paymentTarget = strings.TrimSuffix(paymentTarget, ":latest") + ":cloud"
				}
			}
			if err := c.confirmCloudModelPayment(discoveryCtx, paymentTarget); err != nil {
				if errors.Is(err, errModelPaymentDeclined) {
					pick = true
					continue
				}
				return err
			}
		}
		cloudRec, localRec := c.agentCapableRecommendations(discoveryCtx)
		err = confirmDeprecatedLaunchModel(target, "Pi", "pi", cloudRec, localRec)
		if errors.Is(err, errDeprecatedLaunchModelDeclined) {
			pick = true
			continue
		}
		if err != nil {
			return err
		}
		if model.Remote {
			if err := ensureCloudAuth(ctx, c.apiClient, target); err != nil {
				return err
			}
			if len(model.Capabilities) == 0 {
				// Authentication may make previously unavailable metadata readable.
				model, err = c.piTargetModel(discoveryCtx, target, &remaining)
				if errors.Is(err, context.DeadlineExceeded) && ctx.Err() == nil {
					err = nil
				}
				if err != nil {
					return err
				}
			}
		}
		models, err = c.piModelCatalog(discoveryCtx, model, remaining)
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		break
	}

	configured, err := pi.writeConfig(models, discoveryCtx.Err() != nil)
	if err != nil {
		return fmt.Errorf("setup failed: %w", err)
	}
	if err := config.SaveIntegration("pi", []string{configured}); err != nil {
		return fmt.Errorf("failed to save: %w", err)
	}
	runtimeModel := ""
	if req.ModelOverride != "" || picked {
		runtimeModel = configured
	}
	return launchAfterConfiguration("pi", pi, runtimeModel, models, req)
}

func (p *Pi) currentModel() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	path := filepath.Join(home, ".pi", "agent", "settings.json")
	settings, err := fileutil.ReadJSON(path)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("read Pi config %s: %w", path, err)
	}
	if provider, _ := settings["defaultProvider"].(string); provider != "ollama" {
		return "", nil
	}
	model, _ := settings["defaultModel"].(string)
	return model, nil
}

// piTargetModel retains remote routing metadata even when /show requires sign-in.
func (c *launcherClient) piTargetModel(ctx context.Context, name string, budget *time.Duration) (LaunchModel, error) {
	var inventory []LaunchModel
	if c.inventory != nil {
		inventory = c.inventory.snapshot()
	}
	model, _ := findLaunchModel(inventory, name)
	model.Name = name
	model.Remote = model.Remote || isCloudModelName(name)
	if model.Remote {
		if disabled, _ := cloudStatusDisabled(ctx, c.apiClient); disabled {
			return model, fmt.Errorf("cloud is disabled; enable cloud to use %s", name)
		}
	}
	if model.ContextLength > 0 && len(model.Capabilities) > 0 {
		return model, nil
	}
	showCtx, cancel := context.WithTimeout(ctx, *budget)
	defer cancel()
	started := time.Now()
	show, err := c.apiClient.Show(showCtx, &api.ShowRequest{Model: name})
	*budget -= time.Since(started)
	if err != nil {
		if showCtx.Err() != nil {
			return model, showCtx.Err()
		}
		if len(inventory) == 0 {
			inventory, _ = c.modelInventory().Load(ctx)
			if listed, ok := findLaunchModel(inventory, name); ok {
				model = listed
				model.Name = name
				model.Remote = model.Remote || isCloudModelName(name)
			}
		}
		var status api.StatusError
		if model.Remote && !(errors.As(err, &status) && status.StatusCode == http.StatusNotFound) {
			return model, nil
		}
		return model, err
	}
	if show.RemoteModel != "" {
		model.RemoteModel = show.RemoteModel
	}
	if !model.Remote && (show.RemoteModel != "" || show.RemoteHost != "") {
		model.Remote = true
		if disabled, _ := cloudStatusDisabled(ctx, c.apiClient); disabled {
			return model, fmt.Errorf("cloud is disabled; enable cloud to use %s", name)
		}
	}
	if len(show.Capabilities) > 0 {
		model.Capabilities = slices.Clone(show.Capabilities)
	}
	if model.ContextLength <= 0 {
		model.ContextLength, _ = modelInfoContextLength(show.ModelInfo)
	}
	return model, nil
}

func piSupportsChat(model LaunchModel) bool {
	return len(model.Capabilities) == 0 || model.HasCapability("completion") || model.HasCapability("tools")
}

func (c *launcherClient) piSelectableModels(ctx context.Context, items []ModelItem) []ModelItem {
	inventory, _ := c.modelInventory().Load(ctx)
	cloudDisabled, _ := cloudStatusDisabled(ctx, c.apiClient)
	return slices.DeleteFunc(items, func(item ModelItem) bool {
		model, _ := findLaunchModel(inventory, item.Name)
		return !piSupportsChat(LaunchModel{Capabilities: item.Capabilities}) || (cloudDisabled && (model.Remote || isCloudModelName(item.Name)))
	})
}

func (c *launcherClient) piModelCatalog(ctx context.Context, primary LaunchModel, budget time.Duration) ([]LaunchModel, error) {
	inventory, err := c.modelInventory().Load(ctx)
	if errors.Is(err, context.DeadlineExceeded) {
		return []LaunchModel{primary}, nil
	}
	if err != nil {
		return nil, err
	}
	cloudDisabled, _ := cloudStatusDisabled(ctx, c.apiClient)
	if model, ok := findLaunchModel(inventory, primary.Name); ok {
		if primary.ContextLength <= 0 {
			primary.ContextLength = model.ContextLength
		}
		if len(primary.Capabilities) == 0 {
			primary.Capabilities = model.Capabilities
		}
		primary.Remote = primary.Remote || model.Remote
	}
	primary.Remote = primary.Remote || isCloudModelName(primary.Name)
	models := []LaunchModel{primary}
	seen := map[string]bool{launchModelRecommendationKey(primary.Name): true}

	for _, model := range inventory {
		key := launchModelRecommendationKey(model.Name)
		if seen[key] || isDeprecatedLaunchModel(model.Name) || (cloudDisabled && (model.Remote || isCloudModelName(model.Name))) {
			continue
		}
		seen[key] = true
		models = append(models, model)
	}
	for _, rec := range c.recommendations(ctx) {
		key := launchModelRecommendationKey(rec.Name)
		if seen[key] || cloudDisabled || !isCloudModelName(rec.Name) || isDeprecatedLaunchModel(rec.Name) {
			continue
		}
		seen[key] = true
		models = append(models, LaunchModel{Name: rec.Name, Remote: true, ContextLength: rec.Details.ContextLength, MaxOutputTokens: rec.MaxOutputTokens})
	}
	// Discovery is best effort: some models may be missing when the timeout expires.
	// TODO: Use a more narrowly scoped model metadata API instead of /show here.
	showCtx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()
	for i := range models {
		models[i] = models[i].WithCloudLimits()
	}
	var wg sync.WaitGroup
	jobs := make(chan int)
	for range min(4, len(models)) {
		wg.Go(func() {
			for i := range jobs {
				model := &models[i]
				if model.ContextLength > 0 && len(model.Capabilities) > 0 {
					continue
				}
				show, err := c.apiClient.Show(showCtx, &api.ShowRequest{Model: model.Name})
				if err != nil {
					var status api.StatusError
					if errors.As(err, &status) && status.StatusCode == http.StatusNotFound {
						model.Name = ""
					}
					continue
				}
				if model.ContextLength <= 0 {
					model.ContextLength, _ = modelInfoContextLength(show.ModelInfo)
				}
				if len(model.Capabilities) == 0 {
					model.Capabilities = slices.Clone(show.Capabilities)
				}
				model.Remote = model.Remote || show.RemoteModel != "" || show.RemoteHost != ""
			}
		})
	}
queue:
	for i := 1; i < len(models); i++ {
		select {
		case jobs <- i:
		case <-showCtx.Done():
			break queue
		}
	}
	close(jobs)
	wg.Wait()
	if err := ctx.Err(); err != nil && !errors.Is(err, context.DeadlineExceeded) {
		return nil, err
	}
	if !piSupportsChat(models[0]) {
		return nil, fmt.Errorf("model %q does not support chat; choose a chat model", primary.Name)
	}
	if cloudDisabled && models[0].Remote {
		return nil, fmt.Errorf("cloud is disabled; enable cloud to use %s", primary.Name)
	}
	if models[0].Name == "" {
		return nil, fmt.Errorf("model %q is no longer available", primary.Name)
	}
	models = slices.DeleteFunc(models, func(model LaunchModel) bool {
		return model.Name == "" || !piSupportsChat(model) || (cloudDisabled && model.Remote)
	})
	return models, nil
}

func (p *Pi) Run(model string, _ []LaunchModel, args []string) error {
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

	if model != "" && !piHasModelArgs(args) {
		args = append([]string{"--provider", "ollama", "--model", model}, args...)
	}
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

func piBaseURL() string {
	return strings.TrimRight(envconfig.Host().String(), "/") + "/v1"
}

func (p *Pi) Edit(models []LaunchModel) error {
	_, err := p.writeConfig(models, false)
	return err
}

func (p *Pi) writeConfig(models []LaunchModel, preserveCatalog bool) (string, error) {
	if len(models) == 0 {
		return "", nil
	}

	primary := models[0].Name
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	configPath := filepath.Join(home, ".pi", "agent", "models.json")
	settingsPath := filepath.Join(home, ".pi", "agent", "settings.json")
	configs := make([]map[string]any, 2)
	for i, path := range []string{configPath, settingsPath} {
		data, err := fileutil.ReadJSON(path)
		if err != nil && !os.IsNotExist(err) {
			return "", fmt.Errorf("read Pi config %s: %w", path, err)
		}
		if data == nil {
			data = make(map[string]any)
		}
		configs[i] = data
	}
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return "", err
	}
	config, settings := configs[0], configs[1]

	providers, ok := config["providers"].(map[string]any)
	if !ok {
		providers = make(map[string]any)
	}

	ollama, ok := providers["ollama"].(map[string]any)
	if !ok {
		ollama = map[string]any{}
	}

	ollama["baseUrl"] = piBaseURL()

	if _, exists := ollama["api"]; !exists {
		ollama["api"] = "openai-completions"
	}
	if _, exists := ollama["apiKey"]; !exists {
		ollama["apiKey"] = "ollama"
	}

	existingModels, ok := ollama["models"].([]any)
	if !ok {
		existingModels = make([]any, 0)
	}

	// Only launch-owned entries are replaced by the current catalog.
	var newModels []any
	manual := make(map[string]string)
	managed := make(map[string]map[string]any)
	found := make(map[string]bool, len(models))
	for _, model := range models {
		found[launchModelRecommendationKey(model.Name)] = true
	}
	for _, entry := range existingModels {
		model, _ := entry.(map[string]any)
		id, _ := model["id"].(string)
		if !isPiOllamaModel(model) {
			newModels = append(newModels, entry)
			key := launchModelRecommendationKey(id)
			if manual[key] == "" || id == models[0].Name {
				manual[key] = id
			}
		} else {
			key := launchModelRecommendationKey(id)
			managed[key] = model
			if preserveCatalog && !found[key] {
				newModels = append(newModels, entry)
			}
		}
	}
	for _, model := range models {
		key := launchModelRecommendationKey(model.Name)
		if id := manual[key]; id != "" {
			if key == launchModelRecommendationKey(models[0].Name) {
				primary = id
			}
			continue
		}
		entry := createConfig(model)
		// A failed metadata lookup must not erase previously known limits or
		// capabilities for a model that remains in the inventory.
		if old := managed[key]; old != nil {
			if model.ContextLength <= 0 && hasContextWindow(old) {
				entry["contextWindow"] = old["contextWindow"]
			}
			if len(model.Capabilities) == 0 {
				for _, key := range []string{"input", "reasoning"} {
					if value, ok := old[key]; ok {
						entry[key] = value
					}
				}
			}
		}
		newModels = append(newModels, entry)
	}

	ollama["models"] = newModels
	providers["ollama"] = ollama
	config["providers"] = providers

	configData, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return "", err
	}
	if err := fileutil.WriteWithBackup(configPath, configData, "pi"); err != nil {
		return "", err
	}

	// The first catalog entry is the separately selected starting model.

	settings["defaultProvider"] = "ollama"
	settings["defaultModel"] = primary

	settingsData, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return "", err
	}
	if err := fileutil.WriteWithBackup(settingsPath, settingsData, "pi"); err != nil {
		return "", err
	}
	return primary, nil
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

	// Returning nil on host drift forces launchEditorIntegration to call Edit.
	if configured, _ := ollama["baseUrl"].(string); configured != "" {
		if strings.TrimRight(configured, "/") != strings.TrimRight(piBaseURL(), "/") {
			return nil
		}
	}

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
