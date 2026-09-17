package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/internal/proxy"
)

const (
	claudeDesktopIntegrationName = "claude-desktop"
	claudeDesktopProfileName     = "Ollama"
	claudeDesktopProfileID       = "00000000-0000-4000-8000-000000000114"
	claudeDesktopGatewayBaseURL  = "http://" + proxy.DefaultClaudeDesktopListenAddr
	claudeDesktopProbeTimeout    = 2 * time.Second
	claudeDesktopModelLabel      = "Default Ollama model"
	claudeDesktopSuccessMessage  = "Claude Desktop profile changed to Ollama."
	claudeDesktopRestoreMessage  = "To restore the usual Claude profile, run: ollama launch claude-desktop --restore"
	claudeDesktopRestoredMessage = "Claude Desktop restored to the usual Claude profile."
)

// Cowork needs unrestricted egress for user-configured plugins and MCP servers.
// Restore removes this override with the rest of the Ollama profile settings.
var claudeDesktopEgressHosts = []string{"*"}

var (
	claudeDesktopGOOS           = runtime.GOOS
	claudeDesktopUserHome       = os.UserHomeDir
	claudeDesktopStat           = os.Stat
	claudeDesktopOpenApp        = defaultClaudeDesktopOpenApp
	claudeDesktopOpenAppPath    = defaultClaudeDesktopOpenAppPath
	claudeDesktopQuitApp        = defaultClaudeDesktopQuitApp
	claudeDesktopIsRunning      = defaultClaudeDesktopRunning
	claudeDesktopRunningAppPath = defaultClaudeDesktopRunningAppPath
	claudeDesktopGlob           = filepath.Glob
	claudeDesktopProbeGateway   = proxy.ProbeClaudeDesktop
)

// ClaudeDesktop configures and launches Claude Desktop in third-party
// inference mode using the Ollama app's local gateway.
type ClaudeDesktop struct{}

// ErrClaudeDesktopRestartConfirmationRequired reports that applying a profile
// change would interrupt a running Claude Desktop process.
var ErrClaudeDesktopRestartConfirmationRequired = errors.New("Claude Desktop restart confirmation is required before changing its profile")

func (c *ClaudeDesktop) String() string { return "Claude Desktop" }

func (c *ClaudeDesktop) Supported() error { return claudeDesktopSupported() }

func (c *ClaudeDesktop) Paths() []string {
	return nil
}

func (c *ClaudeDesktop) AutodiscoveredModel() string {
	return claudeDesktopModelLabel
}

// ConfigureAutodiscovery points Claude Desktop at Ollama's local gateway
// without pinning a model list, so Claude discovers the selected catalog and
// exact Ollama route names the gateway advertises.
func (c *ClaudeDesktop) ConfigureAutodiscovery() error {
	autoMode, err := claudeDesktopAutoModePreference()
	if err != nil {
		return err
	}
	return c.ConfigureAutodiscoveryWithAutoMode(autoMode)
}

// ConfigureAutodiscoveryWithAutoMode writes the managed profile with the
// effective Auto mode state selected by the Ollama app.
func (c *ClaudeDesktop) ConfigureAutodiscoveryWithAutoMode(autoMode bool) error {
	if err := claudeDesktopSupported(); err != nil {
		return err
	}
	if err := ensureClaudeDesktopGateway(); err != nil {
		return err
	}
	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		return err
	}
	return configureClaudeDesktopTargets(targets, claudeDesktopGatewayBaseURL, "ollama", autoMode)
}

func (c *ClaudeDesktop) RestoreHint() string {
	return claudeDesktopRestoreMessage
}

func (c *ClaudeDesktop) ConfigurationSuccessMessage() string {
	return claudeDesktopSuccessMessage + "\n" + claudeDesktopRestoreMessage
}

func (c *ClaudeDesktop) RestoreSuccessMessage() string {
	return claudeDesktopRestoredMessage
}

func (c *ClaudeDesktop) AutodiscoveryConfigured() bool {
	autoMode, err := claudeDesktopAutoModePreference()
	if err != nil {
		return false
	}
	return c.AutodiscoveryConfiguredWithAutoMode(autoMode)
}

// AutodiscoveryConfiguredWithAutoMode reports whether the managed profile has
// the effective Auto mode state selected by the Ollama app.
func (c *ClaudeDesktop) AutodiscoveryConfiguredWithAutoMode(autoMode bool) bool {
	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		return false
	}
	return claudeDesktopTargetsConfigured(targets, autoMode)
}

// UsesOllamaGateway reports whether Claude Desktop is currently routed through
// Ollama's local gateway. It intentionally ignores auxiliary profile settings
// so the gateway can keep serving while those settings are repaired.
func (c *ClaudeDesktop) UsesOllamaGateway() bool {
	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		return false
	}
	return claudeDesktopTargetsUseOllamaGateway(targets)
}

// SetInstalledFromDesktop changes the Claude profile from the native Ollama app.
func (c *ClaudeDesktop) SetInstalledFromDesktop(installed, restart bool) error {
	autoMode := false
	if installed {
		var err error
		autoMode, err = claudeDesktopAutoModePreference()
		if err != nil {
			return err
		}
	}
	return c.SetInstalledFromDesktopWithAutoMode(installed, restart, autoMode)
}

// SetInstalledFromDesktopWithAutoMode changes the Claude profile from the
// native Ollama app with its effective Auto mode state.
func (c *ClaudeDesktop) SetInstalledFromDesktopWithAutoMode(installed, restart, autoMode bool) error {
	if err := claudeDesktopSupported(); err != nil {
		return err
	}

	applyProfile := restoreClaudeDesktopProfile
	if installed {
		applyProfile = func() error { return c.ConfigureAutodiscoveryWithAutoMode(autoMode) }
	}

	running, err := claudeDesktopIsRunning(context.Background())
	if err != nil {
		return fmt.Errorf("check whether Claude Desktop is running: %w", err)
	}
	if !running {
		if err := applyProfile(); err != nil {
			return err
		}
		if installed {
			return claudeDesktopOpenApp()
		}
		return nil
	}
	if !restart {
		return ErrClaudeDesktopRestartConfirmationRequired
	}
	return restartClaudeDesktop(applyProfile)
}

// ApplyProfileChange applies a profile-dependent change immediately. Claude is
// restarted only when it is already running; otherwise the change takes effect
// the next time the user opens it.
func (c *ClaudeDesktop) ApplyProfileChange(change func() error, restartConfirmed bool) error {
	if err := claudeDesktopSupported(); err != nil {
		return err
	}
	running, err := claudeDesktopIsRunning(context.Background())
	if err != nil {
		return fmt.Errorf("check whether Claude Desktop is running: %w", err)
	}
	if !running {
		return change()
	}
	if !restartConfirmed {
		return ErrClaudeDesktopRestartConfirmationRequired
	}
	return restartClaudeDesktop(change)
}

// RestoreForShutdown restores Claude's usual profile without reopening the app.
func (c *ClaudeDesktop) RestoreForShutdown(ctx context.Context) error {
	if err := claudeDesktopSupported(); err != nil {
		return err
	}
	running, err := claudeDesktopIsRunning(ctx)
	if err != nil {
		return fmt.Errorf("check whether Claude Desktop is running: %w", err)
	}
	if !running {
		return restoreClaudeDesktopProfile()
	}
	if err := claudeDesktopQuitApp(ctx); err != nil {
		return fmt.Errorf("quit Claude Desktop: %w", err)
	}
	if err := waitForClaudeDesktopExit(ctx); err != nil {
		return err
	}
	return restoreClaudeDesktopProfile()
}

func restoreClaudeDesktopProfile() error {
	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		return err
	}
	return restoreClaudeDesktopTargets(targets)
}

func (c *ClaudeDesktop) Onboard() error {
	return config.MarkIntegrationOnboarded(claudeDesktopIntegrationName)
}

// ClaudeDesktopModels returns the user's explicitly saved Claude Desktop
// model subset. A nil result means the recommendation source should decide.
func ClaudeDesktopModels() []string {
	configured, err := config.LoadIntegration(claudeDesktopIntegrationName)
	if err != nil {
		return nil
	}
	if len(configured.Aliases) > 0 {
		models := make([]string, 0, len(configured.Aliases))
		for _, route := range proxy.ClaudeDesktopRoutes() {
			if model := strings.TrimSpace(configured.Aliases[route.ID]); model != "" {
				models = append(models, model)
			}
		}
		return models
	}
	return configured.Models
}

// ClaudeDesktopModelMappings returns explicit Claude route assignments. An
// empty map means the legacy ordered model selection or recommendations apply.
func ClaudeDesktopModelMappings() map[string]string {
	configured, err := config.LoadIntegration(claudeDesktopIntegrationName)
	if err != nil || len(configured.Aliases) == 0 {
		return nil
	}
	mappings := make(map[string]string, len(configured.Aliases))
	for _, route := range proxy.ClaudeDesktopRoutes() {
		if model := strings.TrimSpace(configured.Aliases[route.ID]); model != "" {
			mappings[route.ID] = model
		}
	}
	return mappings
}

// SaveClaudeDesktopModels persists the user's explicit Claude Desktop model
// subset in the shared launcher configuration.
func SaveClaudeDesktopModels(models []string) error {
	if len(models) == 0 {
		return errors.New("select at least one Claude Desktop model")
	}
	if err := config.SaveIntegration(claudeDesktopIntegrationName, models); err != nil {
		return err
	}
	return config.SaveAliases(claudeDesktopIntegrationName, nil)
}

// SaveClaudeDesktopModelMappings persists explicit route assignments. Empty
// routes are omitted; sparse mappings and duplicate model values are valid.
func SaveClaudeDesktopModelMappings(mappings map[string]string) error {
	normalized := make(map[string]string)
	models := make([]string, 0, len(mappings))
	for _, route := range proxy.ClaudeDesktopRoutes() {
		if model := strings.TrimSpace(mappings[route.ID]); model != "" {
			normalized[route.ID] = model
			models = append(models, model)
		}
	}
	if len(models) == 0 {
		return errors.New("map at least one Claude Desktop route")
	}
	if err := config.SaveIntegration(claudeDesktopIntegrationName, models); err != nil {
		return err
	}
	return config.SaveAliases(claudeDesktopIntegrationName, normalized)
}

// RestoreClaudeDesktopModels restores a previously captured selection. A nil
// selection restores the implicit recommendation defaults.
func RestoreClaudeDesktopModels(models []string) error {
	if err := config.SaveIntegration(claudeDesktopIntegrationName, models); err != nil {
		return err
	}
	return config.SaveAliases(claudeDesktopIntegrationName, nil)
}

// RestoreClaudeDesktopModelMappings restores a previously captured explicit
// mapping, including the legacy no-mapping state.
func RestoreClaudeDesktopModelMappings(models []string, mappings map[string]string) error {
	if err := config.SaveIntegration(claudeDesktopIntegrationName, models); err != nil {
		return err
	}
	return config.SaveAliases(claudeDesktopIntegrationName, mappings)
}

// ClaudeDesktopAutoModeEnabled reports the user's Claude Desktop auto mode
// preference. It defaults to true when unset and returns configuration read
// failures so callers do not mistake them for an explicit disabled setting.
func ClaudeDesktopAutoModeEnabled() (bool, error) {
	return claudeDesktopAutoModePreference()
}

// SaveClaudeDesktopAutoMode persists the user's Claude Desktop auto mode
// preference in the shared launcher configuration.
func SaveClaudeDesktopAutoMode(enabled bool) error {
	return config.SaveIntegrationAutoMode(claudeDesktopIntegrationName, enabled)
}

func claudeDesktopAutoModePreference() (bool, error) {
	integrationConfig, err := config.LoadIntegration(claudeDesktopIntegrationName)
	if errors.Is(err, os.ErrNotExist) {
		return true, nil
	}
	if err != nil {
		return false, fmt.Errorf("load Claude Desktop auto mode preference: %w", err)
	}
	if integrationConfig.AutoMode == nil {
		return true, nil
	}
	return *integrationConfig.AutoMode, nil
}

func (c *ClaudeDesktop) RequiresInteractiveOnboarding() bool {
	return false
}

func (c *ClaudeDesktop) SkipModelReadiness() bool {
	return true
}

func (c *ClaudeDesktop) Run(_ string, _ []LaunchModel, args []string) error {
	if err := claudeDesktopSupported(); err != nil {
		return err
	}
	if len(args) > 0 {
		return errors.New("claude-desktop does not accept extra arguments")
	}
	if err := ensureClaudeDesktopGateway(); err != nil {
		return err
	}
	return claudeDesktopLaunchOrRestart("Restart Claude Desktop to use Ollama?", c.ConfigureAutodiscovery)
}

func (c *ClaudeDesktop) Restore() error {
	if err := claudeDesktopRestoreSupported(); err != nil {
		return err
	}
	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		return err
	}

	if err := restoreClaudeDesktopTargets(targets); err != nil {
		return err
	}
	return claudeDesktopLaunchOrRestart("Restart Claude Desktop to use the usual Claude profile?", func() error {
		return restoreClaudeDesktopTargets(targets)
	})
}

func configureClaudeDesktopTargets(targets claudeDesktopTargets, baseURL, apiKey string, autoMode bool) error {
	for _, target := range targets.thirdPartyProfiles {
		if err := writeClaudeDesktopGatewayProfile(target.profile, baseURL, apiKey, true, autoMode); err != nil {
			return err
		}
		if err := writeClaudeDesktopMeta(target.meta, claudeDesktopProfileID, claudeDesktopProfileName); err != nil {
			return err
		}
		if err := writeClaudeDesktopDeploymentMode(target.desktopConfig, "3p"); err != nil {
			return err
		}
	}
	for _, path := range targets.normalConfigs {
		if err := writeClaudeDesktopDeploymentMode(path, "3p"); err != nil {
			return err
		}
	}
	return nil
}

func restoreClaudeDesktopTargets(targets claudeDesktopTargets) error {
	for _, path := range targets.normalConfigs {
		if err := writeClaudeDesktopDeploymentMode(path, "1p"); err != nil {
			return err
		}
	}
	for _, target := range targets.thirdPartyProfiles {
		if err := writeClaudeDesktopDeploymentMode(target.desktopConfig, "1p"); err != nil {
			return err
		}
		if err := restoreClaudeDesktopMeta(target.meta); err != nil {
			return err
		}
		if err := restoreClaudeDesktopOllamaProfile(target.profile); err != nil {
			return err
		}
	}
	return nil
}

func claudeDesktopSupported() error {
	if claudeDesktopGOOS == "darwin" {
		return nil
	}
	return errors.New("Claude Desktop launch is only supported on macOS")
}

func claudeDesktopRestoreSupported() error {
	if claudeDesktopGOOS == "darwin" || claudeDesktopGOOS == "windows" {
		return nil
	}
	return errors.New("Claude Desktop restore is only supported on macOS and Windows")
}

func ensureClaudeDesktopGateway() error {
	ctx, cancel := context.WithTimeout(context.Background(), claudeDesktopProbeTimeout)
	defer cancel()
	if err := claudeDesktopProbeGateway(ctx, claudeDesktopGatewayBaseURL); err != nil {
		return fmt.Errorf("Claude gateway is unavailable at %s: %w; restart Ollama and try again", claudeDesktopGatewayBaseURL, err)
	}
	return nil
}

// ClaudeDesktopInstalled reports whether Claude Desktop is installed.
func ClaudeDesktopInstalled() bool {
	if claudeDesktopGOOS == "darwin" {
		return claudeDesktopAppPath() != ""
	}
	if claudeDesktopGOOS != "windows" {
		return false
	}
	running, _ := claudeDesktopIsRunning(context.Background())
	if claudeDesktopAppPath() != "" || running {
		return true
	}
	for _, dir := range claudeDesktopProfileDirCandidates(false) {
		if _, err := claudeDesktopStat(dir); err == nil {
			return true
		}
	}
	return false
}

func claudeDesktopAppPath() string {
	if claudeDesktopGOOS != "darwin" && claudeDesktopGOOS != "windows" {
		return ""
	}
	for _, path := range claudeDesktopAppCandidates() {
		if _, err := claudeDesktopStat(path); err == nil {
			return path
		}
	}
	return ""
}

func claudeDesktopAppCandidates() []string {
	switch claudeDesktopGOOS {
	case "darwin":
		return claudeDesktopDarwinAppCandidates()
	case "windows":
		return claudeDesktopWindowsAppCandidates()
	default:
		return nil
	}
}

func claudeDesktopDarwinAppCandidates() []string {
	candidates := []string{"/Applications/Claude.app"}
	if home, err := claudeDesktopUserHome(); err == nil {
		candidates = append(candidates, filepath.Join(home, "Applications", "Claude.app"))
	}
	return candidates
}

func claudeDesktopWindowsAppCandidates() []string {
	local, err := claudeDesktopLocalAppData()
	if err != nil {
		return nil
	}

	candidates := []string{
		filepath.Join(local, "Programs", "Claude", "Claude.exe"),
		filepath.Join(local, "Programs", "Claude Desktop", "Claude.exe"),
		filepath.Join(local, "Claude", "Claude.exe"),
		filepath.Join(local, "Claude Nest", "Claude.exe"),
		filepath.Join(local, "Claude Desktop", "Claude.exe"),
		filepath.Join(local, "AnthropicClaude", "Claude.exe"),
	}
	for _, pattern := range []string{
		filepath.Join(local, "AnthropicClaude", "app-*", "Claude.exe"),
		filepath.Join(local, "Programs", "Claude", "app-*", "Claude.exe"),
		filepath.Join(local, "Programs", "Claude Desktop", "app-*", "Claude.exe"),
	} {
		matches, _ := claudeDesktopGlob(pattern)
		candidates = append(candidates, matches...)
	}
	return claudeDesktopDedupePaths(candidates)
}

func claudeDesktopDedupePaths(paths []string) []string {
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

type claudeDesktopPaths struct {
	normalConfig  string
	desktopConfig string
	meta          string
	profile       string
}

type claudeDesktopThirdPartyPaths struct {
	desktopConfig string
	meta          string
	profile       string
}

type claudeDesktopTargets struct {
	normalConfigs      []string
	thirdPartyProfiles []claudeDesktopThirdPartyPaths
}

func claudeDesktopConfigPaths() (claudeDesktopPaths, error) {
	switch claudeDesktopGOOS {
	case "darwin":
		return claudeDesktopDarwinConfigPaths()
	case "windows":
		return claudeDesktopWindowsConfigPaths()
	default:
		return claudeDesktopPaths{}, claudeDesktopSupported()
	}
}

func claudeDesktopDarwinConfigPaths() (claudeDesktopPaths, error) {
	normalRoots, thirdPartyRoots, err := claudeDesktopDarwinProfileRoots()
	if err != nil {
		return claudeDesktopPaths{}, err
	}
	normalBase := normalRoots[0]
	thirdPartyBase := thirdPartyRoots[0]
	return claudeDesktopPaths{
		normalConfig:  filepath.Join(normalBase, "claude_desktop_config.json"),
		desktopConfig: filepath.Join(thirdPartyBase, "claude_desktop_config.json"),
		meta:          filepath.Join(thirdPartyBase, "configLibrary", "_meta.json"),
		profile:       filepath.Join(thirdPartyBase, "configLibrary", claudeDesktopProfileID+".json"),
	}, nil
}

func claudeDesktopWindowsConfigPaths() (claudeDesktopPaths, error) {
	normalBase, err := claudeDesktopProfileDir(true)
	if err != nil {
		return claudeDesktopPaths{}, err
	}
	thirdPartyBase, err := claudeDesktopProfileDir(false)
	if err != nil {
		return claudeDesktopPaths{}, err
	}
	return claudeDesktopPaths{
		normalConfig:  filepath.Join(normalBase, "claude_desktop_config.json"),
		desktopConfig: filepath.Join(thirdPartyBase, "claude_desktop_config.json"),
		meta:          filepath.Join(thirdPartyBase, "configLibrary", "_meta.json"),
		profile:       filepath.Join(thirdPartyBase, "configLibrary", claudeDesktopProfileID+".json"),
	}, nil
}

func claudeDesktopProfileDir(normal bool) (string, error) {
	candidates := claudeDesktopProfileDirCandidates(normal)
	if len(candidates) == 0 {
		return "", errors.New("Claude Desktop profile directory could not be resolved")
	}
	for _, candidate := range candidates {
		if _, err := claudeDesktopStat(candidate); err == nil {
			return candidate, nil
		}
	}
	return candidates[0], nil
}

func claudeDesktopProfileDirCandidates(normal bool) []string {
	if claudeDesktopGOOS != "windows" {
		return nil
	}
	normalRoots, thirdPartyRoots, err := claudeDesktopWindowsProfileRoots()
	if err != nil {
		return nil
	}
	if normal {
		return normalRoots
	}
	return thirdPartyRoots
}

func claudeDesktopDarwinProfileRoots() ([]string, []string, error) {
	home, err := claudeDesktopUserHome()
	if err != nil {
		return nil, nil, err
	}
	base := filepath.Join(home, "Library", "Application Support")
	return []string{filepath.Join(base, "Claude")}, []string{filepath.Join(base, "Claude-3p")}, nil
}

func claudeDesktopWindowsProfileRoots() ([]string, []string, error) {
	local, err := claudeDesktopLocalAppData()
	if err != nil {
		return nil, nil, err
	}
	normalRoots := []string{
		filepath.Join(local, "Claude"),
		filepath.Join(local, "Claude Nest"),
	}
	thirdPartyRoots := []string{
		filepath.Join(local, "Claude-3p"),
		filepath.Join(local, "Claude Nest-3p"),
	}
	return normalRoots, thirdPartyRoots, nil
}

func claudeDesktopTargetPaths() (claudeDesktopTargets, error) {
	var (
		normalRoots     []string
		thirdPartyRoots []string
		err             error
	)

	switch claudeDesktopGOOS {
	case "darwin":
		normalRoots, thirdPartyRoots, err = claudeDesktopDarwinProfileRoots()
	case "windows":
		normalRoots, thirdPartyRoots, err = claudeDesktopWindowsProfileRoots()
	default:
		err = claudeDesktopSupported()
	}
	if err != nil {
		return claudeDesktopTargets{}, err
	}

	return newClaudeDesktopTargets(normalRoots, thirdPartyRoots), nil
}

func newClaudeDesktopTargets(normalRoots, thirdPartyRoots []string) claudeDesktopTargets {
	targets := claudeDesktopTargets{}
	for _, root := range claudeDesktopDedupePaths(normalRoots) {
		targets.normalConfigs = append(targets.normalConfigs, filepath.Join(root, "claude_desktop_config.json"))
	}
	for _, root := range claudeDesktopDedupePaths(thirdPartyRoots) {
		targets.thirdPartyProfiles = append(targets.thirdPartyProfiles, claudeDesktopThirdPartyPaths{
			desktopConfig: filepath.Join(root, "claude_desktop_config.json"),
			meta:          filepath.Join(root, "configLibrary", "_meta.json"),
			profile:       filepath.Join(root, "configLibrary", claudeDesktopProfileID+".json"),
		})
	}
	return targets
}

func claudeDesktopLocalAppData() (string, error) {
	if local := strings.TrimSpace(os.Getenv("LOCALAPPDATA")); local != "" {
		return local, nil
	}
	if home := strings.TrimSpace(os.Getenv("USERPROFILE")); home != "" {
		return filepath.Join(home, "AppData", "Local"), nil
	}
	home, err := claudeDesktopUserHome()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, "AppData", "Local"), nil
}

func writeClaudeDesktopDeploymentMode(path, mode string) error {
	cfg, err := readClaudeDesktopJSONAllowMissing(path)
	if err != nil {
		return fmt.Errorf("parse Claude Desktop config: %w", err)
	}
	cfg["deploymentMode"] = mode
	return writeClaudeDesktopJSON(path, cfg)
}

func writeClaudeDesktopMeta(path, id, name string) error {
	meta, err := readClaudeDesktopJSONAllowMissing(path)
	if err != nil {
		return fmt.Errorf("parse Claude Desktop config metadata: %w", err)
	}

	meta["appliedId"] = id
	entries := make([]any, 0)
	for _, entry := range claudeDesktopAnySlice(meta["entries"]) {
		entryMap, _ := entry.(map[string]any)
		if entryMap == nil {
			entries = append(entries, entry)
			continue
		}
		if entryID, _ := entryMap["id"].(string); entryID == id {
			continue
		}
		entries = append(entries, entryMap)
	}
	entries = append(entries, map[string]any{
		"id":   id,
		"name": name,
	})
	meta["entries"] = entries
	return writeClaudeDesktopJSON(path, meta)
}

func writeClaudeDesktopGatewayProfile(path, baseURL, apiKey string, forceChooser, autoMode bool) error {
	cfg, err := readClaudeDesktopJSONAllowMissing(path)
	if err != nil {
		return fmt.Errorf("parse Claude Desktop Ollama profile: %w", err)
	}
	cfg["inferenceProvider"] = "gateway"
	cfg["inferenceGatewayBaseUrl"] = baseURL
	cfg["inferenceGatewayApiKey"] = apiKey
	cfg["inferenceGatewayAuthScheme"] = "bearer"
	cfg["deploymentDisplayName"] = claudeDesktopProfileName
	cfg["chatTabEnabled"] = true
	delete(cfg, "inferenceModels")
	cfg["disableDeploymentModeChooser"] = forceChooser
	cfg["coworkEgressAllowedHosts"] = claudeDesktopEgressHosts
	cfg["disableEssentialTelemetry"] = true
	cfg["disableNonessentialTelemetry"] = true
	cfg["autoModeEnabled"] = autoMode
	return writeClaudeDesktopJSON(path, cfg)
}

func restoreClaudeDesktopMeta(path string) error {
	meta, err := readClaudeDesktopJSONAllowMissing(path)
	if err != nil {
		return fmt.Errorf("parse Claude Desktop config metadata: %w", err)
	}
	if len(meta) == 0 {
		return nil
	}

	changed := false
	if appliedID, _ := meta["appliedId"].(string); appliedID == claudeDesktopProfileID {
		delete(meta, "appliedId")
		changed = true
	}

	entries := claudeDesktopAnySlice(meta["entries"])
	if entries != nil {
		filtered := make([]any, 0, len(entries))
		for _, entry := range entries {
			entryMap, _ := entry.(map[string]any)
			if entryID, _ := entryMap["id"].(string); entryID == claudeDesktopProfileID {
				changed = true
				continue
			}
			filtered = append(filtered, entry)
		}
		meta["entries"] = filtered
	}

	if !changed {
		return nil
	}
	return writeClaudeDesktopJSON(path, meta)
}

func restoreClaudeDesktopOllamaProfile(path string) error {
	cfg, err := readClaudeDesktopJSONAllowMissing(path)
	if err != nil {
		return fmt.Errorf("parse Claude Desktop Ollama profile: %w", err)
	}
	if len(cfg) == 0 {
		return nil
	}
	cfg["disableDeploymentModeChooser"] = false
	delete(cfg, "inferenceProvider")
	delete(cfg, "inferenceGatewayBaseUrl")
	delete(cfg, "inferenceGatewayAuthScheme")
	delete(cfg, "deploymentDisplayName")
	delete(cfg, "inferenceModels")
	delete(cfg, "coworkEgressAllowedHosts")
	delete(cfg, "autoModeEnabled")
	delete(cfg, "disableEssentialTelemetry")
	delete(cfg, "disableNonessentialTelemetry")
	return writeClaudeDesktopJSON(path, cfg)
}

func readClaudeDesktopAppliedID(path string) string {
	meta, err := readClaudeDesktopJSON(path)
	if err != nil {
		return ""
	}
	applied, _ := meta["appliedId"].(string)
	return applied
}

func readClaudeDesktopDeploymentMode(path string) string {
	cfg, err := readClaudeDesktopJSON(path)
	if err != nil {
		return ""
	}
	mode, _ := cfg["deploymentMode"].(string)
	return mode
}

func claudeDesktopTargetsConfigured(targets claudeDesktopTargets, autoMode bool) bool {
	if !claudeDesktopTargetsUseOllamaGateway(targets) {
		return false
	}
	for _, target := range targets.thirdPartyProfiles {
		if !claudeDesktopThirdPartyProfileConfigured(target, autoMode) {
			return false
		}
	}
	return true
}

func claudeDesktopTargetsUseOllamaGateway(targets claudeDesktopTargets) bool {
	if len(targets.normalConfigs) == 0 || len(targets.thirdPartyProfiles) == 0 {
		return false
	}
	for _, path := range targets.normalConfigs {
		if readClaudeDesktopDeploymentMode(path) != "3p" {
			return false
		}
	}
	for _, target := range targets.thirdPartyProfiles {
		if readClaudeDesktopDeploymentMode(target.desktopConfig) != "3p" {
			return false
		}
		if !claudeDesktopThirdPartyProfileUsesOllamaGateway(target) {
			return false
		}
	}
	return true
}

func claudeDesktopThirdPartyProfileConfigured(target claudeDesktopThirdPartyPaths, autoMode bool) bool {
	if !claudeDesktopThirdPartyProfileUsesOllamaGateway(target) {
		return false
	}

	cfg, err := readClaudeDesktopJSON(target.profile)
	if err != nil {
		return false
	}
	if s, _ := cfg["inferenceGatewayApiKey"].(string); strings.TrimSpace(s) == "" {
		return false
	}
	if s, _ := cfg["deploymentDisplayName"].(string); s != claudeDesktopProfileName {
		return false
	}
	egressHosts := claudeDesktopAnySlice(cfg["coworkEgressAllowedHosts"])
	if len(egressHosts) != len(claudeDesktopEgressHosts) {
		return false
	}
	for i, host := range egressHosts {
		if host != claudeDesktopEgressHosts[i] {
			return false
		}
	}
	if disabled, _ := cfg["disableEssentialTelemetry"].(bool); !disabled {
		return false
	}
	if disabled, _ := cfg["disableNonessentialTelemetry"].(bool); !disabled {
		return false
	}
	if enabled, ok := cfg["autoModeEnabled"].(bool); !ok || enabled != autoMode {
		return false
	}
	return true
}

func claudeDesktopThirdPartyProfileUsesOllamaGateway(target claudeDesktopThirdPartyPaths) bool {
	if readClaudeDesktopAppliedID(target.meta) != claudeDesktopProfileID {
		return false
	}

	cfg, err := readClaudeDesktopJSON(target.profile)
	if err != nil {
		return false
	}
	if s, _ := cfg["inferenceProvider"].(string); s != "gateway" {
		return false
	}
	if s, _ := cfg["inferenceGatewayBaseUrl"].(string); !claudeDesktopGatewayURLMatches(s) {
		return false
	}
	return true
}

func claudeDesktopGatewayURLMatches(value string) bool {
	parsed, err := url.Parse(strings.TrimSpace(value))
	if err != nil || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return false
	}
	if strings.TrimRight(parsed.EscapedPath(), "/") != "" {
		return false
	}
	return strings.EqualFold(parsed.Scheme, "http") && strings.EqualFold(parsed.Host, proxy.DefaultClaudeDesktopListenAddr)
}

func readClaudeDesktopJSONAllowMissing(path string) (map[string]any, error) {
	cfg, err := readClaudeDesktopJSON(path)
	if errors.Is(err, os.ErrNotExist) {
		return map[string]any{}, nil
	}
	return cfg, err
}

func readClaudeDesktopJSON(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	if cfg == nil {
		cfg = map[string]any{}
	}
	return cfg, nil
}

func writeClaudeDesktopJSON(path string, cfg any) error {
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return fileutil.WriteWithBackup(path, data)
}

func claudeDesktopAnySlice(value any) []any {
	switch v := value.(type) {
	case []any:
		return v
	case nil:
		return nil
	default:
		return nil
	}
}

func claudeDesktopLaunchOrRestart(prompt string, reapplyProfile func() error) error {
	running, err := claudeDesktopIsRunning(context.Background())
	if err != nil {
		return fmt.Errorf("check whether Claude Desktop is running: %w", err)
	}
	if !running {
		return claudeDesktopOpenApp()
	}
	restart, err := ConfirmPrompt(prompt)
	if err != nil {
		return err
	}
	if !restart {
		fmt.Fprintln(os.Stderr, "\nQuit and reopen Claude Desktop when you're ready for the profile change to take effect.")
		return nil
	}
	return restartClaudeDesktop(reapplyProfile)
}

func restartClaudeDesktop(reapplyProfile func() error) error {
	restartAppPath := ""
	if claudeDesktopGOOS == "windows" {
		restartAppPath = claudeDesktopRunningAppPath()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := claudeDesktopQuitApp(ctx); err != nil {
		return fmt.Errorf("quit Claude Desktop: %w", err)
	}
	if err := waitForClaudeDesktopExit(ctx); err != nil {
		return err
	}
	// Claude persists settings while shutting down. Reapply the profile after
	// exit so its last write cannot restore stale surface or gateway values.
	if err := reapplyProfile(); err != nil {
		reapplyErr := fmt.Errorf("reapply Claude Desktop profile: %w", err)
		if openErr := openClaudeDesktopAfterRestart(restartAppPath); openErr != nil {
			return errors.Join(reapplyErr, fmt.Errorf("reopen Claude Desktop after profile failure: %w", openErr))
		}
		return reapplyErr
	}
	return openClaudeDesktopAfterRestart(restartAppPath)
}

func openClaudeDesktopAfterRestart(restartAppPath string) error {
	if restartAppPath != "" {
		return claudeDesktopOpenAppPath(restartAppPath)
	}
	return claudeDesktopOpenApp()
}

func waitForClaudeDesktopExit(ctx context.Context) error {
	for {
		running, err := claudeDesktopIsRunning(ctx)
		if err != nil {
			return fmt.Errorf("check whether Claude Desktop is running: %w", err)
		}
		if !running {
			return nil
		}
		select {
		case <-ctx.Done():
			return errors.New("Claude Desktop did not quit; quit it manually and re-run the command")
		case <-time.After(200 * time.Millisecond):
		}
	}
}

// ClaudeDesktopRunning reports whether Claude Desktop is open.
func ClaudeDesktopRunning() bool {
	running, _ := claudeDesktopIsRunning(context.Background())
	return running
}

// Running reports whether Claude Desktop is currently open.
func (c *ClaudeDesktop) Running() bool {
	return ClaudeDesktopRunning()
}

// OpenClaudeDesktop brings the installed Claude Desktop app to the foreground.
func OpenClaudeDesktop() error {
	if err := claudeDesktopSupported(); err != nil {
		return err
	}
	return claudeDesktopOpenApp()
}

// Open launches or foregrounds Claude Desktop.
func (c *ClaudeDesktop) Open() error {
	return OpenClaudeDesktop()
}

func defaultClaudeDesktopRunning(ctx context.Context) (bool, error) {
	var (
		out []byte
		err error
	)
	switch claudeDesktopGOOS {
	case "darwin":
		out, err = exec.CommandContext(ctx, "pgrep", "-f", "Claude.app/Contents/MacOS/Claude").Output()
		if exitErr := (*exec.ExitError)(nil); errors.As(err, &exitErr) && exitErr.ExitCode() == 1 && ctx.Err() == nil {
			return false, nil
		}
	case "windows":
		out, err = exec.CommandContext(ctx, "powershell.exe", "-NoProfile", "-Command", `(Get-Process claude -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1).Id`).Output()
	default:
		return false, nil
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		return false, ctxErr
	}
	if err != nil {
		return false, err
	}
	return strings.TrimSpace(string(out)) != "", nil
}

func defaultClaudeDesktopOpenApp() error {
	switch claudeDesktopGOOS {
	case "windows":
		if path := claudeDesktopAppPath(); path != "" {
			return claudeDesktopOpenAppPath(path)
		}
		if path := claudeDesktopRunningAppPath(); path != "" {
			return claudeDesktopOpenAppPath(path)
		}
		return errors.New("Claude Desktop executable was not found; open Claude Desktop manually once and re-run 'ollama launch claude-desktop --restore'")
	case "darwin":
		path := claudeDesktopAppPath()
		if path == "" {
			return errors.New("Claude Desktop app was not found")
		}
		return openClaudeDesktopDarwin(path)
	default:
		return claudeDesktopSupported()
	}
}

func defaultClaudeDesktopOpenAppPath(path string) error {
	switch claudeDesktopGOOS {
	case "windows":
		return exec.Command("powershell.exe", "-NoProfile", "-Command", "Start-Process -FilePath "+quotePowerShellString(path)).Run()
	case "darwin":
		return openClaudeDesktopDarwin(path)
	default:
		return claudeDesktopSupported()
	}
}

func claudeDesktopDarwinOpenArgs(path string) []string {
	return []string{path}
}

func openClaudeDesktopDarwin(path string) error {
	cmd := exec.Command("/usr/bin/open", claudeDesktopDarwinOpenArgs(path)...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func defaultClaudeDesktopRunningAppPath() string {
	if claudeDesktopGOOS != "windows" {
		return ""
	}
	script := `(Get-Process claude -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 -and $_.Path } | Select-Object -First 1 -ExpandProperty Path)`
	out, err := exec.Command("powershell.exe", "-NoProfile", "-Command", script).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func defaultClaudeDesktopQuitApp(ctx context.Context) error {
	if claudeDesktopGOOS == "windows" {
		script := `Get-Process claude -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | ForEach-Object { [void]$_.CloseMainWindow() }`
		return exec.CommandContext(ctx, "powershell.exe", "-NoProfile", "-Command", script).Run()
	}
	return exec.CommandContext(ctx, "osascript", "-e", `tell application "Claude" to quit`).Run()
}

func quotePowerShellString(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}
