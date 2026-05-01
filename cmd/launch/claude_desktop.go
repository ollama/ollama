package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/internal/modelref"
	"golang.org/x/term"
)

const (
	claudeDesktopIntegrationName = "claude-desktop"
	claudeDesktopProfileName     = "Ollama"
	claudeDesktopProfileID       = "00000000-0000-4000-8000-000000000114"
	claudeDesktopGatewayBaseURL  = "https://ollama.com"
)

var (
	claudeDesktopGOOS           = runtime.GOOS
	claudeDesktopUserHome       = os.UserHomeDir
	claudeDesktopStat           = os.Stat
	claudeDesktopOpenApp        = defaultClaudeDesktopOpenApp
	claudeDesktopOpenAppPath    = defaultClaudeDesktopOpenAppPath
	claudeDesktopQuitApp        = defaultClaudeDesktopQuitApp
	claudeDesktopIsRunning      = defaultClaudeDesktopIsRunning
	claudeDesktopRunningAppPath = defaultClaudeDesktopRunningAppPath
	claudeDesktopGlob           = filepath.Glob
	claudeDesktopSleep          = time.Sleep
	claudeDesktopHTTPClient     = http.DefaultClient
	claudeDesktopPromptAPIKey   = promptClaudeDesktopAPIKey
	claudeDesktopValidateAPIKey = validateClaudeDesktopAPIKey
)

// ClaudeDesktop configures and launches Claude Desktop in third-party
// inference mode using Ollama Cloud as the gateway.
type ClaudeDesktop struct{}

func (c *ClaudeDesktop) String() string { return "Claude Desktop" }

func (c *ClaudeDesktop) Supported() error { return claudeDesktopSupported() }

func (c *ClaudeDesktop) Paths() []string {
	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		return nil
	}
	return []string{paths.normalConfig, paths.desktopConfig, paths.meta, paths.profile}
}

func (c *ClaudeDesktop) Configure(model string) error {
	return c.ConfigureWithModels(model, []string{model})
}

func (c *ClaudeDesktop) ConfigureWithModels(model string, models []string) error {
	if err := claudeDesktopSupported(); err != nil {
		return err
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return fmt.Errorf("model is required")
	}

	key, err := claudeDesktopAPIKey()
	if err != nil {
		return err
	}
	if err := claudeDesktopValidateAPIKey(context.Background(), key); err != nil {
		return err
	}

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		return err
	}

	if err := writeClaudeDesktopDeploymentMode(paths.normalConfig, "3p"); err != nil {
		return err
	}
	if err := writeClaudeDesktopDeploymentMode(paths.desktopConfig, "3p"); err != nil {
		return err
	}
	if err := writeClaudeDesktopMeta(paths.meta, claudeDesktopProfileID, claudeDesktopProfileName); err != nil {
		return err
	}
	if err := writeClaudeDesktopGatewayProfile(paths.profile, claudeDesktopGatewayModels(model, models), key, true); err != nil {
		return err
	}
	return nil
}

func (c *ClaudeDesktop) CurrentModel() string {
	if err := claudeDesktopSupported(); err != nil {
		return ""
	}
	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		return ""
	}
	if appliedID := readClaudeDesktopAppliedID(paths.meta); appliedID != "" && appliedID != claudeDesktopProfileID {
		return ""
	}

	cfg, err := readClaudeDesktopJSON(paths.profile)
	if err != nil {
		return ""
	}
	if s, _ := cfg["inferenceProvider"].(string); s != "gateway" {
		return ""
	}
	if s, _ := cfg["inferenceGatewayBaseUrl"].(string); strings.TrimRight(s, "/") != claudeDesktopGatewayBaseURL {
		return ""
	}
	models := claudeDesktopStringSlice(cfg["inferenceModels"])
	if len(models) == 0 {
		return ""
	}
	return models[0]
}

func (c *ClaudeDesktop) Onboard() error {
	return config.MarkIntegrationOnboarded(claudeDesktopIntegrationName)
}

func (c *ClaudeDesktop) RequiresInteractiveOnboarding() bool {
	return false
}

func (c *ClaudeDesktop) SkipModelReadiness() bool {
	return true
}

func (c *ClaudeDesktop) Run(_ string, args []string) error {
	if err := claudeDesktopSupported(); err != nil {
		return err
	}
	if len(args) > 0 {
		return fmt.Errorf("claude-desktop does not accept extra arguments")
	}
	return claudeDesktopLaunchOrRestart("Restart Claude Desktop to use Ollama?")
}

func (c *ClaudeDesktop) Restore() error {
	if err := claudeDesktopSupported(); err != nil {
		return err
	}
	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		return err
	}
	if err := writeClaudeDesktopDeploymentMode(paths.normalConfig, "1p"); err != nil {
		return err
	}
	if err := writeClaudeDesktopDeploymentMode(paths.desktopConfig, "1p"); err != nil {
		return err
	}
	if err := disableClaudeDesktopLaunchProfileForce(paths.profile); err != nil {
		return err
	}
	return claudeDesktopLaunchOrRestart("Restart Claude Desktop to use the usual Claude profile?")
}

func claudeDesktopSupported() error {
	switch claudeDesktopGOOS {
	case "darwin", "windows":
		return nil
	default:
		return fmt.Errorf("Claude Desktop launch is only supported on macOS and Windows")
	}
}

func claudeDesktopInstalled() bool {
	if claudeDesktopAppPath() != "" {
		return true
	}
	if claudeDesktopGOOS == "windows" && claudeDesktopIsRunning() {
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
		candidates := []string{"/Applications/Claude.app"}
		if home, err := claudeDesktopUserHome(); err == nil {
			candidates = append(candidates, filepath.Join(home, "Applications", "Claude.app"))
		}
		return candidates
	case "windows":
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
	default:
		return nil
	}
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

func claudeDesktopConfigPaths() (claudeDesktopPaths, error) {
	switch claudeDesktopGOOS {
	case "darwin":
		home, err := claudeDesktopUserHome()
		if err != nil {
			return claudeDesktopPaths{}, err
		}
		base := filepath.Join(home, "Library", "Application Support", "Claude-3p")
		return claudeDesktopPaths{
			normalConfig:  filepath.Join(home, "Library", "Application Support", "Claude", "claude_desktop_config.json"),
			desktopConfig: filepath.Join(base, "claude_desktop_config.json"),
			meta:          filepath.Join(base, "configLibrary", "_meta.json"),
			profile:       filepath.Join(base, "configLibrary", claudeDesktopProfileID+".json"),
		}, nil
	case "windows":
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
	default:
		return claudeDesktopPaths{}, claudeDesktopSupported()
	}
}

func claudeDesktopProfileDir(normal bool) (string, error) {
	candidates := claudeDesktopProfileDirCandidates(normal)
	if len(candidates) == 0 {
		return "", fmt.Errorf("Claude Desktop profile directory could not be resolved")
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
	local, err := claudeDesktopLocalAppData()
	if err != nil {
		return nil
	}
	if normal {
		return []string{
			filepath.Join(local, "Claude"),
			filepath.Join(local, "Claude Nest"),
		}
	}
	return []string{
		filepath.Join(local, "Claude-3p"),
		filepath.Join(local, "Claude Nest-3p"),
	}
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

func claudeDesktopAPIKey() (string, error) {
	if key := strings.TrimSpace(os.Getenv("OLLAMA_API_KEY")); key != "" {
		return key, nil
	}
	if !isInteractiveSession() || currentLaunchConfirmPolicy.requireYesMessage {
		return "", missingClaudeDesktopAPIKeyError()
	}
	key, err := claudeDesktopPromptAPIKey()
	if err != nil {
		return "", err
	}
	key = strings.TrimSpace(key)
	if key == "" {
		return "", missingClaudeDesktopAPIKeyError()
	}
	return key, nil
}

func missingClaudeDesktopAPIKeyError() error {
	return fmt.Errorf("OLLAMA_API_KEY is required for Claude Desktop. Create an API key at https://ollama.com/settings/keys, then re-run with OLLAMA_API_KEY set")
}

func promptClaudeDesktopAPIKey() (string, error) {
	fmt.Fprint(os.Stderr, "Enter Ollama API key (input hidden): ")
	key, err := term.ReadPassword(int(os.Stdin.Fd()))
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return "", err
	}
	return string(key), nil
}

func validateClaudeDesktopAPIKey(ctx context.Context, key string) error {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, claudeDesktopGatewayBaseURL+"/api/tags", nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Accept", "application/json")

	resp, err := claudeDesktopHTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("validate Ollama API key: %w", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4<<10))

	switch {
	case resp.StatusCode >= 200 && resp.StatusCode < 300:
		return nil
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		return fmt.Errorf("Ollama API key was rejected; create a valid key at https://ollama.com/settings/keys")
	default:
		return fmt.Errorf("validate Ollama API key: unexpected status %d", resp.StatusCode)
	}
}

func claudeDesktopGatewayModels(primary string, models []string) []string {
	out := make([]string, 0, len(models)+1)
	add := func(model string, allowPlain bool) {
		model = strings.TrimSpace(model)
		if model == "" {
			return
		}
		if base, ok := modelref.StripCloudSourceTag(model); ok {
			out = append(out, base)
			return
		}
		if allowPlain {
			out = append(out, model)
		}
	}

	add(primary, true)
	for _, model := range models {
		add(model, false)
	}
	return dedupeModelList(out)
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

func writeClaudeDesktopGatewayProfile(path string, models []string, apiKey string, forceChooser bool) error {
	cfg, err := readClaudeDesktopJSONAllowMissing(path)
	if err != nil {
		return fmt.Errorf("parse Claude Desktop Ollama profile: %w", err)
	}
	cfg["inferenceProvider"] = "gateway"
	cfg["inferenceGatewayBaseUrl"] = claudeDesktopGatewayBaseURL
	cfg["inferenceGatewayApiKey"] = apiKey
	cfg["inferenceGatewayAuthScheme"] = "bearer"
	cfg["inferenceModels"] = models
	cfg["disableDeploymentModeChooser"] = forceChooser
	return writeClaudeDesktopJSON(path, cfg)
}

func disableClaudeDesktopLaunchProfileForce(path string) error {
	cfg, err := readClaudeDesktopJSONAllowMissing(path)
	if err != nil {
		return fmt.Errorf("parse Claude Desktop Ollama profile: %w", err)
	}
	if len(cfg) == 0 {
		return nil
	}
	cfg["disableDeploymentModeChooser"] = false
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

func claudeDesktopStringSlice(value any) []string {
	switch v := value.(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
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

func claudeDesktopLaunchOrRestart(prompt string) error {
	if !claudeDesktopIsRunning() {
		return claudeDesktopOpenApp()
	}
	restartAppPath := ""
	if claudeDesktopGOOS == "windows" {
		restartAppPath = claudeDesktopRunningAppPath()
	}

	restart, err := ConfirmPrompt(prompt)
	if err != nil {
		return err
	}
	if !restart {
		fmt.Fprintln(os.Stderr, "\nQuit and reopen Claude Desktop when you're ready for the profile change to take effect.")
		return nil
	}

	if err := claudeDesktopQuitApp(); err != nil {
		return fmt.Errorf("quit Claude Desktop: %w", err)
	}
	if err := waitForClaudeDesktopExit(30 * time.Second); err != nil {
		return err
	}
	if restartAppPath != "" {
		return claudeDesktopOpenAppPath(restartAppPath)
	}
	return claudeDesktopOpenApp()
}

func waitForClaudeDesktopExit(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !claudeDesktopIsRunning() {
			return nil
		}
		claudeDesktopSleep(200 * time.Millisecond)
	}
	return fmt.Errorf("Claude Desktop did not quit; quit it manually and re-run the command")
}

func defaultClaudeDesktopIsRunning() bool {
	var cmd *exec.Cmd
	switch claudeDesktopGOOS {
	case "darwin":
		cmd = exec.Command("pgrep", "-f", "Claude.app/Contents/MacOS/Claude")
	case "windows":
		cmd = exec.Command("powershell.exe", "-NoProfile", "-Command", `(Get-Process claude -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1).Id`)
	default:
		return false
	}
	out, err := cmd.Output()
	if claudeDesktopGOOS == "windows" {
		return err == nil && strings.TrimSpace(string(out)) != ""
	}
	return err == nil && strings.TrimSpace(string(out)) != ""
}

func defaultClaudeDesktopOpenApp() error {
	if claudeDesktopGOOS == "windows" {
		if path := claudeDesktopAppPath(); path != "" {
			return claudeDesktopOpenAppPath(path)
		}
		if path := claudeDesktopRunningAppPath(); path != "" {
			return claudeDesktopOpenAppPath(path)
		}
		return fmt.Errorf("Claude Desktop executable was not found; open Claude Desktop manually once and re-run 'ollama launch claude-desktop'")
	}
	cmd := exec.Command("open", "-a", "Claude")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func defaultClaudeDesktopOpenAppPath(path string) error {
	if claudeDesktopGOOS == "windows" {
		return exec.Command("powershell.exe", "-NoProfile", "-Command", "Start-Process -FilePath "+quotePowerShellString(path)).Run()
	}
	cmd := exec.Command("open", "-a", "Claude")
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

func defaultClaudeDesktopQuitApp() error {
	if claudeDesktopGOOS == "windows" {
		script := `Get-Process claude -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | ForEach-Object { [void]$_.CloseMainWindow() }`
		return exec.Command("powershell.exe", "-NoProfile", "-Command", script).Run()
	}
	return exec.Command("osascript", "-e", `tell application "Claude" to quit`).Run()
}

func quotePowerShellString(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}
