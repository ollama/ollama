package launch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/envconfig"
)

const gooseInstallURL = "https://goose-docs.ai/docs/getting-started/installation/"

var (
	// gooseGOOS allows tests to override the detected operating system.
	gooseGOOS = runtime.GOOS

	// Test seams for platform-specific Goose app detection and launch.
	gooseUserHome    = os.UserHomeDir
	gooseStatFn      = os.Stat
	gooseGlob        = filepath.Glob
	gooseOpenPath    = defaultGooseOpenPath
	gooseOpenStartID = defaultGooseOpenStartID
	gooseRunPath     = defaultGooseRunningAppPath
	gooseStartID     = defaultGooseStartAppID
	gooseCommand     = exec.Command
	gooseIsRunning   = defaultGooseIsRunning
	gooseQuitApp     = defaultGooseQuitApp
	gooseSleep       = time.Sleep
	gooseSaveAppPath = config.SaveIntegrationAppPath
)

// Goose implements Runner for the Goose desktop app integration.
//
// Goose ships with a built-in `ollama` provider, so this launcher just sets
// the standard GOOSE_PROVIDER / GOOSE_MODEL / OLLAMA_HOST env vars and
// launches the Goose desktop app.
type Goose struct{}

func (g *Goose) String() string { return "Goose Desktop" }

func (g *Goose) Supported() error {
	switch gooseGOOS {
	case "darwin", "windows":
		return nil
	default:
		return fmt.Errorf("Goose Desktop launch is only supported on macOS and Windows; use 'ollama launch goose-cli' for the Goose CLI")
	}
}

// GooseCLI implements Runner for the Goose CLI integration.
type GooseCLI struct{}

func (g *GooseCLI) String() string { return "Goose CLI" }

// envVars returns the environment variables that configure Goose to use
// the local Ollama server as its model provider.
func (g *Goose) envVars(model string) []string {
	env := []string{
		"GOOSE_PROVIDER=ollama",
		"OLLAMA_HOST=" + envconfig.Host().String(),
	}
	if model != "" {
		env = append(env, "GOOSE_MODEL="+model)
	}
	return env
}

// desktopAppAvailable reports whether the Goose desktop app is available.
func (g *Goose) desktopAppAvailable() bool {
	if gooseDesktopAppPath() != "" {
		return true
	}
	if gooseGOOS != "windows" {
		return false
	}
	if path := gooseRunPath(); path != "" {
		gooseRememberDesktopAppPath(path)
		return true
	}
	return gooseStartID() != ""
}

func gooseDesktopAppPath() string {
	for _, path := range gooseDesktopAppCandidates() {
		if _, err := gooseStatFn(path); err == nil {
			return path
		}
	}
	return ""
}

func gooseDesktopAppCandidates() []string {
	switch gooseGOOS {
	case "darwin":
		return gooseDarwinAppCandidates()
	case "windows":
		return gooseDedupePaths(append(gooseWindowsAppCandidates(), gooseSavedAppCandidates()...))
	default:
		return nil
	}
}

func gooseDarwinAppCandidates() []string {
	candidates := []string{"/Applications/Goose.app"}
	if home, err := gooseUserHome(); err == nil {
		candidates = append(candidates, filepath.Join(home, "Applications", "Goose.app"))
	}
	return candidates
}

func gooseWindowsAppCandidates() []string {
	var candidates []string
	if local, err := gooseLocalAppData(); err == nil {
		candidates = append(candidates,
			filepath.Join(local, "Programs", "Goose", "Goose.exe"),
			filepath.Join(local, "Programs", "goose", "Goose.exe"),
			filepath.Join(local, "Goose", "Goose.exe"),
			filepath.Join(local, "goose", "Goose.exe"),
			filepath.Join(local, "Block", "goose", "Goose.exe"),
		)
		for _, pattern := range []string{
			filepath.Join(local, "Programs", "Goose", "app-*", "Goose.exe"),
			filepath.Join(local, "Programs", "goose", "app-*", "Goose.exe"),
			filepath.Join(local, "Goose", "app-*", "Goose.exe"),
			filepath.Join(local, "goose", "app-*", "Goose.exe"),
			filepath.Join(local, "Block", "goose", "app-*", "Goose.exe"),
		} {
			matches, _ := gooseGlob(pattern)
			candidates = append(candidates, matches...)
		}
	}

	for _, root := range []string{os.Getenv("ProgramFiles"), os.Getenv("ProgramFiles(x86)")} {
		root = strings.TrimSpace(root)
		if root == "" {
			continue
		}
		candidates = append(candidates,
			filepath.Join(root, "Goose", "Goose.exe"),
			filepath.Join(root, "goose", "Goose.exe"),
			filepath.Join(root, "Block", "goose", "Goose.exe"),
		)
	}
	return gooseDedupePaths(candidates)
}

func gooseSavedAppCandidates() []string {
	cfg, err := config.LoadIntegration("goose-desktop")
	if err != nil || strings.TrimSpace(cfg.AppPath) == "" {
		return nil
	}
	return []string{cfg.AppPath}
}

func gooseLocalAppData() (string, error) {
	if local := strings.TrimSpace(os.Getenv("LOCALAPPDATA")); local != "" {
		return local, nil
	}
	if profile := strings.TrimSpace(os.Getenv("USERPROFILE")); profile != "" {
		return filepath.Join(profile, "AppData", "Local"), nil
	}
	home, err := gooseUserHome()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, "AppData", "Local"), nil
}

func gooseDedupePaths(paths []string) []string {
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

// runDesktopApp launches the Goose desktop app with the provider env vars
// baked into the spawned process.
func (g *Goose) runDesktopApp(model string) error {
	env := g.envVars(model)
	switch gooseGOOS {
	case "darwin":
		if path := gooseDesktopAppPath(); path != "" {
			shouldLaunch, err := gooseRestartIfRunning()
			if err != nil {
				return err
			}
			if !shouldLaunch {
				return nil
			}
			return gooseOpenPath(path, env)
		}
		return gooseDesktopNotInstalledError()
	case "windows":
		if path := gooseDesktopAppPath(); path != "" {
			shouldLaunch, err := gooseRestartIfRunning()
			if err != nil {
				return err
			}
			if !shouldLaunch {
				return nil
			}
			return gooseOpenPath(path, env)
		}
		if path := gooseRunPath(); path != "" {
			gooseRememberDesktopAppPath(path)
			shouldLaunch, err := gooseRestartIfRunning()
			if err != nil {
				return err
			}
			if !shouldLaunch {
				return nil
			}
			return gooseOpenPath(path, env)
		}
		if appID := gooseStartID(); appID != "" {
			shouldLaunch, err := gooseRestartIfRunning()
			if err != nil {
				return err
			}
			if !shouldLaunch {
				return nil
			}
			return gooseOpenStartID(appID, env)
		}
		return gooseDesktopNotInstalledError()
	default:
		return fmt.Errorf("Goose desktop app is only supported on macOS and Windows; use 'ollama launch goose-cli' for the Goose CLI")
	}
}

func gooseRestartIfRunning() (bool, error) {
	if !gooseIsRunning() {
		return true, nil
	}

	restart, err := ConfirmPrompt("Goose Desktop is already running. Restart it so Ollama can apply the selected provider and model?")
	if err != nil {
		return false, err
	}
	if !restart {
		fmt.Fprintln(os.Stderr, "\nQuit Goose Desktop and re-run 'ollama launch goose' when you're ready for the selected provider and model to take effect.")
		return false, nil
	}

	if err := gooseQuitApp(); err != nil {
		return false, fmt.Errorf("quit Goose Desktop: %w", err)
	}
	if err := waitForGooseDesktopExit(30 * time.Second); err != nil {
		return false, err
	}
	return true, nil
}

func waitForGooseDesktopExit(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !gooseIsRunning() {
			return nil
		}
		gooseSleep(200 * time.Millisecond)
	}
	return fmt.Errorf("Goose Desktop did not quit; quit it manually and re-run 'ollama launch goose'")
}

func gooseRememberDesktopAppPath(path string) {
	if gooseGOOS != "windows" || strings.TrimSpace(path) == "" {
		return
	}
	_ = gooseSaveAppPath("goose-desktop", path)
}

func gooseDesktopNotInstalledError() error {
	switch gooseGOOS {
	case "windows":
		return fmt.Errorf("Goose Desktop wasn't found.\n\nIf you downloaded a zip or extracted Goose manually, open Goose once, then run:\n  ollama launch goose\n\nOr install Goose Desktop:\n  %s\n\nIf you prefer the CLI, run:\n  ollama launch goose-cli", gooseInstallURL)
	case "darwin":
		return fmt.Errorf("Goose Desktop wasn't found. Install Goose Desktop from %s, then re-run 'ollama launch goose'", gooseInstallURL)
	default:
		return (&Goose{}).Supported()
	}
}

func defaultGooseOpenPath(path string, env []string) error {
	var cmd *exec.Cmd
	switch gooseGOOS {
	case "darwin":
		args := make([]string, 0, 1+len(env)*2)
		for _, e := range env {
			args = append(args, "--env", e)
		}
		args = append(args, path)
		cmd = gooseCommand("open", args...)
	case "windows":
		cmd = gooseCommand("powershell.exe", "-NoProfile", "-Command", "Start-Process -FilePath "+quotePowerShellString(path))
	default:
		return fmt.Errorf("Goose desktop app is only supported on macOS and Windows")
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), env...)
	return cmd.Run()
}

func defaultGooseOpenStartID(appID string, env []string) error {
	cmd := gooseCommand("powershell.exe", "-NoProfile", "-Command", "Start-Process "+quotePowerShellString(`shell:AppsFolder\`+appID))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), env...)
	return cmd.Run()
}

func defaultGooseRunningAppPath() string {
	if gooseGOOS != "windows" {
		return ""
	}
	script := `(Get-Process Goose -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 -and $_.Path } | Select-Object -First 1 -ExpandProperty Path)`
	out, err := gooseCommand("powershell.exe", "-NoProfile", "-Command", script).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func defaultGooseStartAppID() string {
	if gooseGOOS != "windows" {
		return ""
	}
	script := `(Get-StartApps Goose | Where-Object { $_.Name -eq 'Goose' -or $_.Name -like 'Goose*' } | Select-Object -First 1 -ExpandProperty AppID)`
	out, err := gooseCommand("powershell.exe", "-NoProfile", "-Command", script).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func defaultGooseIsRunning() bool {
	switch gooseGOOS {
	case "darwin":
		out, err := gooseCommand("pgrep", "-f", "Goose.app/Contents/MacOS/Goose").Output()
		return err == nil && strings.TrimSpace(string(out)) != ""
	case "windows":
		out, err := gooseCommand("powershell.exe", "-NoProfile", "-Command", `Get-Process Goose -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id`).Output()
		return err == nil && strings.TrimSpace(string(out)) != ""
	default:
		return false
	}
}

func defaultGooseQuitApp() error {
	switch gooseGOOS {
	case "darwin":
		return gooseCommand("osascript", "-e", `tell application "Goose" to quit`).Run()
	case "windows":
		script := `Get-Process Goose -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 } | ForEach-Object { [void]$_.CloseMainWindow() }`
		return gooseCommand("powershell.exe", "-NoProfile", "-Command", script).Run()
	default:
		return nil
	}
}

// runCLI launches `goose session` with the integration's env vars and
// passes through any extra args provided by the user.
func (g *Goose) runCLI(model string, extra []string) error {
	bin, err := exec.LookPath("goose")
	if err != nil {
		return fmt.Errorf("goose is not installed, install from %s", gooseInstallURL)
	}

	args := append([]string{"session"}, extra...)
	cmd := gooseCommand(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), g.envVars(model)...)
	return cmd.Run()
}

// Run launches the Goose desktop app with the given model.
func (g *Goose) Run(model string, _ []LaunchModel, args []string) error {
	if len(args) > 0 {
		return fmt.Errorf("goose desktop does not accept extra arguments; use 'ollama launch goose-cli -- ...' to pass arguments to Goose CLI")
	}
	return g.runDesktopApp(model)
}

// Run launches Goose CLI in session mode with the given model.
func (g *GooseCLI) Run(model string, _ []LaunchModel, args []string) error {
	return (&Goose{}).runCLI(model, args)
}

// installed reports whether the Goose desktop app is available on this machine.
func (g *Goose) installed() bool {
	return g.desktopAppAvailable()
}

// installed reports whether the `goose` CLI is available on this machine.
func (g *GooseCLI) installed() bool {
	_, err := exec.LookPath("goose")
	return err == nil
}
