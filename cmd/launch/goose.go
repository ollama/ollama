package launch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

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
)

// Goose implements Runner for the Goose coding agent integration.
//
// Goose ships with a built-in `ollama` provider, so this launcher just sets
// the standard GOOSE_PROVIDER / GOOSE_MODEL / OLLAMA_HOST env vars and
// launches either the Goose.app desktop bundle (on macOS, when present) or
// falls back to the `goose` CLI in `session` mode.
type Goose struct{}

func (g *Goose) String() string { return "Goose" }

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
	return gooseGOOS == "windows" && (gooseRunPath() != "" || gooseStartID() != "")
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
		return gooseWindowsAppCandidates()
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
			return gooseOpenPath(path, env)
		}
		return fmt.Errorf("Goose.app was not found")
	case "windows":
		if path := gooseDesktopAppPath(); path != "" {
			return gooseOpenPath(path, env)
		}
		if path := gooseRunPath(); path != "" {
			return gooseOpenPath(path, env)
		}
		if appID := gooseStartID(); appID != "" {
			return gooseOpenStartID(appID, env)
		}
		return fmt.Errorf("Goose executable was not found; open Goose manually once and re-run 'ollama launch goose'")
	default:
		return g.runCLI(model, nil)
	}
}

func defaultGooseOpenPath(path string, env []string) error {
	var cmd *exec.Cmd
	switch gooseGOOS {
	case "darwin":
		cmd = exec.Command("open", path)
	case "windows":
		cmd = exec.Command("powershell.exe", "-NoProfile", "-Command", "Start-Process -FilePath "+quotePowerShellString(path))
	default:
		return fmt.Errorf("Goose desktop app is only supported on macOS and Windows")
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), env...)
	return cmd.Run()
}

func defaultGooseOpenStartID(appID string, env []string) error {
	cmd := exec.Command("powershell.exe", "-NoProfile", "-Command", "Start-Process "+quotePowerShellString(`shell:AppsFolder\`+appID))
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
	out, err := exec.Command("powershell.exe", "-NoProfile", "-Command", script).Output()
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
	out, err := exec.Command("powershell.exe", "-NoProfile", "-Command", script).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// runCLI launches `goose session` with the integration's env vars and
// passes through any extra args provided by the user.
func (g *Goose) runCLI(model string, extra []string) error {
	bin, err := exec.LookPath("goose")
	if err != nil {
		return fmt.Errorf("goose is not installed, install from https://block.github.io/goose/docs/getting-started/installation/")
	}

	args := append([]string{"session"}, extra...)
	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), g.envVars(model)...)
	return cmd.Run()
}

// Run launches Goose with the given model. On macOS, if the Goose.app desktop
// bundle is installed it is preferred; otherwise the `goose` CLI is used.
func (g *Goose) Run(model string, _ []LaunchModel, args []string) error {
	if g.desktopAppAvailable() {
		return g.runDesktopApp(model)
	}
	return g.runCLI(model, args)
}

// installed reports whether either the Goose desktop app or the `goose` CLI
// is available on this machine.
func (g *Goose) installed() bool {
	if g.desktopAppAvailable() {
		return true
	}
	_, err := exec.LookPath("goose")
	return err == nil
}
