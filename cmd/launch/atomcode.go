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

// AtomCode implements Runner for the AtomCode CLI integration.
type AtomCode struct{}

func (a *AtomCode) String() string { return "AtomCode" }

func (a *AtomCode) findPath() (string, error) {
	if p, err := exec.LookPath("atomcode"); err == nil {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	name := "atomcode"
	if runtime.GOOS == "windows" {
		name = "atomcode.exe"
	}
	for _, fallback := range []string{
		filepath.Join(home, ".local", "bin", name),
	} {
		if _, err := os.Stat(fallback); err == nil {
			return fallback, nil
		}
	}
	return "", fmt.Errorf("atomcode binary not found")
}

func (a *AtomCode) Run(model string, models []LaunchModel, args []string) error {
	bin, err := ensureAtomCodeInstalled()
	if err != nil {
		return err
	}

	configPath, err := atomCodeConfigPath()
	if err != nil {
		return err
	}
	if err := writeAtomCodeConfig(configPath, model, models); err != nil {
		return fmt.Errorf("failed to configure atomcode: %w", err)
	}

	runArgs := []string{"--config", configPath, "--provider", atomCodeProvider, "--model", model}
	runArgs = append(runArgs, args...)

	cmd := exec.Command(bin, runArgs...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func ensureAtomCodeInstalled() (string, error) {
	if path, err := (&AtomCode{}).findPath(); err == nil {
		return path, nil
	}

	if err := checkAtomCodeInstallerDependencies(); err != nil {
		return "", err
	}

	ok, err := ConfirmPrompt("AtomCode is not installed. Install now?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("atomcode installation cancelled")
	}

	bin, args, err := atomCodeInstallerCommand(runtime.GOOS)
	if err != nil {
		return "", err
	}

	fmt.Fprintf(os.Stderr, "\nInstalling AtomCode...\n")
	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install atomcode: %w", err)
	}

	path, err := (&AtomCode{}).findPath()
	if err != nil {
		return "", fmt.Errorf("atomcode was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sAtomCode installed successfully%s\n\n", ansiGreen, ansiReset)
	return path, nil
}

func checkAtomCodeInstallerDependencies() error {
	switch runtime.GOOS {
	case "windows":
		if _, err := exec.LookPath("powershell"); err != nil {
			return fmt.Errorf("atomcode is not installed and required dependencies are missing\n\nInstall the following first:\n  PowerShell: https://learn.microsoft.com/powershell/\n\nThen re-run:\n  ollama launch atomcode")
		}
	default:
		var missing []string
		if _, err := exec.LookPath("curl"); err != nil {
			missing = append(missing, "curl: https://curl.se/")
		}
		if _, err := exec.LookPath("bash"); err != nil {
			missing = append(missing, "bash: https://www.gnu.org/software/bash/")
		}
		if len(missing) > 0 {
			return fmt.Errorf("atomcode is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch atomcode", strings.Join(missing, "\n  "))
		}
	}
	return nil
}

func atomCodeInstallerCommand(goos string) (string, []string, error) {
	switch goos {
	case "windows":
		return "powershell", []string{
			"-NoProfile",
			"-ExecutionPolicy",
			"Bypass",
			"-Command",
			"irm https://raw.atomgit.com/atomgit_atomcode/atomcode/raw/main/scripts/install.ps1 | iex",
		}, nil
	case "darwin", "linux":
		return "bash", []string{
			"-c",
			"curl -fsSL https://raw.atomgit.com/atomgit_atomcode/atomcode/raw/main/scripts/install.sh | sh",
		}, nil
	default:
		return "", nil, fmt.Errorf("unsupported platform for atomcode install: %s", goos)
	}
}

const atomCodeProvider = "ollama"

// atomCodeConfigPath returns the launcher-owned config that routes AtomCode to
// Ollama. It is regenerated on every launch so the user's own
// ~/.atomcode/config.toml is never modified.
func atomCodeConfigPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".atomcode", "ollama-launch.toml"), nil
}

func writeAtomCodeConfig(configPath, model string, models []LaunchModel) error {
	var b strings.Builder
	fmt.Fprintf(&b, "default_provider = %q\n\n", atomCodeProvider)
	fmt.Fprintf(&b, "[providers.%s]\n", atomCodeProvider)
	fmt.Fprintf(&b, "type = %q\n", atomCodeProvider)
	fmt.Fprintf(&b, "model = %q\n", model)
	fmt.Fprintf(&b, "base_url = %q\n", strings.TrimRight(envconfig.Host().String(), "/"))
	if ctx := atomCodeModelContext(models, model); ctx > 0 {
		fmt.Fprintf(&b, "context_window = %d\n", ctx)
	}

	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}
	return os.WriteFile(configPath, []byte(b.String()), 0o644)
}

func atomCodeModelContext(models []LaunchModel, model string) int {
	for _, m := range models {
		if m.Name == model && m.ContextLength > 0 {
			return m.ContextLength
		}
	}
	return 0
}
