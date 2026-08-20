package launch

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"

	"github.com/ollama/ollama/envconfig"
)

// Poolside implements Runner for Poolside's CLI.
type Poolside struct{}

const (
	poolsideUnixInstallScript    = "curl -fsSL https://downloads.poolside.ai/pool/install.sh | sh"
	poolsideWindowsInstallScript = "irm https://downloads.poolside.ai/pool/install.ps1 | iex"
)

var poolsideGOOS = runtime.GOOS

func (p *Poolside) String() string { return "Pool" }

func (p *Poolside) args(model string, extra []string) []string {
	var args []string
	if model != "" {
		args = append(args, "-m", model)
	}
	args = append(args, extra...)
	return args
}

func (p *Poolside) Run(model string, _ []LaunchModel, args []string) error {
	bin, err := ensurePoolsideInstalled()
	if err != nil {
		return err
	}

	cmd := exec.Command(bin, p.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(),
		"POOLSIDE_STANDALONE_BASE_URL="+envconfig.Host().String()+"/v1",
		"POOLSIDE_API_KEY=ollama",
	)
	return cmd.Run()
}

func ensurePoolsideInstalled() (string, error) {
	if path, err := exec.LookPath("pool"); err == nil {
		return path, nil
	}

	if poolsideGOOS == "windows" {
		return ensurePoolsideInstalledWindows()
	}
	return ensurePoolsideInstalledUnix()
}

func ensurePoolsideInstalledUnix() (string, error) {
	if _, err := exec.LookPath("curl"); err != nil {
		return "", fmt.Errorf("pool is not installed and curl is required to install it\n\nInstall curl, then re-run:\n  ollama launch pool")
	}
	if _, err := exec.LookPath("sh"); err != nil {
		return "", fmt.Errorf("pool is not installed and sh is required to install it\n\nInstall a POSIX shell, then re-run:\n  ollama launch pool")
	}

	ok, err := ConfirmPrompt("Pool is not installed. Install now?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("pool installation cancelled")
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Pool...\n")
	cmd := exec.Command("sh", "-c", poolsideUnixInstallScript)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install pool: %w", err)
	}

	path, err := exec.LookPath("pool")
	if err != nil {
		return "", fmt.Errorf("pool was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sPool installed successfully%s\n\n", ansiGreen, ansiReset)
	return path, nil
}

func ensurePoolsideInstalledWindows() (string, error) {
	if _, err := exec.LookPath("powershell"); err != nil {
		return "", fmt.Errorf("pool is not installed and PowerShell is required to install it\n\nInstall PowerShell, then re-run:\n  ollama launch pool")
	}

	ok, err := ConfirmPrompt("Pool is not installed. Install now?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("pool installation cancelled")
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Pool...\n")
	cmd := exec.Command("powershell", "-NoProfile", "-Command", poolsideWindowsInstallScript)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install pool: %w", err)
	}

	path, err := exec.LookPath("pool")
	if err != nil {
		return "", fmt.Errorf("pool was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sPool installed successfully%s\n\n", ansiGreen, ansiReset)
	return path, nil
}
