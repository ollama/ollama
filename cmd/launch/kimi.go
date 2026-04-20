package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// Kimi implements Runner for Kimi Code CLI integration.
type Kimi struct{}

const (
	kimiDefaultModelAlias     = "ollama"
	kimiDefaultMaxContextSize = 32768
)

var (
	kimiGOOS             = runtime.GOOS
	kimiModelShowTimeout = 5 * time.Second
)

func (k *Kimi) String() string { return "Kimi Code CLI" }

func (k *Kimi) args(config string, extra []string) []string {
	args := []string{"--config", config}
	args = append(args, extra...)
	return args
}

func (k *Kimi) Run(model string, args []string) error {
	if strings.TrimSpace(model) == "" {
		return fmt.Errorf("model is required")
	}
	if err := validateKimiPassthroughArgs(args); err != nil {
		return err
	}

	config, err := buildKimiInlineConfig(model, resolveKimiMaxContextSize(model))
	if err != nil {
		return fmt.Errorf("failed to build kimi config: %w", err)
	}

	bin, err := ensureKimiInstalled()
	if err != nil {
		return err
	}

	cmd := exec.Command(bin, k.args(config, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func validateKimiPassthroughArgs(args []string) error {
	for _, arg := range args {
		switch {
		case arg == "--config", strings.HasPrefix(arg, "--config="):
			return fmt.Errorf("conflicting extra argument %q: ollama launch kimi manages --config", arg)
		case arg == "--config-file", strings.HasPrefix(arg, "--config-file="):
			return fmt.Errorf("conflicting extra argument %q: ollama launch kimi manages --config-file", arg)
		case arg == "--model", strings.HasPrefix(arg, "--model="):
			return fmt.Errorf("conflicting extra argument %q: ollama launch kimi manages --model", arg)
		case arg == "-m", strings.HasPrefix(arg, "-m="):
			return fmt.Errorf("conflicting extra argument %q: ollama launch kimi manages -m/--model", arg)
		}
	}
	return nil
}

func buildKimiInlineConfig(model string, maxContextSize int) (string, error) {
	cfg := map[string]any{
		"default_model": kimiDefaultModelAlias,
		"providers": map[string]any{
			kimiDefaultModelAlias: map[string]any{
				"type":     "openai_legacy",
				"base_url": envconfig.Host().String() + "/v1",
				"api_key":  "ollama",
			},
		},
		"models": map[string]any{
			kimiDefaultModelAlias: map[string]any{
				"provider":         kimiDefaultModelAlias,
				"model":            model,
				"max_context_size": maxContextSize,
			},
		},
	}

	data, err := json.Marshal(cfg)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func resolveKimiMaxContextSize(model string) int {
	if l, ok := lookupCloudModelLimit(model); ok {
		return l.Context
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return kimiDefaultMaxContextSize
	}

	ctx, cancel := context.WithTimeout(context.Background(), kimiModelShowTimeout)
	defer cancel()
	resp, err := client.Show(ctx, &api.ShowRequest{Model: model})
	if err != nil {
		return kimiDefaultMaxContextSize
	}

	if n, ok := modelInfoContextLength(resp.ModelInfo); ok {
		return n
	}

	return kimiDefaultMaxContextSize
}

func modelInfoContextLength(modelInfo map[string]any) (int, bool) {
	for key, val := range modelInfo {
		if !strings.HasSuffix(key, ".context_length") {
			continue
		}
		switch v := val.(type) {
		case float64:
			if v > 0 {
				return int(v), true
			}
		case int:
			if v > 0 {
				return v, true
			}
		case int64:
			if v > 0 {
				return int(v), true
			}
		}
	}
	return 0, false
}

func ensureKimiInstalled() (string, error) {
	if _, err := exec.LookPath("kimi"); err == nil {
		return "kimi", nil
	}

	if err := checkKimiInstallerDependencies(); err != nil {
		return "", err
	}

	ok, err := ConfirmPrompt("Kimi is not installed. Install now?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("kimi installation cancelled")
	}

	bin, args, err := kimiInstallerCommand(kimiGOOS)
	if err != nil {
		return "", err
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Kimi...\n")
	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install kimi: %w", err)
	}

	if _, err := exec.LookPath("kimi"); err != nil {
		return "", fmt.Errorf("kimi was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sKimi installed successfully%s\n\n", ansiGreen, ansiReset)
	return "kimi", nil
}

func checkKimiInstallerDependencies() error {
	switch kimiGOOS {
	case "windows":
		if _, err := exec.LookPath("powershell"); err != nil {
			return fmt.Errorf("kimi is not installed and required dependencies are missing\n\nInstall the following first:\n  PowerShell: https://learn.microsoft.com/powershell/\n\nThen re-run:\n  ollama launch kimi")
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
			return fmt.Errorf("kimi is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch kimi", strings.Join(missing, "\n  "))
		}
	}
	return nil
}

func kimiInstallerCommand(goos string) (string, []string, error) {
	switch goos {
	case "windows":
		return "powershell", []string{
			"-NoProfile",
			"-ExecutionPolicy",
			"Bypass",
			"-Command",
			"Invoke-RestMethod https://code.kimi.com/install.ps1 | Invoke-Expression",
		}, nil
	case "darwin", "linux":
		return "bash", []string{
			"-c",
			"curl -LsSf https://code.kimi.com/install.sh | bash",
		}, nil
	default:
		return "", nil, fmt.Errorf("unsupported platform for kimi install: %s", goos)
	}
}
