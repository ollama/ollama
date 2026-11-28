package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/types/model"
	"golang.org/x/mod/semver"
)

const (
	openCodeInstallScript = "curl -fsSL https://opencode.ai/v2/install | bash"
	openCodePluginID      = "ollama.integration"
	openCodePluginSource  = "git+https://github.com/ollama/opencode-ollama.git#main"
	openCodeStartupEnv    = "OLLAMA_OPENCODE_STARTUP"
	openCodeMinVersion    = "v2.4.1"
)

var openCodeGOOS = runtime.GOOS

type OpenCode struct{}

type openCodeStartup struct {
	Model            string `json:"model"`
	ModelWarning     string `json:"modelWarning,omitempty"`
	CheckModelAccess bool   `json:"checkModelAccess,omitempty"`
	AutoApprove      bool   `json:"autoApprove,omitempty"`
}

func (o *OpenCode) String() string { return "OpenCode" }

func (o *OpenCode) Run(model string, _ []LaunchModel, args []string) error {
	req := IntegrationLaunchRequest{ModelOverride: model, ExtraArgs: args}
	return o.launch(context.Background(), req, launchIntegrationPolicy(req))
}

func (o *OpenCode) launch(ctx context.Context, req IntegrationLaunchRequest, policy LaunchPolicy) error {
	if req.ConfigureOnly && (req.ModelOverride != "" || len(req.ExtraArgs) > 0) {
		return errors.New("--config installs the OpenCode plugin. Use --model and extra arguments when launching OpenCode")
	}
	args := req.ExtraArgs
	var startupEnv string
	if req.ModelOverride != "" {
		startup, err := openCodeStartupModel(req.ModelOverride, policy.Confirm == LaunchConfirmAutoApprove)
		if err != nil {
			return err
		}
		switch {
		case len(args) == 0:
			data, err := json.Marshal(startup)
			if err != nil {
				return err
			}
			startupEnv = openCodeStartupEnv + "=" + string(data)
		case args[0] == "run":
			args = append([]string{"run", "--model", "ollama/" + startup.Model}, args[1:]...)
		default:
			return errors.New("use --model to open a new OpenCode chat or with the OpenCode run command")
		}
	}

	restoreConfirm := withLaunchConfirmPolicy(policy.confirmPolicy())
	defer restoreConfirm()
	path, err := ensureOpenCodeInstalled()
	if err != nil {
		return err
	}
	version := openCodeVersion(ctx, path)
	if semver.Major(version) != "v2" || semver.Compare(version, openCodeMinVersion) < 0 {
		return errors.New("ollama launch opencode requires OpenCode 2.4.1 or newer in the 2.x series. Update OpenCode and try again")
	}
	if err := ensureOpenCodePlugin(ctx, path); err != nil {
		return fmt.Errorf("prepare Ollama OpenCode plugin: %w", err)
	}
	if req.ConfigureOnly {
		fmt.Fprintln(os.Stdout, "OpenCode is ready. Run opencode to choose a model in /models.")
		return nil
	}

	cmd := exec.CommandContext(ctx, path, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if startupEnv != "" {
		cmd.Env = append(os.Environ(), startupEnv)
	}
	return cmd.Run()
}

func openCodeStartupModel(requested string, autoApprove bool) (openCodeStartup, error) {
	pullName, cloud, err := modelref.NormalizePullName(requested)
	if err != nil {
		return openCodeStartup{}, err
	}
	name := model.ParseName(pullName)
	if !name.IsValid() {
		return openCodeStartup{}, fmt.Errorf("invalid Ollama model name %q", requested)
	}
	return openCodeStartup{
		Model:            name.DisplayShortest(),
		ModelWarning:     deprecatedLaunchModelPrompt(requested, "OpenCode", "", "", ""),
		CheckModelAccess: cloud,
		AutoApprove:      autoApprove,
	}, nil
}

// findOpenCode returns the opencode binary path, checking PATH first then the
// curl installer location (~/.opencode/bin) which may not be on PATH yet.
func findOpenCode() (string, bool) {
	if p, err := exec.LookPath("opencode"); err == nil {
		return p, true
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", false
	}
	name := "opencode"
	if openCodeGOOS == "windows" {
		name = "opencode.exe"
	}
	fallback := filepath.Join(home, ".opencode", "bin", name)
	if _, err := os.Stat(fallback); err == nil {
		return fallback, true
	}
	return "", false
}

func openCodeVersion(ctx context.Context, path string) string {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, path, "--version")
	cmd.WaitDelay = time.Second
	data, err := cmd.Output()
	if err != nil {
		return ""
	}
	fields := strings.Fields(string(data))
	if len(fields) == 0 {
		return ""
	}
	version := "v" + strings.TrimPrefix(fields[len(fields)-1], "v")
	if semver.IsValid(version) {
		return version
	}
	return ""
}

func ensureOpenCodeInstalled() (string, error) {
	if opencodePath, ok := findOpenCode(); ok {
		return opencodePath, nil
	}

	if err := checkOpenCodeInstallerDependencies(); err != nil {
		return "", err
	}

	ok, err := ConfirmPrompt("OpenCode is not installed. Install now?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("opencode installation cancelled")
	}

	bin, args, err := openCodeInstallerCommand(openCodeGOOS)
	if err != nil {
		return "", err
	}

	fmt.Fprintf(os.Stderr, "\nInstalling OpenCode...\n")
	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install opencode: %w", err)
	}

	opencodePath, ok := findOpenCode()
	if !ok {
		return "", fmt.Errorf("opencode was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sOpenCode installed successfully%s\n\n", ansiGreen, ansiReset)
	return opencodePath, nil
}

func checkOpenCodeInstallerDependencies() error {
	switch openCodeGOOS {
	case "windows":
		if _, err := exec.LookPath("npm"); err != nil {
			return fmt.Errorf("opencode is not installed and required dependencies are missing\n\nInstall the following first:\n  npm (Node.js): https://nodejs.org/\n\nThen re-run:\n  ollama launch opencode")
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
			return fmt.Errorf("opencode is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch opencode", strings.Join(missing, "\n  "))
		}
	}
	return nil
}

func openCodeInstallerCommand(goos string) (string, []string, error) {
	switch goos {
	case "windows":
		return "npm", []string{"install", "-g", "@opencode/cli@2"}, nil
	case "darwin", "linux":
		return "bash", []string{"-c", "set -o pipefail; " + openCodeInstallScript}, nil
	default:
		return "", nil, fmt.Errorf("unsupported platform for opencode install: %s", goos)
	}
}

func ensureOpenCodePlugin(ctx context.Context, path string) error {
	list := exec.CommandContext(ctx, path, "plugin", "list")
	// An npm launcher can leave a child holding the output pipe when canceled.
	list.WaitDelay = time.Second
	output, err := list.Output()
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if err != nil {
		var failure *exec.ExitError
		if errors.As(err, &failure) && len(failure.Stderr) > 0 {
			return fmt.Errorf("list OpenCode plugins: %w: %s", err, strings.TrimSpace(string(failure.Stderr)))
		}
		return fmt.Errorf("list OpenCode plugins: %w", err)
	}

	for line := range strings.SplitSeq(string(output), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		// OpenCode can list a resolved package before its plugin ID is registered.
		if fields[0] == openCodePluginID || (fields[0] == "-" && fields[1] != "-" && fields[2] == openCodePluginSource) {
			return nil
		}
	}

	cmd := exec.CommandContext(ctx, path, "plugin", "add", openCodePluginSource)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	cmd.WaitDelay = time.Second
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("install %s: %w", openCodePluginSource, err)
	}
	return nil
}
