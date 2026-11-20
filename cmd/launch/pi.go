package launch

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"golang.org/x/mod/semver"
	"golang.org/x/term"
)

// Pi implements Runner for Pi (Pi Coding Agent) integration.
type Pi struct{}

const (
	piNpmPackage   = "@earendil-works/pi-coding-agent"
	piMinVersion   = "1.5.2"
	piOllamaSource = "git:github.com/ollama/pi-ollama"
)

func (p *Pi) String() string { return "Pi" }

func (p *Pi) Run(model string, _ []LaunchModel, args []string) error {
	forward := piLaunchArgs(model, args)
	bin, err := preparePi(IntegrationLaunchRequest{ExtraArgs: args})
	if err != nil {
		return err
	}
	cmd := exec.Command(bin, forward...)
	cmd.Env = append(os.Environ(), "OLLAMA_LAUNCH_PI=1")
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

func preparePi(req IntegrationLaunchRequest) (string, error) {
	offlineEnv := strings.ToLower(os.Getenv("PI_OFFLINE"))
	offline := slices.Contains(req.ExtraArgs, "--offline") || offlineEnv == "1" || offlineEnv == "true" || offlineEnv == "yes"
	if offline {
		if _, err := exec.LookPath("pi"); err != nil {
			return "", fmt.Errorf("Pi is missing and offline setup was requested; install Pi online first")
		}
	}
	bin, err := ensurePiInstalled()
	if err != nil {
		return "", err
	}
	if err := checkPiVersion(bin); err != nil {
		return "", err
	}
	if req.ConfigureOnly || (!offline && !piPluginDeclared()) {
		if err := configurePi(bin, req, offline); err != nil {
			return "", err
		}
	}
	return bin, nil
}

// Recognize the source Launch installs even when its checkout is missing.
// Pi handles other source formats, manual extensions and project trust.
func piPluginDeclared() bool {
	dir := os.Getenv("PI_CODING_AGENT_DIR")
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return false
		}
		dir = filepath.Join(home, ".pi", "agent")
	}
	data, err := os.ReadFile(filepath.Join(dir, "settings.json"))
	var settings struct{ Packages []any }
	if err != nil || json.Unmarshal(data, &settings) != nil {
		return false
	}
	for _, pkg := range settings.Packages {
		if filtered, ok := pkg.(map[string]any); ok {
			pkg = filtered["source"]
		}
		if source, ok := pkg.(string); ok && (source == piOllamaSource || strings.HasPrefix(source, piOllamaSource+"@")) {
			return true
		}
	}
	return false
}

func checkPiVersion(bin string) error {
	output, err := exec.Command(bin, "--version").Output()
	version := "v" + strings.TrimPrefix(strings.TrimSpace(string(output)), "v")
	upgrade := "Update Pi:\n  npm install -g " + piNpmPackage + "@latest"
	if err != nil || !semver.IsValid(version) {
		return fmt.Errorf("could not verify Pi %s or newer\n\n%s", piMinVersion, upgrade)
	}
	if semver.Compare(version, "v"+piMinVersion) < 0 {
		return fmt.Errorf("Pi %s or newer is required (found %s)\n\n%s", piMinVersion, strings.TrimPrefix(version, "v"), upgrade)
	}
	return nil
}

func piLaunchArgs(model string, args []string) []string {
	var forward []string
	if model != "" {
		if !strings.HasPrefix(model, "ollama/") {
			model = "ollama/" + model
		}
		forward = append(forward, "--model", model)
	}
	return append(forward, args...)
}

//go:embed pi_setup.mjs
var piSetupExtension []byte

func configurePi(bin string, req IntegrationLaunchRequest, offline bool) error {
	model := ""
	if req.ConfigureOnly {
		model = req.ModelOverride
	}
	if model == "ollama/" {
		return fmt.Errorf("missing model name after ollama/")
	}
	dir, err := os.MkdirTemp("", "ollama-pi-setup-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	probe := filepath.Join(dir, "setup.mjs")
	if err := os.WriteFile(probe, piSetupExtension, 0o600); err != nil {
		return err
	}
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	input, err := json.Marshal(map[string]any{
		"cwd": cwd, "agentDir": os.Getenv("PI_CODING_AGENT_DIR"), "args": req.ExtraArgs,
		"offline": offline, "configureOnly": req.ConfigureOnly, "model": strings.TrimPrefix(model, "ollama/"), "plugin": piOllamaSource,
	})
	if err != nil {
		return err
	}
	// Load only our setup extension. It uses the installed Pi SDK to inspect
	// the real profile without running user extensions or starting a session.
	cmd := exec.Command(bin, "--no-extensions", "--no-approve", "--no-mcp", "-e", probe, "--help")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "PI_CODING_AGENT_DIR="+dir, "PI_OFFLINE=1")
	cmd.Stdin = bytes.NewReader(input)
	if err := runPiSetupCommand(cmd, "ollama-pi-ready"); err != nil {
		return fmt.Errorf("Pi setup failed: %w", err)
	}
	return nil
}

func ensurePiInstalled() (string, error) {
	if _, err := exec.LookPath("pi"); err == nil {
		return "pi", nil
	}

	if _, err := exec.LookPath("npm"); err != nil {
		return "", fmt.Errorf("pi is not installed and required dependencies are missing\n\nInstall the following first:\n  npm (Node.js): https://nodejs.org/\n\nThen re-run:\n  ollama launch pi")
	}

	ok, err := ConfirmPrompt("Install Pi with npm?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("pi installation cancelled")
	}

	if err := runPiSetupCommand(exec.Command("npm", "install", "-g", piNpmPackage+"@latest"), ""); err != nil {
		return "", fmt.Errorf("failed to install pi: %w", err)
	}
	if _, err := exec.LookPath("pi"); err != nil {
		return "", fmt.Errorf("pi was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}
	return "pi", nil
}

func runPiSetupCommand(cmd *exec.Cmd, marker string) error {
	if isInteractiveSession() && term.IsTerminal(int(os.Stderr.Fd())) {
		// Keep stdin in normal mode so Ctrl+C interrupts the installer.
		defer defaultSpinner("Preparing Pi").Stop()
	}
	var diagnostics bytes.Buffer
	cmd.Stderr = &diagnostics
	out, err := cmd.Output()
	if err == nil && marker != "" {
		// Pi can print help and exit successfully if the setup extension failed to load.
		out = bytes.TrimSpace(out)
		if string(out[bytes.LastIndexByte(out, '\n')+1:]) != marker {
			err = fmt.Errorf("Pi did not complete Ollama plugin setup")
		}
	}
	if err == nil {
		return nil
	}
	msg := strings.TrimSpace(diagnostics.String())
	if msg == "" && marker == "" {
		msg = strings.TrimSpace(string(out))
	}
	if msg == "" {
		return err
	}
	return fmt.Errorf("%w: %s", err, msg)
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
