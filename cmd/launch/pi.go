package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/api"
	"golang.org/x/mod/semver"
)

// Pi implements Runner for Pi (Pi Coding Agent) integration.
type Pi struct{}

const (
	piNpmPackage = "@earendil-works/pi-coding-agent"
	piOllamaPkg  = "@ollama/pi"
	piMinVersion = "1.5.2"
)

var piOllamaSource = "git:github.com/ollama/pi-ollama"

func (p *Pi) String() string { return "Pi" }

func (p *Pi) Run(model string, _ []LaunchModel, args []string) error {
	forward := piLaunchArgs(model, args)
	bin, err := preparePi()
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "\n%sLaunching Pi...%s\n\n", ansiGray, ansiReset)
	cmd := exec.Command(bin, forward...)
	cmd.Env = append(os.Environ(), "OLLAMA_LAUNCH_PI=1")
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

func preparePi() (string, error) {
	fmt.Fprintf(os.Stderr, "\n%sPreparing Pi...%s\n", ansiGray, ansiReset)

	fmt.Fprintf(os.Stderr, "%sChecking Pi installation...%s\n", ansiGray, ansiReset)
	bin, err := ensurePiInstalled()
	if err != nil {
		return "", err
	}
	if err := checkPiVersion(bin); err != nil {
		return "", err
	}
	if err := ensurePiOllamaPackage(bin); err != nil {
		return "", err
	}
	return bin, nil
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

func piOllamaInstalled(output string) bool {
	for _, line := range strings.Split(output, "\n") {
		path := strings.TrimSpace(line)
		if !filepath.IsAbs(path) {
			continue
		}
		data, err := os.ReadFile(filepath.Join(path, "package.json"))
		if err != nil {
			continue
		}
		var pkg struct{ Name string }
		if json.Unmarshal(data, &pkg) != nil || pkg.Name != piOllamaPkg {
			continue
		}
		if _, err := os.Stat(filepath.Join(path, "index.ts")); err == nil {
			return true
		}
	}
	return false
}

func ensurePiOllamaPackage(bin string) error {
	list := func() (bool, error) {
		cmd := exec.Command(bin, "list")
		cmd.Env = append(os.Environ(), "FORCE_COLOR=0")
		output, err := cmd.CombinedOutput()
		if err != nil {
			return false, fmt.Errorf("could not check Pi packages: %w; run 'pi list' for details", err)
		}
		return piOllamaInstalled(string(output)), nil
	}
	installed, err := list()
	if err != nil || installed {
		return err
	}
	fmt.Fprintf(os.Stderr, "%sInstalling Ollama plugin...%s\n", ansiGray, ansiReset)
	cmd := exec.Command(bin, "install", piOllamaSource)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("could not install the Ollama plugin: %w; try 'pi install %s'", err, piOllamaSource)
	}
	installed, err = list()
	if err != nil {
		return err
	}
	if !installed {
		return fmt.Errorf("the Ollama plugin is missing after installation; run 'pi install %s'", piOllamaSource)
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
	if err := runQuietCommand("npm", "install", "-g", piNpmPackage+"@latest"); err != nil {
		return fmt.Errorf("failed to install pi: %w", err)
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
