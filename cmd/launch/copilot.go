package launch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"

	"github.com/ollama/ollama/envconfig"
)

// Copilot implements Runner for GitHub Copilot CLI integration.
type Copilot struct{}

func (c *Copilot) String() string { return "Copilot CLI" }

const (
	copilotFallbackPromptTokens = 4_096
	copilotDefaultOutputTokens  = 64_000
)

func (c *Copilot) args(model string, extra []string) []string {
	var args []string
	if model != "" {
		args = append(args, "--model", model)
	}
	args = append(args, extra...)
	return args
}

func (c *Copilot) findPath() (string, error) {
	if p, err := exec.LookPath("copilot"); err == nil {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	name := "copilot"
	if runtime.GOOS == "windows" {
		name = "copilot.exe"
	}
	fallback := filepath.Join(home, ".local", "bin", name)
	if _, err := os.Stat(fallback); err != nil {
		return "", err
	}
	return fallback, nil
}

func (c *Copilot) Run(model string, models []LaunchModel, args []string) error {
	copilotPath, err := c.findPath()
	if err != nil {
		return fmt.Errorf("copilot is not installed, install from https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli")
	}

	cmd := exec.Command(copilotPath, c.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	cmd.Env = append(os.Environ(), c.envVars(model, models)...)

	return cmd.Run()
}

// envVars returns the environment variables that configure Copilot CLI
// to use Ollama as its model provider.
func (c *Copilot) envVars(model string, models []LaunchModel) []string {
	promptTokens, outputTokens := copilotTokenLimits(model, models)
	env := []string{
		"COPILOT_PROVIDER_BASE_URL=" + envconfig.Host().String() + "/v1",
		"COPILOT_PROVIDER_API_KEY=",
		"COPILOT_PROVIDER_WIRE_API=responses",
		"COPILOT_PROVIDER_MAX_PROMPT_TOKENS=" + strconv.Itoa(promptTokens),
		"COPILOT_PROVIDER_MAX_OUTPUT_TOKENS=" + strconv.Itoa(outputTokens),
	}

	if model != "" {
		env = append(env, "COPILOT_MODEL="+model)
	}

	return env
}

func copilotTokenLimits(model string, models []LaunchModel) (int, int) {
	launchModel := copilotLaunchModel(model, models)

	promptTokens := launchModel.ContextLength
	if promptTokens <= 0 {
		promptTokens = launchModel.Details.ContextLength
	}
	if !isCloudModelName(launchModel.Name) && launchModel.Details.Format != "safetensors" {
		if ctxLen := envconfig.ContextLength(); ctxLen > 0 {
			promptTokens = int(ctxLen)
		}
	}
	if promptTokens <= 0 {
		promptTokens = copilotFallbackPromptTokens
	}

	outputTokens := launchModel.MaxOutputTokens
	if outputTokens <= 0 {
		outputTokens = copilotDefaultOutputTokens
	}

	return promptTokens, outputTokens
}

func copilotLaunchModel(model string, models []LaunchModel) LaunchModel {
	if model != "" {
		if launchModel, ok := findLaunchModel(models, model); ok {
			return launchModel.WithCloudLimits()
		}
		return fallbackLaunchModel(model)
	}
	if len(models) > 0 {
		return models[0].WithCloudLimits()
	}
	return LaunchModel{}
}
