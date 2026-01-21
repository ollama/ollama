package integrations

import (
	"fmt"
	"os/exec"
	"strings"

	"golang.org/x/mod/semver"
)

var codexIntegration = &integrationDef{
	Name:        "Codex",
	DisplayName: "Codex",
	Command:     "codex",
	EnvVars: func(model string) []envVar {
		return []envVar{}
	},
	Args: func(model string) []string {
		if model == "" {
			return []string{"--oss"}
		}
		return []string{"--oss", "-m", model}
	},
	CheckInstall: checkCodexVersion,
}

func checkCodexVersion() error {
	if _, err := exec.LookPath("codex"); err != nil {
		return fmt.Errorf("codex is not installed. Install with: npm install -g @openai/codex")
	}

	out, err := exec.Command("codex", "--version").Output()
	if err != nil {
		return fmt.Errorf("failed to get codex version: %w", err)
	}

	// Parse output like "codex-cli 0.87.0"
	fields := strings.Fields(strings.TrimSpace(string(out)))
	if len(fields) < 2 {
		return fmt.Errorf("unexpected codex version output: %s", string(out))
	}

	version := "v" + fields[len(fields)-1]
	minVersion := "v0.81.0"

	if semver.Compare(version, minVersion) < 0 {
		return fmt.Errorf("codex version %s is too old, minimum required is %s. Update with: npm update -g @openai/codex", fields[len(fields)-1], "0.81.0")
	}

	return nil
}
