package config

var claudeIntegration = &integration{
	Name:        "Claude",
	DisplayName: "Claude Code",
	Command:     "claude",
	EnvVars: func(model string) []envVar {
		return []envVar{
			{Name: "ANTHROPIC_BASE_URL", Value: "http://localhost:11434"},
			{Name: "ANTHROPIC_API_KEY", Value: ""}, // Must be set to skip error message in Claude
			{Name: "ANTHROPIC_AUTH_TOKEN", Value: "ollama"},
		}
	},
	Args: func(model string) []string {
		if model == "" {
			return nil
		}
		return []string{"--model", model}
	},
	CheckInstall: checkCommand("claude", "install from https://code.claude.com/docs/en/quickstart"),
}
