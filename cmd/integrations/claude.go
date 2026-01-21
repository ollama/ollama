package integrations

var claudeIntegration = &integrationDef{
	Name:        "Claude",
	DisplayName: "Claude Code",
	Command:     "claude",
	EnvVars: func(model string) []envVar {
		return []envVar{
			{Name: "ANTHROPIC_BASE_URL", Value: "http://localhost:11434"},
			{Name: "ANTHROPIC_API_KEY", Value: "ollama"},
			{Name: "ANTHROPIC_AUTH_TOKEN", Value: "ollama"},
		}
	},
	Args: func(model string) []string {
		if model == "" {
			return nil
		}
		return []string{"--model", model}
	},
	CheckInstall: checkCommand("claude", "Install with: npm install -g @anthropic-ai/claude-code"),
}
