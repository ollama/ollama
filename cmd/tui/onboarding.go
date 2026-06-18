package tui

// RunAgentSignInOnboarding asks whether the user wants to sign in before
// starting the root agent flow.
func RunAgentSignInOnboarding() (bool, error) {
	return RunConfirmWithOptions(
		"Sign in to use web search and cloud models with Ollama, Claude Code, OpenClaw, Hermes and more?\n\nYou can keep using local models without signing in.",
		ConfirmOptions{
			YesLabel:    "Sign in",
			NoLabel:     "Not now",
			PlainPrompt: true,
		},
	)
}
