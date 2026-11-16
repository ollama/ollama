package launch

// Pi's native scoped picker starts with Ollama models; Tab still shows all providers.
// Apply it only for this launch, without replacing an explicit Pi model scope.
func piUseDefaultModelScope(args []string) bool {
	if piControlsLaunch(args) || piHasModelArgs(args) {
		return false
	}
	settings, err := piModelSettings()
	if err != nil {
		return false
	}
	if _, exists := settings["enabledModels"]; exists {
		return false
	}
	provider, _ := settings["defaultProvider"].(string)
	return provider == "" || provider == "ollama"
}
