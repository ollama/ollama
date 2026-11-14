package launch

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/cmd/internal/fileutil"
)

// Pi merges these model preferences with project values taking precedence.
func piModelSettings() (map[string]any, error) {
	agentDir, err := piAgentDir()
	if err != nil {
		return nil, err
	}
	cwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	settings := make(map[string]any)
	for i, path := range []string{filepath.Join(agentDir, "settings.json"), filepath.Join(cwd, ".pi", "settings.json")} {
		values, err := fileutil.ReadJSON(path)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			if i == 0 {
				return nil, fmt.Errorf("read Pi config %s: %w", path, err)
			}
			// Pi reports project errors and continues with global preferences.
			return settings, err
		}
		maps.Copy(settings, values)
	}
	return settings, nil
}

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
