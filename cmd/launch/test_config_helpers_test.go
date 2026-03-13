package launch

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/cmd/config"
)

var (
	integrations       map[string]Runner
	integrationAliases map[string]bool
	integrationOrder   = launcherIntegrationOrder
)

func init() {
	integrations = buildTestIntegrations()
	integrationAliases = buildTestIntegrationAliases()
}

func buildTestIntegrations() map[string]Runner {
	result := make(map[string]Runner, len(integrationSpecsByName))
	for name, spec := range integrationSpecsByName {
		result[strings.ToLower(name)] = spec.Runner
	}
	return result
}

func buildTestIntegrationAliases() map[string]bool {
	result := make(map[string]bool)
	for _, spec := range integrationSpecs {
		for _, alias := range spec.Aliases {
			result[strings.ToLower(alias)] = true
		}
	}
	return result
}

func setTestHome(t *testing.T, dir string) {
	t.Helper()
	setLaunchTestHome(t, dir)
}

func SaveIntegration(appName string, models []string) error {
	return config.SaveIntegration(appName, models)
}

func LoadIntegration(appName string) (*config.IntegrationConfig, error) {
	return config.LoadIntegration(appName)
}

func SaveAliases(appName string, aliases map[string]string) error {
	return config.SaveAliases(appName, aliases)
}

func LastModel() string {
	return config.LastModel()
}

func SetLastModel(model string) error {
	return config.SetLastModel(model)
}

func LastSelection() string {
	return config.LastSelection()
}

func SetLastSelection(selection string) error {
	return config.SetLastSelection(selection)
}

func IntegrationModel(appName string) string {
	return config.IntegrationModel(appName)
}

func IntegrationModels(appName string) []string {
	return config.IntegrationModels(appName)
}

func integrationOnboarded(appName string) error {
	return config.MarkIntegrationOnboarded(appName)
}
