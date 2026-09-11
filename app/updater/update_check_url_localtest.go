//go:build (windows || darwin) && updater_localtest

package updater

import (
	"os"
	"strings"
)

func updateCheckURLBase() string {
	configuredURL := configuredUpdateCheckURLBase()
	if configuredURL != defaultUpdateCheckURLBase {
		return configuredURL
	}
	if rawURL := strings.TrimSpace(os.Getenv("OLLAMA_TEST_UPDATE_URL")); rawURL != "" {
		return rawURL
	}
	return configuredURL
}
