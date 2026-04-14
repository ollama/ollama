package cloud

import (
	"github.com/ollama/ollama/envconfig"
)

const DisabledMessagePrefix = "ollama cloud is disabled"

// Status returns whether cloud is disabled and the source of the decision.
// Source is one of: "none", "env", "config", "both".
func Status() (disabled bool, source string) {
	return envconfig.NoCloud(), envconfig.NoCloudSource()
}

func Disabled() bool {
	return envconfig.NoCloud()
}

func DisabledError(operation string) string {
	if operation == "" {
		return DisabledMessagePrefix
	}

	return DisabledMessagePrefix + ": " + operation
}
