package defaults

import (
	"log/slog"
	"os"
)

const (
	ENV_OLLAMA_DEFAULT_REGISTRY_ENDPOINT = "OLLAMA_DEFAULT_REGISTRY_ENDPOINT"
	ENV_OLLAMA_DEFAULT_NAMESPACE         = "OLLAMA_DEFAULT_NAMESPACE"
	ENV_OLLAMA_DEFAULT_PROTOCOL_SCHEME   = "OLLAMA_DEFAULT_PROTOCOL_SCHEME"
	ENV_OLLAMA_DEFAULT_TAG               = "OLLAMA_DEFAULT_TAG"
	ENV_OLLAMA_UPDATE_CHECK_ENDPOINT     = "OLLAMA_UPDATE_CHECK_ENDPOINT"
)

// setFromEnvString searches for a environment variable by name. If the value is
// not empty, setFromEnvString will update the destination variable to the value
// found. When the logging level is that of "debug", it will log anytime changes
// are made. Set the sensitive value to true if the value is secure material,
// such as a password.
func setFromEnvString(name string, dest *string, sensitive bool) bool {
	if value := os.Getenv(name); value != "" {
		// handle debugging output for sensitive values
		if sensitive {
			// log only the environment variable name in the debug message
			slog.Debug("setFromEnvString", "evironment_varialbe_name", name)

			// set the value and return true to indicate the mutation happened
			*dest = value
			return true
		}

		// log all the details in the debug message
		slog.Debug("setFromEnvString", "evironment_varialbe_name", name, "new_value", value, "old_value", *dest)

		// set the value and return true to indicate the mutation happened
		*dest = value
		return true
	}

	// return false to indicate no mutation happened
	return false
}
