package paths

import (
	"os"
)

const ollamaHomeEnvVar = "OLLAMA_HOME"

func OllamaHomeDir() (string, error) {
	ev := os.Getenv(ollamaHomeEnvVar)
	if len(ev) > 0 {
		return ev, nil
	}

	return os.UserHomeDir()
}
