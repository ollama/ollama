package build

import (
	"os"
	"path/filepath"
	"sync"
)

var (
	defaultDir = sync.OnceValues(func() (string, error) {
		dir := os.Getenv("OLLAMA_MODELS")
		if dir == "" {
			home, err := os.UserHomeDir()
			if err != nil {
				return "", err
			}
			dir = filepath.Join(home, ".ollama", "models")
		}
		return dir, nil
	})
)

// DefaultDir returns the default directory for models. It returns the value
// of the OLLAMA_MODELS environment variable if set; otherwise it returns
// "$HOME/.ollama/models".
func DefaultDir() (string, error) {
	return defaultDir()
}
