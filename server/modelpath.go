package server

import (
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var (
	ErrInvalidDigestFormat = errors.New("invalid digest format")
)

// modelsDir returns the value of the OLLAMA_MODELS environment variable or the user's home directory if OLLAMA_MODELS is not set.
// The models directory is where Ollama stores its model files and manifests.
func modelsDir() (string, error) {
	if models, exists := os.LookupEnv("OLLAMA_MODELS"); exists {
		return models, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "models"), nil
}

func GetBlobsPath(digest string) (string, error) {
	dir, err := modelsDir()
	if err != nil {
		return "", err
	}

	// only accept actual sha256 digests
	pattern := "^sha256[:-][0-9a-fA-F]{64}$"
	re := regexp.MustCompile(pattern)

	if digest != "" && !re.MatchString(digest) {
		return "", ErrInvalidDigestFormat
	}

	digest = strings.ReplaceAll(digest, ":", "-")
	path := filepath.Join(dir, "blobs", digest)
	dirPath := filepath.Dir(path)
	if digest == "" {
		dirPath = path
	}

	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return "", err
	}

	return path, nil
}
