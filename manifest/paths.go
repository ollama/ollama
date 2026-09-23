package manifest

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

var ErrInvalidDigestFormat = errors.New("invalid digest format")

// a manifest spells a digest with ":", a filename with "-"
var digestPattern = regexp.MustCompile(`^sha256[:-][0-9a-fA-F]{64}$`)

// ValidateDigest reports whether digest names a blob.
func ValidateDigest(digest string) error {
	if !digestPattern.MatchString(digest) {
		return ErrInvalidDigestFormat
	}

	return nil
}

func Path() (string, error) {
	path := filepath.Join(envconfig.Models(), "manifests")
	if err := os.MkdirAll(path, 0o755); err != nil {
		return "", fmt.Errorf("%w: ensure path elements are traversable", err)
	}

	return path, nil
}

// PathForName returns the path to the manifest file for a specific model name.
func PathForName(n model.Name) (string, error) {
	if !n.IsValid() {
		return "", os.ErrNotExist
	}

	manifests, err := Path()
	if err != nil {
		return "", err
	}

	return filepath.Join(manifests, n.Filepath()), nil
}

func BlobsPath(digest string) (string, error) {
	// the empty digest names the blobs directory
	if digest != "" {
		if err := ValidateDigest(digest); err != nil {
			return "", err
		}
	}

	digest = strings.ReplaceAll(digest, ":", "-")
	path := filepath.Join(envconfig.Models(), "blobs", digest)
	dirPath := filepath.Dir(path)
	if digest == "" {
		dirPath = path
	}

	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return "", fmt.Errorf("%w: ensure path elements are traversable", err)
	}

	return path, nil
}

// PruneDirectory removes empty directories recursively.
func PruneDirectory(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}

	if info.IsDir() && info.Mode()&os.ModeSymlink == 0 {
		entries, err := os.ReadDir(path)
		if err != nil {
			return err
		}

		for _, entry := range entries {
			if err := PruneDirectory(filepath.Join(path, entry.Name())); err != nil {
				return err
			}
		}

		entries, err = os.ReadDir(path)
		if err != nil {
			return err
		}

		if len(entries) > 0 {
			return nil
		}

		return os.Remove(path)
	}

	return nil
}
