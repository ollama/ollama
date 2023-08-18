package server

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

type ModelPath struct {
	ProtocolScheme string
	Registry       string
	Namespace      string
	Repository     string
	Tag            string
}

const (
	DefaultRegistry       = "registry.ollama.ai"
	DefaultNamespace      = "library"
	DefaultTag            = "latest"
	DefaultProtocolScheme = "https"
)

var protocolPattern = regexp.MustCompile(`^.*://`)

func ParseModelPath(name string) ModelPath {
	name = protocolPattern.ReplaceAllString(name, "")

	slashParts := strings.Split(name, "/")

	registry := DefaultRegistry
	namespace := DefaultNamespace
	repository := ""
	tag := DefaultTag

	switch len(slashParts) {
	case 3:
		registry = slashParts[0]
		namespace = slashParts[1]
		repository = slashParts[2]
	case 2:
		namespace = slashParts[0]
		repository = slashParts[1]
	case 1:
		repository = slashParts[0]
	default:
		fmt.Println("Invalid image format.")
		return ModelPath{}
	}

	repoParts := strings.Split(repository, ":")
	if len(repoParts) == 2 {
		repository = repoParts[0]
		tag = repoParts[1]
	}

	return ModelPath{
		ProtocolScheme: DefaultProtocolScheme,
		Registry:       registry,
		Namespace:      namespace,
		Repository:     repository,
		Tag:            tag,
	}
}

func (mp ModelPath) GetNamespaceRepository() string {
	return fmt.Sprintf("%s/%s", mp.Namespace, mp.Repository)
}

func (mp ModelPath) GetFullTagname() string {
	return fmt.Sprintf("%s/%s/%s:%s", mp.Registry, mp.Namespace, mp.Repository, mp.Tag)
}

func (mp ModelPath) GetShortTagname() string {
	if mp.Registry == DefaultRegistry {
		if mp.Namespace == DefaultNamespace {
			return fmt.Sprintf("%s:%s", mp.Repository, mp.Tag)
		}
		return fmt.Sprintf("%s/%s:%s", mp.Namespace, mp.Repository, mp.Tag)
	}
	return fmt.Sprintf("%s/%s/%s:%s", mp.Registry, mp.Namespace, mp.Repository, mp.Tag)
}

func (mp ModelPath) GetManifestPath(createDir bool) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	path := filepath.Join(home, ".ollama", "models", "manifests", mp.Registry, mp.Namespace, mp.Repository, mp.Tag)
	if createDir {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return "", err
		}
	}

	return path, nil
}

func GetManifestPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(home, ".ollama", "models", "manifests"), nil
}

func GetBlobsPath(digest string) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	if runtime.GOOS == "windows" {
		digest = strings.ReplaceAll(digest, ":", "-")
	}

	path := filepath.Join(home, ".ollama", "models", "blobs", digest)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}

	return path, nil
}
