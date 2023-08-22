package server

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
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

var (
	ErrInvalidImageFormat = errors.New("invalid image format")
	ErrInvalidProtocol    = errors.New("invalid protocol scheme")
	ErrInsecureProtocol   = errors.New("insecure protocol http")
)

func ParseModelPath(name string, allowInsecure bool) (ModelPath, error) {
	mp := ModelPath{
		ProtocolScheme: DefaultProtocolScheme,
		Registry:       DefaultRegistry,
		Namespace:      DefaultNamespace,
		Repository:     "",
		Tag:            DefaultTag,
	}

	protocol, rest, didSplit := strings.Cut(name, "://")
	if didSplit {
		if protocol == "https" || protocol == "http" && allowInsecure {
			mp.ProtocolScheme = protocol
			name = rest
		} else if protocol == "http" && !allowInsecure {
			return ModelPath{}, ErrInsecureProtocol
		} else {
			return ModelPath{}, ErrInvalidProtocol
		}
	}

	slashParts := strings.Split(name, "/")
	switch len(slashParts) {
	case 3:
		mp.Registry = slashParts[0]
		mp.Namespace = slashParts[1]
		mp.Repository = slashParts[2]
	case 2:
		mp.Namespace = slashParts[0]
		mp.Repository = slashParts[1]
	case 1:
		mp.Repository = slashParts[0]
	default:
		return ModelPath{}, ErrInvalidImageFormat
	}

	if repo, tag, didSplit := strings.Cut(mp.Repository, ":"); didSplit {
		mp.Repository = repo
		mp.Tag = tag
	}

	return mp, nil
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
