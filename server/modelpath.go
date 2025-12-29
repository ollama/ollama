package server

import (
	"errors"
	"fmt"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

type ModelPath struct {
	ProtocolScheme string
	Registry       string
	Namespace      string
	Kind           string // Optional: "skill", "agent", or empty for models
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
	ErrInvalidImageFormat  = errors.New("invalid image format")
	ErrInvalidDigestFormat = errors.New("invalid digest format")
	ErrInvalidProtocol     = errors.New("invalid protocol scheme")
	ErrInsecureProtocol    = errors.New("insecure protocol http")
	ErrModelPathInvalid    = errors.New("invalid model path")
)

func ParseModelPath(name string) ModelPath {
	mp := ModelPath{
		ProtocolScheme: DefaultProtocolScheme,
		Registry:       DefaultRegistry,
		Namespace:      DefaultNamespace,
		Kind:           "",
		Repository:     "",
		Tag:            DefaultTag,
	}

	before, after, found := strings.Cut(name, "://")
	if found {
		mp.ProtocolScheme = before
		name = after
	}

	name = strings.ReplaceAll(name, string(os.PathSeparator), "/")
	parts := strings.Split(name, "/")
	switch len(parts) {
	case 4:
		// host/namespace/kind/model or host/namespace/model:tag with kind
		mp.Registry = parts[0]
		mp.Namespace = parts[1]
		if model.ValidKinds[parts[2]] {
			mp.Kind = parts[2]
			mp.Repository = parts[3]
		} else {
			// Not a valid kind, treat as old format with extra part
			mp.Repository = parts[3]
		}
	case 3:
		// Could be: host/namespace/model OR namespace/kind/model
		if model.ValidKinds[parts[1]] {
			// namespace/kind/model
			mp.Namespace = parts[0]
			mp.Kind = parts[1]
			mp.Repository = parts[2]
		} else {
			// host/namespace/model
			mp.Registry = parts[0]
			mp.Namespace = parts[1]
			mp.Repository = parts[2]
		}
	case 2:
		// Could be: namespace/model OR kind/model
		if model.ValidKinds[parts[0]] {
			// kind/model (library skill)
			mp.Kind = parts[0]
			mp.Repository = parts[1]
		} else {
			// namespace/model
			mp.Namespace = parts[0]
			mp.Repository = parts[1]
		}
	case 1:
		mp.Repository = parts[0]
	}

	if repo, tag, found := strings.Cut(mp.Repository, ":"); found {
		mp.Repository = repo
		mp.Tag = tag
	}

	return mp
}

func (mp ModelPath) GetNamespaceRepository() string {
	if mp.Kind != "" {
		return fmt.Sprintf("%s/%s/%s", mp.Namespace, mp.Kind, mp.Repository)
	}
	return fmt.Sprintf("%s/%s", mp.Namespace, mp.Repository)
}

func (mp ModelPath) GetFullTagname() string {
	if mp.Kind != "" {
		return fmt.Sprintf("%s/%s/%s/%s:%s", mp.Registry, mp.Namespace, mp.Kind, mp.Repository, mp.Tag)
	}
	return fmt.Sprintf("%s/%s/%s:%s", mp.Registry, mp.Namespace, mp.Repository, mp.Tag)
}

func (mp ModelPath) GetShortTagname() string {
	if mp.Registry == DefaultRegistry {
		if mp.Namespace == DefaultNamespace {
			if mp.Kind != "" {
				return fmt.Sprintf("%s/%s:%s", mp.Kind, mp.Repository, mp.Tag)
			}
			return fmt.Sprintf("%s:%s", mp.Repository, mp.Tag)
		}
		if mp.Kind != "" {
			return fmt.Sprintf("%s/%s/%s:%s", mp.Namespace, mp.Kind, mp.Repository, mp.Tag)
		}
		return fmt.Sprintf("%s/%s:%s", mp.Namespace, mp.Repository, mp.Tag)
	}
	if mp.Kind != "" {
		return fmt.Sprintf("%s/%s/%s/%s:%s", mp.Registry, mp.Namespace, mp.Kind, mp.Repository, mp.Tag)
	}
	return fmt.Sprintf("%s/%s/%s:%s", mp.Registry, mp.Namespace, mp.Repository, mp.Tag)
}

// GetManifestPath returns the path to the manifest file for the given model path, it is up to the caller to create the directory if it does not exist.
func (mp ModelPath) GetManifestPath() (string, error) {
	name := model.Name{
		Host:      mp.Registry,
		Namespace: mp.Namespace,
		Kind:      mp.Kind,
		Model:     mp.Repository,
		Tag:       mp.Tag,
	}
	if !name.IsValid() {
		return "", fs.ErrNotExist
	}
	return filepath.Join(envconfig.Models(), "manifests", name.Filepath()), nil
}

func (mp ModelPath) BaseURL() *url.URL {
	return &url.URL{
		Scheme: mp.ProtocolScheme,
		Host:   mp.Registry,
	}
}

func GetManifestPath() (string, error) {
	path := filepath.Join(envconfig.Models(), "manifests")
	if err := os.MkdirAll(path, 0o755); err != nil {
		return "", fmt.Errorf("%w: ensure path elements are traversable", err)
	}

	return path, nil
}

func GetBlobsPath(digest string) (string, error) {
	// only accept actual sha256 digests
	pattern := "^sha256[:-][0-9a-fA-F]{64}$"
	re := regexp.MustCompile(pattern)

	if digest != "" && !re.MatchString(digest) {
		return "", ErrInvalidDigestFormat
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
