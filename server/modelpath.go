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
	case 3:
		mp.Registry = parts[0]
		mp.Namespace = parts[1]
		mp.Repository = parts[2]
	case 2:
		mp.Namespace = parts[0]
		mp.Repository = parts[1]
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

// GetManifestPath returns the path to the manifest file for the given model path, it is up to the caller to create the directory if it does not exist.
func (mp ModelPath) GetManifestPath() (string, error) {
	name := model.Name{
		Host:      mp.Registry,
		Namespace: mp.Namespace,
		Model:     mp.Repository,
		Tag:       mp.Tag,
	}
	if !name.IsValid() {
		return "", fs.ErrNotExist
	}
	
	// Use the new multi-path search function
	return FindManifestPath(name)
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

// GetManifestPaths returns all manifest directories from all model paths
func GetManifestPaths() ([]string, error) {
	modelPaths := envconfig.ModelPaths()
	manifestPaths := make([]string, 0, len(modelPaths))
	
	for _, modelPath := range modelPaths {
		path := filepath.Join(modelPath, "manifests")
		manifestPaths = append(manifestPaths, path)
	}
	
	return manifestPaths, nil
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

// GetBlobsPaths returns all blob directories from all model paths
func GetBlobsPaths() ([]string, error) {
	modelPaths := envconfig.ModelPaths()
	blobsPaths := make([]string, 0, len(modelPaths))
	
	for _, modelPath := range modelPaths {
		path := filepath.Join(modelPath, "blobs")
		blobsPaths = append(blobsPaths, path)
	}
	
	return blobsPaths, nil
}

// FindBlobPath searches for a blob file across all model paths and returns the first found path
func FindBlobPath(digest string) (string, error) {
	// only accept actual sha256 digests
	pattern := "^sha256[:-][0-9a-fA-F]{64}$"
	re := regexp.MustCompile(pattern)

	if digest != "" && !re.MatchString(digest) {
		return "", ErrInvalidDigestFormat
	}

	digest = strings.ReplaceAll(digest, ":", "-")
	modelPaths := envconfig.ModelPaths()
	
	// Search through all model paths
	for _, modelPath := range modelPaths {
		path := filepath.Join(modelPath, "blobs", digest)
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}
	
	// If not found, return the path in the first (primary) model directory
	if len(modelPaths) > 0 {
		path := filepath.Join(modelPaths[0], "blobs", digest)
		dirPath := filepath.Dir(path)
		if digest == "" {
			dirPath = path
		}

		if err := os.MkdirAll(dirPath, 0o755); err != nil {
			return "", fmt.Errorf("%w: ensure path elements are traversable", err)
		}
		return path, nil
	}
	
	return "", fmt.Errorf("no model paths configured")
}

// FindManifestPath searches for a manifest file across all model paths and returns the first found path
func FindManifestPath(name model.Name) (string, error) {
	if !name.IsValid() {
		return "", fs.ErrNotExist
	}
	
	modelPaths := envconfig.ModelPaths()
	filePath := name.Filepath()
	
	// Search through all model paths
	for _, modelPath := range modelPaths {
		path := filepath.Join(modelPath, "manifests", filePath)
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}
	
	// If not found, return the path in the first (primary) model directory
	if len(modelPaths) > 0 {
		path := filepath.Join(modelPaths[0], "manifests", filePath)
		return path, nil
	}
	
	return "", fmt.Errorf("no model paths configured")
}
