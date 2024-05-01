package server

import (
	"cmp"
	"fmt"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/client/registry"
	"github.com/ollama/ollama/types/model"
)

// cache is a simple demo disk cache. it does not validate anything
type cache struct {
	dir string
}

func defaultCache() registry.Cache {
	homeDir, _ := os.UserHomeDir()
	if homeDir == "" {
		panic("could not determine home directory")
	}
	modelsDir := cmp.Or(
		os.Getenv("OLLAMA_MODELS"),
		filepath.Join(homeDir, ".ollama", "models"),
	)
	return &cache{modelsDir}
}

func invalidDigest(digest string) error {
	return fmt.Errorf("invalid digest: %s", digest)
}

func (c *cache) OpenLayer(d model.Digest) (registry.ReadAtSeekCloser, error) {
	return os.Open(c.LayerFile(d))
}

func (c *cache) LayerFile(d model.Digest) string {
	return filepath.Join(c.dir, "blobs", d.String())
}

func (c *cache) PutLayerFile(d model.Digest, fromPath string) error {
	if !d.IsValid() {
		return invalidDigest(d.String())
	}
	bfile := c.LayerFile(d)
	dir, _ := filepath.Split(bfile)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	return os.Rename(fromPath, bfile)
}

func (c *cache) ManifestData(name model.Name) []byte {
	if !name.IsFullyQualified() {
		return nil
	}
	data, err := os.ReadFile(filepath.Join(c.dir, "manifests", name.Filepath()))
	if err != nil {
		return nil
	}
	return data
}

func (c *cache) SetManifestData(name model.Name, data []byte) error {
	if !name.IsFullyQualified() {
		return fmt.Errorf("invalid name: %s", name)
	}
	filep := filepath.Join(c.dir, "manifests", name.Filepath())
	dir, _ := filepath.Split(filep)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	return os.WriteFile(filep, data, 0644)
}
