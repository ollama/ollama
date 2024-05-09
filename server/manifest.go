package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/types/model"
)

type Manifest struct {
	ManifestV2
	Digest string `json:"-"`
}

func (m *Manifest) Size() (size int64) {
	for _, layer := range append(m.Layers, m.Config) {
		size += layer.Size
	}

	return
}

func ParseNamedManifest(name model.Name) (*Manifest, error) {
	if !name.IsFullyQualified() {
		return nil, model.Unqualified(name)
	}

	manifests, err := GetManifestPath()
	if err != nil {
		return nil, err
	}

	var manifest ManifestV2
	manifestfile, err := os.Open(filepath.Join(manifests, name.Filepath()))
	if err != nil {
		return nil, err
	}

	sha256sum := sha256.New()
	if err := json.NewDecoder(io.TeeReader(manifestfile, sha256sum)).Decode(&manifest); err != nil {
		return nil, err
	}

	return &Manifest{
		ManifestV2: manifest,
		Digest:     fmt.Sprintf("%x", sha256sum.Sum(nil)),
	}, nil
}

func WriteManifest(name string, config *Layer, layers []*Layer) error {
	manifest := ManifestV2{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Config:        config,
		Layers:        layers,
	}

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(manifest); err != nil {
		return err
	}

	modelpath := ParseModelPath(name)
	manifestPath, err := modelpath.GetManifestPath()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(manifestPath), 0o755); err != nil {
		return err
	}

	return os.WriteFile(manifestPath, b.Bytes(), 0o644)
}
