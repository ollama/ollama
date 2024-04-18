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

	filepath string
	digest   string
}

func (m *Manifest) Size() (size int64) {
	for _, layer := range append(m.Layers, m.Config) {
		size += layer.Size
	}

	return
}

func (m *Manifest) Remove() error {
	if err := os.Remove(m.filepath); err != nil {
		return err
	}

	for _, layer := range append(m.Layers, m.Config) {
		if err := layer.Remove(); err != nil {
			return err
		}
	}

	manifests, err := GetManifestPath()
	if err != nil {
		return err
	}

	return PruneDirectory(manifests)
}

func ParseNamedManifest(n model.Name) (*Manifest, error) {
	if !n.IsFullyQualified() {
		return nil, model.Unqualified(n)
	}

	manifests, err := GetManifestPath()
	if err != nil {
		return nil, err
	}

	p := filepath.Join(manifests, n.Filepath())

	var m ManifestV2
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	sha256sum := sha256.New()
	if err := json.NewDecoder(io.TeeReader(f, sha256sum)).Decode(&m); err != nil {
		return nil, err
	}

	return &Manifest{
		ManifestV2: m,
		filepath:   p,
		digest:     fmt.Sprintf("%x", sha256sum.Sum(nil)),
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

func Manifests() (map[model.Name]*Manifest, error) {
	manifests, err := GetManifestPath()
	if err != nil {
		return nil, err
	}

	// TODO(mxyng): use something less brittle
	matches, err := filepath.Glob(fmt.Sprintf("%s/*/*/*/*", manifests))
	if err != nil {
		return nil, err
	}

	ms := make(map[model.Name]*Manifest)
	for _, match := range matches {
		rel, err := filepath.Rel(manifests, match)
		if err != nil {
			return nil, err
		}

		n := model.ParseNameFromFilepath(rel)
		if n.IsValid() {
			m, err := ParseNamedManifest(n)
			if err != nil {
				return nil, err
			}

			ms[n] = m
		}
	}

	return ms, nil
}
