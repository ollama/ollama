package server

import (
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
	stat     os.FileInfo
	digest   string `json:"-"`
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

func ParseNamedManifest(name model.Name) (*Manifest, error) {
	manifests, err := GetManifestPath()
	if err != nil {
		return nil, err
	}

	manifestpath := filepath.Join(manifests, name.FilepathNoBuild())

	var manifest ManifestV2
	manifestfile, err := os.Open(manifestpath)
	if err != nil {
		return nil, err
	}
	defer manifestfile.Close()

	stat, err := manifestfile.Stat()
	if err != nil {
		return nil, err
	}

	sha256sum := sha256.New()
	if err := json.NewDecoder(io.TeeReader(manifestfile, sha256sum)).Decode(&manifest); err != nil {
		return nil, err
	}

	return &Manifest{
		ManifestV2: manifest,
		filepath:   manifestpath,
		stat:       stat,
		digest:     fmt.Sprintf("%x", sha256sum.Sum(nil)),
	}, nil
}

func NewManifest(name model.Name, config *Layer, layers []*Layer) (*Manifest, error) {
	manifests, err := GetManifestPath()
	if err != nil {
		return nil, err
	}

	manifestpath := filepath.Join(manifests, name.FilepathNoBuild())
	if err := os.MkdirAll(filepath.Dir(manifestpath), 0o755); err != nil {
		return nil, err
	}

	manifestfile, err := os.Create(manifestpath)
	if err != nil {
		return nil, err
	}
	defer manifestfile.Close()

	manifest := Manifest{
		ManifestV2: ManifestV2{
			SchemaVersion: 2,
			MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
			Config:        config,
			Layers:        layers,
		},
	}

	sha256sum := sha256.New()
	if err := json.NewEncoder(io.MultiWriter(manifestfile, sha256sum)).Encode(manifest); err != nil {
		return nil, err
	}

	manifest.filepath = manifestpath
	manifest.stat, err = manifestfile.Stat()
	if err != nil {
		return nil, err
	}

	manifest.digest = fmt.Sprintf("%x", sha256sum.Sum(nil))
	return &manifest, nil
}

type iter_Seq2[A, B any] func(func(A, B) bool)

func Manifests() iter_Seq2[model.Name, *Manifest] {
	return func(yield func(model.Name, *Manifest) bool) {
		manifests, err := GetManifestPath()
		if err != nil {
			return
		}

		// TODO(mxyng): use something less brittle
		matches, err := filepath.Glob(fmt.Sprintf("%s/*/*/*/*", manifests))
		if err != nil {
			return
		}

		for _, match := range matches {
			rel, err := filepath.Rel(manifests, match)
			if err != nil {
				return
			}

			name := model.ParseNameFromFilepath(rel, "")
			if name.IsValid() {
				manifest, err := ParseNamedManifest(name)
				if err != nil {
					return
				}

				if !yield(name, manifest) {
					return
				}
			}
		}
	}
}
