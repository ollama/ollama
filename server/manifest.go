package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/types/model"
)

type Manifest struct {
	SchemaVersion int     `json:"schemaVersion"`
	MediaType     string  `json:"mediaType"`
	Config        Layer   `json:"config"`
	Layers        []Layer `json:"layers"`

	filepath string
	fi       os.FileInfo
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

	manifests, err := GetManifestPath()
	if err != nil {
		return err
	}

	return PruneDirectory(manifests)
}

func (m *Manifest) RemoveLayers() error {
	for _, layer := range append(m.Layers, m.Config) {
		if layer.Digest != "" {
			if err := layer.Remove(); errors.Is(err, os.ErrNotExist) {
				slog.Debug("layer does not exist", "digest", layer.Digest)
			} else if err != nil {
				return err
			}
		}
	}

	return nil
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

	var m Manifest
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}

	sha256sum := sha256.New()
	if err := json.NewDecoder(io.TeeReader(f, sha256sum)).Decode(&m); err != nil {
		return nil, err
	}

	m.filepath = p
	m.fi = fi
	m.digest = hex.EncodeToString(sha256sum.Sum(nil))

	return &m, nil
}

func WriteManifest(name model.Name, config Layer, layers []Layer) error {
	manifests, err := GetManifestPath()
	if err != nil {
		return err
	}

	p := filepath.Join(manifests, name.Filepath())
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}

	f, err := os.Create(p)
	if err != nil {
		return err
	}
	defer f.Close()

	m := Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Config:        config,
		Layers:        layers,
	}

	return json.NewEncoder(f).Encode(m)
}

func Manifests() (map[model.Name]*Manifest, error) {
	manifests, err := GetManifestPath()
	if err != nil {
		return nil, err
	}

	// TODO(mxyng): use something less brittle
	matches, err := filepath.Glob(filepath.Join(manifests, "*", "*", "*", "*"))
	if err != nil {
		return nil, err
	}

	ms := make(map[model.Name]*Manifest)
	for _, match := range matches {
		fi, err := os.Stat(match)
		if err != nil {
			return nil, err
		}

		if !fi.IsDir() {
			rel, err := filepath.Rel(manifests, match)
			if err != nil {
				slog.Warn("bad filepath", "path", match, "error", err)
				continue
			}

			n := model.ParseNameFromFilepath(rel)
			if !n.IsValid() {
				slog.Warn("bad manifest name", "path", rel)
				continue
			}

			m, err := ParseNamedManifest(n)
			if syntax := &(json.SyntaxError{}); errors.As(err, &syntax) {
				slog.Warn("bad manifest", "name", n, "error", err)
				continue
			} else if err != nil {
				return nil, fmt.Errorf("%s: %w", n, err)
			}

			ms[n] = m
		}
	}

	return ms, nil
}
