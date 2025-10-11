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

	// Try to find manifest in any of the model paths
	manifestPath, err := FindManifestPath(n)
	if err != nil {
		return nil, err
	}

	var m Manifest
	f, err := os.Open(manifestPath)
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

	m.filepath = manifestPath
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

func Manifests(continueOnError bool) (map[model.Name]*Manifest, error) {
	manifestPaths, err := GetManifestPaths()
	if err != nil {
		return nil, err
	}

	ms := make(map[model.Name]*Manifest)
	
	// Search through all manifest directories
	for _, manifestDir := range manifestPaths {
		// TODO(mxyng): use something less brittle
		matches, err := filepath.Glob(filepath.Join(manifestDir, "*", "*", "*", "*"))
		if err != nil {
			if !continueOnError {
				return nil, err
			}
			slog.Warn("failed to glob manifests", "path", manifestDir, "error", err)
			continue
		}

		for _, match := range matches {
			fi, err := os.Stat(match)
			if err != nil {
				if !continueOnError {
					return nil, err
				}
				slog.Warn("failed to stat manifest", "path", match, "error", err)
				continue
			}

			if !fi.IsDir() {
				rel, err := filepath.Rel(manifestDir, match)
				if err != nil {
					if !continueOnError {
						return nil, fmt.Errorf("%s %w", match, err)
					}
					slog.Warn("bad filepath", "path", match, "error", err)
					continue
				}

				n := model.ParseNameFromFilepath(rel)
				if !n.IsValid() {
					if !continueOnError {
						return nil, fmt.Errorf("%s %w", rel, err)
					}
					slog.Warn("bad manifest name", "path", rel)
					continue
				}

				// Skip if we already found this model in a higher priority path
				if _, exists := ms[n]; exists {
					continue
				}

				m, err := ParseNamedManifest(n)
				if err != nil {
					if !continueOnError {
						return nil, fmt.Errorf("%s %w", n, err)
					}
					slog.Warn("bad manifest", "name", n, "error", err)
					continue
				}

				ms[n] = m
			}
		}
	}

	return ms, nil
}
