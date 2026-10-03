package manifest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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

func (m *Manifest) Digest() string {
	return m.digest
}

func (m *Manifest) FileInfo() os.FileInfo {
	return m.fi
}

// ConfigLayer returns the JSON layer stored under configPath.
func (m *Manifest) ConfigLayer(configPath string) (Layer, bool) {
	for _, layer := range m.Layers {
		if layer.MediaType == "application/vnd.ollama.image.json" && layer.Name == configPath {
			return layer, true
		}
	}
	return Layer{}, false
}

// ReadConfig returns the contents of the JSON layer stored under configPath.
func (m *Manifest) ReadConfig(configPath string) ([]byte, error) {
	layer, ok := m.ConfigLayer(configPath)
	if !ok {
		return nil, fmt.Errorf("config %q not found in manifest", configPath)
	}
	blobPath, err := BlobsPath(layer.Digest)
	if err != nil {
		return nil, err
	}
	return os.ReadFile(blobPath)
}

// ReadConfigJSON reads and unmarshals a config layer as JSON.
func (m *Manifest) ReadConfigJSON(configPath string, v any) error {
	data, err := m.ReadConfig(configPath)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, v)
}

// TensorLayers returns the layers holding tensor data.
func (m *Manifest) TensorLayers() []Layer {
	var layers []Layer
	for _, layer := range m.Layers {
		if layer.MediaType == MediaTypeImageTensor {
			layers = append(layers, layer)
		}
	}
	return layers
}

func (m *Manifest) Remove() error {
	if err := os.Remove(m.filepath); err != nil {
		return err
	}

	manifests, err := Path()
	if err != nil {
		return err
	}

	return PruneDirectory(manifests)
}

// RemoveLayers deletes the layers no other manifest references. The digests it
// removed are returned even alongside an error, so callers can clean up
// anything derived from them.
func (m *Manifest) RemoveLayers() ([]string, error) {
	ms, err := Manifests(true)
	if err != nil {
		return nil, err
	}

	// Build set of digests still in use by other manifests
	inUse := make(map[string]struct{})
	for _, other := range ms {
		for _, layer := range append(other.Layers, other.Config) {
			if layer.Digest != "" {
				inUse[layer.Digest] = struct{}{}
			}
		}
	}

	// Remove layers not used by any other manifest
	var removed []string
	for _, layer := range append(m.Layers, m.Config) {
		if layer.Digest == "" {
			continue
		}
		if _, used := inUse[layer.Digest]; used {
			continue
		}
		blob, err := BlobsPath(layer.Digest)
		if err != nil {
			return removed, err
		}
		if err := os.Remove(blob); os.IsNotExist(err) {
			slog.Debug("layer does not exist", "digest", layer.Digest)
		} else if err != nil {
			return removed, err
		}
		removed = append(removed, layer.Digest)
	}

	return removed, nil
}

func ParseNamedManifest(n model.Name) (*Manifest, error) {
	if !n.IsFullyQualified() {
		return nil, model.Unqualified(n)
	}

	manifests, err := Path()
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
	manifests, err := Path()
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
	manifests, err := Path()
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
				if !continueOnError {
					return nil, fmt.Errorf("%s %w", match, err)
				}
				slog.Warn("bad filepath", "path", match, "error", err)
				continue
			}

			n := model.ParseNameFromFilepath(rel)
			if !n.IsValid() {
				if !continueOnError {
					return nil, fmt.Errorf("%s: %w", rel, model.ErrUnqualifiedName)
				}
				slog.Warn("bad manifest name", "path", rel)
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

	return ms, nil
}
