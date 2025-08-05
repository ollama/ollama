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
	"time"

	"github.com/ollama/ollama/types/model"
)

// SignatureInfo contains metadata about a model's cryptographic signature
type SignatureInfo struct {
	Format       string    `json:"format"`           // Signature format version (e.g., "oms-v1.0")
	SignatureURI string    `json:"signatureUri"`     // Path to signature file (e.g., "model.sig")
	Verified     bool      `json:"verified"`         // Whether signature has been verified
	Signer       string    `json:"signer,omitempty"` // Identity of the signer
	SignedAt     time.Time `json:"signedAt,omitempty"` // When the model was signed
}

type Manifest struct {
	SchemaVersion int     `json:"schemaVersion"`
	MediaType     string  `json:"mediaType"`
	Config        Layer   `json:"config"`
	Layers        []Layer `json:"layers"`

	// Signature contains cryptographic signature information for this model
	// This field is optional to maintain backward compatibility with unsigned models
	Signature     *SignatureInfo `json:"signature,omitempty"`

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

// Digest returns the SHA-256 digest of this manifest
func (m *Manifest) Digest() string {
	return m.digest
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

	// Update signature verification status if model has signature metadata
	if m.Signature != nil {
		verifier := NewSignatureVerifier()
		result, err := verifier.VerifyManifest(&m)
		if err == nil && result != nil {
			// Update the verified status based on verification result
			m.Signature.Verified = result.Valid
		} else {
			// If verification fails, mark as unverified but don't fail the manifest loading
			m.Signature.Verified = false
			slog.Debug("signature verification failed", "model", n.String(), "error", err)
		}
	}

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

// GetSignatureLayer returns the first signature layer found in this manifest, or nil if none exists
func (m *Manifest) GetSignatureLayer() *Layer {
	for i := range m.Layers {
		if m.Layers[i].IsSignature() {
			return &m.Layers[i]
		}
	}
	return nil
}

// HasSignature returns true if this manifest contains a signature layer
func (m *Manifest) HasSignature() bool {
	return m.GetSignatureLayer() != nil
}

// AddSignatureLayer adds a signature layer to this manifest
func (m *Manifest) AddSignatureLayer(sigLayer Layer) {
	// Remove any existing signature layers first
	layers := make([]Layer, 0, len(m.Layers))
	for _, layer := range m.Layers {
		if !layer.IsSignature() {
			layers = append(layers, layer)
		}
	}
	
	// Add the new signature layer
	m.Layers = append(layers, sigLayer)
}
