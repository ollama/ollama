package server

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
)

// Media type constants for different layer types
const (
	// MediaTypeModelSignature represents an OpenSSF Model Signing (OMS) signature file
	MediaTypeModelSignature = "application/vnd.oms.signature.v1+json"
	// MediaTypeModelConfig represents a model configuration file
	MediaTypeModelConfig = "application/vnd.docker.container.image.v1+json"
	// MediaTypeModelLayer represents a model layer (weights, etc.)
	MediaTypeModelLayer = "application/vnd.ollama.image.model"
)

type Layer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	From      string `json:"from,omitempty"`
	status    string
}

func NewLayer(r io.Reader, mediatype string) (Layer, error) {
	blobs, err := GetBlobsPath("")
	if err != nil {
		return Layer{}, err
	}

	temp, err := os.CreateTemp(blobs, "sha256-")
	if err != nil {
		return Layer{}, err
	}
	defer temp.Close()
	defer os.Remove(temp.Name())

	sha256sum := sha256.New()
	n, err := io.Copy(io.MultiWriter(temp, sha256sum), r)
	if err != nil {
		return Layer{}, err
	}

	if err := temp.Close(); err != nil {
		return Layer{}, err
	}

	digest := fmt.Sprintf("sha256:%x", sha256sum.Sum(nil))
	blob, err := GetBlobsPath(digest)
	if err != nil {
		return Layer{}, err
	}

	status := "using existing layer"
	if _, err := os.Stat(blob); err != nil {
		status = "creating new layer"
		if err := os.Rename(temp.Name(), blob); err != nil {
			return Layer{}, err
		}
		if err := os.Chmod(blob, 0o644); err != nil {
			return Layer{}, err
		}
	}

	return Layer{
		MediaType: mediatype,
		Digest:    digest,
		Size:      n,
		status:    fmt.Sprintf("%s %s", status, digest),
	}, nil
}

func NewLayerFromLayer(digest, mediatype, from string) (Layer, error) {
	if digest == "" {
		return Layer{}, errors.New("creating new layer from layer with empty digest")
	}

	blob, err := GetBlobsPath(digest)
	if err != nil {
		return Layer{}, err
	}

	fi, err := os.Stat(blob)
	if err != nil {
		return Layer{}, err
	}

	return Layer{
		MediaType: mediatype,
		Digest:    digest,
		Size:      fi.Size(),
		From:      from,
		status:    fmt.Sprintf("using existing layer %s", digest),
	}, nil
}

func (l *Layer) Open() (io.ReadSeekCloser, error) {
	if l.Digest == "" {
		return nil, errors.New("opening layer with empty digest")
	}

	blob, err := GetBlobsPath(l.Digest)
	if err != nil {
		return nil, err
	}

	return os.Open(blob)
}

func (l *Layer) Remove() error {
	if l.Digest == "" {
		return nil
	}

	// Ignore corrupt manifests to avoid blocking deletion of layers that are freshly orphaned
	ms, err := Manifests(true)
	if err != nil {
		return err
	}

	for _, m := range ms {
		for _, layer := range append(m.Layers, m.Config) {
			if layer.Digest == l.Digest {
				// something is using this layer
				return nil
			}
		}
	}

	blob, err := GetBlobsPath(l.Digest)
	if err != nil {
		return err
	}

	return os.Remove(blob)
}

// IsSignature returns true if this layer represents a model signature
func (l *Layer) IsSignature() bool {
	return l.MediaType == MediaTypeModelSignature
}

// IsConfig returns true if this layer represents a model configuration
func (l *Layer) IsConfig() bool {
	return l.MediaType == MediaTypeModelConfig
}

// IsModelLayer returns true if this layer represents model data (weights, etc.)
func (l *Layer) IsModelLayer() bool {
	return l.MediaType == MediaTypeModelLayer
}

// NewSignatureLayer creates a new Layer for a signature file
func NewSignatureLayer(r io.Reader) (Layer, error) {
	return NewLayer(r, MediaTypeModelSignature)
}

// NewSignatureLayerFromFile creates a signature layer from an existing signature file
func NewSignatureLayerFromFile(sigFilePath string) (Layer, error) {
	file, err := os.Open(sigFilePath)
	if err != nil {
		return Layer{}, fmt.Errorf("failed to open signature file: %w", err)
	}
	defer file.Close()
	
	return NewSignatureLayer(file)
}
