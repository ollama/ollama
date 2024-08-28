package server

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
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

// Prune removes the layer from the filesystem if it is not referenced any manifest.
func (l *Layer) Prune() error {
	if l.Digest == "" {
		return nil
	}

	ms, err := Manifests()
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

	slog.Debug("pruning layer", "digest", l.Digest)
	return os.Remove(blob)
}

func Layers() (map[string]Layer, error) {
	blobs, err := GetBlobsPath("")
	if err != nil {
		return nil, err
	}

	// TODO(mxyng): use something less brittle
	matches, err := filepath.Glob(filepath.Join(blobs, "*"))
	if err != nil {
		return nil, err
	}

	layers := make(map[string]Layer)
	for _, match := range matches {
		rel, err := filepath.Rel(blobs, match)
		if err != nil {
			slog.Warn("bad filepath", "path", match, "error", err)
			continue
		}

		// TODO(mxyng): this should ideally use model.Digest but
		// that's currently incompatible with the manifest digest
		digest := strings.Replace(rel, "sha256-", "sha256:", 1)
		layer, err := NewLayerFromLayer(digest, "", "")
		if err != nil {
			slog.Warn("bad blob", "digest", digest, "error", err)
			layer = Layer{Digest: rel}
		}

		layers[digest] = layer
	}

	return layers, nil
}
