package compatmigrate

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func convertedLegacyShadowName(digest string) (model.Name, error) {
	hex := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(digest)), "sha256:")
	if hex == "" {
		return model.Name{}, errors.New("converted manifest digest is empty")
	}
	name := model.ParseName(manifest.RunnerLlamaCPP + ":" + hex)
	if !name.IsFullyQualified() {
		return model.Name{}, fmt.Errorf("invalid converted manifest shadow name for digest %q", digest)
	}
	return name, nil
}

// removeConvertedChildBlobs removes the blobs written for a converted child
// that will not be referenced by a manifest list: its config and layer blobs
// plus, when already written, the child manifest blob itself. Blobs shared
// with the source model remain referenced by the source manifest and are
// skipped by RemoveUnreferencedBlobs.
func removeConvertedChildBlobs(child *manifest.Manifest, manifestDigest string) {
	if child == nil {
		return
	}

	digests := make([]string, 0, len(child.Layers)+2)
	if manifestDigest != "" {
		digests = append(digests, manifestDigest)
	}
	if child.Config.Digest != "" {
		digests = append(digests, child.Config.Digest)
	}
	for _, layer := range child.Layers {
		digests = append(digests, layer.Digest)
	}

	if _, err := manifest.RemoveUnreferencedBlobs(digests...); err != nil {
		slog.Warn("could not remove aborted migration blobs", "error", err)
	}
}

// removeConvertedReference discards output that could not be installed.
func removeConvertedReference(ref manifest.Manifest) {
	digest := ref.BlobDigest()
	if digest == "" {
		return
	}

	child := &manifest.Manifest{}
	if path, err := manifest.BlobsPath(digest); err == nil {
		if data, err := os.ReadFile(path); err == nil {
			if err := json.Unmarshal(data, child); err != nil {
				child = &manifest.Manifest{}
			}
		}
	}
	removeConvertedChildBlobs(child, digest)
}

func resolveChildManifest(child manifest.Manifest) (*manifest.Manifest, error) {
	if child.MediaType == manifest.MediaTypeManifestList {
		return nil, errors.New("nested manifest lists are not supported")
	}

	resolved, ok, err := manifest.ResolveManifestReference(child)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, os.ErrNotExist
	}
	if resolved.MediaType == manifest.MediaTypeManifestList {
		return nil, errors.New("nested manifest lists are not supported")
	}

	if err := manifest.FillMetadata(resolved); err != nil {
		return nil, err
	}
	return resolved, nil
}

func manifestBlobsExist(m *manifest.Manifest) bool {
	if m == nil {
		return false
	}
	if m.Config.Digest != "" && !blobExists(m.Config.Digest) {
		return false
	}
	hasModelLayer := false
	for _, layer := range m.Layers {
		if layer.Digest == "" {
			return false
		}
		if !blobExists(layer.Digest) {
			return false
		}
		if layer.MediaType == manifest.MediaTypeImageModel {
			hasModelLayer = true
		}
	}
	return hasModelLayer
}

func blobExists(digest string) bool {
	path, err := manifest.BlobsPath(digest)
	if err != nil {
		return false
	}
	_, err = os.Stat(path)
	return err == nil
}
