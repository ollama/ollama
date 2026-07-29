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

func writeCompatibilityManifestList(name model.Name, source *manifest.Manifest, manifests []manifest.Manifest) (bool, error) {
	if source != nil {
		if err := writeLegacySourceManifest(name, source); err != nil {
			return false, err
		}
	}

	if err := manifest.WriteManifestListPreserveLegacy(name, manifests); err != nil {
		return false, err
	}
	return true, nil
}

func writeLegacySourceManifest(name model.Name, source *manifest.Manifest) error {
	// TODO: remove this downgrade anchor once rollback to pre-manifest-list
	// Ollama versions is no longer supported.
	data, err := json.Marshal(source)
	if err != nil {
		return err
	}
	return manifest.WriteLegacyManifestData(name, data)
}

func writeConvertedLegacyShadow(digest string, data []byte) error {
	// TODO: remove this shadow tag once rollback to pre-manifest-list Ollama
	// versions is no longer supported.
	name, err := convertedLegacyShadowName(digest)
	if err != nil {
		return err
	}
	return manifest.WriteLegacyManifestData(name, data)
}

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

// removeConvertedReference undoes a completed conversion whose manifest list
// was never written: it drops the legacy shadow tag first (the shadow retains
// the converted blobs) and then the blobs only the converted child references.
func removeConvertedReference(ref manifest.Manifest) {
	digest := ref.BlobDigest()
	if digest == "" {
		return
	}

	if shadow, err := convertedLegacyShadowName(digest); err == nil {
		if err := manifest.RemoveNamed(shadow); err != nil && !errors.Is(err, os.ErrNotExist) {
			slog.Warn("could not remove converted manifest shadow", "digest", digest, "error", err)
		}
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

func manifestReferenceForChild(child *manifest.Manifest) (manifest.Manifest, error) {
	data, err := json.Marshal(child)
	if err != nil {
		return manifest.Manifest{}, err
	}
	digest, err := manifest.WriteManifestBlob(data)
	if err != nil {
		return manifest.Manifest{}, err
	}
	return manifest.NewManifestReference(digest, child.Runner, child.Format)
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

func isRunnerFormat(m *manifest.Manifest, runner, format string) bool {
	if m == nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(m.Runner), runner) &&
		strings.EqualFold(strings.TrimSpace(m.Format), format)
}
