package compatmigrate

import (
	"encoding/json"
	"errors"
	"fmt"
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

	parent := manifest.Manifest{
		SchemaVersion: 2,
		MediaType:     manifest.MediaTypeManifestList,
		Manifests:     manifests,
	}
	data, err := json.Marshal(parent)
	if err != nil {
		return false, err
	}
	if err := manifest.WriteManifestDataPreserveLegacy(name, data); err != nil {
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
	name := model.ParseName("llamacpp:" + hex)
	if !name.IsFullyQualified() {
		return model.Name{}, fmt.Errorf("invalid converted manifest shadow name for digest %q", digest)
	}
	return name, nil
}

func resolveChildManifest(child manifest.Manifest) (*manifest.Manifest, error) {
	if child.MediaType == manifest.MediaTypeManifestList {
		return nil, errors.New("nested manifest lists are not supported")
	}

	resolved := child
	if resolved.Config.Digest == "" && len(resolved.Layers) == 0 && resolved.BlobDigest() != "" {
		blobPath, err := manifest.BlobsPath(resolved.BlobDigest())
		if err != nil {
			return nil, err
		}
		data, err := os.ReadFile(blobPath)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(data, &resolved); err != nil {
			return nil, err
		}
		if resolved.MediaType == manifest.MediaTypeManifestList {
			return nil, errors.New("nested manifest lists are not supported")
		}
		if resolved.Runner == "" {
			resolved.Runner = child.Runner
		}
		if resolved.Format == "" {
			resolved.Format = child.Format
		}
	}

	if err := fillManifestMetadata(&resolved); err != nil {
		return nil, err
	}
	return &resolved, nil
}

func fillManifestMetadata(m *manifest.Manifest) error {
	if m.Runner != "" && m.Format != "" {
		m.Runner = strings.ToLower(strings.TrimSpace(m.Runner))
		m.Format = strings.ToLower(strings.TrimSpace(m.Format))
		return nil
	}

	config, err := readManifestConfig(m.Config.Digest)
	if err != nil {
		return err
	}
	manifest.FillMetadataForConfig(m, config)
	if m.Runner == "" || m.Format == "" {
		return errors.New("manifest is missing runner or format metadata")
	}
	return nil
}

func readManifestConfig(digest string) (model.ConfigV2, error) {
	var config model.ConfigV2
	if digest == "" {
		return config, errors.New("manifest is missing config digest")
	}

	configPath, err := manifest.BlobsPath(digest)
	if err != nil {
		return config, err
	}
	f, err := os.Open(configPath)
	if err != nil {
		return config, err
	}
	defer f.Close()

	return config, json.NewDecoder(f).Decode(&config)
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
		if layer.MediaType == "application/vnd.ollama.image.model" {
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
