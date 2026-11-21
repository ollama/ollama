package compatmigrate

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"slices"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// RetireConvertedModels reclaims Phase 2 legacy copies without converting anything.
func RetireConvertedModels() error {
	models, err := manifest.Manifests(true)
	if err != nil {
		return err
	}
	for name := range models {
		if _, err := retireConvertedModel(name, ""); err != nil {
			slog.Warn("could not retire legacy model", "model", name.DisplayShortest(), "error", err)
		}
	}
	return nil
}

func retireConvertedModel(name model.Name, selectedDigest string) (string, error) {
	data, err := manifest.ReadManifestData(name)
	if err != nil {
		return "", err
	}
	var parent manifest.Manifest
	if err := json.Unmarshal(data, &parent); err != nil {
		return "", err
	}
	if parent.MediaType != manifest.MediaTypeManifestList {
		return "", nil
	}
	anchorData, err := manifest.ReadLegacyManifestData(name)
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	} else if err != nil {
		return "", err
	}
	var anchor manifest.Manifest
	if err := json.Unmarshal(anchorData, &anchor); err != nil {
		return "", err
	}

	// A user-created list may intentionally pair different GGUFs. Require the
	// Phase 2 source anchor and converted shadow, not just two runner labels.
	linked := make(map[string]bool)
	var sourceLayers []manifest.Layer
	for _, layer := range anchor.Layers {
		if layer.MediaType == manifest.MediaTypeManifest || layer.MediaType == manifest.MediaTypeManifestList {
			linked[manifest.NormalizeDigest(layer.Digest)] = true
		} else {
			sourceLayers = append(sourceLayers, layer)
		}
	}
	if !linked[fmt.Sprintf("sha256:%x", sha256.Sum256(data))] {
		return "", nil
	}

	legacyIndex := -1
	var converted string
	for i, ref := range parent.Manifests {
		digest, err := manifest.ChildManifestDigest(ref)
		if err != nil {
			return "", err
		}
		if !linked[manifest.NormalizeDigest(digest)] || ref.Format != manifest.FormatGGUF {
			continue
		}
		child, err := resolveChildManifest(ref)
		if err != nil || !manifestBlobsExist(child) {
			continue
		}
		switch ref.Runner {
		case manifest.RunnerGGML:
			if child.Config != anchor.Config || !slices.Equal(child.Layers, sourceLayers) {
				continue
			}
			if legacyIndex != -1 {
				return "", nil
			}
			legacyIndex = i
		case manifest.RunnerLlamaCPP:
			shadow, err := convertedLegacyShadowName(digest)
			if err != nil {
				return "", err
			}
			shadowData, err := manifest.ReadManifestData(shadow)
			if err != nil || !manifest.SameDigest(fmt.Sprintf("sha256:%x", sha256.Sum256(shadowData)), digest) {
				continue
			}
			runner, err := RunnerForManifest(name, child)
			if err != nil || runner != manifest.RunnerLlamaCPP {
				continue
			}
			if converted != "" {
				return "", nil
			}
			converted = digest
		}
	}
	if legacyIndex == -1 || converted == "" {
		return "", nil
	}
	legacyDigest, err := manifest.ChildManifestDigest(parent.Manifests[legacyIndex])
	if err != nil {
		return "", err
	}
	if selectedDigest != "" && !manifest.SameDigest(selectedDigest, legacyDigest) && !manifest.SameDigest(selectedDigest, converted) {
		return "", nil
	}
	parent.Manifests = slices.Delete(parent.Manifests, legacyIndex, legacyIndex+1)
	if err := replaceConvertedModel(name, data, &parent); err != nil {
		return "", err
	}
	return converted, nil
}

func replaceConvertedModel(name model.Name, previous []byte, replacement *manifest.Manifest) error {
	candidates, err := manifest.ReferencedBlobDigestsForName(name)
	if errors.Is(err, os.ErrNotExist) {
		return manifest.ErrManifestChanged
	}
	if err != nil {
		return err
	}
	data, err := json.Marshal(replacement)
	if err != nil {
		return err
	}
	if err := manifest.ReplaceManifestData(name, previous, data); err != nil {
		return err
	}
	if _, err := manifest.RemoveUnreferencedBlobs(candidates...); err != nil {
		// Installation succeeded. Startup GC can retry removal (e.g. an old
		// runner may still have a source file open on Windows).
		slog.Warn("could not remove retired migration blobs", "model", name.DisplayShortest(), "error", err)
	}
	return nil
}
