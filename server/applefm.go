package server

import (
	"context"
	"errors"
	"io/fs"
	"log/slog"

	"github.com/ollama/ollama/applefm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// appleFMModelName is the name Apple's on-device Foundation Model is listed as.
const appleFMModelName = "apple-foundation-model"

var appleFMAvailable = applefm.Available

// registerAppleFoundationModel adds the on-device model to the local model
// list when the operating system provides it, and removes it when it no
// longer does. The manifest only holds a config, since the weights are
// managed by macOS. A different model with the same name is left alone.
func registerAppleFoundationModel(ctx context.Context) error {
	name := model.ParseName(appleFMModelName)
	existing, err := manifest.ParseNamedManifest(name)
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	if existing != nil && existing.Format != applefm.ModelFormat {
		slog.Warn("apple foundation model not registered, another model has its name", "model", name.DisplayShortest())
		return nil
	}

	if err := appleFMAvailable(ctx); err != nil {
		slog.Debug("apple foundation model not registered", "error", err)
		if existing != nil {
			slog.Info("removing unavailable apple foundation model", "model", name.DisplayShortest())
			_, err := manifest.RemoveNamed(name)
			return err
		}
		return nil
	}

	layer, err := createConfigLayer(model.ConfigV2{
		ModelFormat:  applefm.ModelFormat,
		ModelFamily:  applefm.ModelFormat,
		Capabilities: []string{string(model.CapabilityCompletion), string(model.CapabilityVision)},
		ContextLen:   applefm.ContextLength,
	})
	if err != nil {
		return err
	}
	if existing != nil && existing.Config.Digest == layer.Digest {
		return nil
	}

	slog.Info("registering apple foundation model", "model", name.DisplayShortest())
	return manifest.WriteManifestWithMetadata(name, *layer, nil, applefm.RunnerName, applefm.ModelFormat)
}
