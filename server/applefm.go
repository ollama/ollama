package server

import (
	"context"
	"log/slog"

	"github.com/ollama/ollama/applefm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// appleFMModelName is the name Apple's on-device Foundation Model is listed as.
const appleFMModelName = "apple-foundation-model"

var appleFMAvailable = applefm.Available

// registerAppleFoundationModel adds the on-device model to the local model
// list when the operating system provides it. The manifest only holds a
// config, since the weights are managed by macOS.
func registerAppleFoundationModel(ctx context.Context) error {
	if err := appleFMAvailable(ctx); err != nil {
		slog.Debug("apple foundation model not registered", "error", err)
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

	name := model.ParseName(appleFMModelName)
	if mf, err := manifest.ParseNamedManifest(name); err == nil && mf.Config.Digest == layer.Digest {
		return nil
	}

	slog.Info("registering apple foundation model", "model", name.DisplayShortest())
	return manifest.WriteManifestWithMetadata(name, *layer, nil, applefm.RunnerName, applefm.ModelFormat)
}
