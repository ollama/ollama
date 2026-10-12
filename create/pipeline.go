package create

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/types/model"
)

// PipelineOptions controls the source-specific stages of a safetensors import.
type PipelineOptions struct {
	Quantize      string
	Parser        string
	Renderer      string
	Requires      string
	DraftDir      string
	DraftQuantize string
	Validation    MLXValidationOptions
}

// Create imports a safetensors model through the full pipeline: read the
// source into an inventory, classify it, plan the output blobs, write them
// through store, import the config files, and write the manifest. It is the
// shared local and server entry point; the caller supplies blob storage (store)
// and manifest assembly (writeManifest). store, writeManifest, and fn must be
// non-nil.
func Create(ctx context.Context, modelName, modelDir string, opts PipelineOptions, store BlobStore, writeManifest ManifestWriter, fn func(status string)) error {
	defer releaseMLXCache()

	if err := checkContext(ctx); err != nil {
		return err
	}
	inv, err := ReadInventory(modelDir)
	if err != nil {
		return fmt.Errorf("read model: %w", err)
	}
	if opts.Renderer == "clef" {
		inv, err = prepareClefInventory(inv)
		if err != nil {
			return err
		}
	}
	modelConfig, err := inferSafetensorsConfig(modelDir, inv.Config, opts.Parser, opts.Renderer)
	if err != nil {
		return err
	}
	if opts.Requires != "" {
		modelConfig.Requires, err = validateRequires(opts.Requires)
		if err != nil {
			return err
		}
	}
	if err := validateMLXSource(inv.Config, false, opts.Validation); err != nil {
		return err
	}
	if err := checkContext(ctx); err != nil {
		return err
	}
	if inv.Config.Architecture() == "LayaForDecision" && opts.Quantize != "" {
		return fmt.Errorf("Laya currently requires unquantized weights")
	}
	class, err := Classify(inv, opts.Quantize)
	if err != nil {
		return err
	}
	policy, err := newTensorImportTransform(inv)
	if err != nil {
		return fmt.Errorf("build quantization policy for %q: %w", inv.Config.Architecture(), err)
	}
	// Arch-specific policies may fill manifest-config fields the shared
	// inference can't derive from an arch-specific config layout.
	if aug, ok := policy.(interface{ augmentConfig(*model.ConfigV2) error }); ok {
		if err := aug.augmentConfig(&modelConfig); err != nil {
			return fmt.Errorf("augment config for %q: %w", inv.Config.Architecture(), err)
		}
	}
	specs, err := Plan(inv, class, policy)
	if err != nil {
		return fmt.Errorf("plan model: %w", err)
	}

	var draftLayers []LayerInfo
	if opts.DraftDir != "" {
		draftLayers, err = createDraftLayers(ctx, opts.DraftDir, "draft.", "draft/", opts.DraftQuantize, opts.Validation, store, fn)
		if err != nil {
			return err
		}
	}

	fn(fmt.Sprintf("importing %s (%d tensors%s)", modelName, len(inv.Tensors), quantizeStatus(class)))
	layers, err := WriteBlobs(ctx, specs, modelDir, store)
	if err != nil {
		return err
	}

	// Import config files (config.json, tokenizer, etc.) as JSON blobs.
	configLayers, configLayer, err := importConfigBlobs(ctx, modelDir, "", inv.RawConfig, store, fn)
	if err != nil {
		return err
	}
	layers = append(layers, configLayers...)
	layers = append(layers, draftLayers...)
	if configLayer.Digest == "" {
		configLayer, err = store.WriteBlob(bytes.NewReader(inv.RawConfig), mediaTypeImageJSON, "config.json")
		if err != nil {
			return err
		}
		layers = append(layers, configLayer)
	}
	if err := checkContext(ctx); err != nil {
		return err
	}

	fn(fmt.Sprintf("writing manifest for %s", modelName))
	if err := writeManifest(ctx, modelName, ManifestInfo{ModelConfig: modelConfig, ConfigLayer: configLayer, Layers: layers, Class: class}); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	fn(fmt.Sprintf("successfully imported %s with %d layers", modelName, len(layers)))
	return nil
}

func checkContext(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("nil context")
	}
	return ctx.Err()
}

const mediaTypeImageJSON = "application/vnd.ollama.image.json"

// importConfigBlobs writes every .json in modelDir (except the shard index) as an
// image.json blob, prefixing each blob name with namePrefix, and returns the
// resulting layers along with the config.json layer (zero value if absent). The
// target import passes "" for namePrefix; a draft import passes "draft/" so its
// config sits beside the target's. configOverride replaces only config.json
// when an importer has normalized its contents.
func importConfigBlobs(ctx context.Context, modelDir, namePrefix string, configOverride json.RawMessage, store BlobStore, fn func(status string)) ([]LayerInfo, LayerInfo, error) {
	names, err := SafetensorsConfigFiles(modelDir)
	if err != nil {
		return nil, LayerInfo{}, err
	}
	var layers []LayerInfo
	var configLayer LayerInfo
	for _, name := range names {
		if err := checkContext(ctx); err != nil {
			return nil, LayerInfo{}, err
		}
		if !strings.HasSuffix(name, ".json") || name == "model.safetensors.index.json" {
			continue
		}
		fn(fmt.Sprintf("importing config %s", name))
		var f io.ReadCloser
		if name == "config.json" && configOverride != nil {
			f = io.NopCloser(bytes.NewReader(configOverride))
		} else {
			f, err = os.Open(filepath.Join(modelDir, name))
			if err != nil {
				return nil, LayerInfo{}, fmt.Errorf("open %s: %w", name, err)
			}
		}
		layer, err := store.WriteBlob(ReaderWithContext(ctx, f), mediaTypeImageJSON, namePrefix+name)
		closeErr := f.Close()
		if err != nil {
			return nil, LayerInfo{}, fmt.Errorf("write config %s: %w", name, err)
		}
		if closeErr != nil {
			return nil, LayerInfo{}, fmt.Errorf("close config %s: %w", name, closeErr)
		}
		if name == "config.json" {
			configLayer = layer
		}
		layers = append(layers, layer)
	}
	return layers, configLayer, nil
}

func quantizeStatus(c Classification) string {
	switch c.Kind {
	case SourceBlockFP8:
		return ", converting fp8 to mxfp8"
	case SourcePrequantized:
		return ", preserving source quantization"
	default:
		if c.Quantize != "" {
			return ", quantizing to " + c.Quantize
		}
		return ""
	}
}
