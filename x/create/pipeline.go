package create

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Create imports a safetensors model through the full pipeline: read the
// source into an inventory, classify it, plan the output blobs, write them
// through store, import the config files, and write the manifest. It is the
// server-side entry point — the caller supplies blob storage (store) and
// manifest assembly (writeManifest).
func Create(modelName, modelDir, quantize string, store BlobStore, writeManifest ManifestWriter, fn func(status string)) error {
	defer sweepMLX()

	inv, err := ReadInventory(modelDir)
	if err != nil {
		return fmt.Errorf("read model: %w", err)
	}
	class, err := Classify(inv, quantize)
	if err != nil {
		return err
	}
	policy, err := newTensorImportTransform(inv)
	if err != nil {
		return fmt.Errorf("build quantization policy for %q: %w", inv.Config.Architecture(), err)
	}
	specs, err := Plan(inv, class, policy)
	if err != nil {
		return fmt.Errorf("plan model: %w", err)
	}

	fn(fmt.Sprintf("importing %s (%d tensors%s)", modelName, len(inv.Tensors), quantizeStatus(class)))
	layers, err := WriteBlobs(specs, modelDir, store)
	if err != nil {
		return err
	}

	// Import config files (config.json, tokenizer, etc.) as JSON blobs.
	configLayers, configLayer, err := importConfigBlobs(modelDir, "", store, fn)
	if err != nil {
		return err
	}
	layers = append(layers, configLayers...)
	if configLayer.Digest == "" {
		return fmt.Errorf("config.json not found in %s", modelDir)
	}

	fn(fmt.Sprintf("writing manifest for %s", modelName))
	if err := writeManifest(modelName, configLayer, layers); err != nil {
		return fmt.Errorf("write manifest: %w", err)
	}
	fn(fmt.Sprintf("successfully imported %s with %d layers", modelName, len(layers)))
	return nil
}

const mediaTypeImageJSON = "application/vnd.ollama.image.json"

// importConfigBlobs writes every .json in modelDir (except the shard index) as an
// image.json blob, prefixing each blob name with namePrefix, and returns the
// resulting layers along with the config.json layer (zero value if absent). The
// target import passes "" for namePrefix; a draft import passes "draft/" so its
// config sits beside the target's.
func importConfigBlobs(modelDir, namePrefix string, store BlobStore, fn func(status string)) ([]LayerInfo, LayerInfo, error) {
	entries, err := os.ReadDir(modelDir)
	if err != nil {
		return nil, LayerInfo{}, err
	}
	var layers []LayerInfo
	var configLayer LayerInfo
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") || entry.Name() == "model.safetensors.index.json" {
			continue
		}
		name := entry.Name()
		fn(fmt.Sprintf("importing config %s", name))
		f, err := os.Open(filepath.Join(modelDir, name))
		if err != nil {
			return nil, LayerInfo{}, fmt.Errorf("open %s: %w", name, err)
		}
		layer, err := store.WriteBlob(f, mediaTypeImageJSON, namePrefix+name)
		f.Close()
		if err != nil {
			return nil, LayerInfo{}, fmt.Errorf("write config %s: %w", name, err)
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

// StoreFromLayerCreator adapts a LayerCreator-style function to a BlobStore, so
// a caller that already has a blob-writing callback can drive the pipeline.
func StoreFromLayerCreator(fn LayerCreator) BlobStore {
	return layerCreatorStore{fn}
}

type layerCreatorStore struct{ fn LayerCreator }

func (s layerCreatorStore) WriteBlob(r io.Reader, mediaType, name string) (LayerInfo, error) {
	return s.fn(r, mediaType, name)
}
