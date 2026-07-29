package compatmigrate

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func loadSourceModelFromManifest(source model.Name, mf *manifest.Manifest) (*SourceModel, error) {
	var config model.ConfigV2
	if mf.Config.Digest != "" {
		configPath, err := manifest.BlobsPath(mf.Config.Digest)
		if err != nil {
			return nil, err
		}

		f, err := os.Open(configPath)
		if err != nil {
			return nil, err
		}
		defer f.Close()

		if err := json.NewDecoder(f).Decode(&config); err != nil {
			return nil, err
		}
	}

	var modelDigest string
	var projectorDigest string
	for _, layer := range mf.Layers {
		switch layer.MediaType {
		case manifest.MediaTypeImageModel:
			modelDigest = layer.Digest
		case manifest.MediaTypeImageProjector:
			projectorDigest = layer.Digest
		}
	}
	if modelDigest == "" {
		return nil, fmt.Errorf("source manifest %s has no GGUF model layer", source.DisplayShortest())
	}

	modelPath, err := manifest.BlobsPath(modelDigest)
	if err != nil {
		return nil, err
	}

	g, err := gguf.Open(modelPath)
	if err != nil {
		return nil, err
	}

	src := &SourceModel{
		Source:         source,
		Manifest:       mf,
		Config:         config,
		GGUFPath:       modelPath,
		GGUF:           g,
		GGUFData:       g.ReaderAt(),
		GGUFDataOffset: g.TensorDataOffset(),
	}

	if projectorDigest != "" {
		projectorPath, err := manifest.BlobsPath(projectorDigest)
		if err != nil {
			g.Close()
			return nil, err
		}

		projector, err := gguf.Open(projectorPath)
		if err != nil {
			g.Close()
			return nil, err
		}

		src.ProjectorPath = projectorPath
		src.ProjectorGGUF = projector
		src.ProjectorData = projector.ReaderAt()
		src.ProjectorDataOffset = projector.TensorDataOffset()
	}

	return src, nil
}

func (src *SourceModel) Close() error {
	var err error
	if src.GGUF != nil {
		err = src.GGUF.Close()
	}
	if src.ProjectorGGUF != nil {
		if closeErr := src.ProjectorGGUF.Close(); err == nil {
			err = closeErr
		}
	}
	return err
}

func convertedManifest(src *SourceModel, result *Result) (_ *manifest.Manifest, err error) {
	// Track blobs this conversion writes so a failure partway through does not
	// strand a multi-GB blob until the next startup prune. Layers cloned from
	// the source model are excluded — they stay referenced by its manifest.
	var written []string
	defer func() {
		if err != nil && len(written) > 0 {
			if _, removeErr := manifest.RemoveUnreferencedBlobs(written...); removeErr != nil {
				slog.Warn("could not remove partial migration blobs", "error", removeErr)
			}
		}
	}()

	var layers []manifest.Layer

	modelLayer, err := writeGGUFLayer(result.ModelKV, result.ModelTensors, manifest.MediaTypeImageModel)
	if err != nil {
		return nil, err
	}
	layers = append(layers, modelLayer)
	written = append(written, modelLayer.Digest)

	if len(result.ProjectorTensors) > 0 {
		projectorLayer, err := writeGGUFLayer(result.ProjectorKV, result.ProjectorTensors, manifest.MediaTypeImageProjector)
		if err != nil {
			return nil, err
		}
		layers = append(layers, projectorLayer)
		written = append(written, projectorLayer.Digest)
	} else if result.PreserveProjector {
		projectorLayer, err := copySourceProjectorLayer(src)
		if err != nil {
			return nil, err
		}
		layers = append(layers, projectorLayer)
	}

	ancillary, err := copyAncillaryLayers(src)
	if err != nil {
		return nil, err
	}
	layers = append(layers, ancillary...)

	config := src.Config
	config.ModelFormat = "gguf"
	config.ModelFamily = result.ModelKV.String("general.architecture")
	config.ModelFamilies = []string{config.ModelFamily}
	if len(result.ProjectorTensors) > 0 || result.PreserveProjector {
		config.ModelFamilies = append(config.ModelFamilies, "clip")
	}
	if result.ClearRenderer {
		config.Renderer = ""
	} else if result.Renderer != "" {
		config.Renderer = result.Renderer
	}
	if result.ClearParser {
		config.Parser = ""
	} else if result.Parser != "" {
		config.Parser = result.Parser
	}
	if result.Requires != "" {
		config.Requires = result.Requires
	}
	config.RemoteHost = ""
	config.RemoteModel = ""

	configLayer, err := newConfigLayer(config)
	if err != nil {
		return nil, err
	}
	written = append(written, configLayer.Digest)

	return &manifest.Manifest{
		SchemaVersion: 2,
		MediaType:     manifest.MediaTypeManifest,
		Config:        configLayer,
		Layers:        layers,
		Runner:        manifest.RunnerLlamaCPP,
		Format:        manifest.FormatGGUF,
	}, nil
}

func newConfigLayer(config model.ConfigV2) (manifest.Layer, error) {
	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(config); err != nil {
		return manifest.Layer{}, fmt.Errorf("failed to encode config: %w", err)
	}
	layer, err := manifest.NewLayer(&b, manifest.MediaTypeImageConfig)
	if err != nil {
		return manifest.Layer{}, fmt.Errorf("failed to create config layer: %w", err)
	}
	return layer, nil
}

func copyAncillaryLayers(src *SourceModel) ([]manifest.Layer, error) {
	var layers []manifest.Layer
	for _, layer := range src.Manifest.Layers {
		switch layer.MediaType {
		case manifest.MediaTypeImageModel,
			manifest.MediaTypeImageProjector,
			manifest.MediaTypeImageAdapter,
			manifest.MediaTypeImageEmbed:
			continue
		}

		cloned, err := manifest.NewLayerFromLayer(layer.Digest, layer.MediaType, src.Source.DisplayShortest())
		if err != nil {
			return nil, err
		}
		cloned.Name = layer.Name
		layers = append(layers, cloned)
	}

	return layers, nil
}

func copySourceProjectorLayer(src *SourceModel) (manifest.Layer, error) {
	for _, layer := range src.Manifest.Layers {
		if layer.MediaType != manifest.MediaTypeImageProjector {
			continue
		}
		cloned, err := manifest.NewLayerFromLayer(layer.Digest, layer.MediaType, src.Source.DisplayShortest())
		if err != nil {
			return manifest.Layer{}, err
		}
		cloned.Name = layer.Name
		return cloned, nil
	}
	return manifest.Layer{}, fmt.Errorf("source manifest %s has no projector layer", src.Source.DisplayShortest())
}

func writeGGUFLayer(kv ggml.KV, tensors []*ggml.Tensor, mediaType string) (manifest.Layer, error) {
	blobs, err := manifest.BlobsPath("")
	if err != nil {
		return manifest.Layer{}, err
	}

	f, err := os.CreateTemp(blobs, "compat-migrate-*.gguf")
	if err != nil {
		return manifest.Layer{}, err
	}
	defer os.Remove(f.Name())

	if err := ggml.WriteGGUF(f, kv, tensors); err != nil {
		f.Close()
		return manifest.Layer{}, err
	}
	if err := f.Close(); err != nil {
		return manifest.Layer{}, err
	}

	return manifest.NewLayerFromFile(f.Name(), mediaType)
}
