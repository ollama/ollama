package create

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// SafetensorsManifestOptions describes the config and Modelfile-derived layers
// shared by local and server-side safetensors create.
type SafetensorsManifestOptions struct {
	ModelDir string

	BaseConfig   *model.ConfigV2
	Quantize     string
	Capabilities []string
	MinVersion   string
	Requires     string
	Parser       string
	Renderer     string
	DraftDir     string

	Template   string
	System     string
	License    any
	Parameters map[string]any

	ExtraLayers         func([]manifest.Layer) ([]manifest.Layer, error)
	BeforeWriteManifest func()
	IncludeRootFSDiffs  bool
}

// NewSafetensorsManifestWriter returns a ManifestWriter that builds the shared
// safetensors config and Modelfile-derived manifest layers.
func NewSafetensorsManifestWriter(opts SafetensorsManifestOptions) ManifestWriter {
	return func(modelName string, _ LayerInfo, layers []LayerInfo) error {
		name := model.ParseName(modelName)
		if !name.IsValid() {
			return fmt.Errorf("invalid model name: %s", modelName)
		}

		config := model.ConfigV2{}
		if opts.BaseConfig != nil {
			config = *opts.BaseConfig
		}
		config.ModelFormat = "safetensors"
		if opts.Quantize != "" || config.FileType == "" {
			config.FileType = strings.ToLower(strings.TrimSpace(opts.Quantize))
		}
		if opts.Capabilities != nil {
			config.Capabilities = append([]string(nil), opts.Capabilities...)
		}
		// TODO: find a better way to detect image input support. For now,
		// preserve the existing Flux2KleinPipeline vision capability behavior.
		if opts.ModelDir != "" {
			modelIndex := filepath.Join(opts.ModelDir, "model_index.json")
			if data, err := os.ReadFile(modelIndex); err == nil {
				var cfg struct {
					ClassName string `json:"_class_name"`
				}
				if json.Unmarshal(data, &cfg) == nil && cfg.ClassName == "Flux2KleinPipeline" {
					config.Capabilities = append(config.Capabilities, "vision")
				}
			}
		}
		if opts.MinVersion != "" {
			config.Requires = opts.MinVersion
		}
		if opts.Requires != "" {
			config.Requires = opts.Requires
		}
		config.Parser = opts.Parser
		config.Renderer = opts.Renderer
		if opts.DraftDir != "" {
			draft, err := safetensorsDraftMetadata(opts.DraftDir)
			if err != nil {
				return err
			}
			config.Draft = draft
		}

		manifestLayers := layerInfoToManifestLayers(layers)
		modelfileLayers, err := safetensorsModelfileLayers(opts.Template, opts.System, opts.License, opts.Parameters)
		if err != nil {
			return err
		}
		manifestLayers = append(manifestLayers, modelfileLayers...)
		if opts.ExtraLayers != nil {
			manifestLayers, err = opts.ExtraLayers(manifestLayers)
			if err != nil {
				return err
			}
		}

		configLayer, err := safetensorsConfigLayer(manifestLayers, config, opts.IncludeRootFSDiffs)
		if err != nil {
			return err
		}
		if opts.BeforeWriteManifest != nil {
			opts.BeforeWriteManifest()
		}
		return manifest.WriteManifest(name, configLayer, manifestLayers)
	}
}

func layerInfoToManifestLayers(layers []LayerInfo) []manifest.Layer {
	out := make([]manifest.Layer, 0, len(layers))
	for _, l := range layers {
		out = append(out, manifest.Layer{
			MediaType: l.MediaType,
			Digest:    l.Digest,
			Size:      l.Size,
			Name:      l.Name,
		})
	}
	return out
}

func safetensorsConfigLayer(layers []manifest.Layer, config model.ConfigV2, includeRootFSDiffs bool) (manifest.Layer, error) {
	if includeRootFSDiffs {
		digests := make([]string, len(layers))
		for i, layer := range layers {
			digests[i] = layer.Digest
		}
		config.RootFS.DiffIDs = digests
	}

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(config); err != nil {
		return manifest.Layer{}, fmt.Errorf("failed to encode config: %w", err)
	}
	layer, err := manifest.NewLayer(&b, "application/vnd.docker.container.image.v1+json")
	if err != nil {
		return manifest.Layer{}, fmt.Errorf("failed to create config layer: %w", err)
	}
	return layer, nil
}

func safetensorsDraftMetadata(draftDir string) (*model.Draft, error) {
	configPath := filepath.Join(draftDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read draft config %s: %w", configPath, err)
	}

	var cfg struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse draft config %s: %w", configPath, err)
	}
	arch := ""
	if len(cfg.Architectures) > 0 {
		arch = cfg.Architectures[0]
	}
	if arch == "" {
		arch = cfg.ModelType
	}
	if arch == "" {
		return nil, fmt.Errorf("draft architecture not found in %s", configPath)
	}

	return &model.Draft{
		ModelFormat:  "safetensors",
		Architecture: arch,
		TensorPrefix: "draft.",
		Config:       "draft/config.json",
	}, nil
}

func safetensorsModelfileLayers(template, system string, license any, parameters map[string]any) ([]manifest.Layer, error) {
	var layers []manifest.Layer
	var err error
	if template != "" {
		layers, err = appendTextLayer(layers, "application/vnd.ollama.image.template", template)
		if err != nil {
			return nil, fmt.Errorf("failed to create template layer: %w", err)
		}
	}
	if system != "" {
		layers, err = appendTextLayer(layers, "application/vnd.ollama.image.system", system)
		if err != nil {
			return nil, fmt.Errorf("failed to create system layer: %w", err)
		}
	}
	if license != nil {
		switch l := license.(type) {
		case string:
			if l != "" {
				layers, err = appendTextLayer(layers, "application/vnd.ollama.image.license", l)
				if err != nil {
					return nil, fmt.Errorf("failed to create license layer: %w", err)
				}
			}
		default:
			var licenses []string
			b, _ := json.Marshal(l)
			if err := json.Unmarshal(b, &licenses); err != nil {
				return nil, err
			}
			for _, v := range licenses {
				layers, err = appendTextLayer(layers, "application/vnd.ollama.image.license", v)
				if err != nil {
					return nil, fmt.Errorf("failed to create license layer: %w", err)
				}
			}
		}
	}
	if len(parameters) > 0 {
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(parameters); err != nil {
			return nil, fmt.Errorf("failed to encode parameters: %w", err)
		}
		layer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.params")
		if err != nil {
			return nil, fmt.Errorf("failed to create params layer: %w", err)
		}
		layers = append(layers, layer)
	}
	return layers, nil
}

func appendTextLayer(layers []manifest.Layer, mediaType, value string) ([]manifest.Layer, error) {
	layer, err := manifest.NewLayer(bytes.NewReader([]byte(value)), mediaType)
	if err != nil {
		return nil, err
	}
	return append(layers, layer), nil
}
