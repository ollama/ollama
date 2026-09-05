package create

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"golang.org/x/mod/semver"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

var (
	ErrBadTemplate     = errors.New("template error")
	ErrInvalidRequires = errors.New("invalid requires version")
)

// SafetensorsManifestOptions describes the config and Modelfile-derived layers
// shared by local and server-side safetensors create.
type SafetensorsManifestOptions struct {
	MinVersion string
	DraftDir   string

	Template   string
	System     string
	License    any
	Parameters map[string]any
	Messages   []api.Message

	BeforeWriteManifest func()
}

// ModelfileLayerOptions contains the Modelfile values that overlay inherited
// manifest layers during create.
type ModelfileLayerOptions struct {
	Template   string
	System     string
	License    any
	Parameters map[string]any
	Messages   []api.Message
}

// NewSafetensorsManifestWriter returns a ManifestWriter that builds the shared
// safetensors config and Modelfile-derived manifest layers.
func NewSafetensorsManifestWriter(opts SafetensorsManifestOptions) ManifestWriter {
	return func(ctx context.Context, modelName string, info ManifestInfo) error {
		if err := checkContext(ctx); err != nil {
			return err
		}
		name := model.ParseName(modelName)
		if !name.IsValid() {
			return fmt.Errorf("invalid model name: %s", modelName)
		}

		config := info.ModelConfig
		config.ModelFormat = "safetensors"
		if config.Requires != "" {
			var err error
			config.Requires, err = validateRequires(config.Requires)
			if err != nil {
				return err
			}
		}
		fileType := info.Class.Quantize
		if fileType != "" || config.FileType == "" {
			config.FileType = fileType
		}
		if opts.MinVersion != "" && semver.Compare(
			"v"+strings.TrimPrefix(config.Requires, "v"),
			"v"+strings.TrimPrefix(opts.MinVersion, "v"),
		) < 0 {
			config.Requires = opts.MinVersion
		}
		if opts.DraftDir != "" {
			draft, err := safetensorsDraftMetadata(opts.DraftDir)
			if err != nil {
				return err
			}
			config.Draft = draft
		}

		manifestLayers, err := ApplyModelfileLayers(layerInfoToManifestLayers(info.Layers), ModelfileLayerOptions{
			Template:   opts.Template,
			System:     opts.System,
			License:    opts.License,
			Parameters: opts.Parameters,
			Messages:   opts.Messages,
		})
		if err != nil {
			return err
		}
		configLayer, err := safetensorsConfigLayer(config)
		if err != nil {
			return err
		}
		if opts.BeforeWriteManifest != nil {
			opts.BeforeWriteManifest()
		}
		if err := checkContext(ctx); err != nil {
			return err
		}
		return manifest.WriteManifest(name, configLayer, manifestLayers)
	}
}

func validateRequires(value string) (string, error) {
	requires := "v" + strings.TrimPrefix(value, "v")
	if !semver.IsValid(requires) {
		return "", fmt.Errorf("%w: %q is not valid semantic version", ErrInvalidRequires, value)
	}
	return strings.TrimPrefix(requires, "v"), nil
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

func safetensorsConfigLayer(config model.ConfigV2) (manifest.Layer, error) {
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

// ApplyModelfileLayers overlays explicit Modelfile values onto inherited
// layers. Singleton values are replaced, parameters are merged, and licenses
// are appended.
func ApplyModelfileLayers(layers []manifest.Layer, opts ModelfileLayerOptions) ([]manifest.Layer, error) {
	if opts.Template != "" {
		if _, err := template.Parse(opts.Template); err != nil {
			return nil, fmt.Errorf("%w: %s", ErrBadTemplate, err)
		}
		layers = removeLayersByMediaType(layers, "application/vnd.ollama.image.prompt")
		layers = removeLayersByMediaType(layers, "application/vnd.ollama.image.template")
		var err error
		layers, err = appendTextLayer(layers, "application/vnd.ollama.image.template", opts.Template)
		if err != nil {
			return nil, fmt.Errorf("failed to create template layer: %w", err)
		}
	}

	if opts.System != "" {
		layers = removeLayersByMediaType(layers, "application/vnd.ollama.image.system")
		var err error
		layers, err = appendTextLayer(layers, "application/vnd.ollama.image.system", opts.System)
		if err != nil {
			return nil, fmt.Errorf("failed to create system layer: %w", err)
		}
	}

	if opts.License != nil {
		switch l := opts.License.(type) {
		case string:
			if l != "" {
				var err error
				layers, err = appendTextLayer(layers, "application/vnd.ollama.image.license", l)
				if err != nil {
					return nil, fmt.Errorf("failed to create license layer: %w", err)
				}
			}
		default:
			var licenses []string
			b, err := json.Marshal(l)
			if err != nil {
				return nil, fmt.Errorf("failed to encode licenses: %w", err)
			}
			if err := json.Unmarshal(b, &licenses); err != nil {
				return nil, err
			}
			for _, v := range licenses {
				var err error
				layers, err = appendTextLayer(layers, "application/vnd.ollama.image.license", v)
				if err != nil {
					return nil, fmt.Errorf("failed to create license layer: %w", err)
				}
			}
		}
	}

	if len(opts.Parameters) > 0 {
		parameters := make(map[string]any)
		for _, layer := range layers {
			if layer.MediaType != "application/vnd.ollama.image.params" {
				continue
			}

			f, err := layer.Open()
			if err != nil {
				return nil, fmt.Errorf("failed to open inherited parameters: %w", err)
			}
			var inherited map[string]any
			decodeErr := json.NewDecoder(f).Decode(&inherited)
			closeErr := f.Close()
			if decodeErr != nil {
				return nil, fmt.Errorf("failed to decode inherited parameters: %w", decodeErr)
			}
			if closeErr != nil {
				return nil, fmt.Errorf("failed to close inherited parameters: %w", closeErr)
			}
			for k, v := range inherited {
				if _, exists := parameters[k]; !exists {
					parameters[k] = v
				}
			}
		}
		maps.Copy(parameters, opts.Parameters)
		layers = removeLayersByMediaType(layers, "application/vnd.ollama.image.params")

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

	if len(opts.Messages) > 0 {
		layers = removeLayersByMediaType(layers, "application/vnd.ollama.image.messages")
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(opts.Messages); err != nil {
			return nil, fmt.Errorf("failed to encode messages: %w", err)
		}
		layer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.messages")
		if err != nil {
			return nil, fmt.Errorf("failed to create messages layer: %w", err)
		}
		layers = append(layers, layer)
	}
	return layers, nil
}

func removeLayersByMediaType(layers []manifest.Layer, mediaType string) []manifest.Layer {
	return slices.DeleteFunc(layers, func(layer manifest.Layer) bool {
		return layer.MediaType == mediaType
	})
}

func appendTextLayer(layers []manifest.Layer, mediaType, value string) ([]manifest.Layer, error) {
	layer, err := manifest.NewLayer(bytes.NewReader([]byte(value)), mediaType)
	if err != nil {
		return nil, err
	}
	return append(layers, layer), nil
}
