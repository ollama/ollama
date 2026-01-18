// Package client provides client-side model creation for safetensors-based models.
//
// This package is in x/ because the safetensors model storage format is under development.
// It also exists to break an import cycle: server imports x/create, so x/create
// cannot import server. This sub-package can import server because server doesn't
// import it.
package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/create"
)

// MinOllamaVersion is the minimum Ollama version required for safetensors models.
const MinOllamaVersion = "0.14.0"

// ModelfileConfig holds configuration extracted from a Modelfile.
type ModelfileConfig struct {
	Template string
	System   string
	License  string
}

// CreateOptions holds all options for model creation.
type CreateOptions struct {
	ModelName string
	ModelDir  string
	Quantize  string           // "fp8" for quantization
	Modelfile *ModelfileConfig // template/system/license from Modelfile
}

// CreateModel imports a model from a local directory.
// This creates blobs and manifest directly on disk, bypassing the HTTP API.
// Automatically detects model type (safetensors LLM vs image gen) and routes accordingly.
func CreateModel(opts CreateOptions, p *progress.Progress) error {
	// Detect model type
	isSafetensors := create.IsSafetensorsModelDir(opts.ModelDir)
	isImageGen := create.IsTensorModelDir(opts.ModelDir)

	if !isSafetensors && !isImageGen {
		return fmt.Errorf("%s is not a supported model directory (needs config.json + *.safetensors or model_index.json)", opts.ModelDir)
	}

	// Determine model type settings
	var modelType, spinnerKey string
	var capabilities []string
	if isSafetensors {
		modelType = "safetensors model"
		spinnerKey = "create"
		capabilities = []string{"completion"}
	} else {
		modelType = "image generation model"
		spinnerKey = "imagegen"
		capabilities = []string{"image"}
	}

	// Set up progress spinner
	statusMsg := "importing " + modelType
	spinner := progress.NewSpinner(statusMsg)
	p.Add(spinnerKey, spinner)

	progressFn := func(msg string) {
		spinner.Stop()
		statusMsg = msg
		spinner = progress.NewSpinner(statusMsg)
		p.Add(spinnerKey, spinner)
	}

	// Create the model using shared callbacks
	var err error
	if isSafetensors {
		err = create.CreateSafetensorsModel(
			opts.ModelName, opts.ModelDir, opts.Quantize,
			newLayerCreator(), newTensorLayerCreator(),
			newManifestWriter(opts, capabilities),
			progressFn,
		)
	} else {
		err = create.CreateImageGenModel(
			opts.ModelName, opts.ModelDir, opts.Quantize,
			newLayerCreator(), newTensorLayerCreator(),
			newManifestWriter(opts, capabilities),
			progressFn,
		)
	}

	spinner.Stop()
	if err != nil {
		return err
	}

	fmt.Printf("Created %s '%s'\n", modelType, opts.ModelName)
	return nil
}

// newLayerCreator returns a LayerCreator callback for creating config/JSON layers.
func newLayerCreator() create.LayerCreator {
	return func(r io.Reader, mediaType, name string) (create.LayerInfo, error) {
		layer, err := server.NewLayer(r, mediaType)
		if err != nil {
			return create.LayerInfo{}, err
		}

		return create.LayerInfo{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      name,
		}, nil
	}
}

// newTensorLayerCreator returns a QuantizingTensorLayerCreator callback for creating tensor layers.
// When quantize is non-empty, returns multiple layers (weight + scales + optional qbias).
func newTensorLayerCreator() create.QuantizingTensorLayerCreator {
	return func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]create.LayerInfo, error) {
		if quantize != "" {
			return createQuantizedLayers(r, name, dtype, shape, quantize)
		}
		return createUnquantizedLayer(r, name)
	}
}

// createQuantizedLayers quantizes a tensor and returns the resulting layers.
func createQuantizedLayers(r io.Reader, name, dtype string, shape []int32, quantize string) ([]create.LayerInfo, error) {
	if !QuantizeSupported() {
		return nil, fmt.Errorf("quantization requires MLX support")
	}

	// Quantize the tensor
	qweightData, scalesData, qbiasData, _, _, _, err := quantizeTensor(r, name, dtype, shape, quantize)
	if err != nil {
		return nil, fmt.Errorf("failed to quantize %s: %w", name, err)
	}

	// Create layer for quantized weight
	weightLayer, err := server.NewLayer(bytes.NewReader(qweightData), server.MediaTypeImageTensor)
	if err != nil {
		return nil, err
	}

	// Create layer for scales
	scalesLayer, err := server.NewLayer(bytes.NewReader(scalesData), server.MediaTypeImageTensor)
	if err != nil {
		return nil, err
	}

	layers := []create.LayerInfo{
		{
			Digest:    weightLayer.Digest,
			Size:      weightLayer.Size,
			MediaType: weightLayer.MediaType,
			Name:      name,
		},
		{
			Digest:    scalesLayer.Digest,
			Size:      scalesLayer.Size,
			MediaType: scalesLayer.MediaType,
			Name:      name + "_scale",
		},
	}

	// Add qbiases layer if present (affine mode)
	if qbiasData != nil {
		qbiasLayer, err := server.NewLayer(bytes.NewReader(qbiasData), server.MediaTypeImageTensor)
		if err != nil {
			return nil, err
		}
		layers = append(layers, create.LayerInfo{
			Digest:    qbiasLayer.Digest,
			Size:      qbiasLayer.Size,
			MediaType: qbiasLayer.MediaType,
			Name:      name + "_qbias",
		})
	}

	return layers, nil
}

// createUnquantizedLayer creates a single tensor layer without quantization.
func createUnquantizedLayer(r io.Reader, name string) ([]create.LayerInfo, error) {
	layer, err := server.NewLayer(r, server.MediaTypeImageTensor)
	if err != nil {
		return nil, err
	}

	return []create.LayerInfo{
		{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      name,
		},
	}, nil
}

// newManifestWriter returns a ManifestWriter callback for writing the model manifest.
func newManifestWriter(opts CreateOptions, capabilities []string) create.ManifestWriter {
	return func(modelName string, config create.LayerInfo, layers []create.LayerInfo) error {
		name := model.ParseName(modelName)
		if !name.IsValid() {
			return fmt.Errorf("invalid model name: %s", modelName)
		}

		// Create config blob with version requirement
		configData := model.ConfigV2{
			ModelFormat:  "safetensors",
			Capabilities: capabilities,
			Requires:     MinOllamaVersion,
		}
		configJSON, err := json.Marshal(configData)
		if err != nil {
			return fmt.Errorf("failed to marshal config: %w", err)
		}

		// Create config layer blob
		configLayer, err := server.NewLayer(bytes.NewReader(configJSON), "application/vnd.docker.container.image.v1+json")
		if err != nil {
			return fmt.Errorf("failed to create config layer: %w", err)
		}

		// Convert LayerInfo to server.Layer
		serverLayers := make([]server.Layer, 0, len(layers))
		for _, l := range layers {
			serverLayers = append(serverLayers, server.Layer{
				MediaType: l.MediaType,
				Digest:    l.Digest,
				Size:      l.Size,
				Name:      l.Name,
			})
		}

		// Add Modelfile layers if present
		if opts.Modelfile != nil {
			modelfileLayers, err := createModelfileLayers(opts.Modelfile)
			if err != nil {
				return err
			}
			serverLayers = append(serverLayers, modelfileLayers...)
		}

		return server.WriteManifest(name, configLayer, serverLayers)
	}
}

// createModelfileLayers creates layers for template, system, and license from Modelfile config.
func createModelfileLayers(mf *ModelfileConfig) ([]server.Layer, error) {
	var layers []server.Layer

	if mf.Template != "" {
		layer, err := server.NewLayer(bytes.NewReader([]byte(mf.Template)), "application/vnd.ollama.image.template")
		if err != nil {
			return nil, fmt.Errorf("failed to create template layer: %w", err)
		}
		layers = append(layers, layer)
	}

	if mf.System != "" {
		layer, err := server.NewLayer(bytes.NewReader([]byte(mf.System)), "application/vnd.ollama.image.system")
		if err != nil {
			return nil, fmt.Errorf("failed to create system layer: %w", err)
		}
		layers = append(layers, layer)
	}

	if mf.License != "" {
		layer, err := server.NewLayer(bytes.NewReader([]byte(mf.License)), "application/vnd.ollama.image.license")
		if err != nil {
			return nil, fmt.Errorf("failed to create license layer: %w", err)
		}
		layers = append(layers, layer)
	}

	return layers, nil
}
