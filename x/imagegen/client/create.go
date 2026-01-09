// Package client provides client-side model creation for tensor-based models.
//
// This package is in x/ because the tensor model storage format is under development.
// It also exists to break an import cycle: server imports x/imagegen, so x/imagegen
// cannot import server. This sub-package can import server because server doesn't
// import it.
//
// TODO (jmorganca): This is temporary. When tensor models are promoted to production:
//  1. Add proper API endpoints for tensor model creation
//  2. Move tensor extraction to server-side
//  3. Remove this package
//  4. Follow the same client→server pattern as regular model creation
package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/imagegen"
)

// MinOllamaVersion is the minimum Ollama version required for image generation models.
const MinOllamaVersion = "0.14.0"

// CreateModel imports a tensor-based model from a local directory.
// This creates blobs and manifest directly on disk, bypassing the HTTP API.
//
// TODO (jmorganca): Replace with API-based creation when promoted to production.
func CreateModel(modelName, modelDir string, p *progress.Progress) error {
	if !imagegen.IsTensorModelDir(modelDir) {
		return fmt.Errorf("%s is not an image generation model directory (model_index.json not found)", modelDir)
	}

	status := "importing image generation model"
	spinner := progress.NewSpinner(status)
	p.Add("imagegen", spinner)

	// Create layer callback for config files
	createLayer := func(r io.Reader, mediaType, name string) (imagegen.LayerInfo, error) {
		layer, err := server.NewLayer(r, mediaType)
		if err != nil {
			return imagegen.LayerInfo{}, err
		}
		layer.Name = name

		return imagegen.LayerInfo{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      name,
		}, nil
	}

	// Create tensor layer callback for individual tensors
	// name is path-style: "component/tensor_name"
	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32) (imagegen.LayerInfo, error) {
		layer, err := server.NewLayer(r, server.MediaTypeImageTensor)
		if err != nil {
			return imagegen.LayerInfo{}, err
		}
		layer.Name = name
		layer.Dtype = dtype
		layer.Shape = shape

		return imagegen.LayerInfo{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      name,
			Dtype:     dtype,
			Shape:     shape,
		}, nil
	}

	// Create manifest writer callback
	writeManifest := func(modelName string, config imagegen.LayerInfo, layers []imagegen.LayerInfo) error {
		name := model.ParseName(modelName)
		if !name.IsValid() {
			return fmt.Errorf("invalid model name: %s", modelName)
		}

		// Create a proper config blob with version requirement
		configData := model.ConfigV2{
			ModelFormat:  "safetensors",
			Capabilities: []string{"image"},
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

		// Convert LayerInfo to server.Layer (include the original model_index.json in layers)
		serverLayers := make([]server.Layer, len(layers))
		for i, l := range layers {
			serverLayers[i] = server.Layer{
				MediaType: l.MediaType,
				Digest:    l.Digest,
				Size:      l.Size,
				TensorMeta: server.TensorMeta{
					Name:  l.Name,
					Dtype: l.Dtype,
					Shape: l.Shape,
				},
			}
		}

		return server.WriteManifest(name, configLayer, serverLayers)
	}

	// Progress callback
	progressFn := func(msg string) {
		spinner.Stop()
		status = msg
		spinner = progress.NewSpinner(status)
		p.Add("imagegen", spinner)
	}

	err := imagegen.CreateModel(modelName, modelDir, createLayer, createTensorLayer, writeManifest, progressFn)
	spinner.Stop()
	if err != nil {
		return err
	}

	fmt.Printf("Created image generation model '%s'\n", modelName)
	return nil
}
