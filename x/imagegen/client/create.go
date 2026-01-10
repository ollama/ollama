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
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/server"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/imagegen"
)

// MinOllamaVersion is the minimum Ollama version required for image generation models.
const MinOllamaVersion = "0.14.0"

// CreateModel imports a tensor-based model from a local directory.
func CreateModel(modelName, modelDir string, p *progress.Progress) error {
	return CreateModelFromModelfile(modelName, modelDir, nil, p)
}

// CreateModelFromModelfile imports a tensor-based model using Modelfile commands.
// Extracts LICENSE, REQUIRES, and PARAMETER commands from the Modelfile.
func CreateModelFromModelfile(modelName, modelDir string, commands []parser.Command, p *progress.Progress) error {
	if !imagegen.IsTensorModelDir(modelDir) {
		return fmt.Errorf("%s is not an image generation model directory (model_index.json not found)", modelDir)
	}

	// Extract metadata from Modelfile commands
	var licenses []string
	var requires string
	params := make(map[string]any)

	for _, c := range commands {
		switch c.Name {
		case "license":
			licenses = append(licenses, c.Args)
		case "requires":
			requires = c.Args
		case "model":
			// skip - already handled by caller
		default:
			// Treat as parameter (steps, width, height, seed, etc.)
			ps, err := api.FormatParams(map[string][]string{c.Name: {c.Args}})
			if err == nil {
				for k, v := range ps {
					params[k] = v
				}
			}
		}
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
		return imagegen.LayerInfo{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      name,
		}, nil
	}

	// Create tensor layer callback
	createTensorLayer := func(r io.Reader, name, dtype string, shape []int32) (imagegen.LayerInfo, error) {
		layer, err := server.NewLayer(r, server.MediaTypeImageTensor)
		if err != nil {
			return imagegen.LayerInfo{}, err
		}
		return imagegen.LayerInfo{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      name,
		}, nil
	}

	// Create manifest writer callback
	writeManifest := func(modelName string, config imagegen.LayerInfo, layers []imagegen.LayerInfo) error {
		name := model.ParseName(modelName)
		if !name.IsValid() {
			return fmt.Errorf("invalid model name: %s", modelName)
		}

		// Use Modelfile REQUIRES if specified, otherwise use minimum
		if requires == "" {
			requires = MinOllamaVersion
		}

		configData := model.ConfigV2{
			ModelFormat:  "safetensors",
			Capabilities: []string{"image"},
			Requires:     requires,
		}
		configJSON, err := json.Marshal(configData)
		if err != nil {
			return fmt.Errorf("failed to marshal config: %w", err)
		}

		configLayer, err := server.NewLayer(bytes.NewReader(configJSON), "application/vnd.docker.container.image.v1+json")
		if err != nil {
			return fmt.Errorf("failed to create config layer: %w", err)
		}

		// Convert to server.Layer
		serverLayers := make([]server.Layer, len(layers))
		for i, l := range layers {
			serverLayers[i] = server.Layer{
				MediaType: l.MediaType,
				Digest:    l.Digest,
				Size:      l.Size,
				Name:      l.Name,
			}
		}

		// Add license layers
		for _, license := range licenses {
			layer, err := server.NewLayer(strings.NewReader(license), "application/vnd.ollama.image.license")
			if err != nil {
				return fmt.Errorf("failed to create license layer: %w", err)
			}
			serverLayers = append(serverLayers, layer)
		}

		// Add parameters layer
		if len(params) > 0 {
			paramsJSON, err := json.Marshal(params)
			if err != nil {
				return fmt.Errorf("failed to marshal parameters: %w", err)
			}
			layer, err := server.NewLayer(bytes.NewReader(paramsJSON), "application/vnd.ollama.image.params")
			if err != nil {
				return fmt.Errorf("failed to create params layer: %w", err)
			}
			serverLayers = append(serverLayers, layer)
		}

		return server.WriteManifest(name, configLayer, serverLayers)
	}

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
