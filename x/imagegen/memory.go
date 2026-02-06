// Package imagegen provides experimental image generation capabilities for Ollama.
//
// This package is in x/ because the tensor model storage format is under development.
// The goal is to integrate these capabilities into the main Ollama packages once
// the format is stable.
//
// TODO (jmorganca): Integrate into main packages when stable:
//   - CLI commands → cmd/
//   - API endpoints → api/
//   - Model creation → server/
package imagegen

import (
	"encoding/json"
	"fmt"
	"runtime"

	"github.com/ollama/ollama/x/imagegen/manifest"
)

// SupportedBackends lists the backends that support image generation.
var SupportedBackends = []string{"metal", "cuda", "cpu"}

// CheckPlatformSupport validates that image generation is supported on the current platform.
// Returns nil if supported, or an error describing why it's not supported.
func CheckPlatformSupport() error {
	switch runtime.GOOS {
	case "darwin":
		// macOS: Metal is supported via MLX
		if runtime.GOARCH != "arm64" {
			return fmt.Errorf("image generation on macOS requires Apple Silicon (arm64), got %s", runtime.GOARCH)
		}
		return nil
	case "linux", "windows":
		// Linux/Windows: CUDA support (requires mlx or cuda build)
		// The actual backend availability is checked at runtime
		return nil
	default:
		return fmt.Errorf("image generation is not supported on %s", runtime.GOOS)
	}
}

// ResolveModelName checks if a model name is a known image generation model.
// Returns the normalized model name if found, empty string otherwise.
func ResolveModelName(modelName string) string {
	modelManifest, err := manifest.LoadManifest(modelName)
	if err == nil && modelManifest.HasTensorLayers() {
		return modelName
	}
	return ""
}

// DetectModelType reads model_index.json and returns the model type.
// Checks both "architecture" (Ollama format) and "_class_name" (diffusers format).
// Returns empty string if detection fails.
func DetectModelType(modelName string) string {
	modelManifest, err := manifest.LoadManifest(modelName)
	if err != nil {
		return ""
	}

	data, err := modelManifest.ReadConfig("model_index.json")
	if err != nil {
		return ""
	}

	var index struct {
		Architecture string `json:"architecture"`
		ClassName    string `json:"_class_name"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		return ""
	}

	// Prefer architecture (Ollama format), fall back to _class_name (diffusers)
	if index.Architecture != "" {
		return index.Architecture
	}
	return index.ClassName
}
