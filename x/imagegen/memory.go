// Package imagegen provides image generation capabilities for Ollama.
package imagegen

import (
	"encoding/json"
	"fmt"
	"runtime"
)

// GB is a convenience constant for gigabytes.
const GB = 1024 * 1024 * 1024

// SupportedBackends lists the backends that support image generation.
var SupportedBackends = []string{"metal", "cuda", "cpu"}

// modelVRAMEstimates maps pipeline class names to their estimated VRAM requirements.
var modelVRAMEstimates = map[string]uint64{
	"ZImagePipeline":    21 * GB, // ~21GB for Z-Image (text encoder + transformer + VAE)
	"FluxPipeline":      21 * GB, // ~21GB for Flux (same architecture)
	"QwenImagePipeline": 80 * GB, // TODO: verify actual requirements, using conservative estimate for now
}

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

// CheckMemoryRequirements validates that there's enough memory for image generation.
// Returns nil if memory is sufficient, or an error if not.
func CheckMemoryRequirements(modelName string, availableMemory uint64) error {
	required := EstimateVRAM(modelName)
	if availableMemory < required {
		return fmt.Errorf("insufficient memory for image generation: need %d GB, have %d GB",
			required/GB, availableMemory/GB)
	}
	return nil
}

// ResolveModelName checks if a model name is a known image generation model.
// Returns the normalized model name if found, empty string otherwise.
func ResolveModelName(modelName string) string {
	manifest, err := LoadManifest(modelName)
	if err == nil && manifest.HasImageTensorLayers() {
		return modelName
	}
	return ""
}

// EstimateVRAM returns the estimated VRAM needed for an image generation model.
// Returns a conservative default of 21GB if the model type cannot be determined.
func EstimateVRAM(modelName string) uint64 {
	manifest, err := LoadManifest(modelName)
	if err != nil {
		return 21 * GB
	}

	data, err := manifest.ReadConfig("model_index.json")
	if err != nil {
		return 21 * GB
	}

	// Parse just the class name
	var index struct {
		ClassName string `json:"_class_name"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		return 21 * GB
	}

	if estimate, ok := modelVRAMEstimates[index.ClassName]; ok {
		return estimate
	}
	return 21 * GB
}

// IsImageGenModel checks if the given model name is an image generation model.
func IsImageGenModel(modelName string) bool {
	return ResolveModelName(modelName) != ""
}
