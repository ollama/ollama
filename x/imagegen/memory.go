// Package imagegen provides image generation capabilities for Ollama.
package imagegen

import "encoding/json"

// GB is a convenience constant for gigabytes.
const GB = 1024 * 1024 * 1024

// modelVRAMEstimates maps pipeline class names to their estimated VRAM requirements.
var modelVRAMEstimates = map[string]uint64{
	"ZImagePipeline":    12 * GB, // ~12GB for Z-Image
	"FluxPipeline":      12 * GB, // ~12GB for Flux (same architecture)
	"QwenImagePipeline": 16 * GB, // ~16GB for Qwen-Image
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
// Returns a conservative default of 16GB if the model type cannot be determined.
func EstimateVRAM(modelName string) uint64 {
	manifest, err := LoadManifest(modelName)
	if err != nil {
		return 16 * GB
	}

	data, err := manifest.ReadConfig("model_index.json")
	if err != nil {
		return 16 * GB
	}

	// Parse just the class name
	var index struct {
		ClassName string `json:"_class_name"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		return 16 * GB
	}

	if estimate, ok := modelVRAMEstimates[index.ClassName]; ok {
		return estimate
	}
	return 16 * GB
}

// IsImageGenModel checks if the given model name is an image generation model.
func IsImageGenModel(modelName string) bool {
	return ResolveModelName(modelName) != ""
}
