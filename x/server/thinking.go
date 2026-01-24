package server

import (
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// IsSafetensorsThinkingModel checks if a safetensors model supports thinking
// based on its architecture from config.json.
func IsSafetensorsThinkingModel(name model.Name) bool {
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return false
	}

	var config struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := mf.ReadConfigJSON("config.json", &config); err != nil {
		return false
	}

	// Determine architecture
	arch := config.ModelType
	if arch == "" && len(config.Architectures) > 0 {
		arch = config.Architectures[0]
	}
	if arch == "" {
		return false
	}

	archLower := strings.ToLower(arch)

	// List of architectures that support thinking
	thinkingArchitectures := []string{
		"glm4moe",  // GLM-4 MoE models
		"deepseek", // DeepSeek models
		"qwen3",    // Qwen3 models
	}

	for _, thinkArch := range thinkingArchitectures {
		if strings.Contains(archLower, thinkArch) {
			return true
		}
	}

	return false
}
