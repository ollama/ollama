package create

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// InferSafetensorsCapabilitiesFromDir derives the runtime capabilities for a
// Hugging Face style safetensors model directory.
func InferSafetensorsCapabilitiesFromDir(modelDir string) []string {
	data, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return []string{"completion"}
	}

	caps, err := InferSafetensorsCapabilities(data)
	if err != nil {
		return []string{"completion"}
	}
	return caps
}

// InferSafetensorsCapabilities derives capabilities that can be known from
// config.json alone. Generic tool and thinking support is also inferred at
// runtime from template variables and registered parser capabilities.
func InferSafetensorsCapabilities(configJSON []byte) ([]string, error) {
	caps := []string{"completion"}

	if hasVisionConfig(configJSON) {
		caps = append(caps, "vision")
	}
	if hasAudioConfig(configJSON) {
		caps = append(caps, "audio")
	}

	thinking, err := SupportsThinkingConfig(configJSON)
	if err != nil {
		return nil, err
	}
	if thinking {
		caps = append(caps, "thinking")
	}

	return caps, nil
}

// InferSafetensorsCapabilitiesFromFiles derives runtime capabilities from
// uploaded safetensors config blobs keyed by their manifest layer names.
func InferSafetensorsCapabilitiesFromFiles(files map[string][]byte) ([]string, error) {
	if data, ok := files["model_index.json"]; ok {
		var cfg map[string]any
		if err := json.Unmarshal(data, &cfg); err != nil {
			return nil, err
		}
		return []string{"image"}, nil
	}

	if data, ok := files["config.json"]; ok {
		return InferSafetensorsCapabilities(data)
	}

	return nil, fmt.Errorf("config.json or model_index.json not found")
}

// SupportsThinkingConfig checks the small set of model families whose thinking
// support must be inferred from config.json metadata at create time. This is a
// fallback for models that lack a template/parser signal, not the generic
// runtime capability path.
func SupportsThinkingConfig(configJSON []byte) (bool, error) {
	var cfg struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := json.Unmarshal(configJSON, &cfg); err != nil {
		return false, err
	}

	thinkingConfigFallbackFamilies := []string{
		"glm4moe",
		"deepseek",
		"qwen3",
	}

	for _, arch := range cfg.Architectures {
		archLower := strings.ToLower(arch)
		if slices.ContainsFunc(thinkingConfigFallbackFamilies, func(family string) bool {
			return strings.Contains(archLower, family)
		}) {
			return true, nil
		}
	}

	typeLower := strings.ToLower(cfg.ModelType)
	if typeLower != "" && slices.ContainsFunc(thinkingConfigFallbackFamilies, func(family string) bool {
		return strings.Contains(typeLower, family)
	}) {
		return true, nil
	}

	return false, nil
}

func hasVisionConfig(configJSON []byte) bool {
	var cfg struct {
		VisionConfig *map[string]any `json:"vision_config"`
	}
	return json.Unmarshal(configJSON, &cfg) == nil && cfg.VisionConfig != nil
}

func hasAudioConfig(configJSON []byte) bool {
	var cfg struct {
		AudioConfig *map[string]any `json:"audio_config"`
	}
	return json.Unmarshal(configJSON, &cfg) == nil && cfg.AudioConfig != nil
}
