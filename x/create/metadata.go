package create

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	modelparsers "github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/model"
)

// inferSafetensorsConfig derives the manifest config shared by local and
// server-side safetensors imports. Explicit parser and renderer values take
// precedence over the inferred values.
func inferSafetensorsConfig(modelDir string, cfg sourceModelConfig, parserOverride, rendererOverride string) (model.ConfigV2, error) {
	chatTemplate, err := readChatTemplateStrict(modelDir)
	if err != nil {
		return model.ConfigV2{}, err
	}

	parserName, err := parserNameForConfig(modelDir, cfg, chatTemplate)
	if err != nil {
		return model.ConfigV2{}, err
	}
	if parserOverride != "" {
		parserName = parserOverride
	}

	rendererName, err := rendererNameForConfig(modelDir, cfg, chatTemplate)
	if err != nil {
		return model.ConfigV2{}, err
	}
	if rendererOverride != "" {
		rendererName = rendererOverride
	}

	capabilities := inferSafetensorsCapabilitiesFromConfig(cfg, chatTemplate, parserName)
	generationDefaults, err := readHFGenerationDefaults(modelDir)
	if err != nil {
		return model.ConfigV2{}, err
	}

	return model.ConfigV2{
		ModelFormat:        "safetensors",
		Parser:             parserName,
		Renderer:           rendererName,
		Capabilities:       capabilities,
		GenerationDefaults: generationDefaults,
	}, nil
}

func readHFGenerationDefaults(modelDir string) (model.GenerationDefaults, error) {
	path := filepath.Join(modelDir, "generation_config.json")
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	} else if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}

	defaults, err := model.ParseHFGenerationDefaults(data)
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return defaults, nil
}

func readChatTemplateStrict(modelDir string) (string, error) {
	tokenizerConfig := filepath.Join(modelDir, "tokenizer_config.json")
	if data, err := os.ReadFile(tokenizerConfig); err == nil {
		var cfg struct {
			ChatTemplate string `json:"chat_template"`
		}
		if err := json.Unmarshal(data, &cfg); err != nil {
			return "", fmt.Errorf("parse %s: %w", tokenizerConfig, err)
		}
		if cfg.ChatTemplate != "" {
			return cfg.ChatTemplate, nil
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("read %s: %w", tokenizerConfig, err)
	}

	chatTemplatePath := filepath.Join(modelDir, "chat_template.jinja")
	data, err := os.ReadFile(chatTemplatePath)
	if err == nil {
		return string(data), nil
	}
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}
	return "", fmt.Errorf("read %s: %w", chatTemplatePath, err)
}

func inferSafetensorsCapabilitiesFromConfig(cfg sourceModelConfig, chatTemplate, parserName string) []string {
	capabilities := []string{"completion"}

	caps := detectCapabilitiesFromConfig(cfg, chatTemplate)
	if caps.vision {
		capabilities = append(capabilities, "vision")
	}
	if caps.audio {
		capabilities = append(capabilities, "audio")
	}

	var builtinParser modelparsers.Parser
	if parserName != "" {
		builtinParser = modelparsers.ParserForName(parserName)
	}
	if builtinParser != nil && builtinParser.HasToolSupport() {
		capabilities = append(capabilities, "tools")
	}
	if caps.thinking || (builtinParser != nil && builtinParser.HasThinkingSupport()) {
		capabilities = append(capabilities, "thinking")
	}

	return capabilities
}

type modelCapabilities struct {
	vision   bool
	audio    bool
	thinking bool
}

func detectCapabilitiesFromConfig(cfg sourceModelConfig, chatTemplate string) modelCapabilities {
	return modelCapabilities{
		vision: cfg.VisionConfig != nil || cfg.HasVision,
		audio:  cfg.AudioConfig != nil || cfg.SoundConfig != nil,
		thinking: thinking.TemplateSupportsThinking(chatTemplate) ||
			alwaysSupportsThinking(cfg.Architectures, cfg.ModelType),
	}
}

func alwaysSupportsThinking(architectures []string, modelType string) bool {
	if isQwen35Family(modelType) || isQwen4Family(modelType) {
		return true
	}
	for _, arch := range architectures {
		if isQwen35Family(arch) || isQwen4Family(arch) {
			return true
		}
	}
	return false
}

func isQwen35Family(s string) bool {
	s = strings.ToLower(s)
	return strings.Contains(s, "qwen3_5") || strings.Contains(s, "qwen3next")
}

func isQwen4Family(s string) bool {
	s = strings.ToLower(s)
	return strings.Contains(s, "qwen4exp") || strings.Contains(s, "qwen4_exp")
}

func qwen35RendererNameFromTemplate(chatTemplate string) string {
	if strings.Contains(chatTemplate, "resolved_reasoning_effort") &&
		strings.Contains(chatTemplate, "preserve_thinking") {
		return "qwen3.8"
	}
	return "qwen3.5"
}

func lagunaRendererParserNameFromTemplate(modelDir, chatTemplate string) (string, error) {
	const poolsideV1Marker = "laguna_glm_thinking_v8"

	if strings.Contains(chatTemplate, poolsideV1Marker) {
		return "poolside-v1", nil
	}

	chatTemplatePath := filepath.Join(modelDir, "chat_template.jinja")
	data, err := os.ReadFile(chatTemplatePath)
	if err == nil && strings.Contains(string(data), poolsideV1Marker) {
		return "poolside-v1", nil
	}
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("read %s: %w", chatTemplatePath, err)
	}
	return "laguna", nil
}

func nemotronRendererParserNameFromTemplate(modelDir, chatTemplate string) (string, error) {
	const v35Marker = "{reasoning effort: efficient}"

	chatTemplatePath := filepath.Join(modelDir, "chat_template.jinja")
	data, err := os.ReadFile(chatTemplatePath)
	if err == nil && strings.Contains(string(data), v35Marker) {
		return "nemotron-3.5-nano", nil
	}
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("read %s: %w", chatTemplatePath, err)
	}
	if strings.Contains(chatTemplate, v35Marker) {
		return "nemotron-3.5-nano", nil
	}
	return "nemotron-3-nano", nil
}

func sourceConfigIdentifiers(cfg sourceModelConfig) []string {
	ids := append([]string(nil), cfg.Architectures...)
	return append(ids, cfg.ModelType, cfg.LLMConfig.ModelType)
}

func parserNameForConfig(modelDir string, cfg sourceModelConfig, chatTemplate string) (string, error) {
	for _, id := range sourceConfigIdentifiers(cfg) {
		name, err := parserNameForIdentifier(modelDir, id, chatTemplate)
		if err != nil || name != "" {
			return name, err
		}
	}
	return "", nil
}

func parserNameForIdentifier(modelDir, s, chatTemplate string) (string, error) {
	s = strings.ToLower(s)
	switch {
	case strings.HasPrefix(s, "museglimmer") || s == "muse_glimmer":
		return "glimmer", nil
	case strings.Contains(s, "laguna"):
		return lagunaRendererParserNameFromTemplate(modelDir, chatTemplate)
	case strings.Contains(s, "cohere2moe") || strings.Contains(s, "cohere2_moe"):
		return "cohere", nil
	case strings.Contains(s, "glm4") || strings.Contains(s, "glm-4"):
		return "glm-4.7", nil
	case strings.Contains(s, "deepseek"):
		return "deepseek3", nil
	case strings.Contains(s, "gemma4"):
		return "gemma4", nil
	case isQwen4Family(s), isQwen35Family(s):
		return "qwen3.5", nil
	case strings.Contains(s, "qwen3"):
		return "qwen3", nil
	case strings.Contains(s, "nemotronh") || strings.Contains(s, "nemotron_h"):
		return nemotronRendererParserNameFromTemplate(modelDir, chatTemplate)
	default:
		return "", nil
	}
}

func rendererNameForConfig(modelDir string, cfg sourceModelConfig, chatTemplate string) (string, error) {
	for _, id := range sourceConfigIdentifiers(cfg) {
		name, err := rendererNameForIdentifier(modelDir, id, chatTemplate)
		if err != nil || name != "" {
			return name, err
		}
	}
	return "", nil
}

func rendererNameForIdentifier(modelDir, s, chatTemplate string) (string, error) {
	s = strings.ToLower(s)
	switch {
	case strings.HasPrefix(s, "museglimmer") || s == "muse_glimmer":
		return "glimmer", nil
	case strings.Contains(s, "laguna"):
		return lagunaRendererParserNameFromTemplate(modelDir, chatTemplate)
	case strings.Contains(s, "cohere2moe") || strings.Contains(s, "cohere2_moe"):
		return "cohere", nil
	case strings.Contains(s, "gemma4"):
		return "gemma4", nil
	case strings.Contains(s, "glm4") || strings.Contains(s, "glm-4"):
		return "glm-4.7", nil
	case strings.Contains(s, "deepseek"):
		return "deepseek3", nil
	case isQwen4Family(s):
		return "qwen3.8", nil
	case isQwen35Family(s):
		return qwen35RendererNameFromTemplate(chatTemplate), nil
	case strings.Contains(s, "qwen3"):
		return "qwen3-coder", nil
	case strings.Contains(s, "nemotronh") || strings.Contains(s, "nemotron_h"):
		return nemotronRendererParserNameFromTemplate(modelDir, chatTemplate)
	default:
		return "", nil
	}
}
