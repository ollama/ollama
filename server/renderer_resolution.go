package server

import (
	"strconv"
	"strings"

	"github.com/ollama/ollama/format"
)

const (
	gemma4RendererLegacy = "gemma4"
	gemma4RendererSmall  = "gemma4-small"
	gemma4RendererLarge  = "gemma4-large"
	qwen35Default        = "qwen3.5"
	qwen35Instruct       = "qwen3.5-instruct"
	qwen35Thinking       = "qwen3.5-thinking"

	// Gemma 4 small templates cover the e2b/e4b family, while 12b/26b/31b use
	// the large template. Default to the small prompt unless the model is
	// clearly in the large range.
	gemma4LargeMinParameterCount = 12_000_000_000
)

func resolveRendererName(m *Model) string {
	if m == nil || m.Config.Renderer == "" {
		return ""
	}

	switch m.Config.Renderer {
	case gemma4RendererLegacy:
		return resolveGemma4Renderer(m)
	case qwen35Default:
		return resolveQwen35Variant(m)
	default:
		return m.Config.Renderer
	}
}

func resolveParserName(m *Model) string {
	if m == nil || m.Config.Parser == "" {
		return ""
	}

	switch m.Config.Parser {
	case qwen35Default:
		return resolveQwen35Variant(m)
	default:
		return m.Config.Parser
	}
}

func resolveGemma4Renderer(m *Model) string {
	if m == nil || m.Config.Renderer != gemma4RendererLegacy {
		if m == nil {
			return gemma4RendererLegacy
		}
		return m.Config.Renderer
	}

	if renderer, ok := gemma4RendererFromName(m.ShortName); ok {
		return renderer
	}

	if renderer, ok := gemma4RendererFromName(m.Name); ok {
		return renderer
	}

	if parameterCount, ok := parseHumanParameterCount(m.Config.ModelType); ok {
		return gemma4RendererForParameterCount(parameterCount)
	}

	return gemma4RendererSmall
}

func gemma4RendererForParameterCount(parameterCount uint64) string {
	if parameterCount >= gemma4LargeMinParameterCount {
		return gemma4RendererLarge
	}

	return gemma4RendererSmall
}

func gemma4RendererFromName(name string) (string, bool) {
	lower := strings.ToLower(name)
	switch {
	case strings.Contains(lower, "e2b"), strings.Contains(lower, "e4b"):
		return gemma4RendererSmall, true
	case strings.Contains(lower, "12b"), strings.Contains(lower, "26b"), strings.Contains(lower, "31b"):
		return gemma4RendererLarge, true
	default:
		return "", false
	}
}

func resolveQwen35Variant(m *Model) string {
	if m == nil {
		return qwen35Default
	}

	for _, name := range []string{m.ShortName, m.Name, m.Config.BaseName, m.Config.RemoteModel} {
		if variant, ok := qwen35VariantFromName(name); ok {
			return variant
		}
	}

	return qwen35Default
}

func qwen35VariantFromName(name string) (string, bool) {
	lower := strings.ToLower(name)
	if idx := strings.LastIndex(lower, "/"); idx != -1 {
		lower = lower[idx+1:]
	}

	if !strings.Contains(lower, "qwen3.5") && !strings.Contains(lower, "qwen35") {
		return "", false
	}

	switch {
	case strings.Contains(lower, "instruct"),
		strings.Contains(lower, "non-thinking"),
		strings.Contains(lower, "non_thinking"),
		strings.Contains(lower, "nothink"),
		strings.Contains(lower, "no-think"):
		return qwen35Instruct, true
	case strings.Contains(lower, "thinking"),
		strings.Contains(lower, "think"):
		return qwen35Thinking, true
	default:
		return "", false
	}
}

func parseHumanParameterCount(s string) (uint64, bool) {
	if s == "" {
		return 0, false
	}

	unit := strings.ToUpper(s[len(s)-1:])
	var multiplier float64
	switch unit {
	case "B":
		multiplier = float64(format.Billion)
	case "M":
		multiplier = float64(format.Million)
	case "K":
		multiplier = float64(format.Thousand)
	default:
		return 0, false
	}

	value, err := strconv.ParseFloat(s[:len(s)-1], 64)
	if err != nil {
		return 0, false
	}

	return uint64(value * multiplier), true
}

func isGemma4Renderer(renderer string) bool {
	switch renderer {
	case gemma4RendererLegacy, gemma4RendererSmall, gemma4RendererLarge:
		return true
	default:
		return false
	}
}
