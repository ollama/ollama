// Package capabilities infers model capability tags from reusable model metadata.
package capabilities

import (
	"slices"
	"strings"

	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/types/model"
)

// Parser is the capability-reporting subset of model/parsers.Parser.
type Parser interface {
	HasToolSupport() bool
	HasThinkingSupport() bool
}

// Append adds capability if it is non-empty and not already present.
func Append(capabilities []model.Capability, capability model.Capability) []model.Capability {
	if capability == "" || slices.Contains(capabilities, capability) {
		return capabilities
	}
	return append(capabilities, capability)
}

// AppendAll appends capabilities, preserving first-seen order and uniqueness.
func AppendAll(capabilities []model.Capability, values ...model.Capability) []model.Capability {
	for _, capability := range values {
		capabilities = Append(capabilities, capability)
	}
	return capabilities
}

// FromChatTemplate returns capabilities inferred from a GGUF Jinja chat template.
func FromChatTemplate(chatTemplate string) []model.Capability {
	return AppendChatTemplate(nil, chatTemplate)
}

// AppendChatTemplate appends capabilities inferred from a GGUF Jinja chat template.
func AppendChatTemplate(capabilities []model.Capability, chatTemplate string) []model.Capability {
	if chatTemplate == "" {
		return capabilities
	}

	if ChatTemplateHasToolSupport(chatTemplate) {
		capabilities = Append(capabilities, model.CapabilityTools)
	}
	if ChatTemplateHasThinkingSupport(chatTemplate) {
		capabilities = Append(capabilities, model.CapabilityThinking)
	}

	return capabilities
}

// ChatTemplateHasToolSupport reports whether a GGUF Jinja chat template appears to support tools.
func ChatTemplateHasToolSupport(chatTemplate string) bool {
	return strings.Contains(chatTemplate, "tools") || strings.Contains(chatTemplate, "tool_call")
}

// ChatTemplateHasToolRoundTrip reports whether a GGUF Jinja chat template appears to support tool calls and tool responses.
func ChatTemplateHasToolRoundTrip(chatTemplate string) bool {
	if !ChatTemplateHasToolSupport(chatTemplate) {
		return false
	}

	toolCalls := strings.Contains(chatTemplate, "tool_calls") || strings.Contains(chatTemplate, "assistant_tool_call")
	return toolCalls && (strings.Contains(chatTemplate, "tool_response") ||
		strings.Contains(chatTemplate, "tool_results") ||
		strings.Contains(chatTemplate, "role'] == 'tool'") ||
		strings.Contains(chatTemplate, `role'] == "tool"`) ||
		strings.Contains(chatTemplate, `role"] == 'tool'`) ||
		strings.Contains(chatTemplate, `role"] == "tool"`) ||
		strings.Contains(chatTemplate, `message.role == 'tool'`) ||
		strings.Contains(chatTemplate, `message.role == "tool"`) ||
		strings.Contains(chatTemplate, "ipython"))
}

// ChatTemplateHasThinkingSupport reports whether a GGUF Jinja chat template appears to support thinking.
func ChatTemplateHasThinkingSupport(chatTemplate string) bool {
	if strings.Contains(chatTemplate, "<think>") && strings.Contains(chatTemplate, "</think>") {
		return true
	}

	// Some Qwen/DeepSeek templates strip prior reasoning by splitting assistant
	// content at </think>; llama.cpp can still extract reasoning from them.
	return (strings.Contains(chatTemplate, "content.split('</think>')") ||
		strings.Contains(chatTemplate, `content.split("</think>")`)) &&
		!strings.Contains(chatTemplate, "reasoning_content") &&
		!strings.Contains(chatTemplate, "<SPECIAL_12>")
}

// FromGoTemplate returns capabilities inferred from an Ollama Go template.
func FromGoTemplate(t *template.Template) ([]model.Capability, error) {
	if t == nil {
		return nil, nil
	}

	v, err := t.Vars()
	if err != nil {
		return nil, err
	}

	var capabilities []model.Capability
	if slices.Contains(v, "tools") {
		capabilities = Append(capabilities, model.CapabilityTools)
	}
	if slices.Contains(v, "suffix") {
		capabilities = Append(capabilities, model.CapabilityInsert)
	}

	openingTag, closingTag := thinking.InferTags(t.Template)
	if openingTag != "" && closingTag != "" {
		capabilities = Append(capabilities, model.CapabilityThinking)
	}

	return capabilities, nil
}

// GoTemplateHasToolRoundTrip reports whether an Ollama Go template appears to support tool calls and tool responses.
func GoTemplateHasToolRoundTrip(t *template.Template) bool {
	if t == nil {
		return false
	}

	v, err := t.Vars()
	if err != nil || !slices.Contains(v, "tools") || !slices.Contains(v, "toolcalls") {
		return false
	}

	raw := t.String()
	return strings.Contains(raw, `eq .Role "tool"`) ||
		strings.Contains(raw, "tool_response") ||
		strings.Contains(raw, "TOOL_RESULTS")
}

// FromParser returns capabilities reported by a parser implementation.
func FromParser(p Parser) []model.Capability {
	if p == nil {
		return nil
	}

	var capabilities []model.Capability
	if p.HasToolSupport() {
		capabilities = Append(capabilities, model.CapabilityTools)
	}
	if p.HasThinkingSupport() {
		capabilities = Append(capabilities, model.CapabilityThinking)
	}

	return capabilities
}

// HasMore reports whether candidate contains more capability tags than current.
func HasMore(candidate, current []model.Capability) bool {
	return len(candidate) > len(current)
}

// Same reports whether candidate and current contain the same capability tags, ignoring order.
func Same(candidate, current []model.Capability) bool {
	if len(candidate) != len(current) {
		return false
	}
	for _, c := range candidate {
		if !slices.Contains(current, c) {
			return false
		}
	}
	return true
}
