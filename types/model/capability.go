package model

import "strings"

type Capability string

const (
	CapabilityCompletion = Capability("completion")
	CapabilityTools      = Capability("tools")
	CapabilityInsert     = Capability("insert")
	CapabilityVision     = Capability("vision")
	CapabilityEmbedding  = Capability("embedding")
	CapabilityThinking   = Capability("thinking")
	CapabilityImage      = Capability("image")
	CapabilityAudio      = Capability("audio")
)

func (c Capability) String() string {
	return string(c)
}

// ChatTemplateHasToolSupport reports whether a Jinja chat template references
// tools or tool calls.
func ChatTemplateHasToolSupport(chatTemplate string) bool {
	return strings.Contains(chatTemplate, "tools") || strings.Contains(chatTemplate, "tool_call")
}

// ChatTemplateHasThinkingSupport reports whether a Jinja chat template emits
// thinking blocks.
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
