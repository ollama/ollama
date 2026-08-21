package parsers

import (
	"strings"

	"github.com/ollama/ollama/api"
)

// BailingParser parses output from Bailing MoE V3 (Ling) models. The output
// grammar matches GLM-4.6/4.7: reasoning terminated by </think>, free-form
// content, and <tool_call>{name}<arg_key>...</arg_key><arg_value>...</arg_value>
// blocks. The Bailing prompt prefills "<think>" when thinking is enabled and
// "<think></think>" when disabled, so the model output starts directly with
// thinking content (no opening tag) in the enabled case.
type BailingParser struct {
	GLM46Parser
}

func (p *BailingParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.callIndex = 0

	thinkingEnabled := thinkValue == nil || thinkValue.Bool()

	if lastMessage != nil && lastMessage.Role == "assistant" {
		// Mirror the renderer's semantic split: an inline <think> block in
		// Content counts as reasoning, not content, so the prompt may end
		// inside an open thinking block even when Content is non-empty.
		reasoning, content := splitBailingPrefill(lastMessage)
		switch {
		case content != "" || len(lastMessage.ToolCalls) > 0:
			// The prompt ends mid-content (or after a tool call).
			p.state = glm46ParserState_CollectingContent
		case reasoning != "":
			if thinkingEnabled {
				// The renderer leaves the thinking block open.
				p.state = glm46ParserState_CollectingThinking
			} else {
				// The renderer closes the block; the model continues content.
				p.state = glm46ParserState_CollectingContent
			}
		default:
			// A bare prefill renders exactly like the generation prompt.
			p.state = bailingInitialState(thinkingEnabled)
		}
		return tools
	}

	p.state = bailingInitialState(thinkingEnabled)
	return tools
}

func bailingInitialState(thinkingEnabled bool) glm46ParserState {
	if thinkingEnabled {
		// The prompt ends with <think>; output starts inside the block.
		return glm46ParserState_CollectingThinking
	}
	// The prompt ends with <think></think>; output is content, but tolerate a
	// stray opening tag.
	return glm46ParserState_LookingForThinkingOpen
}

// splitBailingPrefill splits a prefilled assistant message into reasoning and
// content with the same semantics as the Bailing renderer: an explicit
// Thinking field wins, otherwise an inline <think>...</think> block is
// extracted from Content.
func splitBailingPrefill(message *api.Message) (reasoning, content string) {
	reasoning = message.Thinking
	content = message.Content
	if reasoning != "" || !strings.Contains(content, "</think>") {
		return reasoning, content
	}

	parts := strings.Split(content, "</think>")
	first := strings.TrimRight(parts[0], "\n")
	if open := strings.LastIndex(first, "<think>"); open != -1 {
		first = first[open+len("<think>"):]
	}
	reasoning = strings.TrimLeft(first, "\n")
	content = strings.TrimLeft(parts[len(parts)-1], "\n")
	return reasoning, content
}
