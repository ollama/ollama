package renderers

import (
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// GraniteThinkingRenderer implements the "granite thinking" chat template —
// the ChatML-with-<think> template used starting with Granite 4.2. It is not
// tied to a specific Granite version: older Granite models (4.1 and earlier)
// use a different, non-thinking template and must not be routed here (see
// create/metadata.go's graniteThinkingTemplateName, which gates on the
// template actually containing <think>).
//
// The template is a system turn (always present, even if empty) optionally
// followed by a JSON tool catalog, then the conversation, with
// <think>...</think> reasoning and Qwen3-Coder-style
// <tool_call><function=...> tool calls.
type GraniteThinkingRenderer struct{}

func (r *GraniteThinkingRenderer) LeadingBOS() string {
	return ""
}

func (r *GraniteThinkingRenderer) Thinking() *model.Thinking {
	return &model.Thinking{Values: []any{false, true}, Default: true}
}

func (r *GraniteThinkingRenderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var sb strings.Builder

	isThinking := true
	if think != nil && think.Value != nil {
		isThinking = think.Bool()
	}

	var systemMessage string
	loopMessages := messages
	if len(messages) > 0 && messages[0].Role == "system" {
		systemMessage = messages[0].Content
		loopMessages = messages[1:]
	}

	lastUserIdx := -1
	for i, m := range loopMessages {
		if m.Role == "user" {
			lastUserIdx = i
		}
	}

	// The system turn is always emitted, even when empty, matching the
	// reference template exactly (it never gates "<|im_start|>system\n" on
	// systemMessage or tools being non-empty).
	sb.WriteString(imStartTag + "system\n" + systemMessage)
	if len(tools) > 0 {
		if systemMessage != "" {
			sb.WriteString("\n\n")
		}
		sb.WriteString("# Tools\n\nYou have access to the following functions:\n\n<tools>")
		for _, tool := range tools {
			sb.WriteString("\n")
			if b, err := marshalWithSpaces(tool.Function); err == nil {
				sb.Write(b)
			}
		}
		sb.WriteString(qwen35ToolPostamble)
	}
	sb.WriteString(imEndTag + "\n")

	for i, message := range loopMessages {
		switch message.Role {
		case "assistant":
			fullReasoning := i >= lastUserIdx
			var contentBlock string
			if fullReasoning && message.Thinking != "" {
				contentBlock = "<think>\n" + message.Thinking + "\n</think>\n" + message.Content
			} else {
				contentBlock = "<think></think>" + message.Content
			}
			contentBlock = strings.TrimSpace(contentBlock)

			sb.WriteString(imStartTag + "assistant\n")
			if len(message.ToolCalls) > 0 {
				if contentBlock != "" {
					sb.WriteString(contentBlock + "\n")
				}
				for _, toolCall := range message.ToolCalls {
					sb.WriteString("<tool_call>\n<function=" + toolCall.Function.Name + ">\n")
					for name, value := range toolCall.Function.Arguments.All() {
						sb.WriteString("<parameter=" + name + ">\n")
						sb.WriteString(formatToolCallArgument(value))
						sb.WriteString("\n</parameter>\n")
					}
					sb.WriteString("</function>\n</tool_call>\n")
				}
			} else {
				sb.WriteString(contentBlock)
			}
			sb.WriteString(imEndTag + "\n")

		case "tool":
			if i == 0 || loopMessages[i-1].Role != "tool" {
				sb.WriteString(imStartTag + "user")
			}
			sb.WriteString("\n<tool_response>\n" + message.Content + "\n</tool_response>\n")
			if i == len(loopMessages)-1 || loopMessages[i+1].Role != "tool" {
				sb.WriteString(imEndTag + "\n")
			}

		default: // "user", "system" (non-leading), or any other role
			sb.WriteString(imStartTag + message.Role + "\n" + message.Content + imEndTag + "\n")
		}
	}

	sb.WriteString(imStartTag + "assistant\n")
	if isThinking {
		sb.WriteString("<think>\n")
	} else {
		sb.WriteString("<think></think>")
	}

	return sb.String(), nil
}
