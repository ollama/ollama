package renderers

import (
	"encoding/json"
	"strings"

	"github.com/ollama/ollama/api"
)

type DeepSeek3Variant int

const (
	Deepseek31 DeepSeek3Variant = iota
)

type DeepSeek3Renderer struct {
	IsThinking bool
	Variant    DeepSeek3Variant
}

func (r *DeepSeek3Renderer) LeadingBOS() string {
	return "<｜begin▁of▁sentence｜>"
}

func (r *DeepSeek3Renderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	segments, err := r.RenderSegments(messages, tools, thinkValue)
	if err != nil {
		return "", err
	}
	return JoinSegments(segments), nil
}

func (r *DeepSeek3Renderer) RenderSegments(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) ([]Segment, error) {
	var sb segmentBuilder

	// thinking is enabled: model must support it AND user must request it
	thinking := r.IsThinking && (thinkValue != nil && thinkValue.Bool())

	sb.control("<｜begin▁of▁sentence｜>")

	// extract system messages first
	isFirstSystemPrompt := true
	for _, message := range messages {
		if message.Role == "system" {
			if isFirstSystemPrompt {
				isFirstSystemPrompt = false
			} else {
				sb.content("\n\n")
			}
			sb.content(message.Content)
		}
	}

	// tool definitions
	if len(tools) > 0 {
		sb.control("\n\n## Tools\nYou have access to the following tools:\n")

		for _, tool := range tools {
			sb.control("\n### ")
			sb.content(tool.Function.Name)
			sb.control("\nDescription: ")
			sb.content(tool.Function.Description)

			// parameters as JSON
			parametersJSON, err := json.Marshal(tool.Function.Parameters)
			if err == nil {
				sb.control("\n\nParameters: ")
				sb.content(string(parametersJSON))
				sb.control("\n")
			}
		}

		// usage instructions
		sb.control("\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n")
		sb.control("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>\n\n")
		sb.control("Where:\n\n")
		sb.control("- `tool_call_name` must be an exact match to one of the available tools\n")
		sb.control("- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema\n")
		sb.control("- For multiple tool calls, chain them directly without separators or spaces\n")
	}

	// state tracking
	isTool := false
	isLastUser := false

	// Find the index of the last user message to determine which assistant message is "current"
	lastUserIndex := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			lastUserIndex = i
			break
		}
	}

	for i, message := range messages {
		switch message.Role {
		case "user":
			isTool = false
			isLastUser = true
			sb.control("<｜User｜>")
			sb.content(message.Content)

		case "assistant":
			if len(message.ToolCalls) > 0 {
				if isLastUser {
					sb.control("<｜Assistant｜></think>")
				}
				isLastUser = false
				isTool = false

				if message.Content != "" {
					sb.content(message.Content)
				}

				sb.control("<｜tool▁calls▁begin｜>")
				for _, toolCall := range message.ToolCalls {
					sb.control("<｜tool▁call▁begin｜>")
					sb.content(toolCall.Function.Name)
					sb.control("<｜tool▁sep｜>")

					argsJSON, _ := json.Marshal(toolCall.Function.Arguments)
					sb.content(string(argsJSON))
					sb.control("<｜tool▁call▁end｜>")
				}
				sb.control("<｜tool▁calls▁end｜><｜end▁of▁sentence｜>")
			} else {
				if isLastUser {
					sb.control("<｜Assistant｜>")
					hasThinking := message.Thinking != ""

					// only use <think> for the current turn (after last user message)
					isCurrentTurn := i > lastUserIndex
					if hasThinking && thinking && isCurrentTurn {
						sb.control("<think>")
					} else {
						sb.control("</think>")
					}
				}
				isLastUser = false

				content := message.Content
				if isTool {
					isTool = false
				} else if strings.HasPrefix(content, "<think>") {
					// Strip a replayed reasoning block from clients that resend
					// the model's raw output as content. Content that merely
					// mentions "</think>" without a leading "<think>" is
					// ordinary text and is preserved verbatim (#17248).
					if parts := strings.SplitN(content, "</think>", 2); len(parts) > 1 {
						content = parts[1]
					}
				}
				sb.content(content)
				sb.control("<｜end▁of▁sentence｜>")
			}

		case "tool":
			isLastUser = false
			isTool = true
			sb.control("<｜tool▁output▁begin｜>")
			sb.content(message.Content)
			sb.control("<｜tool▁output▁end｜>")
		}
	}

	if isLastUser && !isTool {
		sb.control("<｜Assistant｜>")
		if thinking {
			sb.control("<think>")
		} else {
			sb.control("</think>")
		}
	}

	return sb.Segments(), nil
}
