package renderers

import (
	"encoding/json"
	"slices"
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

func (r *DeepSeek3Renderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	// thinking is enabled: model must support it AND user must request it
	thinking := r.IsThinking && (thinkValue != nil && thinkValue.Bool())

	// extract system messages first
	var systemPrompt strings.Builder
	isFirstSystemPrompt := true

	for _, message := range messages {
		if message.Role == "system" {
			if isFirstSystemPrompt {
				systemPrompt.WriteString(message.Content)
				isFirstSystemPrompt = false
			} else {
				systemPrompt.WriteString("\n\n" + message.Content)
			}
		}
	}

	sb.WriteString("<｜begin▁of▁sentence｜>")
	sb.WriteString(systemPrompt.String())

	// tool definitions
	if len(tools) > 0 {
		sb.WriteString("\n\n## Tools\nYou have access to the following tools:\n")

		for _, tool := range tools {
			sb.WriteString("\n### " + tool.Function.Name)
			sb.WriteString("\nDescription: " + tool.Function.Description)

			// parameters as JSON
			parametersJSON, err := json.Marshal(tool.Function.Parameters)
			if err == nil {
				sb.WriteString("\n\nParameters: " + string(parametersJSON) + "\n")
			}
		}

		// usage instructions
		sb.WriteString("\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n")
		sb.WriteString("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>\n\n")
		sb.WriteString("Where:\n\n")
		sb.WriteString("- `tool_call_name` must be an exact match to one of the available tools\n")
		sb.WriteString("- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema\n")
		sb.WriteString("- For multiple tool calls, chain them directly without separators or spaces\n")
	}

	// state tracking
	isTool := false
	isLastUser := false

	// Find the index of the last user message to determine which assistant message is "current"
	lastUserIndex := -1
	for i, message := range slices.Backward(messages) {
		if message.Role == "user" {
			lastUserIndex = i
			break
		}
	}

	for i, message := range messages {
		switch message.Role {
		case "user":
			isTool = false
			isLastUser = true
			sb.WriteString("<｜User｜>" + message.Content)

		case "assistant":
			if len(message.ToolCalls) > 0 {
				if isLastUser {
					sb.WriteString("<｜Assistant｜></think>")
				}
				isLastUser = false
				isTool = false

				if message.Content != "" {
					sb.WriteString(message.Content)
				}

				sb.WriteString("<｜tool▁calls▁begin｜>")
				for _, toolCall := range message.ToolCalls {
					sb.WriteString("<｜tool▁call▁begin｜>" + toolCall.Function.Name + "<｜tool▁sep｜>")

					argsJSON, _ := json.Marshal(toolCall.Function.Arguments)
					sb.WriteString(string(argsJSON))
					sb.WriteString("<｜tool▁call▁end｜>")
				}
				sb.WriteString("<｜tool▁calls▁end｜><｜end▁of▁sentence｜>")
			} else {
				if isLastUser {
					sb.WriteString("<｜Assistant｜>")
					hasThinking := message.Thinking != ""

					// only use <think> for the current turn (after last user message)
					isCurrentTurn := i > lastUserIndex
					if hasThinking && thinking && isCurrentTurn {
						sb.WriteString("<think>")
					} else {
						sb.WriteString("</think>")
					}
				}
				isLastUser = false

				content := message.Content
				if isTool {
					sb.WriteString(content + "<｜end▁of▁sentence｜>")
					isTool = false
				} else {
					if strings.Contains(content, "</think>") {
						parts := strings.SplitN(content, "</think>", 2)
						if len(parts) > 1 {
							content = parts[1]
						}
					}
					sb.WriteString(content + "<｜end▁of▁sentence｜>")
				}
			}

		case "tool":
			isLastUser = false
			isTool = true
			sb.WriteString("<｜tool▁output▁begin｜>" + message.Content + "<｜tool▁output▁end｜>")
		}
	}

	if isLastUser && !isTool {
		sb.WriteString("<｜Assistant｜>")
		if thinking {
			sb.WriteString("<think>")
		} else {
			sb.WriteString("</think>")
		}
	}

	return sb.String(), nil
}
