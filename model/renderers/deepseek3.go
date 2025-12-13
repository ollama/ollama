package renderers

import (
	"encoding/json"
	"strings"

	"github.com/ollama/ollama/api"
)

type DeepSeekRenderer struct {
	isThinking bool
}

func (r *DeepSeekRenderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	// thinking is enabled: model must support it AND user must request it
	thinking := r.isThinking && (thinkValue != nil && thinkValue.Bool())

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

	sb.WriteString("<｜begin▁of▁sentence｜>" + systemPrompt.String())

	// state tracking
	isFirst := false
	isTool := false
	isLastUser := false

	for _, message := range messages {
		switch message.Role {
		case "user":
			isTool = false
			isFirst = false
			isLastUser = true
			sb.WriteString("<｜User｜>" + message.Content)

		case "assistant":
			if len(message.ToolCalls) > 0 {
				if isLastUser {
					sb.WriteString("<｜Assistant｜></think>")
				}
				isLastUser = false
				isFirst = false
				isTool = false

				if message.Content != "" {
					sb.WriteString(message.Content)
				}

				sb.WriteString("<｜tool▁calls▁begin｜>")
				for _, toolCall := range message.ToolCalls {
					if !isFirst {
						if message.Content == "" {
							sb.WriteString("<｜tool▁call▁begin｜>" + toolCall.Function.Name + "<｜tool▁sep｜>")
						} else {
							sb.WriteString("<｜tool▁call▁begin｜>" + toolCall.Function.Name + "<｜tool▁sep｜>")
						}
						isFirst = true
					} else {
						sb.WriteString("<｜tool▁call▁begin｜>" + toolCall.Function.Name + "<｜tool▁sep｜>")
					}

					argsJSON, _ := json.Marshal(toolCall.Function.Arguments)
					sb.WriteString(string(argsJSON))
					sb.WriteString("<｜tool▁call▁end｜>")
				}
				sb.WriteString("<｜tool▁calls▁end｜><｜end▁of▁sentence｜>")
			} else {
				if isLastUser {
					sb.WriteString("<｜Assistant｜>")
					// message["prefix"] is defined and message["prefix"] and thinking
					// message.Thinking != "" represents message["prefix"] being defined
					if message.Thinking != "" && thinking {
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
