package renderers

import (
	"encoding/json"
	"strings"

	"github.com/ollama/ollama/api"
)

type CogitoRenderer struct {
	isThinking bool
}

func (r *CogitoRenderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	defaultPrompt := "You are Cogito, an AI assistant created by Deep Cogito, which is an AI research lab based in San Francisco."

	// Determine if thinking is enabled from thinkValue or renderer config
	enableThinking := r.isThinking || (thinkValue != nil && thinkValue.Bool())

	var systemPrompt string
	var conversationMessages []api.Message

	if len(messages) > 0 && messages[0].Role == "system" {
		systemPrompt = messages[0].Content // set ns.system_prompt = raw if raw is string else raw[0].text
		conversationMessages = messages[1:]
	} else {
		conversationMessages = messages
	}

	var finalSystemPrompt string
	if enableThinking {
		finalSystemPrompt = "Enable deep thinking subroutine.\n\n" + defaultPrompt
		if systemPrompt != "" {
			finalSystemPrompt = finalSystemPrompt + "\n\n" + systemPrompt + "\n\n"
		}
	} else {
		finalSystemPrompt = defaultPrompt
		if systemPrompt != "" {
			finalSystemPrompt = finalSystemPrompt + "\n\n" + systemPrompt
		}
	}

	if len(tools) > 0 {
		if finalSystemPrompt != "" {
			finalSystemPrompt = finalSystemPrompt + "\nYou have the following functions available:\n"
			// {%- set ns.system_prompt = ns.system_prompt ~ '
			// You have the following functions available:
			// ' -%}
		} else {
			finalSystemPrompt = "You have the following functions available:\n"
		}

		for _, tool := range tools {
			toolJSON, _ := json.MarshalIndent(tool, "", "    ") // TODO(gguo): double check json format
			finalSystemPrompt = finalSystemPrompt + "```json\n" + string(toolJSON) + "\n```\n"
		}
	}

	sb.WriteString("<｜begin▁of▁sentence｜>" + finalSystemPrompt)

	outputsOpen := false
	isLastUser := false

	for i, message := range conversationMessages {
		switch message.Role {
		case "user":
			isLastUser = true
			sb.WriteString("<｜User｜>" + message.Content + "<｜Assistant｜>") // TODO: this is more complicated in the template

		case "assistant":
			isLastUser = false

			if len(message.ToolCalls) > 0 {
				if message.Content != "" {
					sb.WriteString(message.Content)
				}

				sb.WriteString("<｜tool▁calls▁begin｜>")

				for j, toolCall := range message.ToolCalls {
					sb.WriteString("<｜tool▁call▁begin｜>function<｜tool▁sep｜>" + toolCall.Function.Name)

					argsJSON, _ := json.Marshal(toolCall.Function.Arguments)
					sb.WriteString("\n```json\n" + string(argsJSON) + "\n```")
					sb.WriteString("<｜tool▁call▁end｜>")

					if j < len(message.ToolCalls)-1 {
						sb.WriteString("\n")
					}
				}

				sb.WriteString("<｜tool▁calls▁end｜><｜end▁of▁sentence｜>")
			} else {
				sb.WriteString(message.Content + "<｜end▁of▁sentence｜>")
			}

		case "tool":
			isLastUser = false

			if !outputsOpen {
				sb.WriteString("<｜tool▁outputs▁begin｜>")
				outputsOpen = true
			}

			sb.WriteString("<｜tool▁output▁begin｜>" + message.Content + "<｜tool▁output▁end｜>")

			hasNextTool := i+1 < len(conversationMessages) && conversationMessages[i+1].Role == "tool"
			if hasNextTool {
				sb.WriteString("\n")
			} else {
				sb.WriteString("<｜tool▁outputs▁end｜>")
				outputsOpen = false
			}
		}
	}

	if outputsOpen {
		sb.WriteString("<｜tool▁outputs▁end｜>")
	}

	if !isLastUser {
		sb.WriteString("<｜Assistant｜>")
	}

	if enableThinking {
		sb.WriteString("<think>\n")
	}

	return sb.String(), nil
}
