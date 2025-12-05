package renderers

import (
	"encoding/json"
	"strings"

	"github.com/ollama/ollama/api"
)

const (
	olmo3ThinkingDefaultSystemMessage = "You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai."
	olmo3ThinkingNoFunctionsMessage   = " You do not currently have access to any functions."
)

type Olmo3ThinkingRenderer struct{}

type olmo3ThinkingToolCall struct {
	ID       string                    `json:"id,omitempty"`
	Type     string                    `json:"type,omitempty"`
	Function olmo3ThinkingToolCallFunc `json:"function"`
}

type olmo3ThinkingToolCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func (r *Olmo3ThinkingRenderer) Render(messages []api.Message, tools []api.Tool, _ *api.ThinkValue) (string, error) {
	var sb strings.Builder

	var systemMessage *api.Message
	filteredMessages := make([]api.Message, 0, len(messages))
	for i, message := range messages {
		if message.Role == "system" {
			if systemMessage == nil {
				systemMessage = &messages[i]
			}
			continue
		}
		filteredMessages = append(filteredMessages, message)
	}

	systemContent := olmo3ThinkingDefaultSystemMessage
	if systemMessage != nil {
		systemContent = systemMessage.Content
	}

	sb.WriteString("<|im_start|>system\n")
	sb.WriteString(systemContent)

	if len(tools) > 0 {
		functionsJSON, err := marshalWithSpaces(tools)
		if err != nil {
			return "", err
		}
		sb.WriteString(" <functions>")
		sb.WriteString(string(functionsJSON))
		sb.WriteString("</functions>")
	} else {
		sb.WriteString(olmo3ThinkingNoFunctionsMessage)
		sb.WriteString(" <functions></functions>")
	}
	sb.WriteString("<|im_end|>\n")

	for i, message := range filteredMessages {
		lastMessage := i == len(filteredMessages)-1

		switch message.Role {
		case "user":
			sb.WriteString("<|im_start|>user\n")
			sb.WriteString(message.Content)
			sb.WriteString("<|im_end|>\n")

		case "assistant":
			sb.WriteString("<|im_start|>assistant\n")

			if message.Content != "" {
				sb.WriteString(message.Content)
			}

			if len(message.ToolCalls) > 0 {
				toolCalls := make([]olmo3ThinkingToolCall, len(message.ToolCalls))
				for j, tc := range message.ToolCalls {
					argsJSON, err := json.Marshal(tc.Function.Arguments)
					if err != nil {
						return "", err
					}
					toolCalls[j] = olmo3ThinkingToolCall{
						ID:   tc.ID,
						Type: "function",
						Function: olmo3ThinkingToolCallFunc{
							Name:      tc.Function.Name,
							Arguments: string(argsJSON),
						},
					}
				}
				toolCallsJSON, err := marshalWithSpaces(toolCalls)
				if err != nil {
					return "", err
				}
				sb.WriteString("<function_calls>")
				sb.WriteString(string(toolCallsJSON))
				sb.WriteString("</function_calls>")
			}

			if !lastMessage {
				sb.WriteString("<|im_end|>\n")
			}

		case "tool":
			sb.WriteString("<|im_start|>environment\n")
			sb.WriteString(message.Content)
			sb.WriteString("<|im_end|>\n")
		}
	}

	needsGenerationPrompt := true
	if len(filteredMessages) > 0 {
		lastMsg := filteredMessages[len(filteredMessages)-1]
		if lastMsg.Role == "assistant" && len(lastMsg.ToolCalls) == 0 && lastMsg.Content != "" {
			needsGenerationPrompt = false
		}
	}

	if needsGenerationPrompt {
		sb.WriteString("<|im_start|>assistant\n<think>")
	}

	return sb.String(), nil
}
