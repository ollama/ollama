package renderers

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/ollama/ollama/api"
)

const (
	olmo3DefaultSystemMessage  = "You are a helpful function-calling AI assistant. "
	olmo31DefaultSystemMessage = "You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai. "
	olmo3NoFunctionsMessage    = "You do not currently have access to any functions. "
	olmo3WithFunctionsMessage  = "You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Output any function calls within <function_calls></function_calls> XML tags. Do not make assumptions about what values to plug into functions."
)

type Olmo3Renderer struct {
	UseExtendedSystemMessage bool
}

func (r *Olmo3Renderer) Render(messages []api.Message, tools []api.Tool, _ *api.ThinkValue) (string, error) {
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

	// Render system message
	if systemMessage != nil {
		// Custom system message - single newline after "system"
		sb.WriteString("<|im_start|>system\n")
		sb.WriteString(systemMessage.Content)

		if len(tools) > 0 {
			functionsJSON, err := marshalWithSpaces(tools)
			if err != nil {
				return "", err
			}
			sb.WriteString("<functions>")
			sb.WriteString(string(functionsJSON))
			sb.WriteString("</functions>")
		}
		sb.WriteString("<|im_end|>\n")
	} else {
		// Default system message - single newline after "system"
		sb.WriteString("<|im_start|>system\n")
		if r.UseExtendedSystemMessage {
			sb.WriteString(olmo31DefaultSystemMessage)
		} else {
			sb.WriteString(olmo3DefaultSystemMessage)
		}

		if len(tools) > 0 {
			functionsJSON, err := marshalWithSpaces(tools)
			if err != nil {
				return "", err
			}
			sb.WriteString(olmo3WithFunctionsMessage)
			sb.WriteString("<functions>")
			sb.WriteString(string(functionsJSON))
			sb.WriteString("</functions>")
		} else {
			sb.WriteString(olmo3NoFunctionsMessage)
			sb.WriteString("<functions></functions>")
		}
		sb.WriteString("<|im_end|>\n")
	}

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
				sb.WriteString("<function_calls>")
				for j, tc := range message.ToolCalls {
					// Format as function_name(arg1="value1", arg2="value2")
					sb.WriteString(tc.Function.Name)
					sb.WriteString("(")

					// Get sorted keys for deterministic output
					keys := make([]string, 0, tc.Function.Arguments.Len())
					for k := range tc.Function.Arguments.All() {
						keys = append(keys, k)
					}
					sort.Strings(keys)

					for k, key := range keys {
						if k > 0 {
							sb.WriteString(", ")
						}
						val, _ := tc.Function.Arguments.Get(key)
						value, err := json.Marshal(val)
						if err != nil {
							return "", err
						}
						sb.WriteString(fmt.Sprintf("%s=%s", key, string(value)))
					}
					sb.WriteString(")")

					if j < len(message.ToolCalls)-1 {
						sb.WriteString("\n")
					}
				}
				sb.WriteString("</function_calls>")
			}

			// Add end tag unless it's the last message with content only (prefill)
			if !lastMessage || len(message.ToolCalls) > 0 {
				sb.WriteString("<|im_end|>\n")
			}

		case "tool":
			sb.WriteString("<|im_start|>environment\n")
			sb.WriteString(message.Content)
			sb.WriteString("<|im_end|>\n")
		}
	}

	// Add generation prompt if needed
	needsGenerationPrompt := true
	if len(filteredMessages) > 0 {
		lastMsg := filteredMessages[len(filteredMessages)-1]
		if lastMsg.Role == "assistant" && len(lastMsg.ToolCalls) == 0 && lastMsg.Content != "" {
			needsGenerationPrompt = false
		}
	}

	if needsGenerationPrompt {
		sb.WriteString("<|im_start|>assistant\n")
	}

	return sb.String(), nil
}
