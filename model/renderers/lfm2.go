package renderers

import (
	"encoding/json"
	"strings"

	"github.com/ollama/ollama/api"
)

type LFM2Renderer struct {
	IsThinking bool
}

func (r *LFM2Renderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	// Note: BOS token is added by the tokenizer (add_bos_token: true), not the renderer

	// Extract first system message if present (to combine with tools)
	var firstSystemContent string
	startIdx := 0
	if len(messages) > 0 && messages[0].Role == "system" {
		firstSystemContent = messages[0].Content
		startIdx = 1
	}

	// Append tools to first system content
	if len(tools) > 0 {
		if firstSystemContent != "" {
			firstSystemContent += "\n"
		}
		firstSystemContent += "List of tools: ["
		for i, tool := range tools {
			toolJSON, err := json.Marshal(tool)
			if err != nil {
				return "", err
			}
			firstSystemContent += string(toolJSON)
			if i < len(tools)-1 {
				firstSystemContent += ", "
			}
		}
		firstSystemContent += "]"
	}

	// Output first system block if it has content
	if firstSystemContent != "" {
		sb.WriteString("<|im_start|>system\n")
		sb.WriteString(firstSystemContent)
		sb.WriteString("<|im_end|>\n")
	}

	// Find the index of the last assistant message for thinking stripping
	lastAssistantIndex := -1
	for i := len(messages) - 1; i >= startIdx; i-- {
		if messages[i].Role == "assistant" {
			lastAssistantIndex = i
			break
		}
	}

	// Track whether we need to add generation prompt
	needsGenerationPrompt := len(messages) > 0

	for i := startIdx; i < len(messages); i++ {
		message := messages[i]
		switch message.Role {
		case "system":
			// Additional system messages (after the first) are rendered normally
			sb.WriteString("<|im_start|>system\n")
			sb.WriteString(message.Content)
			sb.WriteString("<|im_end|>\n")

		case "user":
			sb.WriteString("<|im_start|>user\n")
			sb.WriteString(message.Content)
			sb.WriteString("<|im_end|>\n")
			needsGenerationPrompt = true

		case "assistant":
			sb.WriteString("<|im_start|>assistant\n")

			// Check if this is the last assistant message
			isLastAssistant := i == lastAssistantIndex

			// Process content (may need thinking stripped)
			content := message.Content

			// Handle thinking tags in assistant content
			keepPastThinking := r.IsThinking && (thinkValue != nil && thinkValue.Bool())
			if strings.Contains(content, "</think>") {
				parts := strings.SplitN(content, "</think>", 2)
				if len(parts) > 1 {
					if !isLastAssistant && !keepPastThinking {
						// Strip thinking entirely for past assistant messages
						content = strings.TrimSpace(parts[1])
					} else {
						// Preserve thinking but trim whitespace after </think>
						content = parts[0] + "</think>" + strings.TrimLeft(parts[1], " \t\n\r")
					}
				}
			}

			if len(message.ToolCalls) > 0 {
				// Assistant with tool calls - write content first (if any after stripping)
				if content != "" {
					sb.WriteString(content)
				}

				for _, toolCall := range message.ToolCalls {
					sb.WriteString("<|tool_call_start|>")
					toolCallJSON := map[string]any{
						"name":      toolCall.Function.Name,
						"arguments": toolCall.Function.Arguments,
					}
					callJSON, _ := json.Marshal(toolCallJSON)
					sb.WriteString(string(callJSON))
					sb.WriteString("<|tool_call_end|>")
				}
			} else {
				sb.WriteString(content)
			}

			sb.WriteString("<|im_end|>\n")
			needsGenerationPrompt = true // Always add gen prompt after assistant when add_generation_prompt=true

		case "tool":
			// Tool responses are rendered as plain messages per the chat template
			sb.WriteString("<|im_start|>tool\n")
			sb.WriteString(message.Content)
			sb.WriteString("<|im_end|>\n")
			needsGenerationPrompt = true
		}
	}

	// Add generation prompt
	if needsGenerationPrompt {
		sb.WriteString("<|im_start|>assistant\n")
		// Note: Model is a "thinking-only" model - it will output <think> itself
		// We don't add <think> tag to the prompt
	}

	return sb.String(), nil
}
