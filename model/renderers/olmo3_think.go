package renderers

import (
	"strings"

	"github.com/ollama/ollama/api"
)

type Olmo3ThinkVariant int

const (
	// Olmo3Think32B is for allenai/Olmo-3-32B-Think
	Olmo3Think32B Olmo3ThinkVariant = iota
	// Olmo31Think is for allenai/Olmo-3-7B-Think and allenai/Olmo-3.1-32B-Think (includes model info)
	Olmo31Think
)

const (
	olmo3ThinkFunctionsSuffix  = " You do not currently have access to any functions. <functions></functions>"
	olmo3Think32BSystemMessage = "You are a helpful AI assistant."
	olmo31ThinkSystemMessage   = "You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai."
)

type Olmo3ThinkRenderer struct {
	Variant Olmo3ThinkVariant
}

func (r *Olmo3ThinkRenderer) Render(messages []api.Message, _ []api.Tool, _ *api.ThinkValue) (string, error) {
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
		// Skip tool messages - Think models don't support tools
		if message.Role == "tool" {
			continue
		}
		filteredMessages = append(filteredMessages, message)
	}

	sb.WriteString("<|im_start|>system\n")

	if systemMessage != nil {
		sb.WriteString(systemMessage.Content)
		sb.WriteString(olmo3ThinkFunctionsSuffix)
	} else {
		// Default system message varies by variant
		switch r.Variant {
		case Olmo3Think32B:
			sb.WriteString(olmo3Think32BSystemMessage)
		default: // Olmo3Think7B, Olmo31Think use same template - diverges from HF but confirmed difference from team
			sb.WriteString(olmo31ThinkSystemMessage)
		}
	}

	sb.WriteString("<|im_end|>\n")

	for _, message := range filteredMessages {
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
			sb.WriteString("<|im_end|>\n")
		}
	}

	// Always add generation prompt with <think> tag for thinking models
	sb.WriteString("<|im_start|>assistant\n<think>")

	return sb.String(), nil
}
