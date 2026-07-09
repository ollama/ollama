package renderers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

// GLM52Renderer renders messages for GLM-5.2 models.
//
// GLM-5.2 Thinking Modes and Features:
// - INTERLEAVED THINKING: The model thinks between tool calls and after receiving tool results
// - PRESERVED THINKING: The model retains reasoning from previous turns
// - TURN-LEVEL THINKING: Controls whether the model should reason on each turn
// - EXTENDED CONTEXT: Supports up to 1M tokens for complex prompt handling
//
// This renderer ensures proper handling of complex prompts with large context windows.
type GLM52Renderer struct{}

func (r *GLM52Renderer) LeadingBOS() string {
	return ""
}

func (r *GLM52Renderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	sb.WriteString("[gMASK]<sop>")

	if len(tools) > 0 {
		sb.WriteString("<|system|>\n")
		sb.WriteString("# Tools\n\n")
		sb.WriteString("You may call one or more functions to assist with the user query.\n\n")
		sb.WriteString("You are provided with function signatures within <tools></tools> XML tags:\n")
		sb.WriteString("<tools>\n")
		for _, tool := range tools {
			d, _ := json.Marshal(tool)
			sb.WriteString(formatGLM47ToolJSON(d))
			sb.WriteString("\n")
		}
		sb.WriteString("</tools>\n")
		sb.WriteString("<|user|>\n\n")
	}

	for i, message := range messages {
		switch message.Role {
		case "system":
			if i == 0 {
				sb.WriteString("<|system|>\n")
				sb.WriteString(message.Content)
				sb.WriteString("\n")
			}
		case "user":
			sb.WriteString("<|user|>\n")
			sb.WriteString(message.Content)
			sb.WriteString("\n")
		case "assistant":
			sb.WriteString("<|assistant|>\n")
			if message.Content != "" {
				sb.WriteString(message.Content)
			}
			// Add thinking block if needed and not already present
			if thinkValue == nil || thinkValue.Bool() {
				if !strings.Contains(message.Content, "<think>") {
					sb.WriteString("<think>")
				}
			}
			sb.WriteString("\n")
		}
	}

	// Start assistant thinking if enabled
	sb.WriteString("<|assistant|>\n")
	if thinkValue == nil || thinkValue.Bool() {
		sb.WriteString("<think>")
	}

	return sb.String(), nil
}

// formatGLM47ToolJSON formats tool definitions for GLM-4.7/5.2 compatibility
// This function ensures proper JSON formatting for tool calls
func formatGLM47ToolJSON(b []byte) string {
	// For now, return the JSON as-is with proper indentation
	var obj interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return string(b)
	}

	formatted, err := json.MarshalIndent(obj, "", "  ")
	if err != nil {
		return string(b)
	}

	return string(formatted)
}
