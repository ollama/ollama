package renderers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

type Nemotron3NanoRenderer struct {
	IsThinking bool
}

func (r *Nemotron3NanoRenderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	// thinking is enabled: model must support it AND user must request it
	enableThinking := r.IsThinking && (thinkValue != nil && thinkValue.Bool())

	// truncate_history_thinking: drop thinking from historical assistant messages
	truncateHistoryThinking := true

	// Find the last user message index
	lastUserIdx := -1
	for i, msg := range messages {
		if msg.Role == "user" {
			lastUserIdx = i
		}
	}

	// Extract system message if present
	var systemMessage string
	var loopMessages []api.Message
	if len(messages) > 0 && messages[0].Role == "system" {
		systemMessage = messages[0].Content
		loopMessages = messages[1:]
	} else {
		loopMessages = messages
	}

	// Recalculate lastUserIdx for loopMessages
	lastUserIdxInLoop := -1
	for i, msg := range loopMessages {
		if msg.Role == "user" {
			lastUserIdxInLoop = i
		}
	}
	_ = lastUserIdx // silence unused variable warning

	// Write system message - always include the system block
	sb.WriteString("<|im_start|>system\n")
	if systemMessage != "" {
		sb.WriteString(systemMessage)
	}

	if len(tools) > 0 {
		if systemMessage != "" {
			sb.WriteString("\n\n")
		}
		sb.WriteString(r.renderTools(tools))
	}
	sb.WriteString("<|im_end|>\n")

	for i, message := range loopMessages {
		switch message.Role {
		case "assistant":
			// Build content with reasoning handling
			var content string
			if message.Thinking != "" {
				content = "<think>\n" + message.Thinking + "\n</think>\n" + message.Content
			} else {
				content = message.Content
				// Allow downstream logic to handle broken thought, only handle coherent reasoning here
				if !strings.Contains(content, "<think>") && !strings.Contains(content, "</think>") {
					content = "<think></think>" + content
				}
			}

			if len(message.ToolCalls) > 0 {
				// Assistant message with tool calls
				sb.WriteString("<|im_start|>assistant\n")

				includeContent := !(truncateHistoryThinking && i < lastUserIdxInLoop)
				if content != "" {
					if includeContent {
						sb.WriteString(strings.TrimSpace(content) + "\n")
					} else {
						// Truncate thinking
						c := content
						if strings.Contains(c, "</think>") {
							// Keep only content after the last closing think
							parts := strings.Split(c, "</think>")
							c = parts[len(parts)-1]
						} else if strings.Contains(c, "<think>") {
							// If <think> was opened but never closed, drop the trailing think segment
							parts := strings.Split(c, "<think>")
							c = parts[0]
						}
						c = "<think></think>" + strings.TrimSpace(c)
						if len(c) > len("<think></think>") {
							sb.WriteString(c + "\n")
						} else {
							sb.WriteString("<think></think>")
						}
					}
				} else {
					sb.WriteString("<think></think>")
				}

				// Write tool calls
				for _, toolCall := range message.ToolCalls {
					sb.WriteString("<tool_call>\n<function=" + toolCall.Function.Name + ">\n")
					for argName, argValue := range toolCall.Function.Arguments {
						sb.WriteString("<parameter=" + argName + ">\n")
						valueStr := r.formatArgValue(argValue)
						sb.WriteString(valueStr + "\n</parameter>\n")
					}
					sb.WriteString("</function>\n</tool_call>\n")
				}
				sb.WriteString("<|im_end|>\n")
			} else {
				// Assistant message without tool calls
				if !(truncateHistoryThinking && i < lastUserIdxInLoop) {
					sb.WriteString("<|im_start|>assistant\n" + strings.TrimSpace(content) + "<|im_end|>\n")
				} else {
					// Truncate thinking - keep only content after </think>
					c := content
					if strings.Contains(c, "<think>") && strings.Contains(c, "</think>") {
						parts := strings.Split(c, "</think>")
						// Trim the content after </think> before concatenating
						c = "<think></think>" + strings.TrimSpace(parts[len(parts)-1])
					}
					c = strings.TrimSpace(c)
					if c != "" {
						sb.WriteString("<|im_start|>assistant\n" + c + "<|im_end|>\n")
					} else {
						sb.WriteString("<|im_start|>assistant\n<|im_end|>\n")
					}
				}
			}

		case "user", "system":
			sb.WriteString("<|im_start|>" + message.Role + "\n")
			sb.WriteString(message.Content)
			sb.WriteString("<|im_end|>\n")

		case "tool":
			// Check if previous message was also a tool message
			prevWasTool := i > 0 && loopMessages[i-1].Role == "tool"
			nextIsTool := i+1 < len(loopMessages) && loopMessages[i+1].Role == "tool"

			if !prevWasTool {
				sb.WriteString("<|im_start|>user\n")
			}
			sb.WriteString("<tool_response>\n")
			sb.WriteString(message.Content)
			sb.WriteString("\n</tool_response>\n")

			if !nextIsTool {
				sb.WriteString("<|im_end|>\n")
			}

		default:
			sb.WriteString("<|im_start|>" + message.Role + "\n" + message.Content + "<|im_end|>\n")
		}
	}

	// Add generation prompt
	if enableThinking {
		sb.WriteString("<|im_start|>assistant\n<think>\n")
	} else {
		sb.WriteString("<|im_start|>assistant\n<think></think>")
	}

	return sb.String(), nil
}

func (r *Nemotron3NanoRenderer) renderTools(tools []api.Tool) string {
	var sb strings.Builder
	sb.WriteString("# Tools\n\nYou have access to the following functions:\n\n<tools>")

	for _, tool := range tools {
		fn := tool.Function
		sb.WriteString("\n<function>\n<name>" + fn.Name + "</name>")

		if fn.Description != "" {
			sb.WriteString("\n<description>" + strings.TrimSpace(fn.Description) + "</description>")
		}

		sb.WriteString("\n<parameters>")
		if fn.Parameters.Properties != nil {
			for paramName, paramFields := range fn.Parameters.Properties {
				sb.WriteString("\n<parameter>")
				sb.WriteString("\n<name>" + paramName + "</name>")

				if len(paramFields.Type) > 0 {
					sb.WriteString("\n<type>" + strings.Join(paramFields.Type, ", ") + "</type>")
				}

				if paramFields.Description != "" {
					sb.WriteString("\n<description>" + strings.TrimSpace(paramFields.Description) + "</description>")
				}

				if len(paramFields.Enum) > 0 {
					enumJSON, _ := json.Marshal(paramFields.Enum)
					sb.WriteString("\n<enum>" + string(enumJSON) + "</enum>")
				}

				sb.WriteString("\n</parameter>")
			}
		}

		if len(fn.Parameters.Required) > 0 {
			reqJSON, _ := json.Marshal(fn.Parameters.Required)
			sb.WriteString("\n<required>" + string(reqJSON) + "</required>")
		}

		sb.WriteString("\n</parameters>")
		sb.WriteString("\n</function>")
	}

	sb.WriteString("\n</tools>")

	sb.WriteString("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
		"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
		"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" +
		"</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
		"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
		"- Required parameters MUST be specified\n" +
		"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
		"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")

	return sb.String()
}

func (r *Nemotron3NanoRenderer) formatArgValue(value any) string {
	switch v := value.(type) {
	case map[string]any, []any:
		jsonBytes, _ := json.Marshal(v)
		return string(jsonBytes)
	default:
		return fmt.Sprintf("%v", v)
	}
}
