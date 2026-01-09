package renderers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

type Nemotron3NanoRenderer struct{}

func (r *Nemotron3NanoRenderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	// thinking is enabled if user requests it
	enableThinking := thinkValue != nil && thinkValue.Bool()

	// Extract system message if present
	var systemMessage string
	var loopMessages []api.Message
	if len(messages) > 0 && messages[0].Role == "system" {
		systemMessage = messages[0].Content
		loopMessages = messages[1:]
	} else {
		loopMessages = messages
	}

	// Find last user message index for thinking truncation
	lastUserIdx := -1
	for i, msg := range loopMessages {
		if msg.Role == "user" {
			lastUserIdx = i
		}
	}

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
			// Build content with thinking tags
			content := r.buildContent(message)
			shouldTruncate := i < lastUserIdx

			if len(message.ToolCalls) > 0 {
				sb.WriteString("<|im_start|>assistant\n")
				sb.WriteString(r.formatContent(content, shouldTruncate, true))
				r.writeToolCalls(&sb, message.ToolCalls)
				sb.WriteString("<|im_end|>\n")
			} else {
				formatted := r.formatContent(content, shouldTruncate, false)
				sb.WriteString("<|im_start|>assistant\n" + formatted + "<|im_end|>\n")
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
			for paramName, paramFields := range fn.Parameters.Properties.All() {
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

func (r *Nemotron3NanoRenderer) buildContent(message api.Message) string {
	// The parser always extracts thinking into the Thinking field,
	// so Content will never have <think> tags embedded
	if message.Thinking != "" {
		return "<think>\n" + message.Thinking + "\n</think>\n" + message.Content
	}
	return "<think></think>" + message.Content
}

func (r *Nemotron3NanoRenderer) formatContent(content string, truncate bool, addNewline bool) string {
	if content == "" {
		return "<think></think>"
	}

	if !truncate {
		if addNewline {
			return strings.TrimSpace(content) + "\n"
		}
		return strings.TrimSpace(content)
	}

	// Truncate thinking - keep only content after </think>
	c := content
	if strings.Contains(c, "</think>") {
		parts := strings.Split(c, "</think>")
		c = parts[len(parts)-1]
	} else if strings.Contains(c, "<think>") {
		parts := strings.Split(c, "<think>")
		c = parts[0]
	}
	c = "<think></think>" + strings.TrimSpace(c)

	if addNewline && len(c) > len("<think></think>") {
		return c + "\n"
	}
	if c == "<think></think>" {
		return c
	}
	return strings.TrimSpace(c)
}

func (r *Nemotron3NanoRenderer) writeToolCalls(sb *strings.Builder, toolCalls []api.ToolCall) {
	for _, tc := range toolCalls {
		sb.WriteString("<tool_call>\n<function=" + tc.Function.Name + ">\n")
		for name, value := range tc.Function.Arguments.All() {
			sb.WriteString("<parameter=" + name + ">\n" + r.formatArgValue(value) + "\n</parameter>\n")
		}
		sb.WriteString("</function>\n</tool_call>\n")
	}
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
