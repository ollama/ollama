package renderers

import (
	"fmt"
	"sort"
	"strings"

	"github.com/ollama/ollama/api"
)

type FunctionGemmaRenderer struct{}

const defaultSystemMessage = "You can do function calling with the following functions:"

func (r *FunctionGemmaRenderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	sb.WriteString("<bos>")

	var systemMessage string
	var loopMessages []api.Message
	if len(messages) > 0 && (messages[0].Role == "system" || messages[0].Role == "developer") {
		systemMessage = messages[0].Content
		loopMessages = messages[1:]
	} else {
		loopMessages = messages
	}

	if systemMessage != "" || len(tools) > 0 {
		sb.WriteString("<start_of_turn>developer\n")
		if systemMessage != "" {
			sb.WriteString(strings.TrimSpace(systemMessage))
		}
		if len(tools) > 0 {
			if systemMessage != "" {
				sb.WriteString("\n")
			}
			if strings.TrimSpace(systemMessage) != defaultSystemMessage {
				// Only add default message if user does not provide it
				sb.WriteString(defaultSystemMessage)
			}
		}
		for _, tool := range tools {
			sb.WriteString(r.renderToolDeclaration(tool))
		}
		sb.WriteString("<end_of_turn>\n")
	}

	// Track previous message type for tool response handling
	prevMessageType := ""

	for i, message := range loopMessages {
		switch message.Role {
		case "assistant":
			if prevMessageType != "tool_response" {
				sb.WriteString("<start_of_turn>model\n")
			}
			prevMessageType = ""

			if message.Content != "" {
				sb.WriteString(strings.TrimSpace(message.Content))
			}

			if len(message.ToolCalls) > 0 {
				for _, tc := range message.ToolCalls {
					sb.WriteString(r.formatToolCall(tc))
				}
				// After tool calls, expect tool responses
				if i+1 < len(loopMessages) && loopMessages[i+1].Role == "tool" {
					sb.WriteString("<start_function_response>")
					prevMessageType = "tool_call"
				} else {
					sb.WriteString("<end_of_turn>\n")
				}
			} else {
				sb.WriteString("<end_of_turn>\n")
			}

		case "user":
			if prevMessageType != "tool_response" {
				sb.WriteString("<start_of_turn>user\n")
			}
			prevMessageType = ""
			sb.WriteString(strings.TrimSpace(message.Content))
			sb.WriteString("<end_of_turn>\n")

		case "tool":
			toolName := ""
			// Find the tool name from the previous assistant's tool call
			for j := i - 1; j >= 0; j-- {
				if loopMessages[j].Role == "assistant" && len(loopMessages[j].ToolCalls) > 0 {
					// Count how many tool messages came before this one
					toolIdx := 0
					for k := j + 1; k < i; k++ {
						if loopMessages[k].Role == "tool" {
							toolIdx++
						}
					}
					if toolIdx < len(loopMessages[j].ToolCalls) {
						toolName = loopMessages[j].ToolCalls[toolIdx].Function.Name
					}
					break
				}
			}

			if prevMessageType != "tool_call" {
				sb.WriteString("<start_function_response>")
			}
			sb.WriteString("response:" + toolName + "{" + r.formatArgValue(message.Content) + "}<end_function_response>")
			prevMessageType = "tool_response"

		default:
			sb.WriteString("<start_of_turn>" + message.Role + "\n")
			sb.WriteString(strings.TrimSpace(message.Content))
			sb.WriteString("<end_of_turn>\n")
		}
	}

	if prevMessageType != "tool_response" {
		sb.WriteString("<start_of_turn>model\n")
	}

	return sb.String(), nil
}

func (r *FunctionGemmaRenderer) renderToolDeclaration(tool api.Tool) string {
	var sb strings.Builder

	fn := tool.Function
	sb.WriteString("<start_function_declaration>declaration:" + fn.Name + "{")
	sb.WriteString("description:<escape>" + fn.Description + "<escape>")

	if fn.Parameters.Properties != nil || fn.Parameters.Type != "" {
		sb.WriteString(",parameters:{")

		needsComma := false

		// Only include properties:{} if there are actual properties
		if fn.Parameters.Properties != nil && fn.Parameters.Properties.Len() > 0 {
			sb.WriteString("properties:{")
			r.writeProperties(&sb, fn.Parameters.Properties)
			sb.WriteString("}")
			needsComma = true
		}

		if len(fn.Parameters.Required) > 0 {
			if needsComma {
				sb.WriteString(",")
			}
			sb.WriteString("required:[")
			for i, req := range fn.Parameters.Required {
				if i > 0 {
					sb.WriteString(",")
				}
				sb.WriteString("<escape>" + req + "<escape>")
			}
			sb.WriteString("]")
			needsComma = true
		}

		if fn.Parameters.Type != "" {
			if needsComma {
				sb.WriteString(",")
			}
			sb.WriteString("type:<escape>" + strings.ToUpper(fn.Parameters.Type) + "<escape>")
		}

		sb.WriteString("}")
	}

	sb.WriteString("}<end_function_declaration>")
	return sb.String()
}

func (r *FunctionGemmaRenderer) writeProperties(sb *strings.Builder, props *api.ToolPropertiesMap) {
	keys := make([]string, 0, props.Len())
	for k := range props.All() {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	first := true
	for _, name := range keys {
		prop, _ := props.Get(name)
		if !first {
			sb.WriteString(",")
		}
		first = false

		sb.WriteString(name + ":{description:<escape>")
		sb.WriteString(prop.Description)
		sb.WriteString("<escape>")

		if len(prop.Type) > 0 {
			sb.WriteString(",type:<escape>" + strings.ToUpper(prop.Type[0]) + "<escape>")
		}

		sb.WriteString("}")
	}
}

func (r *FunctionGemmaRenderer) formatToolCall(tc api.ToolCall) string {
	var sb strings.Builder
	sb.WriteString("<start_function_call>call:" + tc.Function.Name + "{")

	keys := make([]string, 0, tc.Function.Arguments.Len())
	for k := range tc.Function.Arguments.All() {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	first := true
	for _, key := range keys {
		value, _ := tc.Function.Arguments.Get(key)
		if !first {
			sb.WriteString(",")
		}
		first = false
		sb.WriteString(key + ":" + r.formatArgValue(value))
	}

	sb.WriteString("}<end_function_call>")
	return sb.String()
}

func (r *FunctionGemmaRenderer) formatArgValue(value any) string {
	switch v := value.(type) {
	case string:
		return "<escape>" + v + "<escape>"
	case bool:
		if v {
			return "true"
		}
		return "false"
	case float64:
		if v == float64(int64(v)) {
			return fmt.Sprintf("%d", int64(v))
		}
		return fmt.Sprintf("%v", v)
	case int, int64, int32:
		return fmt.Sprintf("%d", v)
	case map[string]any:
		return r.formatMapValue(v)
	case []any:
		return r.formatArrayValue(v)
	default:
		return fmt.Sprintf("%v", v)
	}
}

func (r *FunctionGemmaRenderer) formatMapValue(m map[string]any) string {
	var sb strings.Builder
	sb.WriteString("{")

	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	first := true
	for _, key := range keys {
		if !first {
			sb.WriteString(",")
		}
		first = false
		sb.WriteString(key + ":" + r.formatArgValue(m[key]))
	}

	sb.WriteString("}")
	return sb.String()
}

func (r *FunctionGemmaRenderer) formatArrayValue(arr []any) string {
	var sb strings.Builder
	sb.WriteString("[")

	for i, item := range arr {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(r.formatArgValue(item))
	}

	sb.WriteString("]")
	return sb.String()
}
