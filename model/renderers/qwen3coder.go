package renderers

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/ollama/ollama/api"
)

var (
	imStartTag = "<|im_start|>"
	imEndTag   = "<|im_end|>"
)

// renderAdditionalKeys renders all JSON fields except the ones in handledKeys
// This follows the same approach from the reference implementation, which gives
// a particular key ordering
func renderAdditionalKeys(obj any, handledKeys map[string]bool) string {
	data, err := json.Marshal(obj)
	if err != nil {
		return ""
	}

	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return ""
	}

	var sb strings.Builder
	for key, value := range m {
		if handledKeys[key] {
			continue
		}

		// Check if value is a map or array (needs JSON serialization)
		switch v := value.(type) {
		case map[string]any, []any:
			jsonBytes, _ := json.Marshal(v)
			// TODO(drifkin): it would be nice to format the JSON here similarly to
			// python's default json.dumps behavior (spaces after commas and colons).
			// This would let us be byte-for-byte compatible with the reference
			// implementation for most common inputs
			jsonStr := string(jsonBytes)
			sb.WriteString("\n<" + key + ">" + jsonStr + "</" + key + ">")
		case nil:
			continue
		default:
			// Simple types, convert to string
			sb.WriteString("\n<" + key + ">" + fmt.Sprintf("%v", value) + "</" + key + ">")
		}
	}

	return sb.String()
}

type Qwen3CoderRenderer struct{}

func (r *Qwen3CoderRenderer) Render(messages []api.Message, tools []api.Tool, _ *api.ThinkValue) (string, error) {
	var sb strings.Builder

	// filter out system messages and choose the first (if any) to win
	var systemMessage string
	var filteredMessages []api.Message
	for _, message := range messages {
		if message.Role != "system" {
			filteredMessages = append(filteredMessages, message)
			continue
		}

		if systemMessage == "" {
			systemMessage = message.Content
		}
	}

	if systemMessage != "" || len(tools) > 0 {
		sb.WriteString(imStartTag + "system\n")

		// if we have tools but no system message, match the reference implementation by providing a default system message
		if systemMessage == "" {
			systemMessage = "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks."
		}

		sb.WriteString(systemMessage)

		if len(tools) > 0 {
			sb.WriteString("\n\n# Tools\n\nYou have access to the following functions:\n\n")
			sb.WriteString("<tools>")
			for _, tool := range tools {
				sb.WriteString("\n")
				sb.WriteString("<function>\n")
				sb.WriteString("<name>" + tool.Function.Name + "</name>")
				if tool.Function.Description != "" {
					sb.WriteString("\n<description>" + tool.Function.Description + "</description>")
				}
				sb.WriteString("\n<parameters>")

				for name, prop := range tool.Function.Parameters.Properties.All() {
					sb.WriteString("\n<parameter>")
					sb.WriteString("\n<name>" + name + "</name>")

					if len(prop.Type) > 0 {
						sb.WriteString("\n<type>" + formatToolDefinitionType(prop.Type) + "</type>")
					}

					if prop.Description != "" {
						sb.WriteString("\n<description>" + prop.Description + "</description>")
					}

					// Render any additional keys not already handled
					handledKeys := map[string]bool{
						"type":        true,
						"description": true,
					}
					sb.WriteString(renderAdditionalKeys(prop, handledKeys))

					sb.WriteString("\n</parameter>")
				}

				// Render extra keys for parameters (everything except 'type' and 'properties')
				paramHandledKeys := map[string]bool{
					"type":       true,
					"properties": true,
				}
				sb.WriteString(renderAdditionalKeys(tool.Function.Parameters, paramHandledKeys))

				sb.WriteString("\n</parameters>")
				sb.WriteString("\n</function>")
			}
			sb.WriteString("\n</tools>")
			sb.WriteString("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")
		}

		sb.WriteString(imEndTag + "\n")
	}

	for i, message := range filteredMessages {
		lastMessage := i == len(filteredMessages)-1
		prefill := lastMessage && message.Role == "assistant"
		switch message.Role {
		case "assistant":
			if len(message.ToolCalls) > 0 {
				sb.WriteString(imStartTag + "assistant\n")
				if message.Content != "" {
					sb.WriteString(message.Content + "\n")
				}
				for _, toolCall := range message.ToolCalls {
					sb.WriteString("\n<tool_call>\n<function=" + toolCall.Function.Name + ">")
					for name, value := range toolCall.Function.Arguments.All() {
						valueStr := formatToolCallArgument(value)
						sb.WriteString("\n<parameter=" + name + ">\n" + valueStr + "\n</parameter>")
					}
					sb.WriteString("\n</function>\n</tool_call>")
				}
				sb.WriteString("<|im_end|>\n")
			} else {
				sb.WriteString(imStartTag + "assistant\n")
				sb.WriteString(message.Content)
				if !prefill {
					sb.WriteString(imEndTag + "\n")
				}
			}
		case "tool":
			// consecutive tool responses should share a single `<im_start>user`, but
			// have their own <tool_response> tags

			// only start a new user block if this is the first tool response
			if i == 0 || filteredMessages[i-1].Role != "tool" {
				sb.WriteString(imStartTag + "user\n")
			}

			sb.WriteString("<tool_response>\n")
			sb.WriteString(message.Content)
			sb.WriteString("\n</tool_response>\n")

			// close the user block only if this is the last tool response
			if i == len(filteredMessages)-1 || filteredMessages[i+1].Role != "tool" {
				sb.WriteString(imEndTag + "\n")
			}
		default:
			sb.WriteString(imStartTag + message.Role + "\n")
			sb.WriteString(message.Content)
			sb.WriteString(imEndTag + "\n")
		}

		if lastMessage && !prefill {
			sb.WriteString(imStartTag + "assistant\n")
		}
	}

	return sb.String(), nil
}

func formatToolCallArgument(value any) string {
	if value == nil {
		return "null"
	}

	switch v := value.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	}

	if reflect.TypeOf(value) != nil {
		kind := reflect.TypeOf(value).Kind()
		if kind == reflect.Map || kind == reflect.Slice || kind == reflect.Array {
			if marshalled, err := json.Marshal(value); err == nil {
				return string(marshalled)
			}
		}
	}

	return fmt.Sprintf("%v", value)
}

func formatToolDefinitionType(tp api.PropertyType) string {
	if len(tp) == 0 {
		return "[]"
	}

	if len(tp) == 1 {
		return tp[0]
	}

	// TODO(drifkin): it would be nice to format the JSON here similarly to
	// python's default json.dumps behavior (spaces after commas and colons).
	// This would let us be byte-for-byte compatible with the reference
	// implementation for most common inputs
	jsonBytes, err := json.Marshal(tp)
	if err != nil {
		return "[]"
	}

	return string(jsonBytes)
}
