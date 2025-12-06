package renderers

import (
	"strings"

	"github.com/ollama/ollama/api"
)

const intellect3DefaultSystemMessage = "You are INTELLECT-3, a helpful assistant developed by Prime Intellect, that can interact with a computer to solve tasks."

type Intellect3Renderer struct{}

func (r *Intellect3Renderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
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

		// Use default system message when tools present but no user system message
		if systemMessage == "" && len(tools) > 0 {
			systemMessage = intellect3DefaultSystemMessage
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

				for name, prop := range tool.Function.Parameters.Properties {
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

				// Add thinking tags if present
				if message.Thinking != "" {
					sb.WriteString("<think>" + strings.TrimSpace(message.Thinking) + "</think>\n")
				}

				if message.Content != "" {
					sb.WriteString(strings.TrimSpace(message.Content) + "\n")
				}

				for _, toolCall := range message.ToolCalls {
					sb.WriteString("\n<tool_call>\n<function=" + toolCall.Function.Name + ">")
					for name, value := range toolCall.Function.Arguments {
						valueStr := formatToolCallArgument(value)
						sb.WriteString("\n<parameter=" + name + ">\n" + valueStr + "\n</parameter>")
					}
					sb.WriteString("\n</function>\n</tool_call>")
				}
				sb.WriteString("<|im_end|>\n")
			} else {
				sb.WriteString(imStartTag + "assistant\n")

				// Add thinking tags if present
				if message.Thinking != "" {
					sb.WriteString("<think>" + strings.TrimSpace(message.Thinking) + "</think>\n")
				}

				// Add content if present
				if message.Content != "" {
					sb.WriteString(message.Content)
				}

				if !prefill {
					sb.WriteString(imEndTag + "\n")
				}
			}
		case "tool":
			if i == 0 || filteredMessages[i-1].Role != "tool" {
				sb.WriteString(imStartTag + "user\n")
			}

			sb.WriteString("<tool_response>\n")
			sb.WriteString(message.Content)
			sb.WriteString("\n</tool_response>\n")

			if i == len(filteredMessages)-1 || filteredMessages[i+1].Role != "tool" {
				sb.WriteString(imEndTag + "\n")
			}
		default:
			sb.WriteString(imStartTag + message.Role + "\n")
			sb.WriteString(message.Content)
			sb.WriteString(imEndTag + "\n")
		}

		if lastMessage && !prefill {
			sb.WriteString(imStartTag + "assistant\n<think>")
		}
	}

	return sb.String(), nil
}
