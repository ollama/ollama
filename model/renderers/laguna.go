package renderers

import (
	"strings"

	"github.com/ollama/ollama/api"
)

const (
	lagunaBOS          = "〈|EOS|〉"
	lagunaThoughtOpen  = "<think>"
	lagunaThoughtClose = "</think>"
)

type LagunaRenderer struct{}

func (r *LagunaRenderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var sb strings.Builder
	sb.WriteString(lagunaBOS)

	thinkingEnabled := think == nil || think.Bool()
	systemMessage := ""
	firstMessageIsSystem := len(messages) > 0 && messages[0].Role == "system"
	if firstMessageIsSystem {
		systemMessage = strings.TrimRight(messages[0].Content, "\n")
	}

	sb.WriteString("<system>\n")
	if thinkingEnabled {
		sb.WriteString("You should use chain-of-thought reasoning. Put your reasoning inside <think> </think> tags before your response.")
	} else {
		sb.WriteString("You should respond directly without using chain-of-thought reasoning tags.")
	}
	if strings.TrimSpace(systemMessage) != "" {
		sb.WriteByte('\n')
		sb.WriteString(systemMessage)
	}
	if len(tools) > 0 {
		sb.WriteString("\n\n### Tools\n\n")
		sb.WriteString("You may call functions to assist with the user query.\n")
		sb.WriteString("All available function signatures are listed below:\n")
		sb.WriteString("<available_tools>\n")
		for _, tool := range tools {
			if b, err := marshalWithSpaces(tool); err == nil {
				sb.Write(b)
				sb.WriteByte('\n')
			}
		}
		sb.WriteString("</available_tools>\n\n")
		sb.WriteString("For each function call, return a json object with function name and arguments within '<tool_call>' and '</tool_call>' tags:\n")
		sb.WriteString("<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
	}
	sb.WriteString("\n</system>\n")

	for i, message := range messages {
		if i == 0 && firstMessageIsSystem {
			continue
		}
		content := message.Content
		switch message.Role {
		case "user":
			sb.WriteString("<user>\n")
			sb.WriteString(content)
			sb.WriteString("\n</user>\n")
		case "assistant":
			lastMessage := i == len(messages)-1
			prefill := lastMessage && (content != "" || message.Thinking != "" || len(message.ToolCalls) > 0)
			sb.WriteString("<assistant>\n")
			if thinkingEnabled && message.Thinking != "" {
				sb.WriteString(lagunaThoughtOpen)
				sb.WriteString(message.Thinking)
				sb.WriteString(lagunaThoughtClose)
				sb.WriteByte('\n')
			}
			if strings.Trim(content, "\n") != "" {
				sb.WriteString(strings.Trim(content, "\n"))
				sb.WriteByte('\n')
			}
			for _, toolCall := range message.ToolCalls {
				sb.WriteString("<tool_call>")
				sb.WriteString(toolCall.Function.Name)
				sb.WriteByte('\n')
				for name, value := range toolCall.Function.Arguments.All() {
					sb.WriteString("<arg_key>")
					sb.WriteString(name)
					sb.WriteString("</arg_key>\n")
					sb.WriteString("<arg_value>")
					sb.WriteString(formatToolCallArgument(value))
					sb.WriteString("</arg_value>\n")
				}
				sb.WriteString("</tool_call>\n")
			}
			if !prefill {
				sb.WriteString("</assistant>\n")
			}
		case "tool":
			sb.WriteString("<tool_response>\n")
			sb.WriteString(content)
			sb.WriteString("\n</tool_response>\n")
		case "system":
			sb.WriteString("<system>\n")
			sb.WriteString(content)
			sb.WriteString("\n</system>\n")
		}
	}

	if len(messages) == 0 || messages[len(messages)-1].Role != "assistant" {
		sb.WriteString("<assistant>\n")
	}
	return sb.String(), nil
}
