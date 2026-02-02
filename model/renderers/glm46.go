package renderers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

type GLM46Renderer struct{}

func (r *GLM46Renderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	sb.WriteString("[gMASK]<sop>")

	var lastUserIndex int
	for i, message := range messages {
		if message.Role == "user" {
			lastUserIndex = i
		}
	}

	if len(tools) > 0 {
		sb.WriteString("<|system|>\n")
		sb.WriteString("# Tools\n\n")
		sb.WriteString("You may call one or more functions to assist with the user query.\n\n")
		sb.WriteString("You are provided with function signatures within <tools></tools> XML tags:\n")
		sb.WriteString("<tools>\n")
		for _, tool := range tools {
			d, _ := json.Marshal(tool)
			sb.WriteString(string(d) + "\n")
		}
		sb.WriteString("</tools>\n\n")
		sb.WriteString("For each function call, output the function name and arguments within the following XML format:\n")
		sb.WriteString("<tool_call>{function-name}\n")
		sb.WriteString("<arg_key>{arg-key-1}</arg_key>\n")
		sb.WriteString("<arg_value>{arg-value-1}</arg_value>\n")
		sb.WriteString("<arg_key>{arg-key-2}</arg_key>\n")
		sb.WriteString("<arg_value>{arg-value-2}</arg_value>\n")
		sb.WriteString("...\n")
		sb.WriteString("</tool_call>")
	}

	for i, message := range messages {
		switch message.Role {
		case "user":
			sb.WriteString("<|user|>\n")
			sb.WriteString(message.Content)
			if thinkValue != nil && !thinkValue.Bool() && !strings.HasSuffix(message.Content, "/nothink") {
				sb.WriteString("/nothink")
			}
		case "assistant":
			sb.WriteString("<|assistant|>")
			if i > lastUserIndex {
				if message.Thinking != "" {
					sb.WriteString("\n<think>" + message.Thinking + "</think>")
				} else {
					sb.WriteString("\n<think></think>")
				}
			}
			if message.Content != "" {
				sb.WriteString("\n" + message.Content)
			}
			if len(message.ToolCalls) > 0 {
				for _, toolCall := range message.ToolCalls {
					sb.WriteString("\n<tool_call>" + toolCall.Function.Name + "\n")
					for key, value := range toolCall.Function.Arguments.All() {
						sb.WriteString("<arg_key>" + key + "</arg_key>\n")

						var valueStr string
						if str, ok := value.(string); ok {
							valueStr = str
						} else {
							jsonBytes, err := json.Marshal(value)
							if err != nil {
								valueStr = fmt.Sprintf("%v", value)
							} else {
								valueStr = string(jsonBytes)
							}
						}

						sb.WriteString("<arg_value>" + valueStr + "</arg_value>\n")
					}

					sb.WriteString("</tool_call>")
				}
			}
		case "tool":
			if i == 0 || messages[i-1].Role != "tool" {
				sb.WriteString("<|observation|>")
			}
			sb.WriteString("\n<tool_response>\n")
			sb.WriteString(message.Content)
			sb.WriteString("\n</tool_response>")
		case "system":
			sb.WriteString("<|system|>\n")
			sb.WriteString(message.Content)
		}
	}

	// Add generation prompt
	sb.WriteString("<|assistant|>")
	if thinkValue != nil && !thinkValue.Bool() {
		sb.WriteString("\n<think></think>\n")
	}

	return sb.String(), nil
}
