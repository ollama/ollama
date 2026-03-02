package renderers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

type GlmOcrRenderer struct{}

func (r *GlmOcrRenderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
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
		sb.WriteString("</tools>\n\n")
		sb.WriteString("For each function call, output the function name and arguments within the following XML format:\n")
		sb.WriteString("<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")
	}

	enableThinking := false
	thinkingExplicitlySet := false
	if thinkValue != nil {
		enableThinking = thinkValue.Bool()
		thinkingExplicitlySet = true
	}

	for i, message := range messages {
		switch message.Role {
		case "user":
			sb.WriteString("<|user|>\n")
			sb.WriteString(message.Content)
			if thinkingExplicitlySet && !enableThinking && !strings.HasSuffix(message.Content, "/nothink") {
				sb.WriteString("/nothink")
			}
		case "assistant":
			sb.WriteString("<|assistant|>\n")
			if message.Thinking != "" {
				sb.WriteString("<think>" + strings.TrimSpace(message.Thinking) + "</think>")
			} else {
				sb.WriteString("<think></think>")
			}
			if message.Content != "" {
				sb.WriteString("\n" + strings.TrimSpace(message.Content))
			}
			if len(message.ToolCalls) > 0 {
				for _, toolCall := range message.ToolCalls {
					sb.WriteString("\n<tool_call>" + toolCall.Function.Name)
					sb.WriteString(renderGlmOcrToolArguments(toolCall.Function.Arguments))
					sb.WriteString("</tool_call>")
				}
			}
			sb.WriteString("\n")
		case "tool":
			if i == 0 || messages[i-1].Role != "tool" {
				sb.WriteString("<|observation|>")
			}
			sb.WriteString("\n<tool_response>\n")
			sb.WriteString(message.Content)
			sb.WriteString("\n</tool_response>\n")
		case "system":
			sb.WriteString("<|system|>\n")
			sb.WriteString(message.Content)
			sb.WriteString("\n")
		}
	}

	sb.WriteString("<|assistant|>\n")
	if thinkingExplicitlySet && !enableThinking {
		sb.WriteString("<think></think>\n")
	}

	return sb.String(), nil
}

func renderGlmOcrToolArguments(args api.ToolCallFunctionArguments) string {
	var sb strings.Builder
	for key, value := range args.All() {
		sb.WriteString("<arg_key>" + key + "</arg_key>")
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

		sb.WriteString("<arg_value>" + valueStr + "</arg_value>")
	}

	return sb.String()
}
