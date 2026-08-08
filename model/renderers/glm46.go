package renderers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

type GLM46Renderer struct{}

func (r *GLM46Renderer) LeadingBOS() string {
	return ""
}

func (r *GLM46Renderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	segments, err := r.RenderSegments(messages, tools, thinkValue)
	if err != nil {
		return "", err
	}
	return JoinSegments(segments), nil
}

func (r *GLM46Renderer) RenderSegments(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) ([]Segment, error) {
	var sb segmentBuilder

	sb.control("[gMASK]<sop>")

	var lastUserIndex int
	for i, message := range messages {
		if message.Role == "user" {
			lastUserIndex = i
		}
	}

	if len(tools) > 0 {
		sb.control("<|system|>\n")
		sb.control("# Tools\n\n")
		sb.control("You may call one or more functions to assist with the user query.\n\n")
		sb.control("You are provided with function signatures within <tools></tools> XML tags:\n")
		sb.control("<tools>\n")
		for _, tool := range tools {
			d, _ := json.Marshal(tool)
			sb.content(string(d))
			sb.control("\n")
		}
		sb.control("</tools>\n\n")
		sb.control("For each function call, output the function name and arguments within the following XML format:\n")
		sb.control("<tool_call>{function-name}\n")
		sb.control("<arg_key>{arg-key-1}</arg_key>\n")
		sb.control("<arg_value>{arg-value-1}</arg_value>\n")
		sb.control("<arg_key>{arg-key-2}</arg_key>\n")
		sb.control("<arg_value>{arg-value-2}</arg_value>\n")
		sb.control("...\n")
		sb.control("</tool_call>")
	}

	for i, message := range messages {
		switch message.Role {
		case "user":
			sb.control("<|user|>\n")
			sb.content(message.Content)
			if thinkValue != nil && !thinkValue.Bool() && !strings.HasSuffix(message.Content, "/nothink") {
				sb.control("/nothink")
			}
		case "assistant":
			sb.control("<|assistant|>")
			if i > lastUserIndex {
				if message.Thinking != "" {
					sb.control("\n<think>")
					sb.content(message.Thinking)
					sb.control("</think>")
				} else {
					sb.control("\n<think></think>")
				}
			}
			if message.Content != "" {
				sb.control("\n")
				sb.content(message.Content)
			}
			if len(message.ToolCalls) > 0 {
				for _, toolCall := range message.ToolCalls {
					sb.control("\n<tool_call>")
					sb.content(toolCall.Function.Name)
					sb.control("\n")
					for key, value := range toolCall.Function.Arguments.All() {
						sb.control("<arg_key>")
						sb.content(key)
						sb.control("</arg_key>\n")

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

						sb.control("<arg_value>")
						sb.content(valueStr)
						sb.control("</arg_value>\n")
					}

					sb.control("</tool_call>")
				}
			}
		case "tool":
			if i == 0 || messages[i-1].Role != "tool" {
				sb.control("<|observation|>")
			}
			sb.control("\n<tool_response>\n")
			sb.content(message.Content)
			sb.control("\n</tool_response>")
		case "system":
			sb.control("<|system|>\n")
			sb.content(message.Content)
		}
	}

	// Add generation prompt
	sb.control("<|assistant|>")
	if thinkValue != nil && !thinkValue.Bool() {
		sb.control("\n<think></think>\n")
	}

	return sb.Segments(), nil
}
