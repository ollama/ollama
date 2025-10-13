package renderers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

func marshalWithSpaces(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}

	out := make([]byte, 0, len(b)+len(b)/8)
	inStr, esc := false, false
	for _, c := range b {
		if inStr {
			out = append(out, c)
			if esc {
				esc = false
				continue
			}
			if c == '\\' {
				esc = true
				continue
			}
			if c == '"' {
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
			out = append(out, c)
		case ':':
			out = append(out, ':', ' ')
		case ',':
			out = append(out, ',', ' ')
		default:
			out = append(out, c)
		}
	}
	return out, nil
}

type Qwen3VLRenderer struct {
	isThinking bool
}

func (r *Qwen3VLRenderer) renderContent(content api.Message, doVisionCount bool) string {
	// This assumes all images are at the front of the message - same assumption as ollama/ollama/runner.go
	var subSb strings.Builder
	for range content.Images {
		subSb.WriteString("<|vision_start|><|image_pad|><|vision_end|>")
	}
	// TODO: support videos

	subSb.WriteString(content.Content)
	return subSb.String()
}

func (r *Qwen3VLRenderer) Render(messages []api.Message, tools []api.Tool, _ *api.ThinkValue) (string, error) {
	var sb strings.Builder

	if len(tools) > 0 {
		sb.WriteString(imStartTag + "system\n")
		if len(messages) > 0 && messages[0].Role == "system" {
			sb.WriteString(messages[0].Content + "\n\n")
		}
		sb.WriteString("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>")
		for _, tool := range tools {
			sb.WriteString("\n")
			if b, err := marshalWithSpaces(tool); err == nil {
				sb.Write(b)
			}
		}
		sb.WriteString("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n")
	} else if len(messages) > 0 && messages[0].Role == "system" {
		sb.WriteString("<|im_start|>system\n" + messages[0].Content + "<|im_end|>\n")
	}
	multiStepTool := true
	lastQueryIndex := len(messages) - 1 // so this is the last user message

	for i := len(messages) - 1; i >= 0; i-- {
		message := messages[i]
		if multiStepTool && message.Role == "user" {
			// Check if content starts with <tool_response> and ends with </tool_response>
			content := r.renderContent(message, true)
			if !(strings.HasPrefix(content, "<tool_response>") && strings.HasSuffix(content, "</tool_response>")) {
				multiStepTool = false
				lastQueryIndex = i
			}
		}
	}

	for i, message := range messages {
		content := r.renderContent(message, true)

		lastMessage := i == len(messages)-1
		prefill := lastMessage && message.Role == "assistant"

		fmt.Println("message", i, prefill)

		if message.Role == "user" || message.Role == "system" && i != 0 {
			sb.WriteString("<|im_start|>" + message.Role + "\n" + content + "<|im_end|>\n")
		} else if message.Role == "assistant" {
			contentReasoning := ""

			if r.isThinking {
				if message.Thinking != "" {
					contentReasoning = message.Thinking
				}
			}

			if r.isThinking && i > lastQueryIndex {
				fmt.Println("contentReasoning:", contentReasoning)
				fmt.Println("content:", content)

				if i == len(messages)-1 || contentReasoning != "" {
					fmt.Println("should be in here if we have content reasoning")
					sb.WriteString("<|im_start|>" + message.Role + "\n<think>\n" + strings.Trim(contentReasoning, "\n")) // do we want to add a new line here?
					fmt.Println("<|im_start|>" + message.Role + "\n<think>\n" + strings.Trim(contentReasoning, "\n"))
					if content != "" {
						sb.WriteString("\n</think>\n\n" + strings.TrimLeft(content, "\n"))
						fmt.Println("\n</think>\n\n" + strings.TrimLeft(content, "\n"))
					}
				} else {
					sb.WriteString("<|im_start|>" + message.Role + "\n" + content)
				}
			} else {
				sb.WriteString("<|im_start|>" + message.Role + "\n" + content)
			}

			if len(message.ToolCalls) > 0 {
				for j, toolCall := range message.ToolCalls {
					if j > 0 || content != "" {
						sb.WriteString("\n")
					}

					sb.WriteString("<tool_call>\n{\"name\": \"" + toolCall.Function.Name + "\", \"arguments\": ")
					if b, err := marshalWithSpaces(toolCall.Function.Arguments); err == nil {
						sb.Write(b)
					}
					sb.WriteString("}\n</tool_call>")
				}
			}

			if !prefill {
				sb.WriteString("<|im_end|>\n")
			}
		} else if message.Role == "tool" {
			if i == 0 || messages[i-1].Role != "tool" {
				sb.WriteString("<|im_start|>user")
			}
			sb.WriteString("\n<tool_response>\n" + message.Content + "\n</tool_response>")
			if i == len(messages)-1 || messages[i+1].Role != "tool" {
				sb.WriteString("<|im_end|>\n")
			}
		}

		// prefill at the end
		if lastMessage && !prefill {
			sb.WriteString("<|im_start|>assistant\n")
			if r.isThinking {
				sb.WriteString("<think>\n")
			}
		}
	}

	return sb.String(), nil
}
