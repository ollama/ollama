package renderers

import (
	"strings"

	"github.com/ollama/ollama/api"
)

type Qwen3VLRenderer struct {
	isThinking bool

	useImgTags bool
}

func (r *Qwen3VLRenderer) renderContent(content api.Message) string {
	// This assumes all images are at the front of the message - same assumption as ollama/ollama/runner.go
	var subSb strings.Builder
	for range content.Images {
		// TODO: (jmorganca): how to render this is different for different
		// model backends, and so we should eventually parameterize this or
		// only output a placeholder such as [img]
		if r.useImgTags {
			subSb.WriteString("[img]")
		} else {
			subSb.WriteString("<|vision_start|><|image_pad|><|vision_end|>")
		}
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
			content := r.renderContent(message)
			if !(strings.HasPrefix(content, "<tool_response>") && strings.HasSuffix(content, "</tool_response>")) {
				multiStepTool = false
				lastQueryIndex = i
			}
		}
	}

	for i, message := range messages {
		content := r.renderContent(message)

		lastMessage := i == len(messages)-1
		prefill := lastMessage && message.Role == "assistant"

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
				if i == len(messages)-1 || contentReasoning != "" {
					sb.WriteString("<|im_start|>" + message.Role + "\n<think>\n" + strings.Trim(contentReasoning, "\n")) // do we want to add a new line here?
					if content != "" {
						sb.WriteString("\n</think>\n\n" + strings.TrimLeft(content, "\n"))
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
