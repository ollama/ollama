package renderers

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

const (
	qwen35ThinkOpenTag  = "<think>"
	qwen35ThinkCloseTag = "</think>"
	qwen35ToolPostamble = `
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>`
)

type Qwen35Renderer struct {
	isThinking bool

	emitEmptyThinkOnNoThink bool
	useImgTags              bool
}

func (r *Qwen35Renderer) renderContent(content api.Message, imageOffset int) (string, int) {
	// This assumes all images are at the front of the message - same assumption as ollama/ollama/runner.go
	var subSb strings.Builder
	for range content.Images {
		if r.useImgTags {
			subSb.WriteString(fmt.Sprintf("[img-%d]", imageOffset))
			imageOffset++
		} else {
			subSb.WriteString("<|vision_start|><|image_pad|><|vision_end|>")
		}
	}
	// TODO: support videos

	subSb.WriteString(content.Content)
	return subSb.String(), imageOffset
}

func splitQwen35ReasoningContent(content, messageThinking string, isThinking bool) (reasoning string, remaining string) {
	if isThinking && messageThinking != "" {
		return strings.TrimSpace(messageThinking), content
	}

	if idx := strings.Index(content, qwen35ThinkCloseTag); idx != -1 {
		before := content[:idx]
		if open := strings.LastIndex(before, qwen35ThinkOpenTag); open != -1 {
			reasoning = before[open+len(qwen35ThinkOpenTag):]
		} else {
			reasoning = before
		}
		content = strings.TrimLeft(content[idx+len(qwen35ThinkCloseTag):], "\n")
	}

	return strings.TrimSpace(reasoning), content
}

func (r *Qwen35Renderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var sb strings.Builder

	isThinking := r.isThinking
	if think != nil {
		isThinking = think.Bool()
	}

	if len(tools) > 0 {
		sb.WriteString(imStartTag + "system\n")
		sb.WriteString("# Tools\n\nYou have access to the following functions:\n\n<tools>")
		for _, tool := range tools {
			sb.WriteString("\n")
			if b, err := marshalWithSpaces(tool); err == nil {
				sb.Write(b)
			}
		}
		sb.WriteString(qwen35ToolPostamble)
		if len(messages) > 0 && messages[0].Role == "system" {
			systemContent, _ := r.renderContent(messages[0], 0)
			systemContent = strings.TrimSpace(systemContent)
			if systemContent != "" {
				sb.WriteString("\n\n")
				sb.WriteString(systemContent)
			}
		}
		sb.WriteString(imEndTag + "\n")
	} else if len(messages) > 0 && messages[0].Role == "system" {
		systemContent, _ := r.renderContent(messages[0], 0)
		sb.WriteString(imStartTag + "system\n" + strings.TrimSpace(systemContent) + imEndTag + "\n")
	}

	multiStepTool := true
	lastQueryIndex := len(messages) - 1 // so this is the last user message

	for i := len(messages) - 1; i >= 0; i-- {
		message := messages[i]
		if multiStepTool && message.Role == "user" {
			content, _ := r.renderContent(message, 0)
			content = strings.TrimSpace(content)
			if !(strings.HasPrefix(content, "<tool_response>") && strings.HasSuffix(content, "</tool_response>")) {
				multiStepTool = false
				lastQueryIndex = i
			}
		}
	}

	imageOffset := 0
	for i, message := range messages {
		content, nextImageOffset := r.renderContent(message, imageOffset)
		imageOffset = nextImageOffset
		content = strings.TrimSpace(content)

		lastMessage := i == len(messages)-1
		prefill := lastMessage && message.Role == "assistant"

		if message.Role == "user" || (message.Role == "system" && i != 0) {
			sb.WriteString(imStartTag + message.Role + "\n" + content + imEndTag + "\n")
		} else if message.Role == "assistant" {
			contentReasoning, content := splitQwen35ReasoningContent(content, message.Thinking, isThinking)

			if isThinking && i > lastQueryIndex {
				sb.WriteString(imStartTag + message.Role + "\n<think>\n" + contentReasoning + "\n</think>\n\n" + content)
			} else {
				sb.WriteString(imStartTag + message.Role + "\n" + content)
			}

			if len(message.ToolCalls) > 0 {
				for j, toolCall := range message.ToolCalls {
					if j == 0 {
						if strings.TrimSpace(content) != "" {
							sb.WriteString("\n\n")
						}
					} else {
						sb.WriteString("\n")
					}

					sb.WriteString("<tool_call>\n<function=" + toolCall.Function.Name + ">\n")
					for name, value := range toolCall.Function.Arguments.All() {
						sb.WriteString("<parameter=" + name + ">\n")
						sb.WriteString(formatToolCallArgument(value))
						sb.WriteString("\n</parameter>\n")
					}
					sb.WriteString("</function>\n</tool_call>")
				}
			}

			if !prefill {
				sb.WriteString(imEndTag + "\n")
			}
		} else if message.Role == "tool" {
			if i == 0 || messages[i-1].Role != "tool" {
				sb.WriteString(imStartTag + "user")
			}
			sb.WriteString("\n<tool_response>\n" + content + "\n</tool_response>")
			if i == len(messages)-1 || messages[i+1].Role != "tool" {
				sb.WriteString(imEndTag + "\n")
			}
		}

		// prefill at the end
		if lastMessage && !prefill {
			sb.WriteString(imStartTag + "assistant\n")
			if isThinking {
				sb.WriteString("<think>\n")
			} else if r.emitEmptyThinkOnNoThink {
				sb.WriteString("<think>\n\n</think>\n\n")
			}
		}
	}

	return sb.String(), nil
}
