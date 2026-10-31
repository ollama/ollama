package renderers

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// mimoRenderer renders MiMo V2.6 Distill Qwen message and tool boundaries.
// Its generation prompt opens thinking for this checkpoint.
type mimoRenderer struct {
	useImgTags bool
}

func (r *mimoRenderer) LeadingBOS() string { return "" }

func (r *mimoRenderer) Thinking() *model.Thinking {
	return &model.Thinking{Values: []any{false, true}, Default: true}
}

func (r *mimoRenderer) content(message api.Message, imageOffset int) (string, int) {
	if r.useImgTags {
		return renderContentWithImageTags(message.Content, len(message.Images), imageOffset)
	}

	var sb strings.Builder
	for range message.Images {
		sb.WriteString("<|vision_start|><|image_pad|><|vision_end|>")
	}
	sb.WriteString(message.Content)
	return sb.String(), imageOffset + len(message.Images)
}

func (r *mimoRenderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var sb strings.Builder
	if len(tools) > 0 {
		sb.WriteString("<|im_start|>system\nYou are provided with the following tools:\n\n<tools>")
		for _, tool := range tools {
			definition, err := PythonJSON(tool, 0)
			if err != nil {
				return "", fmt.Errorf("render MiMo tool %q: %w", tool.Function.Name, err)
			}
			sb.WriteByte('\n')
			sb.WriteString(definition)
		}
		sb.WriteString("\n</tools><|im_end|>")
	}

	imageOffset := 0
	for _, message := range messages {
		content, nextOffset := r.content(message, imageOffset)
		imageOffset = nextOffset
		sb.WriteString("<|im_start|>")
		sb.WriteString(message.Role)
		sb.WriteByte('\n')
		if message.Role == "assistant" {
			sb.WriteString("<think>")
			sb.WriteString(message.Thinking)
			sb.WriteString("</think>")
		}
		sb.WriteString(content)
		if message.Role == "assistant" {
			for _, call := range message.ToolCalls {
				sb.WriteString("<tool_call><function=")
				sb.WriteString(call.Function.Name)
				sb.WriteByte('>')
				for key, value := range call.Function.Arguments.All() {
					sb.WriteString("<parameter=")
					sb.WriteString(key)
					sb.WriteByte('>')
					if s, ok := value.(string); ok {
						sb.WriteString(s)
					} else {
						encoded, err := PythonJSON(value, 0)
						if err != nil {
							return "", fmt.Errorf("render MiMo argument %q: %w", key, err)
						}
						sb.WriteString(encoded)
					}
					sb.WriteString("</parameter>")
				}
				sb.WriteString("</function></tool_call>")
			}
		}
		sb.WriteString("<|im_end|>")
	}

	sb.WriteString("<|im_start|>assistant\n")
	if think != nil && !think.Bool() {
		sb.WriteString("<think></think>")
	} else {
		// The publisher template leaves this boundary to the model. This SFT
		// checkpoint can end a turn before closing it, so open thinking here.
		sb.WriteString("<think>\n")
	}
	return sb.String(), nil
}
