package renderers

import (
	"fmt"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

type RippleRenderer struct{}

func (r *RippleRenderer) LeadingBOS() string { return "" }

// Ripple can emit reasoning but has no template control to disable or force it.
func (r *RippleRenderer) Thinking() *model.Thinking { return nil }

func (r *RippleRenderer) Render(messages []api.Message, tools []api.Tool, _ *api.ThinkValue) (string, error) {
	if len(messages) == 0 {
		return "", fmt.Errorf("ripple: expected at least one message")
	}
	for _, message := range messages {
		if len(message.Images) > 0 {
			return "", fmt.Errorf("ripple: text-only messages required")
		}
	}
	var b strings.Builder
	if len(tools) > 0 {
		b.WriteString("<|im_start|>system\n")
		if messages[0].Role == "system" {
			b.WriteString(messages[0].Content + "\n\n")
		}
		b.WriteString("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools> XML tags:\n<tools>")
		for _, tool := range tools {
			encoded, err := PythonJSON(tool, 0)
			if err != nil {
				return "", err
			}
			b.WriteString("\n" + encoded)
		}
		b.WriteString("\n</tools>\n\nFor each function call, return exactly one valid JSON object with function name and arguments inside XML tool-call tags. The arguments field must be a JSON object, not a string, and strings must use valid JSON escaping:\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call><|im_end|>\n")
	}
	for i, message := range messages {
		if i == 0 && message.Role == "system" && len(tools) > 0 {
			continue
		}
		content := message.Content
		if message.Role == "assistant" && message.Thinking != "" && !strings.HasPrefix(strings.TrimLeftFunc(content, func(r rune) bool {
			return unicode.IsSpace(r) || r >= '\x1c' && r <= '\x1f'
		}), "<think>") {
			content = "<think>\n" + message.Thinking + "\n</think>\n" + content
		}
		switch message.Role {
		case "assistant":
			b.WriteString("<|im_start|>assistant")
			if content != "" || len(message.ToolCalls) == 0 {
				b.WriteString("\n" + content)
			}
			for _, call := range message.ToolCalls {
				name, err := PythonJSON(call.Function.Name, 0)
				if err != nil {
					return "", err
				}
				arguments, err := PythonJSON(call.Function.Arguments, 0)
				if err != nil {
					return "", err
				}
				b.WriteString("\n<tool_call>\n{\"name\": " + name + ", \"arguments\": " + arguments + "}\n</tool_call>")
			}
			b.WriteString("<|im_end|>\n")
		case "tool":
			if i == 0 || messages[i-1].Role != "tool" {
				b.WriteString("<|im_start|>user")
			}
			b.WriteString("\n<tool_response>\n" + content + "\n</tool_response>")
			if i == len(messages)-1 || messages[i+1].Role != "tool" {
				b.WriteString("<|im_end|>\n")
			}
		case "system", "user":
			b.WriteString("<|im_start|>" + message.Role + "\n")
			b.WriteString(content)
			b.WriteString("<|im_end|>\n")
		}
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String(), nil
}
