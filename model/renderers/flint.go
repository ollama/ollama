package renderers

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

type FlintRenderer struct{}

func (*FlintRenderer) LeadingBOS() string { return "" }

// The publisher's bare assistant prompt lets the model choose whether to
// reason. Explicit thinking controls add a prefill below.
func (*FlintRenderer) Thinking() *model.Thinking { return nil }

var flintRoleEscaper = strings.NewReplacer(
	"<|begin_of_text|>", "&lt;|begin_of_text|&gt;",
	"<|end_of_text|>", "&lt;|end_of_text|&gt;",
	"<|system|>", "&lt;|system|&gt;",
	"<|developer|>", "&lt;|developer|&gt;",
	"<|user|>", "&lt;|user|&gt;",
	"<|assistant|>", "&lt;|assistant|&gt;",
	"<|observation|>", "&lt;|observation|&gt;",
)

func flintInputText(content string) string {
	content = strings.ReplaceAll(content, "\x00", "")
	content = strings.ReplaceAll(content, "<think>", "")
	content = strings.ReplaceAll(content, "</think>", "")
	return flintRoleEscaper.Replace(content)
}

// The publisher's tool protocol uses literal XML-like delimiters. Remove
// them from untrusted tool results so a result cannot impersonate a call.
var flintToolResultEscaper = strings.NewReplacer(
	"<tool_call>", "", "</tool_call>", "",
	"<tool_response>", "", "</tool_response>", "",
	"<arg_key>", "", "</arg_key>", "",
	"<arg_value>", "", "</arg_value>", "",
)

func (*FlintRenderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var out strings.Builder
	toolsPending := true
	for _, message := range messages {
		if message.Role != "system" && toolsPending {
			if err := writeFlintTools(&out, tools); err != nil {
				return "", err
			}
			toolsPending = false
		}
		switch message.Role {
		case "system":
			out.WriteString("<|system|>")
			out.WriteString(flintInputText(message.Content))
		case "developer":
			out.WriteString("<|developer|>")
			out.WriteString(flintInputText(message.Content))
		case "user":
			out.WriteString("<|user|>")
			out.WriteString(flintInputText(message.Content))
		case "assistant":
			out.WriteString("<|assistant|>")
			if message.Thinking != "" {
				out.WriteString("<think>")
				out.WriteString(message.Thinking)
				out.WriteString("</think>")
			}
			out.WriteString(message.Content)
			for _, call := range message.ToolCalls {
				out.WriteString("<tool_call>")
				out.WriteString(call.Function.Name)
				for name, value := range call.Function.Arguments.All() {
					out.WriteString("<arg_key>")
					out.WriteString(name)
					out.WriteString("</arg_key><arg_value>")
					if s, ok := value.(string); ok {
						out.WriteString(s)
					} else {
						formatted, err := flintPythonJSON(value)
						if err != nil {
							return "", fmt.Errorf("Flint tool %q argument %q: %w", call.Function.Name, name, err)
						}
						out.WriteString(formatted)
					}
					out.WriteString("</arg_value>")
				}
				out.WriteString("</tool_call>")
			}
		case "tool":
			out.WriteString("<|observation|><tool_response>")
			out.WriteString(flintInputText(flintToolResultEscaper.Replace(message.Content)))
			out.WriteString("</tool_response>")
		default:
			return "", fmt.Errorf("Flint renderer does not support message role %q", message.Role)
		}
	}
	if toolsPending {
		if err := writeFlintTools(&out, tools); err != nil {
			return "", err
		}
	}
	out.WriteString("<|assistant|>")
	if think != nil {
		if think.Bool() {
			out.WriteString("<think>")
		} else {
			// Close an empty reasoning section so short completions can answer
			// directly instead of spending their token budget in hidden thought.
			out.WriteString("<think></think>")
		}
	}
	return out.String(), nil
}

func writeFlintTools(out *strings.Builder, tools []api.Tool) error {
	if len(tools) == 0 {
		return nil
	}
	out.WriteString("<|system|>\n# Tools\n\n")
	out.WriteString("You may call one or more functions to assist with the user query.\n\n")
	out.WriteString("You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
	for _, tool := range tools {
		formatted, err := flintPythonJSON(tool.Function)
		if err != nil {
			return fmt.Errorf("Flint tool %q schema: %w", tool.Function.Name, err)
		}
		out.WriteString(formatted)
	}
	out.WriteString("</tools>\n\n")
	out.WriteString("For each function call, output the function name and arguments within the following XML format:\n")
	out.WriteString("<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n")
	return nil
}
