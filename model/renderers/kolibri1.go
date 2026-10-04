package renderers

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

type Kolibri1Renderer struct{}

func (*Kolibri1Renderer) LeadingBOS() string { return "" }
func (*Kolibri1Renderer) Thinking() *model.Thinking {
	return &model.Thinking{Values: []any{false, "low", "medium", "high"}, Default: "high"}
}

func kolibriReasoning(think *api.ThinkValue) string {
	if think != nil && !think.Bool() {
		return "Reasoning is disabled. Proceed straight to answering according to the user's instructions."
	}
	if think != nil && think.IsString() {
		switch think.String() {
		case "low":
			return "Reasoning effort is set to low. Think briefly through only the essential steps in the user's language, then proceed directly to the answer."
		case "medium":
			return "Reasoning effort is set to medium. Think through the task methodically in the user's language, check key assumptions, and provide a well-supported answer."
		}
	}
	return "Reasoning effort is set to high. Think carefully through the task in the user's language, validate key assumptions, consider plausible alternatives, and prioritize correctness and clarity."
}

func (*Kolibri1Renderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var out strings.Builder
	out.WriteString("<|im_start|>system\n")
	if len(messages) > 0 && messages[0].Role == "system" {
		out.WriteString(messages[0].Content)
		out.WriteString("\n\n")
	}
	out.WriteString("# Reasoning effort\n\n" + kolibriReasoning(think))
	if len(tools) > 0 {
		out.WriteString("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>")
		for _, tool := range tools {
			data, err := marshalWithSpaces(tool)
			if err != nil {
				return "", fmt.Errorf("kolibri1 tool schema: %w", err)
			}
			out.WriteByte('\n')
			out.Write(data)
		}
		out.WriteString("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
	}
	out.WriteString("<|im_end|>\n")
	lastQuery := len(messages) - 1
	for i := len(messages) - 1; i >= 0; i-- {
		m := messages[i]
		if m.Role == "user" && !(strings.HasPrefix(m.Content, "<tool_response>") && strings.HasSuffix(m.Content, "</tool_response>")) {
			lastQuery = i
			break
		}
	}
	prefill := false
	for i, m := range messages {
		switch m.Role {
		case "system", "user":
			if i == 0 && m.Role == "system" {
				continue
			}
			out.WriteString(imStartTag + m.Role + "\n" + m.Content + imEndTag + "\n")
		case "assistant":
			out.WriteString(imStartTag + "assistant\n")
			reasoning, content := m.Thinking, m.Content
			if reasoning == "" {
				if j := strings.Index(content, "</think>"); j >= 0 {
					reasoning = content[:j]
					if k := strings.LastIndex(reasoning, "<think>"); k >= 0 {
						reasoning = reasoning[k+len("<think>"):]
					}
					content = strings.TrimLeft(content[strings.LastIndex(content, "</think>")+len("</think>"):], "\n")
				}
			}
			if i > lastQuery {
				out.WriteString("<think>\n")
				if strings.TrimSpace(reasoning) != "" {
					out.WriteString(strings.Trim(reasoning, "\n"))
				}
				out.WriteString("\n</think>\n\n")
			}
			out.WriteString(strings.TrimLeft(content, "\n"))
			for j, call := range m.ToolCalls {
				if j > 0 || content != "" {
					out.WriteByte('\n')
				}
				data, err := marshalWithSpaces(struct {
					Name      string                        `json:"name"`
					Arguments api.ToolCallFunctionArguments `json:"arguments"`
				}{call.Function.Name, call.Function.Arguments})
				if err != nil {
					return "", fmt.Errorf("kolibri1 tool call: %w", err)
				}
				out.WriteString("<tool_call>\n")
				out.Write(data)
				out.WriteString("\n</tool_call>")
			}
			prefill = i == len(messages)-1 && len(m.ToolCalls) == 0
			if !prefill {
				out.WriteString(imEndTag + "\n")
			}
		case "tool":
			if i == 0 || messages[i-1].Role != "tool" {
				out.WriteString(imStartTag + "user")
			}
			out.WriteString("\n<tool_response>\n" + m.Content + "\n</tool_response>")
			if i == len(messages)-1 || messages[i+1].Role != "tool" {
				out.WriteString(imEndTag + "\n")
			}
		}
	}
	if !prefill {
		out.WriteString(imStartTag + "assistant\n")
		if think != nil && !think.Bool() {
			out.WriteString("<think>\n\n</think>\n\n")
		}
	}
	return out.String(), nil
}
