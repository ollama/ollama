package renderers

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

const (
	hy4TurnStart     = "<｜hy_start:opensource｜>"
	hy4TurnRoleEnd   = "<｜hy_middle:opensource｜>"
	hy4TurnEnd       = "<｜hy_end:opensource｜>"
	hy4ReasoningMode = "<｜reasoning_mode:opensource｜>"
	hy4ThinkStart    = "<think:opensource>"
	hy4ThinkEnd      = "</think:opensource>"
	hy4CallsStart    = "<tool_calls:opensource>"
	hy4CallsEnd      = "</tool_calls:opensource>"
	hy4CallStart     = "<tool_call:opensource>"
	hy4CallEnd       = "</tool_call:opensource>"
	hy4ArgKeyStart   = "<arg_key:opensource>"
	hy4ArgKeyEnd     = "</arg_key:opensource>"
	hy4ArgValueStart = "<arg_value:opensource>"
	hy4ArgValueEnd   = "</arg_value:opensource>"
	hy4ResultStart   = "<tool_response:opensource>"
	hy4ResultEnd     = "</tool_response:opensource>"
	hy4Image         = "<｜hy_image▁start:opensource｜><｜hy_image▁pad:opensource｜><｜hy_image▁end:opensource｜>"
)

func hy4Effort(think *api.ThinkValue) string {
	if think != nil {
		switch think.Value {
		case false, "no_think":
			return "no_think"
		case "max":
			return "max"
		}
	}
	return "high"
}

func init() {
	Register("hy4-flash", func() Renderer { return &hy4FlashRenderer{imageTags: RenderImgTags} })
}

type hy4FlashRenderer struct{ imageTags bool }

func (r *hy4FlashRenderer) LeadingBOS() string { return "" }
func (r *hy4FlashRenderer) Thinking() *model.Thinking {
	return &model.Thinking{Values: []any{false, true, "max", "high", "no_think"}, Default: "high"}
}

func (r *hy4FlashRenderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var prompt strings.Builder
	effort := hy4Effort(think)
	first := 0
	system := ""
	images := 0
	if len(messages) > 0 && messages[0].Role == "system" {
		system, images = r.content(messages[0], images)
		first++
	}
	hy4Turn(&prompt, "system")
	if len(tools) > 0 {
		if err := writeHy4Tools(&prompt, tools); err != nil {
			return "", err
		}
		if system != "" {
			prompt.WriteString("\n\n")
		}
	}
	prompt.WriteString(system)
	prompt.WriteString(hy4ReasoningMode)
	prompt.WriteString("reasoning_effort:")
	prompt.WriteString(effort)
	prompt.WriteString(hy4TurnEnd)

	lastUser := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			lastUser = i
			break
		}
	}
	for i := first; i < len(messages); {
		message := messages[i]
		hy4Turn(&prompt, message.Role)
		if message.Role == "tool" {
			start := i
			for i < len(messages) && messages[i].Role == "tool" {
				i++
			}
			results := messages[start:i]
			if start > 0 && messages[start-1].Role == "assistant" {
				results = hy4ToolResults(results, messages[start-1].ToolCalls)
			}
			for _, result := range results {
				var content string
				content, images = r.content(result, images)
				prompt.WriteString(hy4ResultStart)
				prompt.WriteString(content)
				prompt.WriteString(hy4ResultEnd)
			}
		} else {
			content, next := r.content(message, images)
			images = next
			if message.Role == "assistant" {
				prompt.WriteString(hy4ThinkStart)
				if effort != "no_think" && (len(tools) > 0 || i > lastUser) {
					prompt.WriteString(message.Thinking)
				}
				prompt.WriteString(hy4ThinkEnd)
			}
			prompt.WriteString(content)
			if message.Role == "assistant" && len(message.ToolCalls) > 0 {
				if err := writeHy4Calls(&prompt, message.ToolCalls); err != nil {
					return "", err
				}
			}
			i++
		}
		prompt.WriteString(hy4TurnEnd)
	}
	if len(messages) == 0 || messages[len(messages)-1].Role != "assistant" {
		hy4Turn(&prompt, "assistant")
		prompt.WriteString(hy4ThinkStart)
		if effort == "no_think" {
			prompt.WriteString(hy4ThinkEnd)
		}
	}
	return prompt.String(), nil
}

func hy4Turn(prompt *strings.Builder, role string) {
	prompt.WriteString(hy4TurnStart)
	prompt.WriteString(role)
	prompt.WriteString(hy4TurnRoleEnd)
}

func (r *hy4FlashRenderer) content(message api.Message, offset int) (string, int) {
	if r.imageTags {
		return renderContentWithImageTags(message.Content, len(message.Images), offset)
	}
	return strings.Repeat(hy4Image, len(message.Images)) + message.Content, offset + len(message.Images)
}

func hy4ToolResults(results []api.Message, calls []api.ToolCall) []api.Message {
	if len(results) != len(calls) {
		return results
	}
	byID := make(map[string]api.Message, len(results))
	for _, result := range results {
		byID[result.ToolCallID] = result
	}
	if len(byID) != len(results) {
		return results
	}
	ordered := make([]api.Message, 0, len(results))
	for _, call := range calls {
		result, ok := byID[call.ID]
		if !ok || call.ID == "" {
			return results
		}
		ordered = append(ordered, result)
	}
	return ordered
}

func writeHy4Tools(prompt *strings.Builder, tools []api.Tool) error {
	prompt.WriteString("# Tools\n\n")
	prompt.WriteString("You may call one or more functions to assist with the user query.\n\n")
	prompt.WriteString("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n")
	for i, tool := range tools {
		if i > 0 {
			prompt.WriteByte('\n')
		}
		definition, err := PythonJSON(tool.Function, 0)
		if err != nil {
			return fmt.Errorf("encode Hy4 tool %q: %w", tool.Function.Name, err)
		}
		prompt.WriteString(definition)
	}
	prompt.WriteString("\n</tools>\n\nFor function call returns, you should first print ")
	prompt.WriteString(hy4CallsStart)
	prompt.WriteString("\nFor each function call, you should return object like:\n")
	prompt.WriteString(hy4CallStart)
	prompt.WriteString("{function-name}")
	for _, n := range []string{"1", "2"} {
		prompt.WriteString(hy4ArgKeyStart + "{arg-key-" + n + "}" + hy4ArgKeyEnd)
		prompt.WriteString(hy4ArgValueStart + "{arg-value-" + n + "}" + hy4ArgValueEnd)
	}
	prompt.WriteString("..." + hy4CallEnd)
	prompt.WriteString("\nAt the end of function call returns, you should print " + hy4CallsEnd)
	return nil
}

func writeHy4Calls(prompt *strings.Builder, calls []api.ToolCall) error {
	prompt.WriteString(hy4CallsStart)
	for _, call := range calls {
		prompt.WriteString(hy4CallStart)
		prompt.WriteString(call.Function.Name)
		for key, value := range call.Function.Arguments.All() {
			valueText, ok := value.(string)
			if !ok {
				var err error
				valueText, err = PythonJSON(value, 0)
				if err != nil {
					return fmt.Errorf("encode Hy4 argument %q: %w", key, err)
				}
			}
			prompt.WriteString(hy4ArgKeyStart + key + hy4ArgKeyEnd)
			prompt.WriteString(hy4ArgValueStart + valueText + hy4ArgValueEnd)
		}
		prompt.WriteString(hy4CallEnd)
	}
	prompt.WriteString(hy4CallsEnd)
	return nil
}
