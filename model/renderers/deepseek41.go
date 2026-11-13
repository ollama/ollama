package renderers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

type DeepSeekV41Renderer struct{}

func (*DeepSeekV41Renderer) LeadingBOS() string { return "<｜begin▁of▁sentence｜>" }

func (*DeepSeekV41Renderer) Thinking() *model.Thinking {
	return &model.Thinking{Values: []any{false, true, "low", "medium", "high", "max"}, Default: true}
}

type deepseek41Block struct {
	text       string
	images     int
	toolResult bool
	callID     string
}

type deepseek41Turn struct {
	message api.Message
	blocks  []deepseek41Block
}

func deepseek41Turns(messages []api.Message) []deepseek41Turn {
	turns := make([]deepseek41Turn, 0, len(messages))
	var callIDs []string
	for _, message := range messages {
		switch message.Role {
		case "user", "tool":
			block := deepseek41Block{text: message.Content, images: len(message.Images)}
			if message.Role == "tool" {
				block.text = "<tool_result>" + message.Content + "</tool_result>"
				block.toolResult, block.callID = true, message.ToolCallID
			}
			if len(turns) > 0 && turns[len(turns)-1].message.Role == "user" {
				turns[len(turns)-1].blocks = append(turns[len(turns)-1].blocks, block)
			} else {
				turns = append(turns, deepseek41Turn{message: api.Message{Role: "user"}, blocks: []deepseek41Block{block}})
			}
		default:
			turns = append(turns, deepseek41Turn{message: message})
		}
	}
	for i := range turns {
		turn := &turns[i]
		if turn.message.Role == "assistant" && len(turn.message.ToolCalls) > 0 {
			callIDs = callIDs[:0]
			for _, call := range turn.message.ToolCalls {
				callIDs = append(callIDs, call.ID)
			}
		}
		if turn.message.Role != "user" {
			continue
		}
		var results []deepseek41Block
		for _, block := range turn.blocks {
			if block.toolResult {
				results = append(results, block)
			}
		}
		if len(results) < 2 {
			continue
		}
		order := func(b deepseek41Block) int {
			for n, id := range callIDs {
				if id != "" && id == b.callID {
					return n
				}
			}
			return 0
		}
		slices.SortStableFunc(results, func(a, b deepseek41Block) int { return order(a) - order(b) })
		for j, block := range turn.blocks {
			if block.toolResult {
				turn.blocks[j], results = results[0], results[1:]
			}
		}
	}
	return turns
}

func deepseek41Thinking(think *api.ThinkValue) (bool, int, error) {
	if think == nil || think.Value == nil {
		return true, 75, nil
	}
	switch think.Value {
	case true, "high":
		return true, 75, nil
	case false:
		return false, 75, nil
	case "low":
		return true, 50, nil
	case "medium":
		return true, 62, nil
	case "max":
		return true, 100, nil
	default:
		return false, 0, fmt.Errorf("unsupported DeepSeek V4.1 thinking level %v", think.Value)
	}
}

func (r *DeepSeekV41Renderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	thinking, budget, err := deepseek41Thinking(think)
	if err != nil {
		return "", err
	}
	for _, message := range messages {
		if len(message.ToolCalls) > 0 && message.Role != "assistant" {
			return "", fmt.Errorf("DeepSeek V4.1 tool calls require an assistant turn")
		}
		if len(message.Images) > 0 && message.Role == "tool" {
			return "", fmt.Errorf("DeepSeek V4.1 tool images require ordered content blocks")
		}
	}
	turns := deepseek41Turns(messages)
	if len(tools) > 0 && (len(turns) == 0 || turns[0].message.Role != "system") {
		turns = append([]deepseek41Turn{{message: api.Message{Role: "system"}}}, turns...)
	}
	lastUser := -1
	for i, turn := range turns {
		if turn.message.Role == "user" || turn.message.Role == "system" && i > 0 {
			lastUser = i
		}
	}
	var out strings.Builder
	imageOffset := 0
	out.WriteString(r.LeadingBOS())
	for i, turn := range turns {
		message := turn.message
		if strings.Contains(message.Content, "<｜deepseek_image｜>") || strings.Contains(message.Thinking, "<｜deepseek_image｜>") {
			return "", fmt.Errorf("DeepSeek V4.1 image special token cannot appear in message text")
		}
		if i == 0 && (thinking || message.Role == "system") {
			out.WriteString("<｜System｜>")
		}
		if i == 0 && thinking {
			fmt.Fprintf(&out, "Reasoning Effort: %d (range 1-100, the higher the value, the more thorough the reasoning)\n\n", budget)
		}
		switch message.Role {
		case "system":
			if len(message.Images) > 0 {
				return "", fmt.Errorf("DeepSeek V4.1 images require a user turn")
			}
			if i > 0 {
				out.WriteString("<｜System｜>")
			}
			out.WriteString(message.Content)
			if i == 0 && len(tools) > 0 {
				out.WriteString("\n\n")
				if err := deepseek41WriteTools(&out, tools); err != nil {
					return "", err
				}
			}
		case "user":
			out.WriteString("<｜User｜>")
			for j, block := range turn.blocks {
				if j > 0 {
					out.WriteString("\n\n")
				}
				if strings.Contains(block.text, "<｜deepseek_image｜>") {
					return "", fmt.Errorf("DeepSeek V4.1 image special token cannot appear in message text")
				}
				content := block.text
				if block.images > 0 {
					content = renderDeepSeekImages(content, imageOffset, block.images)
					imageOffset += block.images
				}
				out.WriteString(content)
			}
		case "assistant":
			if len(message.Images) > 0 {
				return "", fmt.Errorf("DeepSeek V4.1 images require a user turn")
			}
			if thinking && (len(tools) > 0 || i > lastUser) {
				out.WriteString(message.Thinking)
				out.WriteString("</think>")
			}
			out.WriteString(message.Content)
			if len(message.ToolCalls) > 0 {
				if err := deepseek41WriteCalls(&out, message.ToolCalls); err != nil {
					return "", err
				}
			}
			out.WriteString("<｜end▁of▁sentence｜>")
		default:
			return "", fmt.Errorf("DeepSeek V4.1 does not support role %q", message.Role)
		}
		if i == len(turns)-1 || turns[i+1].message.Role == "assistant" {
			if message.Role == "user" || message.Role == "system" && i > 0 {
				out.WriteString("<｜Assistant｜>")
				if thinking && (len(tools) > 0 || i >= lastUser) {
					out.WriteString("<think>")
				} else {
					out.WriteString("</think>")
				}
			}
		}
	}
	return out.String(), nil
}

func deepseek41WriteCalls(out *strings.Builder, calls []api.ToolCall) error {
	out.WriteString("\n\n<｜DSML｜ calls>\n")
	for i, call := range calls {
		if i > 0 {
			out.WriteByte('\n')
		}
		if err := deepseek41ValidateToolName(call.Function.Name); err != nil {
			return err
		}
		out.WriteString("<｜DSML｜ invoke name=\"")
		out.WriteString(call.Function.Name)
		out.WriteString("\">\n")
		var params []string
		for key, value := range call.Function.Arguments.All() {
			var data string
			stringParam := "false"
			if text, ok := value.(string); ok {
				data, stringParam = text, "true"
			} else {
				var err error
				data, err = deepseek41JSON(value)
				if err != nil {
					return err
				}
			}
			params = append(params, "<｜DSML｜ parameter name=\""+key+"\" string=\""+stringParam+"\">"+data+"</｜DSML｜ parameter>")
		}
		out.WriteString(strings.Join(params, "\n"))
		out.WriteString("\n</｜DSML｜ invoke>")
	}
	out.WriteString("\n</｜DSML｜ calls>")
	return nil
}

func deepseek41WriteTools(out *strings.Builder, tools []api.Tool) error {
	out.WriteString(deepseek41ToolGuide)
	for i, tool := range tools {
		if err := deepseek41ValidateToolName(tool.Function.Name); err != nil {
			return err
		}
		if i > 0 {
			out.WriteByte('\n')
		}
		encoded, err := deepseek41JSON(tool.Function)
		if err != nil {
			return err
		}
		out.WriteString(encoded)
	}
	out.WriteString("\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n")
	return nil
}

func deepseek41JSON(value any) (string, error) {
	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return "", err
	}
	return string(AddJSONSpaces(bytes.TrimSuffix(buf.Bytes(), []byte("\n")))), nil
}

func deepseek41ValidateToolName(name string) error {
	if _, local, qualified := strings.Cut(name, "::"); qualified && strings.Contains(local, "::") {
		return fmt.Errorf("DeepSeek V4.1 tool name %q has more than one namespace separator", name)
	}
	return nil
}

const deepseek41ToolGuide = "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜ calls>\" block like the following:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n...\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n"

// The publisher joins text and image content blocks with blank lines. Ollama
// carries images separately, with optional numbered or generic text markers.
func renderDeepSeekImages(content string, offset, count int) string {
	tags := make([]string, count)
	for i := range tags {
		tags[i] = fmt.Sprintf("[img-%d]", offset+i)
		if !strings.Contains(content, tags[i]) {
			content = strings.Replace(content, "[img]", tags[i], 1)
		}
	}

	used := make([]bool, count)
	parts := make([]string, 0, count*2+1)
	for cursor := 0; cursor < len(content); {
		position, image := len(content), -1
		for i, tag := range tags {
			if at := strings.Index(content[cursor:], tag); at >= 0 && cursor+at < position {
				position, image = cursor+at, i
			}
		}
		if image < 0 {
			parts = append(parts, content[cursor:])
			break
		}
		if position > cursor {
			parts = append(parts, content[cursor:position])
		}
		parts = append(parts, tags[image])
		used[image] = true
		cursor = position + len(tags[image])
	}
	for i, tag := range tags {
		if !used[i] {
			parts = append(parts, tag)
		}
	}
	return strings.Join(parts, "\n\n")
}
