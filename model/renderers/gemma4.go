package renderers

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/ollama/ollama/api"
)

// Gemma4Renderer renders prompts using Gemma 4's chat format with
// <|turn>/<turn|> markers, <|"|> string delimiters, and <|tool>/
// <|tool_call>/<|tool_response> tags for function calling.
type Gemma4Renderer struct {
	useImgTags bool
}

const (
	g4Q = `<|"|>` // Gemma 4 string delimiter
)

func (r *Gemma4Renderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder
	imageOffset := 0

	// BOS token — Gemma 4 models have add_bos_token=false in their tokenizer
	// config, so the tokenizer does not auto-prepend BOS. We must emit it
	// explicitly in the rendered prompt, matching the HF chat template.
	sb.WriteString("<bos>")
	// Extract system message if present.
	var systemMessage string
	var loopMessages []api.Message
	if len(messages) > 0 && (messages[0].Role == "system" || messages[0].Role == "developer") {
		systemMessage = messages[0].Content
		loopMessages = messages[1:]
	} else {
		loopMessages = messages
	}

	// Emit system turn if there's a system message, tools, or thinking.
	hasThink := thinkValue != nil && thinkValue.Bool()
	if systemMessage != "" || len(tools) > 0 || hasThink {
		sb.WriteString("<|turn>system\n")
		if hasThink {
			sb.WriteString("<|think|>")
		}
		if systemMessage != "" {
			sb.WriteString(systemMessage)
		}
		for _, tool := range tools {
			sb.WriteString(r.renderToolDeclaration(tool))
		}
		sb.WriteString("<turn|>\n")
	}

	// inModelTurn tracks whether we're inside an open <|turn>model block.
	// Tool responses are appended inline (no separate turn), and the model
	// turn is only closed when we see a non-tool message or reach the end.
	inModelTurn := false

	for i, message := range loopMessages {
		switch message.Role {
		case "user":
			if inModelTurn {
				// Check if the preceding content was a tool response (no <turn|>
				// between tool response and next user turn per HF reference).
				prevIsToolResponse := i > 0 && loopMessages[i-1].Role == "tool"
				if !prevIsToolResponse {
					sb.WriteString("<turn|>\n")
				}
				inModelTurn = false
			}
			sb.WriteString("<|turn>user\n")
			r.renderContent(&sb, message, &imageOffset)
			sb.WriteString("<turn|>\n")

		case "assistant":
			if inModelTurn {
				sb.WriteString("<turn|>\n")
			}
			sb.WriteString("<|turn>model\n")
			inModelTurn = true
			if message.Content != "" {
				sb.WriteString(message.Content)
			}
			for _, tc := range message.ToolCalls {
				sb.WriteString(r.formatToolCall(tc))
			}

		case "tool":
			// Tool responses are rendered inline in the preceding model turn,
			// matching the reference format from HuggingFace's chat template.
			// Format: <|tool_response>response:NAME{key:value,...}<tool_response|>
			toolName := r.findToolName(loopMessages, i)
			sb.WriteString("<|tool_response>response:" + toolName + "{")
			r.renderToolResponseContent(&sb, message.Content)
			sb.WriteString("}<tool_response|>")
			// Keep the model turn open — it will be closed when we see the
			// next non-tool message or the assistant adds content after the response.

		default:
			if inModelTurn {
				sb.WriteString("<turn|>\n")
				inModelTurn = false
			}
			sb.WriteString("<|turn>" + message.Role + "\n")
			sb.WriteString(message.Content)
			sb.WriteString("<turn|>\n")
		}
	}

	// If the last message is not an open assistant turn, add the generation prompt.
	if !inModelTurn {
		sb.WriteString("<|turn>model\n")
	}

	return sb.String(), nil
}

// renderContent writes a message's content, interleaving [img-N] tags for images.
func (r *Gemma4Renderer) renderContent(sb *strings.Builder, msg api.Message, imageOffset *int) {
	if len(msg.Images) > 0 && r.useImgTags {
		for range msg.Images {
			sb.WriteString(fmt.Sprintf("[img-%d]", *imageOffset))
			*imageOffset++
		}
	}
	sb.WriteString(msg.Content)
}

func (r *Gemma4Renderer) renderToolDeclaration(tool api.Tool) string {
	var sb strings.Builder
	fn := tool.Function

	sb.WriteString("<|tool>declaration:" + fn.Name + "{")
	sb.WriteString("description:" + g4Q + fn.Description + g4Q)

	if fn.Parameters.Properties != nil || fn.Parameters.Type != "" {
		sb.WriteString(",parameters:{")

		needsComma := false

		if fn.Parameters.Properties != nil && fn.Parameters.Properties.Len() > 0 {
			sb.WriteString("properties:{")
			r.writeProperties(&sb, fn.Parameters.Properties)
			sb.WriteString("}")
			needsComma = true
		}

		if len(fn.Parameters.Required) > 0 {
			if needsComma {
				sb.WriteString(",")
			}
			sb.WriteString("required:[")
			for i, req := range fn.Parameters.Required {
				if i > 0 {
					sb.WriteString(",")
				}
				sb.WriteString(g4Q + req + g4Q)
			}
			sb.WriteString("]")
			needsComma = true
		}

		if fn.Parameters.Type != "" {
			if needsComma {
				sb.WriteString(",")
			}
			sb.WriteString("type:" + g4Q + strings.ToUpper(fn.Parameters.Type) + g4Q)
		}

		sb.WriteString("}")
	}

	sb.WriteString("}<tool|>")
	return sb.String()
}

func (r *Gemma4Renderer) writeProperties(sb *strings.Builder, props *api.ToolPropertiesMap) {
	keys := make([]string, 0, props.Len())
	for k := range props.All() {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	first := true
	for _, name := range keys {
		prop, _ := props.Get(name)
		if !first {
			sb.WriteString(",")
		}
		first = false

		sb.WriteString(name + ":{")
		if prop.Description != "" {
			sb.WriteString("description:" + g4Q + prop.Description + g4Q)
		}
		if len(prop.Enum) > 0 {
			if prop.Description != "" {
				sb.WriteString(",")
			}
			sb.WriteString("enum:[")
			for j, e := range prop.Enum {
				if j > 0 {
					sb.WriteString(",")
				}
				sb.WriteString(g4Q + fmt.Sprintf("%v", e) + g4Q)
			}
			sb.WriteString("]")
		}
		if len(prop.Type) > 0 {
			if prop.Description != "" || len(prop.Enum) > 0 {
				sb.WriteString(",")
			}
			sb.WriteString("type:" + g4Q + strings.ToUpper(prop.Type[0]) + g4Q)
		}
		sb.WriteString("}")
	}
}

func (r *Gemma4Renderer) formatToolCall(tc api.ToolCall) string {
	var sb strings.Builder
	sb.WriteString("<|tool_call>call:" + tc.Function.Name + "{")

	keys := make([]string, 0, tc.Function.Arguments.Len())
	for k := range tc.Function.Arguments.All() {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	first := true
	for _, key := range keys {
		value, _ := tc.Function.Arguments.Get(key)
		if !first {
			sb.WriteString(",")
		}
		first = false
		sb.WriteString(key + ":" + r.formatArgValue(value))
	}

	sb.WriteString("}<tool_call|>")
	return sb.String()
}

func (r *Gemma4Renderer) formatArgValue(value any) string {
	switch v := value.(type) {
	case string:
		return g4Q + v + g4Q
	case bool:
		if v {
			return "true"
		}
		return "false"
	case float64:
		if v == float64(int64(v)) {
			return fmt.Sprintf("%d", int64(v))
		}
		return fmt.Sprintf("%v", v)
	case int, int64, int32:
		return fmt.Sprintf("%d", v)
	case map[string]any:
		return r.formatMapValue(v)
	case []any:
		return r.formatArrayValue(v)
	default:
		return fmt.Sprintf("%v", v)
	}
}

func (r *Gemma4Renderer) formatMapValue(m map[string]any) string {
	var sb strings.Builder
	sb.WriteString("{")

	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	first := true
	for _, key := range keys {
		if !first {
			sb.WriteString(",")
		}
		first = false
		sb.WriteString(key + ":" + r.formatArgValue(m[key]))
	}

	sb.WriteString("}")
	return sb.String()
}

func (r *Gemma4Renderer) formatArrayValue(arr []any) string {
	var sb strings.Builder
	sb.WriteString("[")
	for i, item := range arr {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(r.formatArgValue(item))
	}
	sb.WriteString("]")
	return sb.String()
}

// renderToolResponseContent renders tool response content in Gemma 4 format.
// If the content is valid JSON, it renders each field as key:value pairs with
// proper type formatting (strings get <|"|> delimiters, numbers/bools are bare).
// If not valid JSON, wraps the entire content as a single "value" string.
func (r *Gemma4Renderer) renderToolResponseContent(sb *strings.Builder, content string) {
	// Try to parse as JSON object.
	var obj map[string]any
	if err := json.Unmarshal([]byte(content), &obj); err == nil {
		keys := make([]string, 0, len(obj))
		for k := range obj {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		first := true
		for _, key := range keys {
			if !first {
				sb.WriteString(",")
			}
			first = false
			sb.WriteString(key + ":" + r.formatArgValue(obj[key]))
		}
		return
	}

	// Not JSON — wrap as a single string value.
	sb.WriteString("value:" + g4Q + content + g4Q)
}

// findToolName walks backwards from tool message index to find the matching tool call name.
func (r *Gemma4Renderer) findToolName(messages []api.Message, toolIdx int) string {
	for j := toolIdx - 1; j >= 0; j-- {
		if messages[j].Role == "assistant" && len(messages[j].ToolCalls) > 0 {
			toolOffset := 0
			for k := j + 1; k < toolIdx; k++ {
				if messages[k].Role == "tool" {
					toolOffset++
				}
			}
			if toolOffset < len(messages[j].ToolCalls) {
				return messages[j].ToolCalls[toolOffset].Function.Name
			}
			break
		}
	}
	return ""
}
