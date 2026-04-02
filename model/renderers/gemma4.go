package renderers

import (
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
	hasSystemRole := len(messages) > 0 && (messages[0].Role == "system" || messages[0].Role == "developer")
	if hasSystemRole {
		systemMessage = messages[0].Content
		loopMessages = messages[1:]
	} else {
		loopMessages = messages
	}

	// Emit system turn if there's a system/developer role, tools, or thinking.
	hasThink := thinkValue != nil && thinkValue.Bool()
	if hasSystemRole || len(tools) > 0 || hasThink {
		sb.WriteString("<|turn>system\n")
		if hasThink {
			sb.WriteString("<|think|>")
		}
		if systemMessage != "" {
			sb.WriteString(strings.TrimSpace(systemMessage))
		}
		for _, tool := range tools {
			sb.WriteString(r.renderToolDeclaration(tool))
		}
		sb.WriteString("<turn|>\n")
	}

	// Each message gets its own <|turn>role\n ... <turn|>\n block,
	// matching the HF chat template exactly.
	for _, message := range loopMessages {
		switch message.Role {
		case "user":
			sb.WriteString("<|turn>user\n")
			r.renderContent(&sb, message, &imageOffset, true)
			sb.WriteString("<turn|>\n")

		case "assistant":
			sb.WriteString("<|turn>model\n")
			// Tool calls come before content (matching HF template order)
			for _, tc := range message.ToolCalls {
				sb.WriteString(r.formatToolCall(tc))
			}
			// Strip thinking from history (matching HF strip_thinking macro)
			if message.Content != "" {
				sb.WriteString(stripThinking(message.Content))
			}
			sb.WriteString("<turn|>\n")

		case "tool":
			sb.WriteString("<|turn>tool\n")
			sb.WriteString(strings.TrimSpace(message.Content))
			sb.WriteString("<turn|>\n")

		default:
			sb.WriteString("<|turn>" + message.Role + "\n")
			sb.WriteString(strings.TrimSpace(message.Content))
			sb.WriteString("<turn|>\n")
		}
	}

	// Generation prompt
	sb.WriteString("<|turn>model\n")

	return sb.String(), nil
}

// stripThinking removes <|channel>...<channel|> thinking blocks from content,
// matching the HF chat template's strip_thinking macro.
func stripThinking(text string) string {
	var result strings.Builder
	for {
		start := strings.Index(text, "<|channel>")
		if start == -1 {
			result.WriteString(text)
			break
		}
		result.WriteString(text[:start])
		end := strings.Index(text[start:], "<channel|>")
		if end == -1 {
			break
		}
		text = text[start+end+len("<channel|>"):]
	}
	return strings.TrimSpace(result.String())
}

// renderContent writes a message's content, interleaving [img-N] tags for images.
// When trim is true, leading/trailing whitespace is stripped (matching the Jinja2
// template's | trim filter applied to non-model content).
func (r *Gemma4Renderer) renderContent(sb *strings.Builder, msg api.Message, imageOffset *int, trim bool) {
	if len(msg.Images) > 0 && r.useImgTags {
		for range msg.Images {
			sb.WriteString(fmt.Sprintf("[img-%d]", *imageOffset))
			*imageOffset++
		}
	}
	content := msg.Content
	if trim {
		content = strings.TrimSpace(content)
	}
	sb.WriteString(content)
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

		hasContent := false
		if prop.Description != "" {
			sb.WriteString("description:" + g4Q + prop.Description + g4Q)
			hasContent = true
		}

		if len(prop.Type) > 0 {
			typeName := strings.ToUpper(prop.Type[0])

			switch typeName {
			case "STRING":
				if len(prop.Enum) > 0 {
					if hasContent {
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
					hasContent = true
				}

			case "OBJECT":
				// Render nested properties recursively.
				// Note: the leading comma is hardcoded (matching the template),
				// and this does NOT set hasContent — the comma before type:
				// depends only on whether description was present.
				sb.WriteString(",properties:{")
				if prop.Properties != nil && prop.Properties.Len() > 0 {
					r.writeProperties(sb, prop.Properties)
				}
				sb.WriteString("}")
				if len(prop.Required) > 0 {
					sb.WriteString(",required:[")
					for j, req := range prop.Required {
						if j > 0 {
							sb.WriteString(",")
						}
						sb.WriteString(g4Q + req + g4Q)
					}
					sb.WriteString("]")
				}

			case "ARRAY":
				// Render items specification.
				// Same as OBJECT: leading comma is hardcoded, does NOT set hasContent.
				if items, ok := prop.Items.(map[string]any); ok && len(items) > 0 {
					sb.WriteString(",items:{")
					r.writeItemsSpec(sb, items)
					sb.WriteString("}")
				}
			}

			if hasContent {
				sb.WriteString(",")
			}
			sb.WriteString("type:" + g4Q + typeName + g4Q)
		}

		sb.WriteString("}")
	}
}

// writeItemsSpec renders the items specification for array-type properties,
// matching the Jinja2 template's dictsort iteration over items.
func (r *Gemma4Renderer) writeItemsSpec(sb *strings.Builder, items map[string]any) {
	keys := make([]string, 0, len(items))
	for k := range items {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	first := true
	for _, key := range keys {
		value := items[key]
		if value == nil {
			continue
		}
		if !first {
			sb.WriteString(",")
		}
		first = false

		switch key {
		case "type":
			if s, ok := value.(string); ok {
				sb.WriteString("type:" + g4Q + strings.ToUpper(s) + g4Q)
			}
		default:
			sb.WriteString(key + ":" + r.formatArgValue(value))
		}
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
