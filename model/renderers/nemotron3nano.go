package renderers

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
)

type Nemotron3NanoRenderer struct{}

func (r *Nemotron3NanoRenderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder
	imageOffset := 0

	enableThinking := r.resolveThinking(messages, thinkValue)

	// Extract system message if present
	var systemMessage string
	var loopMessages []api.Message
	if len(messages) > 0 && messages[0].Role == "system" {
		systemMessage = r.sanitizeSystemMessage(messages[0].Content)
		loopMessages = messages[1:]
	} else {
		loopMessages = messages
	}

	// Find last user message index for thinking truncation
	lastUserIdx := -1
	for i, msg := range loopMessages {
		if msg.Role == "user" {
			lastUserIdx = i
		}
	}

	sb.WriteString("\n\n\n")
	sb.WriteString("<|im_start|>system\n")
	if systemMessage != "" {
		sb.WriteString(systemMessage)
	}

	if len(tools) > 0 {
		if systemMessage != "" {
			sb.WriteString("\n\n")
		}
		sb.WriteString(r.renderTools(tools))
	}
	sb.WriteString("<|im_end|>\n\n")

	for i, message := range loopMessages {
		switch message.Role {
		case "assistant":
			content := r.buildContent(message)
			shouldTruncate := i < lastUserIdx

			if len(message.ToolCalls) > 0 {
				sb.WriteString("<|im_start|>assistant\n")
				sb.WriteString(r.formatToolCallContent(content, shouldTruncate))
				r.writeToolCalls(&sb, message.ToolCalls)
				sb.WriteString("<|im_end|>\n")
			} else {
				formatted := r.formatAssistantContent(content, shouldTruncate)
				sb.WriteString("<|im_start|>assistant\n")
				sb.WriteString(formatted)
				sb.WriteString("<|im_end|>\n")
			}

		case "user", "system":
			sb.WriteString("<|im_start|>" + message.Role + "\n")
			sb.WriteString(r.renderMessageContent(message, imageOffset))
			imageOffset += len(message.Images)
			sb.WriteString("<|im_end|>\n")

		case "tool":
			// Check if previous message was also a tool message
			prevWasTool := i > 0 && loopMessages[i-1].Role == "tool"
			nextIsTool := i+1 < len(loopMessages) && loopMessages[i+1].Role == "tool"

			if !prevWasTool {
				sb.WriteString("<|im_start|>user\n")
			}
			sb.WriteString("<tool_response>\n")
			sb.WriteString(message.Content)
			sb.WriteString("\n</tool_response>\n")

			if !nextIsTool {
				sb.WriteString("<|im_end|>\n")
			}

		default:
			sb.WriteString("<|im_start|>" + message.Role + "\n" + message.Content + "<|im_end|>\n")
		}
	}

	sb.WriteString("\n")

	// Add generation prompt
	if enableThinking {
		sb.WriteString("<|im_start|>assistant\n<think>\n")
	} else {
		sb.WriteString("<|im_start|>assistant\n<think></think>")
	}

	return sb.String(), nil
}

func (r *Nemotron3NanoRenderer) renderTools(tools []api.Tool) string {
	var sb strings.Builder
	sb.WriteString("# Tools\n\nYou have access to the following functions:\n\n<tools>")

	for _, tool := range tools {
		fn := tool.Function
		sb.WriteString("\n<function>\n<name>" + fn.Name + "</name>")

		if fn.Description != "" {
			sb.WriteString("\n<description>" + strings.TrimSpace(fn.Description) + "</description>")
		}

		sb.WriteString("\n<parameters>")
		if fn.Parameters.Properties != nil {
			for paramName, paramFields := range fn.Parameters.Properties.All() {
				sb.WriteString("\n<parameter>")
				sb.WriteString("\n<name>" + paramName + "</name>")

				if len(paramFields.Type) > 0 {
					sb.WriteString("\n<type>" + r.formatPropertyType(paramFields.Type) + "</type>")
				}

				if paramFields.Description != "" {
					sb.WriteString("\n<description>" + strings.TrimSpace(paramFields.Description) + "</description>")
				}

				if len(paramFields.Enum) > 0 {
					sb.WriteString("\n<enum>" + r.pythonJSON(paramFields.Enum) + "</enum>")
				}

				r.renderToolPropertyExtraKeys(&sb, paramFields)
				sb.WriteString("\n</parameter>")
			}
		}

		r.renderToolParameterExtraKeys(&sb, fn.Parameters)
		if len(fn.Parameters.Required) > 0 {
			sb.WriteString("\n<required>" + r.pythonJSON(fn.Parameters.Required) + "</required>")
		}

		sb.WriteString("\n</parameters>")
		sb.WriteString("\n</function>")
	}

	sb.WriteString("\n</tools>")

	sb.WriteString("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
		"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
		"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" +
		"</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
		"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
		"- Required parameters MUST be specified\n" +
		"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
		"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")

	return sb.String()
}

func (r *Nemotron3NanoRenderer) buildContent(message api.Message) string {
	content := nemotron3NanoRenderContent(message.Content)
	if message.Thinking != "" {
		return "<think>\n" + message.Thinking + "\n</think>\n" + content
	}
	if !strings.Contains(content, "<think>") && !strings.Contains(content, "</think>") {
		return "<think></think>" + content
	}
	return content
}

func (r *Nemotron3NanoRenderer) formatAssistantContent(content string, truncate bool) string {
	if !truncate {
		return strings.TrimSpace(content)
	}

	c := content
	if strings.Contains(c, "<think>") && strings.Contains(c, "</think>") {
		parts := strings.Split(c, "</think>")
		c = "<think></think>" + parts[len(parts)-1]
	}
	return strings.TrimSpace(c)
}

func (r *Nemotron3NanoRenderer) formatToolCallContent(content string, truncate bool) string {
	if strings.TrimSpace(content) == "" {
		return "<think></think>"
	}

	if !truncate {
		return strings.TrimSpace(content) + "\n"
	}

	c := content
	if strings.Contains(c, "</think>") {
		parts := strings.Split(c, "</think>")
		c = parts[len(parts)-1]
	} else if strings.Contains(c, "<think>") {
		parts := strings.Split(c, "<think>")
		c = parts[0]
	}
	c = "<think></think>" + strings.TrimSpace(c)

	return strings.TrimSpace(c) + "\n"
}

func (r *Nemotron3NanoRenderer) writeToolCalls(sb *strings.Builder, toolCalls []api.ToolCall) {
	for _, tc := range toolCalls {
		sb.WriteString("<tool_call>\n<function=" + tc.Function.Name + ">\n")
		for name, value := range tc.Function.Arguments.All() {
			sb.WriteString("<parameter=" + name + ">\n" + r.formatArgValue(value) + "\n</parameter>\n")
		}
		sb.WriteString("</function>\n</tool_call>\n")
	}
}

func (r *Nemotron3NanoRenderer) formatArgValue(value any) string {
	switch v := value.(type) {
	case map[string]any, []any:
		return r.pythonJSON(v)
	default:
		return fmt.Sprintf("%v", v)
	}
}

func (r *Nemotron3NanoRenderer) renderMessageContent(message api.Message, imageOffset int) string {
	content := nemotron3NanoRenderContent(message.Content)
	if len(message.Images) == 0 {
		return content
	}

	if strings.Contains(content, "[img-") {
		return content
	}

	if strings.Contains(content, "[img]") {
		for i := range message.Images {
			content = strings.Replace(content, "[img]", fmt.Sprintf("[img-%d]", imageOffset+i), 1)
		}
		return content
	}

	var sb strings.Builder
	for i := range message.Images {
		sb.WriteString(fmt.Sprintf("[img-%d]", imageOffset+i))
	}
	sb.WriteString(content)
	return sb.String()
}

func nemotron3NanoRenderContent(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var sb strings.Builder
		for _, item := range v {
			obj, ok := item.(map[string]any)
			if !ok {
				bts, _ := json.Marshal(item)
				sb.Write(bts)
				continue
			}

			switch obj["type"] {
			case "image":
				sb.WriteString("<image>")
			case "text":
				if text, ok := obj["text"].(string); ok {
					sb.WriteString(text)
				}
			default:
				bts, _ := json.Marshal(item)
				sb.Write(bts)
			}
		}
		return sb.String()
	default:
		bts, _ := json.Marshal(v)
		return string(bts)
	}
}

func (r *Nemotron3NanoRenderer) resolveThinking(messages []api.Message, thinkValue *api.ThinkValue) bool {
	enableThinking := thinkValue == nil || thinkValue.Bool()
	for _, message := range messages {
		if message.Role != "user" && message.Role != "system" {
			continue
		}
		content := message.Content
		if strings.Contains(strings.ReplaceAll(content, "</think>", ""), "/think") {
			enableThinking = true
		} else if strings.Contains(content, "/no_think") {
			enableThinking = false
		}
	}
	return enableThinking
}

func (r *Nemotron3NanoRenderer) sanitizeSystemMessage(content string) string {
	system := nemotron3NanoRenderContent(content)
	system = strings.ReplaceAll(system, "</think>", "<_end_think>")
	system = strings.ReplaceAll(system, "/think", "")
	system = strings.ReplaceAll(system, "/no_think", "")
	system = strings.ReplaceAll(system, "<_end_think>", "</think>")
	return system
}

func (r *Nemotron3NanoRenderer) formatPropertyType(propertyType api.PropertyType) string {
	if len(propertyType) == 1 {
		return propertyType[0]
	}
	quoted := make([]string, 0, len(propertyType))
	for _, v := range propertyType {
		quoted = append(quoted, "'"+v+"'")
	}
	return "[" + strings.Join(quoted, ", ") + "]"
}

func (r *Nemotron3NanoRenderer) renderToolPropertyExtraKeys(sb *strings.Builder, prop api.ToolProperty) {
	if len(prop.AnyOf) > 0 {
		sb.WriteString("\n<anyOf>" + r.pythonJSON(prop.AnyOf) + "</anyOf>")
	}
	if prop.Items != nil {
		sb.WriteString("\n<items>" + r.pythonJSON(prop.Items) + "</items>")
	}
	if prop.Properties != nil {
		sb.WriteString("\n<properties>" + r.pythonJSON(prop.Properties) + "</properties>")
	}
	if len(prop.Required) > 0 {
		sb.WriteString("\n<required>" + r.pythonJSON(prop.Required) + "</required>")
	}
}

func (r *Nemotron3NanoRenderer) renderToolParameterExtraKeys(sb *strings.Builder, params api.ToolFunctionParameters) {
	if params.Defs != nil {
		sb.WriteString("\n<$defs>" + r.pythonJSON(params.Defs) + "</$defs>")
	}
	if params.Items != nil {
		sb.WriteString("\n<items>" + r.pythonJSON(params.Items) + "</items>")
	}
}

func (r *Nemotron3NanoRenderer) pythonJSON(v any) string {
	switch value := v.(type) {
	case nil:
		return "null"
	case string:
		return strconv.Quote(value)
	case bool:
		if value {
			return "true"
		}
		return "false"
	case int, int8, int16, int32, int64:
		return fmt.Sprintf("%d", reflect.ValueOf(value).Int())
	case uint, uint8, uint16, uint32, uint64:
		return fmt.Sprintf("%d", reflect.ValueOf(value).Uint())
	case float32, float64:
		b, _ := json.Marshal(value)
		return string(b)
	case api.PropertyType:
		return r.pythonJSON([]string(value))
	case []string:
		parts := make([]string, 0, len(value))
		for _, item := range value {
			parts = append(parts, r.pythonJSON(item))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case []any:
		parts := make([]string, 0, len(value))
		for _, item := range value {
			parts = append(parts, r.pythonJSON(item))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case []api.ToolProperty:
		parts := make([]string, 0, len(value))
		for _, item := range value {
			parts = append(parts, r.pythonJSON(item))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case map[string]any:
		keys := make([]string, 0, len(value))
		for key := range value {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		parts := make([]string, 0, len(keys))
		for _, key := range keys {
			parts = append(parts, strconv.Quote(key)+": "+r.pythonJSON(value[key]))
		}
		return "{" + strings.Join(parts, ", ") + "}"
	case *api.ToolPropertiesMap:
		if value == nil {
			return "null"
		}
		parts := make([]string, 0, value.Len())
		for key, prop := range value.All() {
			parts = append(parts, strconv.Quote(key)+": "+r.pythonJSON(prop))
		}
		return "{" + strings.Join(parts, ", ") + "}"
	case api.ToolProperty:
		parts := make([]string, 0, 6)
		if len(value.AnyOf) > 0 {
			parts = append(parts, `"anyOf": `+r.pythonJSON(value.AnyOf))
		}
		if len(value.Type) > 0 {
			if len(value.Type) == 1 {
				parts = append(parts, `"type": `+r.pythonJSON(value.Type[0]))
			} else {
				parts = append(parts, `"type": `+r.pythonJSON([]string(value.Type)))
			}
		}
		if value.Items != nil {
			parts = append(parts, `"items": `+r.pythonJSON(value.Items))
		}
		if value.Description != "" {
			parts = append(parts, `"description": `+r.pythonJSON(value.Description))
		}
		if len(value.Enum) > 0 {
			parts = append(parts, `"enum": `+r.pythonJSON(value.Enum))
		}
		if value.Properties != nil {
			parts = append(parts, `"properties": `+r.pythonJSON(value.Properties))
		}
		if len(value.Required) > 0 {
			parts = append(parts, `"required": `+r.pythonJSON(value.Required))
		}
		return "{" + strings.Join(parts, ", ") + "}"
	default:
		b, err := json.Marshal(value)
		if err != nil {
			return "null"
		}
		var generic any
		if err := json.Unmarshal(b, &generic); err != nil {
			return string(b)
		}
		return r.pythonJSON(generic)
	}
}
