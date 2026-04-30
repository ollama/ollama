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
	useImgTags          bool
	emptyBlockOnNothink bool
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
			sb.WriteString("<|think|>\n")
		}
		if systemMessage != "" {
			sb.WriteString(strings.TrimSpace(systemMessage))
		}
		for _, tool := range tools {
			sb.WriteString(r.renderToolDeclaration(tool))
		}
		sb.WriteString("<turn|>\n")
	}

	lastUserIdx := -1
	for i, message := range loopMessages {
		if message.Role == "user" {
			lastUserIdx = i
		}
	}

	var prevMessageType string

	// Consecutive tool messages are folded into the preceding assistant turn,
	// and adjacent assistant messages continue in the same model turn.
	for i, message := range loopMessages {
		if message.Role == "tool" {
			continue
		}

		prevMessageType = ""
		role := message.Role
		if role == "assistant" {
			role = "model"
		}

		continueSameModelTurn := role == "model" && r.previousNonToolRole(loopMessages, i) == "assistant"
		if !continueSameModelTurn {
			sb.WriteString("<|turn>" + role + "\n")
		}

		if message.Role == "assistant" && message.Thinking != "" && i > lastUserIdx && len(message.ToolCalls) > 0 {
			sb.WriteString("<|channel>thought\n")
			sb.WriteString(message.Thinking)
			sb.WriteString("\n<channel|>")
		}

		if len(message.ToolCalls) > 0 {
			for _, tc := range message.ToolCalls {
				sb.WriteString(r.formatToolCall(tc))
			}
			prevMessageType = "tool_call"
		}

		toolResponsesEmitted := false
		if len(message.ToolCalls) > 0 {
			for k := i + 1; k < len(loopMessages) && loopMessages[k].Role == "tool"; k++ {
				sb.WriteString(r.formatToolResponseBlock(r.toolResponseName(loopMessages[k], message.ToolCalls), loopMessages[k].Content))
				toolResponsesEmitted = true
				prevMessageType = "tool_response"
			}
		}

		messageHadContent := false
		switch role {
		case "model":
			if message.Content != "" || len(message.Images) > 0 {
				message.Content = stripThinking(message.Content)
				r.renderContent(&sb, message, &imageOffset, false)
				messageHadContent = r.messageHasContent(message)
			}
		default:
			r.renderContent(&sb, message, &imageOffset, true)
			message.Content = strings.TrimSpace(message.Content)
			messageHadContent = r.messageHasContent(message)
		}

		if prevMessageType == "tool_call" && !toolResponsesEmitted {
			sb.WriteString("<|tool_response>")
		} else if !(toolResponsesEmitted && !messageHadContent) {
			sb.WriteString("<turn|>\n")
		}
	}

	// Generation prompt.
	if prevMessageType != "tool_response" && prevMessageType != "tool_call" {
		sb.WriteString("<|turn>model\n")
		if r.emptyBlockOnNothink && !hasThink {
			sb.WriteString("<|channel>thought\n<channel|>")
		}
	}

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

func (r *Gemma4Renderer) previousNonToolRole(messages []api.Message, idx int) string {
	for i := idx - 1; i >= 0; i-- {
		if messages[i].Role != "tool" {
			return messages[i].Role
		}
	}
	return ""
}

func (r *Gemma4Renderer) messageHasContent(message api.Message) bool {
	return strings.TrimSpace(message.Content) != "" || len(message.Images) > 0
}

func (r *Gemma4Renderer) toolResponseName(message api.Message, toolCalls []api.ToolCall) string {
	name := message.ToolName
	if name == "" {
		name = "unknown"
	}
	if message.ToolCallID != "" {
		for _, tc := range toolCalls {
			if tc.ID == message.ToolCallID {
				name = tc.Function.Name
				break
			}
		}
	}

	return name
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
			r.writeTypedProperties(&sb, fn.Parameters.Properties)
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

func (r *Gemma4Renderer) writeTypedProperties(sb *strings.Builder, props *api.ToolPropertiesMap) {
	if props == nil || props.Len() == 0 {
		return
	}

	r.writeSchemaProperties(sb, typedSchemaPropertiesMap(props))
}

func typedSchemaPropertiesMap(props *api.ToolPropertiesMap) map[string]any {
	out := make(map[string]any, props.Len())
	for key, prop := range props.All() {
		out[key] = topLevelTypedSchemaValueFromToolProperty(prop)
	}
	return out
}

// writeSchemaItemsSpec renders the items specification for array-type properties,
// matching the Jinja2 template's dictsort iteration over items.
func (r *Gemma4Renderer) writeSchemaItemsSpec(sb *strings.Builder, items map[string]any) {
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
		case "properties":
			sb.WriteString("properties:{")
			if props, ok := r.asSchemaMap(value); ok {
				r.writeSchemaProperties(sb, props)
			}
			sb.WriteString("}")
		case "required":
			sb.WriteString("required:[")
			for i, req := range normalizeStringSlice(value) {
				if i > 0 {
					sb.WriteString(",")
				}
				sb.WriteString(g4Q + req + g4Q)
			}
			sb.WriteString("]")
		case "type":
			typeNames := normalizeTypeNames(value)
			if len(typeNames) == 1 {
				sb.WriteString("type:" + g4Q + typeNames[0] + g4Q)
			} else if len(typeNames) > 1 {
				sb.WriteString("type:[")
				for i, typeName := range typeNames {
					if i > 0 {
						sb.WriteString(",")
					}
					sb.WriteString(g4Q + typeName + g4Q)
				}
				sb.WriteString("]")
			}
		default:
			sb.WriteString(key + ":" + r.formatSchemaValue(value))
		}
	}
}

func (r *Gemma4Renderer) writeSchemaProperties(sb *strings.Builder, props map[string]any) {
	keys := make([]string, 0, len(props))
	for k := range props {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	first := true
	for _, name := range keys {
		if isSchemaStandardKey(name) {
			continue
		}
		prop, ok := r.asSchemaMap(props[name])
		if !ok {
			continue
		}
		if !first {
			sb.WriteString(",")
		}
		first = false

		sb.WriteString(name + ":{")

		addComma := false
		if description, ok := prop["description"].(string); ok && description != "" {
			sb.WriteString("description:" + g4Q + description + g4Q)
			addComma = true
		}

		typeNames := normalizeTypeNames(prop["type"])
		typeName := ""
		if len(typeNames) > 0 {
			typeName = typeNames[0]
		}

		switch typeName {
		case "STRING":
			if enumValues := normalizeSlice(prop["enum"]); len(enumValues) > 0 {
				if addComma {
					sb.WriteString(",")
				} else {
					addComma = true
				}
				sb.WriteString("enum:[")
				for i, value := range enumValues {
					if i > 0 {
						sb.WriteString(",")
					}
					sb.WriteString(g4Q + fmt.Sprintf("%v", value) + g4Q)
				}
				sb.WriteString("]")
			}
		case "ARRAY":
			if items, ok := r.asSchemaMap(prop["items"]); ok && len(items) > 0 {
				if addComma {
					sb.WriteString(",")
				} else {
					addComma = true
				}
				sb.WriteString("items:{")
				r.writeSchemaItemsSpec(sb, items)
				sb.WriteString("}")
			}
		}

		if nullable, ok := prop["nullable"].(bool); ok && nullable {
			if addComma {
				sb.WriteString(",")
			} else {
				addComma = true
			}
			sb.WriteString("nullable:true")
		}

		if typeName == "OBJECT" {
			if nestedProps, ok := r.asSchemaMap(prop["properties"]); ok {
				if addComma {
					sb.WriteString(",")
				} else {
					addComma = true
				}
				sb.WriteString("properties:{")
				r.writeSchemaProperties(sb, nestedProps)
				sb.WriteString("}")
			} else {
				if addComma {
					sb.WriteString(",")
				} else {
					addComma = true
				}
				sb.WriteString("properties:{")
				r.writeSchemaProperties(sb, prop)
				sb.WriteString("}")
			}

			required := normalizeStringSlice(prop["required"])
			if len(required) > 0 {
				if addComma {
					sb.WriteString(",")
				} else {
					addComma = true
				}
				sb.WriteString("required:[")
				for i, req := range required {
					if i > 0 {
						sb.WriteString(",")
					}
					sb.WriteString(g4Q + req + g4Q)
				}
				sb.WriteString("]")
			}
		}

		if len(typeNames) > 0 {
			if addComma {
				sb.WriteString(",")
			}
			if len(typeNames) == 1 {
				sb.WriteString("type:" + g4Q + typeNames[0] + g4Q)
			} else {
				sb.WriteString("type:[")
				for i, name := range typeNames {
					if i > 0 {
						sb.WriteString(",")
					}
					sb.WriteString(g4Q + name + g4Q)
				}
				sb.WriteString("]")
			}
		}

		sb.WriteString("}")
	}
}

func (r *Gemma4Renderer) asSchemaMap(value any) (map[string]any, bool) {
	switch v := value.(type) {
	case map[string]any:
		return v, true
	case *api.ToolPropertiesMap:
		if v == nil {
			return nil, false
		}
		out := make(map[string]any, v.Len())
		for key, prop := range v.All() {
			out[key] = schemaValueFromToolProperty(prop)
		}
		return out, true
	case api.ToolProperty:
		return schemaValueFromToolProperty(v), true
	default:
		return nil, false
	}
}

func schemaValueFromToolProperty(prop api.ToolProperty) map[string]any {
	out := make(map[string]any)
	if len(prop.Type) > 0 {
		if len(prop.Type) == 1 {
			out["type"] = prop.Type[0]
		} else {
			out["type"] = []string(prop.Type)
		}
	} else if unionTypes, ok := simpleAnyOfTypes(prop); ok {
		if len(unionTypes) == 1 {
			out["type"] = unionTypes[0]
		} else {
			out["type"] = []string(unionTypes)
		}
	}
	if prop.Description != "" {
		out["description"] = prop.Description
	}
	if len(prop.Enum) > 0 {
		out["enum"] = prop.Enum
	}
	if prop.Items != nil {
		out["items"] = prop.Items
	}
	if prop.Properties != nil {
		out["properties"] = prop.Properties
	}
	if len(prop.Required) > 0 {
		out["required"] = prop.Required
	}
	return out
}

func topLevelTypedSchemaValueFromToolProperty(prop api.ToolProperty) map[string]any {
	out := make(map[string]any)
	if len(prop.Type) > 0 {
		// api.ToolProperty intentionally models nullability through type unions
		// that include "null" rather than OpenAPI 3.0's nullable:true keyword.
		// Gemma's template accepts nullable:true as well, but our typed top-level
		// tool properties do not carry that field. For multi-type unions, the
		// template stringifies the uppercase list rather than emitting a structured
		// type array. That is odd, but we match upstream here.
		out["type"] = upstreamTypedPropertyTypeValue(prop.Type)
	} else if unionTypes, ok := simpleAnyOfTypes(prop); ok {
		// Gemma's declaration format does not have a dedicated anyOf construct, so
		// we lower simple unions of bare type branches into the same typed union
		// form used for api.PropertyType.
		out["type"] = upstreamTypedPropertyTypeValue(unionTypes)
	}
	if prop.Description != "" {
		out["description"] = prop.Description
	}
	if len(prop.Enum) > 0 {
		out["enum"] = prop.Enum
	}
	if prop.Items != nil {
		out["items"] = prop.Items
	}
	if prop.Properties != nil {
		out["properties"] = typedSchemaPropertiesMap(prop.Properties)
	}
	if len(prop.Required) > 0 {
		out["required"] = prop.Required
	}
	return out
}

func upstreamTypedPropertyTypeValue(types api.PropertyType) string {
	if len(types) == 1 {
		return types[0]
	}

	var sb strings.Builder
	sb.WriteString("[")
	for i, typ := range types {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString("'" + strings.ToUpper(typ) + "'")
	}
	sb.WriteString("]")
	return sb.String()
}

func simpleAnyOfTypes(prop api.ToolProperty) (api.PropertyType, bool) {
	if len(prop.AnyOf) == 0 {
		return nil, false
	}

	var out api.PropertyType
	seen := make(map[string]struct{})
	for _, branch := range prop.AnyOf {
		if !isBareTypeOnlyToolProperty(branch) || len(branch.Type) == 0 {
			return nil, false
		}
		for _, typ := range branch.Type {
			if _, ok := seen[typ]; ok {
				continue
			}
			seen[typ] = struct{}{}
			out = append(out, typ)
		}
	}

	return out, len(out) > 0
}

func isBareTypeOnlyToolProperty(prop api.ToolProperty) bool {
	return len(prop.AnyOf) == 0 &&
		len(prop.Type) > 0 &&
		prop.Items == nil &&
		prop.Description == "" &&
		len(prop.Enum) == 0 &&
		prop.Properties == nil &&
		len(prop.Required) == 0
}

func isSchemaStandardKey(key string) bool {
	switch key {
	case "description", "type", "properties", "required", "nullable":
		return true
	default:
		return false
	}
}

func normalizeTypeNames(value any) []string {
	switch v := value.(type) {
	case string:
		return []string{strings.ToUpper(v)}
	case []string:
		out := make([]string, 0, len(v))
		for _, item := range v {
			out = append(out, strings.ToUpper(item))
		}
		return out
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, strings.ToUpper(s))
			}
		}
		return out
	case api.PropertyType:
		return normalizeTypeNames([]string(v))
	default:
		return nil
	}
}

func normalizeStringSlice(value any) []string {
	switch v := value.(type) {
	case []string:
		return append([]string(nil), v...)
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func normalizeSlice(value any) []any {
	switch v := value.(type) {
	case []any:
		return v
	case []string:
		out := make([]any, 0, len(v))
		for _, item := range v {
			out = append(out, item)
		}
		return out
	default:
		return nil
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

func (r *Gemma4Renderer) formatToolResponseBlock(toolName, response string) string {
	return "<|tool_response>response:" + toolName + "{value:" + r.formatArgValue(response) + "}<tool_response|>"
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

func (r *Gemma4Renderer) formatSchemaValue(value any) string {
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
		return r.formatSchemaMapValue(v)
	case []any:
		return r.formatSchemaArrayValue(v)
	case []string:
		out := make([]any, 0, len(v))
		for _, item := range v {
			out = append(out, item)
		}
		return r.formatSchemaArrayValue(out)
	default:
		return fmt.Sprintf("%v", v)
	}
}

func (r *Gemma4Renderer) formatSchemaMapValue(m map[string]any) string {
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
		sb.WriteString(g4Q + key + g4Q + ":" + r.formatSchemaValue(m[key]))
	}

	sb.WriteString("}")
	return sb.String()
}

func (r *Gemma4Renderer) formatSchemaArrayValue(arr []any) string {
	var sb strings.Builder
	sb.WriteString("[")
	for i, item := range arr {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(r.formatSchemaValue(item))
	}
	sb.WriteString("]")
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
