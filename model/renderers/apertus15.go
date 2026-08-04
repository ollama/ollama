package renderers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

const (
	apertus15BOS             = "<s>"
	apertus15SystemStart     = "<|system_start|>"
	apertus15SystemEnd       = "<|system_end|>"
	apertus15DeveloperStart  = "<|developer_start|>"
	apertus15DeveloperEnd    = "<|developer_end|>"
	apertus15UserStart       = "<|user_start|>"
	apertus15UserEnd         = "<|user_end|>"
	apertus15AssistantStart  = "<|assistant_start|>"
	apertus15AssistantEnd    = "<|assistant_end|>"
	apertus15InnerStart      = "<|inner_prefix|>"
	apertus15InnerEnd        = "<|inner_suffix|>"
	apertus15ToolsStart      = "<|tools_prefix|>"
	apertus15ToolsEnd        = "<|tools_suffix|>"
	apertus15ToolOutputStart = "<|tool_output_start|>"
	apertus15ToolOutputEnd   = "<|tool_output_end|>"
	apertus15Image           = "<|image|>"

	apertus15DefaultSystem = "You are Apertus 1.5 Omni, a multimodal assistant developed by the Swiss AI Initiative. Extended from Apertus 1 via continued pretraining, you understand images and audio and respond in text."
)

type Apertus15Renderer struct {
	useImgTags bool
}

func (r *Apertus15Renderer) LeadingBOS() string {
	return apertus15BOS
}

func (r *Apertus15Renderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var sb strings.Builder
	sb.WriteString(apertus15BOS)

	messageStart := 0
	system := apertus15DefaultSystem
	if len(messages) > 0 && messages[0].Role == "system" {
		system = messages[0].Content
		messageStart = 1
	}
	sb.WriteString(apertus15SystemStart)
	sb.WriteString(system)
	sb.WriteString(apertus15SystemEnd)

	sb.WriteString(apertus15DeveloperStart)
	if think != nil && think.Bool() {
		sb.WriteString("Deliberation: enabled\n")
	} else {
		sb.WriteString("Deliberation: disabled\n")
	}
	if len(tools) == 0 {
		sb.WriteString("Tool Capabilities: disabled")
	} else {
		sb.WriteString("Tool Capabilities:\n")
		if err := renderApertus15Tools(&sb, tools); err != nil {
			return "", err
		}
	}
	sb.WriteString(apertus15DeveloperEnd)

	imageOffset := 0
	inAssistant := false
	inInner := false
	inToolOutput := false
	waitingForToolOutputs := false

	closeToolOutput := func() {
		if inToolOutput {
			sb.WriteString(apertus15ToolOutputEnd)
			inToolOutput = false
		}
	}
	closeAssistant := func() {
		closeToolOutput()
		if inAssistant {
			sb.WriteString(apertus15AssistantEnd)
			inAssistant = false
		}
		inInner = false
		waitingForToolOutputs = false
	}

	for _, message := range messages[messageStart:] {
		switch message.Role {
		case "user":
			closeAssistant()
			sb.WriteString(apertus15UserStart)
			content, nextOffset := r.renderContent(message, imageOffset)
			imageOffset = nextOffset
			sb.WriteString(content)
			sb.WriteString(apertus15UserEnd)
		case "assistant":
			if !inAssistant {
				sb.WriteString(apertus15AssistantStart)
				inAssistant = true
			}
			closeToolOutput()

			if message.Thinking != "" {
				if !inInner {
					sb.WriteString(apertus15InnerStart)
					inInner = true
				}
				sb.WriteString(message.Thinking)
			}

			if message.Content != "" {
				if inInner {
					sb.WriteString(apertus15InnerEnd)
					inInner = false
				}
				sb.WriteString(message.Content)
			}

			if len(message.ToolCalls) > 0 {
				if err := renderApertus15ToolCalls(&sb, message.ToolCalls); err != nil {
					return "", err
				}
				waitingForToolOutputs = true
			}
		case "tool":
			if !inAssistant {
				return "", fmt.Errorf("apertus 1.5 tool message outside assistant turn")
			}
			if !inToolOutput {
				sb.WriteString(apertus15ToolOutputStart)
				inToolOutput = true
			} else {
				sb.WriteString(", ")
			}
			sb.WriteString(message.Content)
			waitingForToolOutputs = false
		case "system":
			return "", fmt.Errorf("apertus 1.5 system message must be the first message")
		default:
			return "", fmt.Errorf("unsupported apertus 1.5 message role %q", message.Role)
		}
	}

	closeToolOutput()
	lastRole := ""
	if len(messages) > 0 {
		lastRole = messages[len(messages)-1].Role
	}
	if inAssistant && !waitingForToolOutputs && lastRole != "assistant" {
		sb.WriteString(apertus15AssistantEnd)
	}
	if lastRole != "assistant" {
		sb.WriteString(apertus15AssistantStart)
	}

	return sb.String(), nil
}

func (r *Apertus15Renderer) renderContent(message api.Message, imageOffset int) (string, int) {
	if r.useImgTags {
		return renderContentWithImageTags(message.Content, len(message.Images), imageOffset)
	}

	var sb strings.Builder
	for range message.Images {
		sb.WriteString(apertus15Image)
	}
	sb.WriteString(message.Content)
	return sb.String(), imageOffset + len(message.Images)
}

func renderApertus15Tools(sb *strings.Builder, tools []api.Tool) error {
	for i, tool := range tools {
		sb.WriteString("// ")
		sb.WriteString(tool.Function.Description)
		sb.WriteByte('\n')
		sb.WriteString("type ")
		sb.WriteString(tool.Function.Name)
		sb.WriteString(" = ")

		params := tool.Function.Parameters
		if params.Properties == nil || params.Properties.Len() == 0 {
			sb.WriteString("() => any;")
		} else {
			sb.WriteString("(_: {\n")
			required := make(map[string]struct{}, len(params.Required))
			for _, name := range params.Required {
				required[name] = struct{}{}
			}
			propertyIndex := 0
			for name, property := range params.Properties.All() {
				if property.Description != "" {
					sb.WriteString("// ")
					sb.WriteString(property.Description)
					sb.WriteByte('\n')
				}
				sb.WriteString(name)
				if _, ok := required[name]; !ok {
					sb.WriteByte('?')
				}
				sb.WriteString(": ")
				typeName, err := apertus15TypeScriptType(property)
				if err != nil {
					return fmt.Errorf("render apertus 1.5 tool %q property %q: %w", tool.Function.Name, name, err)
				}
				sb.WriteString(typeName)
				propertyIndex++
				if propertyIndex < params.Properties.Len() {
					sb.WriteString(",\n")
				} else {
					sb.WriteByte('\n')
				}
			}
			sb.WriteString("}) => any;")
		}
		if i < len(tools)-1 {
			sb.WriteByte('\n')
		}
	}
	return nil
}

func apertus15TypeScriptType(property api.ToolProperty) (string, error) {
	if len(property.AnyOf) > 0 {
		// The source template only recognizes oneOf. Ollama preserves anyOf,
		// so match the template's fallback for this distinct schema keyword.
		return "any", nil
	}

	if len(property.Type) > 1 {
		return strings.Join(property.Type, " | "), nil
	}
	typeName := ""
	if len(property.Type) == 1 {
		typeName = property.Type[0]
	}
	switch typeName {
	case "array":
		if property.Items == nil {
			return "any[]", nil
		}
		encoded, err := json.Marshal(property.Items)
		if err != nil {
			return "", err
		}
		var item api.ToolProperty
		if err := json.Unmarshal(encoded, &item); err != nil {
			return "any[]", nil
		}
		inner, err := apertus15TypeScriptType(item)
		if err != nil {
			return "", err
		}
		if inner == "object | object" || len(inner) > 50 {
			inner = "any"
		}
		return inner + "[]", nil
	case "string":
		if len(property.Enum) == 0 {
			return "string", nil
		}
		values := make([]string, 0, len(property.Enum))
		for _, value := range property.Enum {
			values = append(values, fmt.Sprintf("\"%s\"", apertus15JinjaString(value)))
		}
		return strings.Join(values, " | "), nil
	case "number", "integer":
		return "number", nil
	case "boolean":
		return "boolean", nil
	case "object":
		if property.Properties == nil || property.Properties.Len() == 0 {
			return "object", nil
		}
		return apertus15ObjectType(property.Properties, property.Required)
	default:
		return "any", nil
	}
}

func apertus15JinjaString(value any) string {
	switch value := value.(type) {
	case bool:
		if value {
			return "True"
		}
		return "False"
	case nil:
		return "None"
	default:
		return fmt.Sprint(value)
	}
}

func apertus15ObjectType(properties *api.ToolPropertiesMap, required []string) (string, error) {
	requiredSet := make(map[string]struct{}, len(required))
	for _, name := range required {
		requiredSet[name] = struct{}{}
	}

	var sb strings.Builder
	sb.WriteString("{\n")
	index := 0
	for name, property := range properties.All() {
		sb.WriteString(name)
		if _, ok := requiredSet[name]; !ok {
			sb.WriteByte('?')
		}
		sb.WriteString(": \n                ")
		typeName, err := apertus15TypeScriptType(property)
		if err != nil {
			return "", err
		}
		sb.WriteString(typeName)
		index++
		if index < properties.Len() {
			sb.WriteString(", ")
		}
	}
	sb.WriteByte('}')
	return sb.String(), nil
}

func renderApertus15ToolCalls(sb *strings.Builder, calls []api.ToolCall) error {
	sb.WriteString(apertus15ToolsStart)
	sb.WriteByte('[')
	for i, call := range calls {
		if i > 0 {
			sb.WriteString(", ")
		}
		arguments, err := marshalApertus15Arguments(call.Function.Arguments)
		if err != nil {
			return err
		}
		sb.WriteString("{\"")
		sb.WriteString(call.Function.Name)
		sb.WriteString("\": ")
		sb.Write(arguments)
		sb.WriteByte('}')
	}
	sb.WriteByte(']')
	sb.WriteString(apertus15ToolsEnd)
	return nil
}

func marshalApertus15Arguments(arguments api.ToolCallFunctionArguments) ([]byte, error) {
	var sb strings.Builder
	sb.WriteByte('{')
	index := 0
	for name, value := range arguments.All() {
		if index > 0 {
			sb.WriteString(", ")
		}
		encodedName, err := marshalWithSpacesNoHTMLEscape(name)
		if err != nil {
			return nil, err
		}
		encodedValue, err := marshalWithSpacesNoHTMLEscape(value)
		if err != nil {
			return nil, err
		}
		sb.Write(encodedName)
		sb.WriteString(": ")
		sb.Write(encodedValue)
		index++
	}
	sb.WriteByte('}')
	return []byte(sb.String()), nil
}
