package renderers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	glimmerBOS             = "<|begin_of_text|>"
	glimmerStart           = "<|start|>"
	glimmerMessage         = "<|message|>"
	glimmerEndMessage      = "<|eom|>"
	glimmerEndTurn         = "<|eot|>"
	glimmerDefaultSystem   = "You are a helpful AI assistant."
	glimmerKnowledgeCutoff = "2026-01-04"
)

const glimmerToolDefinitionsPrefix = `In this environment you have access to a set of tools you can use to answer the user's question.

You can invoke a function by writing a "<atem:function_calls>" block like the following:
<atem:function_calls>
<atem:invoke name="$FUNCTION_NAME">
<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>
...
</atem:invoke>
</atem:function_calls>

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
Here are the functions available in JSONSchema format:
// Tool metadata
`

const glimmerToolDefinitionsSuffix = `

Here's an example of how to call a function in the tool set:
(If the tool namespace is not specified, invoke the function directly as ` + "`example_function_name`" + ` rather than ` + "`example_tool_name.example_function_name`" + `)

to=example_tool_name.example_function_name

<atem:function_calls>
<atem:invoke name="example_tool_name.example_function_name">
<atem:parameter name="example_parameter_1">value_1</atem:parameter>
<atem:parameter name="example_parameter_2">This is the value for the second parameter
that can span
"multiple" lines
</atem:parameter>
</atem:invoke>
</atem:function_calls>`

type GlimmerRenderer struct {
	useImgTags  bool
	currentDate string
}

func (r *GlimmerRenderer) LeadingBOS() string {
	return glimmerBOS
}

func glimmerReasoningStrength(think *api.ThinkValue) string {
	if think == nil {
		return "high"
	}
	if !think.Bool() {
		return "none"
	}
	if think.IsString() {
		return think.String()
	}
	return "high"
}

var glimmerSystemReasoningReplacer = strings.NewReplacer(
	"Reasoning effort", "Reasoning strength",
	"Reasoning Effort", "Reasoning Strength",
	"reasoning effort", "reasoning strength",
	"REASONING EFFORT", "REASONING STRENGTH",
)

func glimmerNormalizeSystemReasoning(content string) string {
	return glimmerSystemReasoningReplacer.Replace(content)
}

func glimmerHasSystemReasoning(content string) bool {
	return strings.Contains(strings.ToLower(content), "reasoning strength")
}

func glimmerNamespace(name string) string {
	if namespace, _, ok := strings.Cut(name, "."); ok {
		return namespace
	}
	return name
}

func glimmerNamespaces(tools []api.Tool) []string {
	seen := make(map[string]bool)
	var namespaces []string
	for _, tool := range tools {
		namespace := glimmerNamespace(tool.Function.Name)
		if !seen[namespace] {
			seen[namespace] = true
			namespaces = append(namespaces, namespace)
		}
	}
	return namespaces
}

// glimmerJSON reproduces the Transformers 5.x tojson filter.
func glimmerJSON(v any) (string, error) {
	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(v); err != nil {
		return "", err
	}
	return string(addJSONSpaces(bytes.TrimSuffix(buf.Bytes(), []byte("\n")))), nil
}

func writeGlimmerToolDefinitions(sb *strings.Builder, tools []api.Tool) error {
	sb.WriteString(glimmerToolDefinitionsPrefix)
	for _, namespace := range glimmerNamespaces(tools) {
		name, err := glimmerJSON(namespace)
		if err != nil {
			return err
		}
		description, err := glimmerJSON("")
		if err != nil {
			return err
		}
		sb.WriteString(`{"name": `)
		sb.WriteString(name)
		sb.WriteString(`, "description": `)
		sb.WriteString(description)
		sb.WriteString("}\n")
	}

	sb.WriteString("// Function schemas")
	for _, tool := range tools {
		name, err := glimmerJSON(tool.Function.Name)
		if err != nil {
			return err
		}
		description, err := glimmerJSON(tool.Function.Description)
		if err != nil {
			return err
		}
		parameters, err := glimmerJSON(tool.Function.Parameters)
		if err != nil {
			return err
		}
		sb.WriteString("\n{\"name\": ")
		sb.WriteString(name)
		sb.WriteString(`, "description": `)
		sb.WriteString(description)
		sb.WriteString(`, "parameters": `)
		sb.WriteString(parameters)
		sb.WriteByte('}')
	}
	sb.WriteString(glimmerToolDefinitionsSuffix)
	return nil
}

func writeGlimmerSystemMeta(sb *strings.Builder, tools []api.Tool) {
	sb.WriteString(`# Valid recipients: "self"`)
	for _, namespace := range glimmerNamespaces(tools) {
		sb.WriteString(`, "`)
		sb.WriteString(namespace)
		sb.WriteString(`.*"`)
	}
	sb.WriteString(`, "user".`)
}

func writeGlimmerSystem(sb *strings.Builder, content string, tools []api.Tool, strength, currentDate string, defaultSystem bool) error {
	if !defaultSystem {
		content = glimmerNormalizeSystemReasoning(content)
	}

	writeGlimmerMessageStart(sb, "system")
	sb.WriteString(content)
	if defaultSystem {
		sb.WriteString("\nKnowledge cutoff: ")
		sb.WriteString(glimmerKnowledgeCutoff)
		sb.WriteByte('.')
		sb.WriteString("\nCurrent date: ")
		sb.WriteString(currentDate)
		sb.WriteByte('.')
	}
	if defaultSystem || !glimmerHasSystemReasoning(content) {
		sb.WriteString("\n\nReasoning strength: ")
		sb.WriteString(strength)
		sb.WriteByte('.')
	}
	if len(tools) > 0 {
		sb.WriteString("\n\n")
		if err := writeGlimmerToolDefinitions(sb, tools); err != nil {
			return err
		}
	}
	sb.WriteString("\n\n")
	writeGlimmerSystemMeta(sb, tools)
	sb.WriteString(glimmerEndTurn)
	return nil
}

func writeGlimmerMessageStart(sb *strings.Builder, header string) {
	sb.WriteString(glimmerStart)
	sb.WriteString(header)
	sb.WriteString(glimmerMessage)
}

func writeGlimmerMessage(sb *strings.Builder, header, content, end string) {
	writeGlimmerMessageStart(sb, header)
	sb.WriteString(content)
	sb.WriteString(end)
}

func glimmerToolResultName(message api.Message, messages []api.Message) string {
	if message.ToolName != "" {
		return message.ToolName
	}

	name := message.ToolCallID
	if message.ToolCallID == "" {
		return name
	}
	for _, candidate := range messages {
		for _, call := range candidate.ToolCalls {
			if call.ID == message.ToolCallID {
				name = call.Function.Name
			}
		}
	}
	return name
}

func glimmerCompositeValue(v any) bool {
	if v == nil {
		return false
	}
	switch reflect.TypeOf(v).Kind() {
	case reflect.Array, reflect.Map, reflect.Slice:
		return true
	default:
		return false
	}
}

func writeGlimmerATEM(sb *strings.Builder, call api.ToolCall) error {
	sb.WriteString("<atem:function_calls>\n<atem:invoke name=\"")
	sb.WriteString(call.Function.Name)
	sb.WriteString("\">\n")
	for name, value := range call.Function.Arguments.All() {
		sb.WriteString("<atem:parameter name=\"")
		sb.WriteString(name)
		sb.WriteString("\">")
		switch {
		case value == nil:
			sb.WriteString("null")
		case glimmerCompositeValue(value):
			encoded, err := glimmerJSON(value)
			if err != nil {
				return err
			}
			sb.WriteString(encoded)
		default:
			fmt.Fprint(sb, value)
		}
		sb.WriteString("</atem:parameter>\n")
	}
	sb.WriteString("</atem:invoke>\n</atem:function_calls>")
	return nil
}

func (r *GlimmerRenderer) renderContent(message api.Message, imageOffset int) (string, int) {
	if r.useImgTags {
		return renderContentWithImageTags(message.Content, len(message.Images), imageOffset)
	}

	var sb strings.Builder
	for range message.Images {
		sb.WriteString("<|patch|>")
	}
	sb.WriteString(message.Content)
	return sb.String(), imageOffset + len(message.Images)
}

func (r *GlimmerRenderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var sb strings.Builder
	sb.WriteString(glimmerBOS)

	hasSystem := false
	for _, message := range messages {
		if message.Role == "system" {
			hasSystem = true
			break
		}
	}

	strength := glimmerReasoningStrength(think)
	if !hasSystem {
		currentDate := r.currentDate
		if currentDate == "" {
			currentDate = time.Now().Format(time.DateOnly)
		}
		if err := writeGlimmerSystem(&sb, glimmerDefaultSystem, tools, strength, currentDate, true); err != nil {
			return "", err
		}
	}

	imageOffset := 0
	for i, message := range messages {
		content, nextImageOffset := r.renderContent(message, imageOffset)
		imageOffset = nextImageOffset

		endToken := glimmerEndTurn
		if i+1 < len(messages) && messages[i+1].Role == message.Role {
			endToken = glimmerEndMessage
		}

		switch message.Role {
		case "system":
			if err := writeGlimmerSystem(&sb, content, tools, strength, "", false); err != nil {
				return "", err
			}
		case "user":
			writeGlimmerMessage(&sb, "user", content, glimmerEndTurn)
		case "tool":
			name := glimmerToolResultName(message, messages)
			writeGlimmerMessageStart(&sb, "tool "+name)
			sb.WriteString(`<tool_output name="`)
			sb.WriteString(name)
			sb.WriteString("\">\n")
			sb.WriteString(content)
			sb.WriteString("\n</tool_output>")
			sb.WriteString(glimmerEndTurn)
		case "assistant":
			if message.Thinking != "" {
				writeGlimmerMessage(&sb, "assistant to=self", message.Thinking, glimmerEndMessage)
			}
			// A final thinking-only message is the server's unfinished self
			// segment, represented by recipient=self and end_turn=false in Jinja.
			if len(message.ToolCalls) > 0 {
				for j, call := range message.ToolCalls {
					writeGlimmerMessageStart(&sb, "assistant to="+call.Function.Name)
					if err := writeGlimmerATEM(&sb, call); err != nil {
						return "", err
					}
					if j+1 == len(message.ToolCalls) {
						sb.WriteString(endToken)
					} else {
						sb.WriteString(glimmerEndMessage)
					}
				}
			} else if content != "" || message.Thinking == "" {
				writeGlimmerMessage(&sb, "assistant to=user", content, glimmerEndTurn)
			}
		}
	}

	sb.WriteString(glimmerStart)
	sb.WriteString("assistant")
	return sb.String(), nil
}
