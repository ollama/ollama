package renderers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

type inkling2Renderer struct {
	useImgTags   bool
	imageOffsets map[int][]int
}

func (*inkling2Renderer) LeadingBOS() string { return "" }

// Native messages carry attachment bytes without a MIME type. MP3 can start
// directly with a frame sync when it has no ID3 tag for the MIME sniffer.
func inkling2Audio(data []byte) bool {
	switch http.DetectContentType(data) {
	case "audio/wave", "audio/mpeg":
		return true
	}
	return len(data) >= 2 && data[0] == 0xff && data[1]&0xe0 == 0xe0
}

func (*inkling2Renderer) Thinking() *model.Thinking {
	return &model.Thinking{
		Values:  []any{false, true, "none", "minimal", "low", "medium", "high", "xhigh", "max"},
		Default: "high",
	}
}

func inkling2Effort(think *api.ThinkValue) (string, error) {
	if think == nil || think.Value == nil {
		return "0.9", nil
	}
	value := think.Value
	if name, ok := value.(string); ok {
		// Python str.strip also removes the four ASCII information separators.
		value = strings.TrimFunc(name, func(r rune) bool {
			return unicode.IsSpace(r) || r >= 0x1c && r <= 0x1f
		})
	}
	switch value {
	case false, "none":
		return "0", nil
	case "minimal":
		return "0.1", nil
	case "low":
		return "0.2", nil
	case "medium":
		return "0.7", nil
	case true, "high":
		return "0.9", nil
	case "xhigh", "max":
		return "0.99", nil
	default:
		return "", fmt.Errorf("Unknown reasoning_effort: %v", think.Value)
	}
}

func inkling2Tools(tools []api.Tool) (string, error) {
	specs := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		raw, err := json.Marshal(tool.Function.Parameters)
		if err != nil {
			return "", err
		}
		// Jinja sorts keys recursively for tool declarations. UseNumber
		// retains JSON number types when canonicalizing schema fields.
		var parameters map[string]any
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.UseNumber()
		if err := decoder.Decode(&parameters); err != nil {
			return "", err
		}
		if tool.Function.Parameters.Type == "" {
			delete(parameters, "type")
		}
		if parameters["properties"] == nil {
			delete(parameters, "properties")
		}
		kind := tool.Type
		if kind == "" {
			kind = "function"
		}
		specs = append(specs, map[string]any{
			"description": tool.Function.Description,
			"name":        tool.Function.Name,
			"parameters":  parameters,
			"type":        kind,
		})
	}
	return PythonJSONCompact(specs)
}

var inkling2ImageMarker = regexp.MustCompile(`\[img-[0-9]+\]`)

// RenderInkling2WithImages records each attachment tag's byte offset in the
// prompt. Text may contain identical tags, including an entire image frame.
func RenderInkling2WithImages(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, map[int][]int, error) {
	r := inkling2Renderer{useImgTags: true, imageOffsets: make(map[int][]int)}
	prompt, err := r.Render(messages, tools, think)
	return prompt, r.imageOffsets, err
}

func (r *inkling2Renderer) writeContent(out *strings.Builder, role string, message api.Message, imageOffset int) error {
	writeText := func(text string) {
		out.WriteString(role + "<|content_text|>" + text + "<|end_message|>")
	}
	if len(message.Images) == 0 {
		writeText(message.Content)
		return nil
	}

	content := message.Content
	var prefix strings.Builder
	tags := make(map[string]int, len(message.Images))
	for i := range message.Images {
		tag := "[img-" + strconv.Itoa(imageOffset+i) + "]"
		if !strings.Contains(content, tag) {
			if before, after, ok := strings.Cut(content, "[img]"); ok {
				content = before + tag + after
			} else {
				prefix.WriteString(tag)
			}
		}
		tags[tag] = i
	}
	content = prefix.String() + content
	previous := 0
	for _, span := range inkling2ImageMarker.FindAllStringIndex(content, -1) {
		marker := content[span[0]:span[1]]
		i, ok := tags[marker]
		if !ok {
			continue
		}
		if inkling2Audio(message.Images[i]) {
			return errors.New("Inkling 2 does not accept audio input")
		}
		if message.Role != "user" && message.Role != "tool" {
			return errors.New("Image content is only supported for user and tool messages")
		}
		if span[0] > previous {
			writeText(content[previous:span[0]])
		}
		out.WriteString(role + "<|content_image|>")
		if r.imageOffsets != nil {
			id := imageOffset + i
			r.imageOffsets[id] = append(r.imageOffsets[id], out.Len())
		}
		if !r.useImgTags {
			marker = "<|unused_200054|>"
		}
		out.WriteString(marker + "<|end_message|>")
		previous = span[1]
	}
	if previous < len(content) {
		writeText(content[previous:])
	}
	return nil
}

func (r *inkling2Renderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var out strings.Builder
	if len(tools) > 0 {
		declaration, err := inkling2Tools(tools)
		if err != nil {
			return "", err
		}
		out.WriteString("<|message_system|>tool_declare<|content_xml|>" + declaration + "<|end_message|>")
	}

	writeEffort := func() error {
		effort, err := inkling2Effort(think)
		if err != nil {
			return err
		}
		out.WriteString("<|message_system|><|content_text|>Thinking effort level: " + effort + "<|end_message|>")
		return nil
	}

	effortEmitted := false
	imageOffset := 0
	toolCallNames := make(map[string]string)
	for _, message := range messages {
		if message.Role != "assistant" {
			continue
		}
		for _, call := range message.ToolCalls {
			if call.ID != "" {
				toolCallNames[call.ID] += call.Function.Name
			}
		}
	}
	for _, message := range messages {
		var role string
		switch message.Role {
		case "user":
			role = "<|message_user|>"
		case "assistant":
			role = "<|message_model|>" + message.ToolName
		case "system":
			role = "<|message_system|>"
		case "tool":
			name := message.ToolName
			if name == "" {
				name = toolCallNames[message.ToolCallID]
			}
			role = "<|message_tool|>" + name
		default:
			return "", fmt.Errorf("Unknown message role: %s", message.Role)
		}
		if !effortEmitted && message.Role != "system" {
			if err := writeEffort(); err != nil {
				return "", err
			}
			effortEmitted = true
		}
		if message.Role == "assistant" && message.Thinking != "" {
			out.WriteString(role + "<|content_thinking|>" + message.Thinking + "<|end_message|>")
		}
		if len(message.Images) > 0 || message.Content != "" || !(message.Role == "assistant" && (len(message.ToolCalls) > 0 || message.Thinking != "")) {
			if err := r.writeContent(&out, role, message, imageOffset); err != nil {
				return "", err
			}
			imageOffset += len(message.Images)
		}
		if message.Role == "assistant" {
			for _, call := range message.ToolCalls {
				if message.ToolName != "" && message.ToolName != call.Function.Name {
					return "", fmt.Errorf("assistant name %s does not match its tool call %s", message.ToolName, call.Function.Name)
				}
				// The template sorts argument names; values retain the object
				// order present in their JSON representation.
				arguments := call.Function.Arguments.ToMap()
				if arguments == nil {
					arguments = map[string]any{}
				}
				args, err := PythonJSONCompact(arguments)
				if err != nil {
					return "", fmt.Errorf("Inkling 2 tool call %s: %w", call.Function.Name, err)
				}
				name, err := pythonScalar(call.Function.Name)
				if err != nil {
					return "", err
				}
				body := "{\"name\":" + name + ",\"args\":" + args + "}"
				out.WriteString("<|message_model|>" + call.Function.Name + "<|content_invoke_tool_json|>" + body + "<|end_message|>")
			}
			out.WriteString("<|content_model_end_sampling|>")
		}
	}
	if !effortEmitted {
		if err := writeEffort(); err != nil {
			return "", err
		}
	}
	return out.String(), nil
}
