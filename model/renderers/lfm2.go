package renderers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/ollama/ollama/api"
)

type LFM2Renderer struct {
	IsThinking bool
	useImgTags bool
}

const lfm2BOSToken = "<|startoftext|>"

const (
	lfm2ToolListStartTag     = "<|tool_list_start|>"
	lfm2ToolListEndTag       = "<|tool_list_end|>"
	lfm2ToolCallStartTag     = "<|tool_call_start|>"
	lfm2ToolCallEndTag       = "<|tool_call_end|>"
	lfm2ToolResponseStartTag = "<|tool_response_start|>"
	lfm2ToolResponseEndTag   = "<|tool_response_end|>"
)

func lfm2RenderSystemContent(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var sb strings.Builder
		for _, item := range v {
			obj, ok := item.(map[string]any)
			if !ok {
				continue
			}

			if itemType, _ := obj["type"].(string); itemType == "text" {
				if text, ok := obj["text"].(string); ok {
					sb.WriteString(text)
				}
			}
		}
		return sb.String()
	default:
		return ""
	}
}

func lfm2JSON(v any) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		fallback, _ := json.Marshal(v)
		return string(fallback)
	}

	encoded := bytes.TrimSuffix(buf.Bytes(), []byte{'\n'})

	// HF `tojson` defaults to `json.dumps(..., separators=None)`, which inserts
	// a space after commas and colons.
	var out strings.Builder
	out.Grow(len(encoded) + len(encoded)/8)

	inString := false
	escaped := false
	for i, b := range encoded {
		out.WriteByte(b)

		if inString {
			if escaped {
				escaped = false
				continue
			}
			if b == '\\' {
				escaped = true
				continue
			}
			if b == '"' {
				inString = false
			}
			continue
		}

		if b == '"' {
			inString = true
			continue
		}

		if (b == ':' || b == ',') && i+1 < len(encoded) {
			next := encoded[i+1]
			if next != ' ' && next != '\n' && next != '\r' && next != '\t' {
				out.WriteByte(' ')
			}
		}
	}

	return out.String()
}

func lfm2ImagePlaceholder(useImgTags bool) string {
	if useImgTags {
		return "[img]"
	}

	return "<image>"
}

func lfm2RenderContent(content any, useImgTags bool) string {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var sb strings.Builder
		for _, item := range v {
			obj, ok := item.(map[string]any)
			if !ok {
				sb.WriteString(lfm2JSON(item))
				continue
			}

			itemType, _ := obj["type"].(string)
			switch itemType {
			case "image":
				sb.WriteString(lfm2ImagePlaceholder(useImgTags))
			case "text":
				if text, ok := obj["text"].(string); ok {
					sb.WriteString(text)
				} else {
					sb.WriteString(lfm2JSON(item))
				}
			default:
				sb.WriteString(lfm2JSON(item))
			}
		}
		return sb.String()
	default:
		return lfm2JSON(content)
	}
}

func lfm2ToolSchema(tool api.Tool) any {
	if tool.Function.Name == "" {
		return tool
	}

	// LFM2 templates are typically fed function-schema objects (name/description/parameters).
	return tool.Function
}

func lfm2ToolCallArgument(v any) string {
	return lfm2JSON(v)
}

func lfm2RenderToolCalls(calls []api.ToolCall) string {
	var sb strings.Builder

	sb.WriteString(lfm2ToolCallStartTag)
	sb.WriteString("[")
	for i, tc := range calls {
		if i > 0 {
			sb.WriteString(",")
		}

		sb.WriteString(tc.Function.Name)
		sb.WriteString("(")

		keys := make([]string, 0, tc.Function.Arguments.Len())
		for key := range tc.Function.Arguments.All() {
			keys = append(keys, key)
		}
		sort.Strings(keys)

		for j, key := range keys {
			if j > 0 {
				sb.WriteString(",")
			}
			value, _ := tc.Function.Arguments.Get(key)
			sb.WriteString(key)
			sb.WriteString("=")
			sb.WriteString(lfm2ToolCallArgument(value))
		}

		sb.WriteString(")")
	}
	sb.WriteString("]")
	sb.WriteString(lfm2ToolCallEndTag)

	return sb.String()
}

func (r *LFM2Renderer) renderMessageContent(message api.Message, imageOffset int) string {
	content := lfm2RenderContent(message.Content, r.useImgTags)
	if len(message.Images) == 0 {
		return content
	}

	var sb strings.Builder
	if r.useImgTags {
		for i := range message.Images {
			sb.WriteString(fmt.Sprintf("[img-%d]", imageOffset+i))
		}
	} else {
		placeholder := lfm2ImagePlaceholder(false)
		if strings.Contains(content, placeholder) {
			return content
		}
		for range message.Images {
			sb.WriteString(placeholder)
		}
	}
	sb.WriteString(content)
	return sb.String()
}

func (r *LFM2Renderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	// Follow Liquid tool-use formatting for LFM2 tool wrappers.
	sb.WriteString(lfm2BOSToken)

	// Extract first system message if present (to combine with tools)
	var firstSystemContent string
	startIdx := 0
	if len(messages) > 0 && messages[0].Role == "system" {
		firstSystemContent = lfm2RenderSystemContent(messages[0].Content)
		startIdx = 1
	}

	// Append tools to first system content
	if len(tools) > 0 {
		if firstSystemContent != "" {
			firstSystemContent += "\n"
		}
		firstSystemContent += "List of tools: "
		firstSystemContent += lfm2ToolListStartTag
		firstSystemContent += "["
		for i, tool := range tools {
			firstSystemContent += lfm2JSON(lfm2ToolSchema(tool))
			if i < len(tools)-1 {
				firstSystemContent += ", "
			}
		}
		firstSystemContent += "]"
		firstSystemContent += lfm2ToolListEndTag
	}

	// Output first system block if it has content
	if firstSystemContent != "" {
		sb.WriteString("<|im_start|>system\n")
		sb.WriteString(firstSystemContent)
		sb.WriteString("<|im_end|>\n")
	}

	keepPastThinking := r.IsThinking && (thinkValue != nil && thinkValue.Bool())

	// Find the index of the last assistant message for thinking stripping
	lastAssistantIndex := -1
	for i := len(messages) - 1; i >= startIdx; i-- {
		if messages[i].Role == "assistant" {
			lastAssistantIndex = i
			break
		}
	}

	imageOffset := 0
	for i := range startIdx {
		imageOffset += len(messages[i].Images)
	}

	for i := startIdx; i < len(messages); i++ {
		message := messages[i]
		lastMessage := i == len(messages)-1
		prefill := lastMessage && message.Role == "assistant"

		sb.WriteString("<|im_start|>")
		sb.WriteString(message.Role)
		sb.WriteString("\n")

		content := r.renderMessageContent(message, imageOffset)
		imageOffset += len(message.Images)
		if message.Role == "assistant" && !keepPastThinking && i != lastAssistantIndex {
			if idx := strings.LastIndex(content, "</think>"); idx >= 0 {
				content = strings.TrimSpace(content[idx+len("</think>"):])
			}
		}
		if message.Role == "assistant" && len(message.ToolCalls) > 0 && !strings.Contains(content, lfm2ToolCallStartTag) {
			if strings.TrimSpace(content) == "" {
				content = lfm2RenderToolCalls(message.ToolCalls) + content
			} else {
				content = lfm2RenderToolCalls(message.ToolCalls) + "\n" + content
			}
		}
		if message.Role == "tool" && !strings.Contains(content, lfm2ToolResponseStartTag) {
			content = lfm2ToolResponseStartTag + content + lfm2ToolResponseEndTag
		}

		sb.WriteString(content)
		if !prefill {
			sb.WriteString("<|im_end|>\n")
		}
	}

	needsGenerationPrompt := true
	if len(messages) > 0 && messages[len(messages)-1].Role == "assistant" {
		needsGenerationPrompt = false
	}

	if needsGenerationPrompt {
		// RenderWithRenderer uses add_generation_prompt=true for chat rendering,
		// unless we're prefilling a trailing assistant message.
		sb.WriteString("<|im_start|>assistant\n")
	}

	return sb.String(), nil
}
