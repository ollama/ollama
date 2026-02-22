package renderers

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/ollama/ollama/api"
)

type LFM2Renderer struct {
	IsThinking bool
	useImgTags bool
}

const lfm2BOSToken = "<|startoftext|>"

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

func (r *LFM2Renderer) renderMessageContent(message api.Message) string {
	content := lfm2RenderContent(message.Content, r.useImgTags)
	if len(message.Images) == 0 {
		return content
	}

	// chatPrompt may already have inserted [img] / [img-n] placeholders.
	if strings.Contains(content, "[img]") || strings.Contains(content, "[img-") || strings.Contains(content, "<image>") {
		return content
	}

	var sb strings.Builder
	placeholder := lfm2ImagePlaceholder(r.useImgTags)
	for range message.Images {
		sb.WriteString(placeholder)
	}
	sb.WriteString(content)
	return sb.String()
}

func (r *LFM2Renderer) Render(messages []api.Message, tools []api.Tool, thinkValue *api.ThinkValue) (string, error) {
	var sb strings.Builder

	// Match the source chat_template.jinja.
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
		firstSystemContent += "List of tools: ["
		for i, tool := range tools {
			firstSystemContent += lfm2JSON(tool)
			if i < len(tools)-1 {
				firstSystemContent += ", "
			}
		}
		firstSystemContent += "]"
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

	for i := startIdx; i < len(messages); i++ {
		message := messages[i]
		sb.WriteString("<|im_start|>")
		sb.WriteString(message.Role)
		sb.WriteString("\n")

		content := r.renderMessageContent(message)
		if message.Role == "assistant" && !keepPastThinking && i != lastAssistantIndex {
			if idx := strings.LastIndex(content, "</think>"); idx >= 0 {
				content = strings.TrimSpace(content[idx+len("</think>"):])
			}
		}

		sb.WriteString(content)
		sb.WriteString("<|im_end|>\n")
	}

	// RenderWithRenderer always uses add_generation_prompt=true for chat rendering.
	sb.WriteString("<|im_start|>assistant\n")

	return sb.String(), nil
}
