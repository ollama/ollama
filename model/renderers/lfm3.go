package renderers

import (
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

const (
	lfm3BOS   = "<|startoftext|>"
	lfm3Image = "<|vision_start|><|image|><|vision_end|>"
)

// LFM3Renderer covers three publisher template branches: text and VL (thinking
// by default), Omni (non-reasoning), and Audio (an older template without
// merged tool results; an explicit think selects the README's prefills).
type LFM3Renderer struct {
	useImgTags bool
	noThinking bool
	audio      bool
	speech     bool // Omni and Audio read text aloud
}

func (*LFM3Renderer) LeadingBOS() string { return lfm3BOS }

func (r *LFM3Renderer) Thinking() *model.Thinking {
	// Audio can think after a <think> prefill, but the server defaults thinking
	// on, which would turn transcription requests into reasoning.
	if r.noThinking || r.audio {
		return nil
	}
	return &model.Thinking{Values: []any{true, false}, Default: true}
}

func (r *LFM3Renderer) imageContent(message api.Message, firstImage int, separator string) string {
	content := message.Content
	if r.useImgTags && strings.Contains(content, "[img-") {
		// The server has already placed indexed image tags in this message.
		return content
	}
	// The message becomes a list of media and text parts, which tool results
	// put on separate lines. Unmarked media lead.
	var leading []string
	parts := []string{content} // text at even indices, media at odd
	for i := range message.Images {
		placeholder := lfm3Image
		if r.useImgTags {
			placeholder = fmt.Sprintf("[img-%d]", firstImage+i)
		}
		if !lfm3PlaceMedia(&parts, "[img]", placeholder) &&
			!(r.useImgTags && lfm3PlaceMedia(&parts, lfm3Image, placeholder)) {
			leading = append(leading, placeholder)
		}
	}
	for i, part := range parts {
		if i%2 == 1 || part != "" {
			leading = append(leading, part)
		}
	}
	return strings.Join(leading, separator)
}

// lfm3PlaceMedia replaces the first marker in the text parts with placeholder.
func lfm3PlaceMedia(parts *[]string, marker, placeholder string) bool {
	for i := 0; i < len(*parts); i += 2 {
		before, after, ok := strings.Cut((*parts)[i], marker)
		if ok {
			*parts = slices.Replace(*parts, i, i+1, before, placeholder, after)
			return true
		}
	}
	return false
}

func lfm3ParameterValue(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case nil:
		return "null"
	case bool:
		if v {
			return "true"
		}
		return "false"
	default:
		return lfm2JSON(v)
	}
}

func lfm3RenderToolCalls(calls []api.ToolCall) string {
	var out strings.Builder
	for i, call := range calls {
		if i > 0 {
			out.WriteString("\n\n")
		}
		out.WriteString("<tool_call>\n<function=")
		out.WriteString(call.Function.Name)
		out.WriteString(">\n")
		for name, value := range call.Function.Arguments.All() {
			out.WriteString("<parameter=")
			out.WriteString(name)
			out.WriteString(">\n")
			out.WriteString(lfm3ParameterValue(value))
			out.WriteString("\n</parameter>\n")
		}
		out.WriteString("</function>\n</tool_call>")
	}
	return out.String()
}

func (r *LFM3Renderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var out strings.Builder
	out.WriteString(lfm3BOS)

	first := 0
	system := ""
	if len(messages) > 0 && messages[0].Role == "system" {
		system = r.imageContent(messages[0], 0, "")
		first = 1
	}
	if len(tools) > 0 {
		var listed strings.Builder
		for i, tool := range tools {
			if i > 0 {
				listed.WriteByte('\n')
			}
			listed.WriteString(lfm2JSON(tool.Function))
		}
		if system != "" {
			system += "\n\n"
		}
		system += "<tools>\n" + listed.String() + "\n</tools>"
	}
	if system != "" {
		out.WriteString("<|im_start|>system\n")
		out.WriteString(system)
		out.WriteString("<|im_end|>\n")
	}

	lastUser := -1
	for i := first; i < len(messages); i++ {
		if messages[i].Role == "user" {
			lastUser = i
		}
	}
	imageOffset := 0
	for i := range first {
		imageOffset += len(messages[i].Images)
	}
	// A trailing assistant message with content is a prefill the model
	// continues, as with Transformers' continue_final_message.
	prefill := len(messages) > first && messages[len(messages)-1].Role == "assistant" && messages[len(messages)-1].Content != ""
	// Consecutive results for the same tool call share one tool turn.
	var pendingResults []string
	for i := first; i < len(messages); i++ {
		message := messages[i]
		separator := ""
		if message.Role == "tool" {
			separator = "\n"
		}
		content := r.imageContent(message, imageOffset, separator)
		imageOffset += len(message.Images)
		if !r.audio && message.Role == "tool" && message.ToolCallID != "" && i+1 < len(messages) &&
			messages[i+1].Role == "tool" && messages[i+1].ToolCallID == message.ToolCallID {
			if content != "" {
				pendingResults = append(pendingResults, content)
			}
			continue
		}
		out.WriteString("<|im_start|>")
		out.WriteString(message.Role)
		out.WriteByte('\n')
		switch message.Role {
		case "assistant":
			if i > lastUser {
				switch {
				case message.Thinking != "":
					out.WriteString("<think>")
					out.WriteString(message.Thinking)
					out.WriteString("</think>")
				case !r.audio && (len(message.Images) > 0 || !strings.Contains(content, "</think>")):
					out.WriteString("<think></think>")
				}
			} else {
				if end := strings.LastIndex(content, "</think>"); end >= 0 {
					content = strings.TrimSpace(content[end+len("</think>"):])
				}
				if !r.audio {
					out.WriteString("<think></think>")
				}
			}
			continuing := prefill && i == len(messages)-1
			// The template moves a literal continuation tag after the tool calls.
			const continueTag = "CONTINUE_FINAL_MESSAGE_TAG "
			moveTag := !continuing && strings.HasSuffix(content, continueTag)
			if moveTag {
				content = strings.TrimSuffix(content, continueTag)
			}
			out.WriteString(content)
			out.WriteString(lfm3RenderToolCalls(message.ToolCalls))
			if moveTag {
				out.WriteString(continueTag)
			}
			if continuing {
				return out.String(), nil
			}
		case "tool":
			switch {
			case r.audio && content == "":
			case len(pendingResults) > 0:
				if content != "" {
					pendingResults = append(pendingResults, content)
				}
				content = "<tool_response>\n" + strings.Join(pendingResults, "\n") + "\n</tool_response>"
				pendingResults = nil
			case len(message.Images) > 0 || !strings.Contains(content, "<tool_response>"):
				content = "<tool_response>\n" + content + "\n</tool_response>"
			}
			out.WriteString(content)
		default:
			out.WriteString(content)
		}
		out.WriteString("<|im_end|>\n")
	}
	out.WriteString("<|im_start|>assistant\n")
	if r.audio {
		out.WriteString(lfm3AudioThinkingPrefill(think))
		return out.String(), nil
	}
	thinkingOn := (think != nil && think.Bool()) || (think == nil && !r.noThinking)
	if thinkingOn {
		out.WriteString("<think>")
	} else {
		out.WriteString("<think></think>")
	}
	return out.String(), nil
}

// lfm3AudioThinkingPrefill is the assistant prefill the Audio README uses to
// select thinking, which its template has no switch for. Without an explicit
// choice the prompt ends at the template's generation prompt and the model
// decides whether to reason.
func lfm3AudioThinkingPrefill(think *api.ThinkValue) string {
	switch {
	case think == nil:
		return ""
	case think.Bool():
		return "<think>"
	default:
		return "<think></think>"
	}
}
