package parsers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

const (
	inklingModel     = "<|message_model|>"
	inklingText      = "<|content_text|>"
	inklingThought   = "<|content_thinking|>"
	inklingTool      = "<|content_invoke_tool_json|>"
	inklingToolText  = "<|content_invoke_tool_text|>"
	inklingToolError = "<|content_tool_error|>"
	inklingEnd       = "<|end_message|>"
	inklingEndSample = "<|content_model_end_sampling|>"
	inklingUnframed  = "unframed"
)

type inkling2ToolArguments struct{ api.ToolCallFunctionArguments }

func (a *inkling2ToolArguments) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &a.ToolCallFunctionArguments); err != nil {
		return err
	}
	// The native argument decoder stores numbers as float64. Keep exact IDs
	// in generated tool calls, including IDs in nested objects and arrays.
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var fields map[string]any
	if err := dec.Decode(&fields); err != nil {
		return err
	}
	for key := range a.All() {
		a.Set(key, inkling2ToolValue(fields[key]))
	}
	return nil
}

func inkling2ToolValue(value any) any {
	switch v := value.(type) {
	case json.Number:
		f, _ := v.Float64() // the native decoder already validated the number
		if math.Abs(f) >= 1<<53 && !strings.ContainsAny(v.String(), ".eE") {
			return v
		}
		return f
	case []any:
		for i, item := range v {
			v[i] = inkling2ToolValue(item)
		}
	case map[string]any:
		for key, item := range v {
			v[key] = inkling2ToolValue(item)
		}
	}
	return value
}

type inkling2Parser struct {
	buffer           string
	mode             string
	name             string
	callIndex        int
	suppressThinking bool
	structured       bool
}

func init() {
	Register("inkling2", func() Parser { return &inkling2Parser{} })
}

func (p *inkling2Parser) Init(tools []api.Tool, _ *api.Message, think *api.ThinkValue) []api.Tool {
	*p = inkling2Parser{}
	p.suppressThinking = think != nil && (think.Value == false || think.Value == "none")
	return tools
}

func (*inkling2Parser) HasToolSupport() bool     { return true }
func (*inkling2Parser) HasThinkingSupport() bool { return true }

func (p *inkling2Parser) ThinkingClose() []string {
	if p.mode == inklingText || (p.suppressThinking && p.mode != inklingThought) {
		return nil
	}
	// A thought frame ends before the model opens the visible text frame.
	// Structured output starts inside that text frame.
	return []string{inklingModel + inklingText}
}

// StructuredOutputFrames aligns the stream parser and the grammar with the
// last frame in the rendered prompt. A custom template can open either frame.
func (p *inkling2Parser) StructuredOutputFrames(prompt string) *model.StructuredOutputFrames {
	p.structured = true
	const answer, thought = inklingModel + inklingText, inklingModel + inklingThought
	prompt = strings.TrimRight(prompt, " \t\r\n")
	frames := &model.StructuredOutputFrames{ContentOpen: answer}
	switch {
	case strings.HasSuffix(prompt, answer):
		p.mode = inklingText
		return nil
	case strings.HasSuffix(prompt, thought):
		p.mode = inklingThought
	case strings.HasSuffix(prompt, inklingEnd), strings.HasSuffix(prompt, inklingEndSample):
		// The publisher template ends at a message boundary.
		if p.suppressThinking {
			return frames
		}
		frames.ThinkingOpen = thought
	default:
		// An unframed custom prompt uses the ordinary format grammar.
		if p.suppressThinking {
			p.mode = inklingText
		}
		return nil
	}
	frames.ThinkingClose = inklingEnd
	frames.ThinkingExcludes = p.PreservedTokens()
	return frames
}

func (*inkling2Parser) PreservedTokens() []string {
	return []string{inklingModel, inklingText, inklingThought, inklingTool, inklingToolText, inklingToolError, inklingEnd, inklingEndSample}
}

func (p *inkling2Parser) Add(s string, done bool) (string, string, []api.ToolCall, error) {
	p.buffer += s
	var content, thinking strings.Builder
	var calls []api.ToolCall
	for len(p.buffer) > 0 {
		if p.structured && p.mode == inklingText {
			// The grammar owns the JSON boundary. Marker spellings inside
			// string values are part of the visible JSON.
			content.WriteString(p.buffer)
			p.buffer = ""
			break
		}
		tokens := p.PreservedTokens()
		atBoundary := p.mode == "" || p.mode == inklingEnd || p.mode == inklingEndSample
		searchFrom := 0
		if p.mode == inklingTool {
			// The template allows literal control tokens in JSON strings.
			// Only search for a terminator after the tool call object.
			tokens = []string{inklingEnd, inklingEndSample}
			searchFrom = findJSONEnd(p.buffer) + 1
			if searchFrom == 0 {
				if !done {
					break
				}
				// A token limit can end the stream inside a valid tool call.
				var raw json.RawMessage
				err := json.NewDecoder(strings.NewReader(p.buffer)).Decode(&raw)
				if err == io.EOF || (err == io.ErrUnexpectedEOF && strings.HasPrefix(strings.TrimSpace(p.buffer), "{")) {
					p.buffer = ""
					break
				}
			}
		}
		marker, index := "", len(p.buffer)
		var keep int
		for _, token := range tokens {
			// A raw tool-text or error opener may follow a named header or
			// start a frame by itself. Its spelling inside a body is text.
			if (token == inklingToolText || token == inklingToolError) && p.mode != inklingModel {
				if !atBoundary || !(strings.HasPrefix(p.buffer, token) || strings.HasPrefix(token, p.buffer)) {
					continue
				}
			}
			if i := strings.Index(p.buffer[searchFrom:], token); i >= 0 && searchFrom+i < index {
				marker, index = token, searchFrom+i
			}
			keep = max(keep, overlap(p.buffer, token))
		}
		if marker == "" && !done {
			if p.mode == inklingTool || p.mode == inklingModel {
				break
			}
			index -= keep
		}
		body := p.buffer[:index]
		p.buffer = p.buffer[index+len(marker):]
		switch p.mode {
		case inklingModel:
			p.name = body
		case inklingThought:
			if !p.suppressThinking {
				thinking.WriteString(body)
			}
		case inklingToolError:
			// The model's tool-error frame is suppressed in Ollama responses.
		case inklingTool:
			var invocation struct {
				Name *string               `json:"name"`
				Args inkling2ToolArguments `json:"args"`
			}
			if err := json.Unmarshal([]byte(body), &invocation); err != nil {
				return content.String(), thinking.String(), calls, fmt.Errorf("invalid Inkling2 tool call: %w", err)
			}
			if invocation.Name == nil {
				return content.String(), thinking.String(), calls, fmt.Errorf("missing Inkling2 tool name for header %q", p.name)
			}
			if p.name != "" && p.name != *invocation.Name {
				return content.String(), thinking.String(), calls, fmt.Errorf("invalid Inkling2 tool name %q for header %q", *invocation.Name, p.name)
			}
			calls = append(calls, api.ToolCall{Function: api.ToolCallFunction{
				Index:     p.callIndex,
				Name:      *invocation.Name,
				Arguments: invocation.Args.ToolCallFunctionArguments,
			}})
			p.callIndex++
		default:
			content.WriteString(body)
		}
		if marker == "" {
			if atBoundary && body != "" {
				p.mode = inklingUnframed
			}
			break
		}
		if marker != inklingTool {
			p.name = ""
		}
		p.mode = marker
	}
	return content.String(), thinking.String(), calls, nil
}
