package parsers

import (
	"encoding/json"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

// CohereParser parses output from Cohere North / Command A 2026 models
// (e.g. North-Mini-Code-1.0). The generation prompt ends with
// <|START_THINKING|> (reasoning on) or <|START_THINKING|><|END_THINKING|>
// (reasoning off), so output begins inside the thinking block when reasoning
// is enabled. After thinking, the model emits either
// <|START_TEXT|>content<|END_TEXT|> or an <|START_ACTION|>[...]<|END_ACTION|>
// tool call array, then <|END_OF_TURN_TOKEN|>.
type CohereParser struct {
	state     cohereParserState
	buffer    strings.Builder
	callIndex int
}

type cohereParserState int

const (
	cohereCollectingThinking cohereParserState = iota
	cohereAwaitingBlock
	cohereCollectingContent
	cohereCollectingAction
)

const (
	cohereEndThinking = "<|END_THINKING|>"
	cohereStartText   = "<|START_TEXT|>"
	cohereEndText     = "<|END_TEXT|>"
	cohereStartAction = "<|START_ACTION|>"
	cohereEndAction   = "<|END_ACTION|>"
	cohereEndOfTurn   = "<|END_OF_TURN_TOKEN|>"

	// Legacy response markers from the older Command A chat template, which
	// also ships in these models' tokenizer_config and shows up in sampled
	// output; treated as aliases for START_TEXT / END_TEXT.
	cohereStartResponse = "<|START_RESPONSE|>"
	cohereEndResponse   = "<|END_RESPONSE|>"
)

func (p *CohereParser) HasToolSupport() bool {
	return true
}

func (p *CohereParser) HasThinkingSupport() bool {
	return true
}

func (p *CohereParser) PreservedTokens() []string {
	return []string{
		"<|START_THINKING|>", cohereEndThinking,
		cohereStartText, cohereEndText,
		cohereStartAction, cohereEndAction,
		cohereStartResponse, cohereEndResponse,
	}
}

func (p *CohereParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.buffer.Reset()
	p.callIndex = 0

	// The template enables reasoning by default; nil means default.
	thinkingEnabled := thinkValue == nil || thinkValue.Bool()

	assistantPrefill := lastMessage != nil && lastMessage.Role == "assistant" && lastMessage.Content != ""
	switch {
	case assistantPrefill:
		// The prompt left an open <|START_TEXT|> for continuation.
		p.state = cohereCollectingContent
	case thinkingEnabled:
		p.state = cohereCollectingThinking
	default:
		p.state = cohereAwaitingBlock
	}
	return tools
}

func (p *CohereParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)

	var contentSb, thinkingSb strings.Builder
	for {
		c, t, tc, more := p.eat(done)
		contentSb.WriteString(c)
		thinkingSb.WriteString(t)
		calls = append(calls, tc...)
		if !more {
			break
		}
	}

	for i := range calls {
		calls[i].Function.Index = p.callIndex
		p.callIndex++
	}

	return contentSb.String(), thinkingSb.String(), calls, nil
}

// eat consumes what it can from the buffer for the current state. It returns
// more=true when a state transition happened and the remaining buffer should
// be reprocessed.
func (p *CohereParser) eat(done bool) (content string, thinking string, calls []api.ToolCall, more bool) {
	buf := p.buffer.String()
	if buf == "" {
		return "", "", nil, false
	}

	switch p.state {
	case cohereCollectingThinking:
		if idx := strings.Index(buf, cohereEndThinking); idx != -1 {
			thinking = strings.TrimRightFunc(buf[:idx], unicode.IsSpace)
			p.resetBuffer(buf[idx+len(cohereEndThinking):])
			p.state = cohereAwaitingBlock
			return "", thinking, nil, true
		}
		// Emit all but a possible partial tag / trailing whitespace.
		keep := overlap(buf, cohereEndThinking)
		emitEnd := len(buf) - keep
		emitEnd -= trailingWhitespaceLen(buf[:emitEnd])
		if emitEnd > 0 {
			thinking = buf[:emitEnd]
			p.resetBuffer(buf[emitEnd:])
		}
		return "", thinking, nil, false

	case cohereAwaitingBlock:
		// Between blocks: look for the next text or action opener, skipping
		// whitespace and the end-of-turn token.
		for _, open := range []string{cohereStartText, cohereStartResponse} {
			if idx := strings.Index(buf, open); idx != -1 {
				p.resetBuffer(buf[idx+len(open):])
				p.state = cohereCollectingContent
				return "", "", nil, true
			}
		}
		if idx := strings.Index(buf, cohereStartAction); idx != -1 {
			p.resetBuffer(buf[idx+len(cohereStartAction):])
			p.state = cohereCollectingAction
			return "", "", nil, true
		}
		trimmed := strings.TrimLeftFunc(buf, unicode.IsSpace)
		if strings.HasPrefix(trimmed, cohereEndOfTurn) {
			p.resetBuffer(trimmed[len(cohereEndOfTurn):])
			return "", "", nil, true
		}
		if trimmed == "" {
			if done {
				p.buffer.Reset()
			}
			return "", "", nil, false
		}
		// Wait only while the buffer could still grow into one of the
		// expected tags. Anything else — bare content or an unrecognized
		// tag — streams out as content rather than buffering forever.
		if !done && maybePartialTag(trimmed) {
			return "", "", nil, false
		}
		p.buffer.Reset()
		p.state = cohereCollectingContent
		p.buffer.WriteString(trimmed)
		return "", "", nil, true

	case cohereCollectingContent:
		// END_OF_TURN also closes content for models that skip END_TEXT.
		for _, close := range []string{cohereEndText, cohereEndResponse, cohereEndOfTurn} {
			if idx := strings.Index(buf, close); idx != -1 {
				content = buf[:idx]
				p.resetBuffer(buf[idx+len(close):])
				p.state = cohereAwaitingBlock
				return content, "", nil, true
			}
		}
		keep := max(overlap(buf, cohereEndText), overlap(buf, cohereEndResponse), overlap(buf, cohereEndOfTurn))
		emitEnd := len(buf) - keep
		if emitEnd > 0 {
			content = buf[:emitEnd]
			p.resetBuffer(buf[emitEnd:])
		}
		if done && p.buffer.Len() > 0 {
			content += p.buffer.String()
			p.buffer.Reset()
		}
		return content, "", nil, false

	case cohereCollectingAction:
		if idx := strings.Index(buf, cohereEndAction); idx != -1 {
			payload := buf[:idx]
			p.resetBuffer(buf[idx+len(cohereEndAction):])
			p.state = cohereAwaitingBlock
			calls = parseCohereActions(payload)
			return "", "", calls, true
		}
		if done {
			// Best effort on truncated output.
			calls = parseCohereActions(buf)
			p.buffer.Reset()
			return "", "", calls, false
		}
		return "", "", nil, false
	}

	return "", "", nil, false
}

func (p *CohereParser) resetBuffer(s string) {
	p.buffer.Reset()
	p.buffer.WriteString(s)
}

// maybePartialTag reports whether s is a proper prefix of one of the tags the
// awaiting-block state recognizes — the only case worth waiting on for more
// output before treating the buffer as content.
func maybePartialTag(s string) bool {
	for _, tag := range []string{cohereStartText, cohereStartResponse, cohereStartAction, cohereEndOfTurn} {
		if len(s) < len(tag) && strings.HasPrefix(tag, s) {
			return true
		}
	}
	return false
}

type cohereToolCall struct {
	ToolCallID string                        `json:"tool_call_id"`
	ToolName   string                        `json:"tool_name"`
	Parameters api.ToolCallFunctionArguments `json:"parameters"`
}

func (c cohereToolCall) toolCall() api.ToolCall {
	return api.ToolCall{
		ID: c.ToolCallID,
		Function: api.ToolCallFunction{
			Name:      c.ToolName,
			Arguments: c.Parameters,
		},
	}
}

// parseCohereActions parses the JSON array inside an action block:
// [{"tool_call_id": "0", "tool_name": ..., "parameters": {...}}, ...]
//
// Sampled output occasionally malforms the JSON (a missing comma between
// calls, an unquoted value). When the array as a whole fails to parse, fall
// back to scanning its balanced top-level objects and parse each
// independently, so one bad call doesn't drop its siblings.
func parseCohereActions(payload string) []api.ToolCall {
	payload = strings.TrimSpace(payload)
	if payload == "" {
		return nil
	}

	var parsed []cohereToolCall
	if err := json.Unmarshal([]byte(payload), &parsed); err == nil {
		calls := make([]api.ToolCall, 0, len(parsed))
		for _, c := range parsed {
			calls = append(calls, c.toolCall())
		}
		return calls
	}

	var calls []api.ToolCall
	for _, obj := range scanJSONObjects(payload) {
		var c cohereToolCall
		if err := json.Unmarshal([]byte(obj), &c); err != nil {
			if len(obj) > 200 {
				obj = obj[:200] + "…"
			}
			slog.Warn("cohere action parsing failed", "error", err, "action", obj)
			continue
		}
		calls = append(calls, c.toolCall())
	}
	return calls
}

// scanJSONObjects returns the balanced top-level {...} chunks of s, tracking
// strings and escapes so braces inside values don't split objects.
func scanJSONObjects(s string) []string {
	var objects []string
	depth, start := 0, -1
	inString, escaped := false, false
	for i := range len(s) {
		switch c := s[i]; {
		case escaped:
			escaped = false
		case c == '\\' && inString:
			escaped = true
		case c == '"':
			inString = !inString
		case inString:
		case c == '{':
			if depth == 0 {
				start = i
			}
			depth++
		case c == '}':
			if depth > 0 {
				depth--
				if depth == 0 && start >= 0 {
					objects = append(objects, s[start:i+1])
					start = -1
				}
			}
		}
	}
	return objects
}
