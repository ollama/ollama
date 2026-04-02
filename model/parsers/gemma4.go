package parsers

import (
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

type Gemma4ParserState int

const (
	Gemma4CollectingContent Gemma4ParserState = iota
	Gemma4CollectingThinking
	Gemma4CollectingToolCall
)

const (
	gemma4ThinkingOpenTag  = "<|channel>"
	gemma4ThinkingCloseTag = "<channel|>"
	gemma4ToolCallOpenTag  = "<|tool_call>"
	gemma4ToolCallCloseTag = "<tool_call|>"
)

type Gemma4Parser struct {
	state                 Gemma4ParserState
	buffer                strings.Builder
	hasThinkingSupport    bool
	thinkingEnabled       bool // true when both model supports and user requested thinking
	needsChannelNameStrip bool // true when we just entered thinking and need to strip "thought\n"
}

func (p *Gemma4Parser) HasToolSupport() bool {
	return true
}

func (p *Gemma4Parser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *Gemma4Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	p.thinkingEnabled = p.HasThinkingSupport() && (thinkValue != nil && thinkValue.Bool())

	if !p.thinkingEnabled {
		p.state = Gemma4CollectingContent
		return tools
	}

	if prefill && lastMessage.Content != "" {
		p.state = Gemma4CollectingContent
		return tools
	}

	// When thinking is enabled, start in content mode but we'll switch to
	// thinking when we see <|channel>. The model typically starts with
	// <|channel> immediately when thinking is enabled.
	p.state = Gemma4CollectingContent
	return tools
}

type gemma4Event interface {
	isGemma4Event()
}

type gemma4EventThinkingContent struct {
	content string
}

type gemma4EventContent struct {
	content string
}

type gemma4EventToolCall struct {
	toolCall api.ToolCall
}

func (gemma4EventThinkingContent) isGemma4Event() {}
func (gemma4EventContent) isGemma4Event()         {}
func (gemma4EventToolCall) isGemma4Event()        {}

func (p *Gemma4Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents(done)

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case gemma4EventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		case gemma4EventThinkingContent:
			if p.thinkingEnabled {
				thinkingSb.WriteString(event.content)
			}
			// When thinking is disabled, silently discard channel content
		case gemma4EventContent:
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *Gemma4Parser) parseEvents(done bool) []gemma4Event {
	var all []gemma4Event

	keepLooping := true
	for keepLooping {
		var events []gemma4Event
		events, keepLooping = p.eat(done)
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

// longestOverlap returns the longest overlap between the suffix of bufStr and
// a prefix of any of the given tags.
func longestOverlap(bufStr string, tags ...string) int {
	maxOverlap := 0
	for _, tag := range tags {
		if o := overlap(bufStr, tag); o > maxOverlap {
			maxOverlap = o
		}
	}
	return maxOverlap
}

func (p *Gemma4Parser) eat(done bool) ([]gemma4Event, bool) {
	var events []gemma4Event
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case Gemma4CollectingContent:
		// Check for thinking open tag
		if idx := strings.Index(bufStr, gemma4ThinkingOpenTag); idx != -1 {
			contentBefore := bufStr[:idx]
			remaining := bufStr[idx+len(gemma4ThinkingOpenTag):]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Gemma4CollectingThinking
			p.needsChannelNameStrip = true

			if contentBefore = strings.TrimRightFunc(contentBefore, unicode.IsSpace); len(contentBefore) > 0 {
				events = append(events, gemma4EventContent{content: contentBefore})
			}
			return events, true
		}

		// Check for tool call open tag
		if idx := strings.Index(bufStr, gemma4ToolCallOpenTag); idx != -1 {
			contentBefore := bufStr[:idx]
			remaining := bufStr[idx+len(gemma4ToolCallOpenTag):]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Gemma4CollectingToolCall

			if contentBefore = strings.TrimRightFunc(contentBefore, unicode.IsSpace); len(contentBefore) > 0 {
				events = append(events, gemma4EventContent{content: contentBefore})
			}
			return events, true
		}

		// Check for partial tag overlap
		if !done {
			if overlapLen := longestOverlap(bufStr, gemma4ThinkingOpenTag, gemma4ToolCallOpenTag); overlapLen > 0 {
				beforePartialTag := bufStr[:len(bufStr)-overlapLen]
				trailingLen := trailingWhitespaceLen(beforePartialTag)
				ambiguousStart := len(beforePartialTag) - trailingLen

				unambiguous := bufStr[:ambiguousStart]
				ambiguous := bufStr[ambiguousStart:]
				p.buffer.Reset()
				p.buffer.WriteString(ambiguous)
				if len(unambiguous) > 0 {
					events = append(events, gemma4EventContent{content: unambiguous})
				}
				return events, false
			}
		}

		// No tags found, emit all content
		p.buffer.Reset()
		if len(bufStr) > 0 {
			events = append(events, gemma4EventContent{content: bufStr})
		}
		return events, false

	case Gemma4CollectingThinking:
		// Strip channel name (e.g., "thought\n") after <|channel>.
		// Gemma 4 format: <|channel>thought\n...content...<channel|>
		// In streaming mode, "thought" and "\n" may arrive in separate chunks.
		if p.needsChannelNameStrip {
			if strings.HasPrefix(bufStr, "thought\n") {
				bufStr = bufStr[len("thought\n"):]
				p.buffer.Reset()
				p.buffer.WriteString(bufStr)
				p.needsChannelNameStrip = false
			} else if !done && (bufStr == "thought" || strings.HasPrefix("thought\n", bufStr)) {
				// Partial match — wait for more data.
				return events, false
			} else {
				// No match (different channel name or no newline) — don't strip.
				p.needsChannelNameStrip = false
			}
		}

		if strings.Contains(bufStr, gemma4ThinkingCloseTag) {
			split := strings.SplitN(bufStr, gemma4ThinkingCloseTag, 2)
			thinking := strings.TrimRightFunc(split[0], unicode.IsSpace)
			remaining := strings.TrimLeftFunc(split[1], unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Gemma4CollectingContent

			if len(thinking) > 0 {
				events = append(events, gemma4EventThinkingContent{content: thinking})
			}
			return events, true
		}

		// Check for partial close tag
		if !done {
			if overlapLen := overlap(bufStr, gemma4ThinkingCloseTag); overlapLen > 0 {
				beforePartialTag := bufStr[:len(bufStr)-overlapLen]
				trailingLen := trailingWhitespaceLen(beforePartialTag)
				ambiguousStart := len(beforePartialTag) - trailingLen

				unambiguous := bufStr[:ambiguousStart]
				ambiguous := bufStr[ambiguousStart:]
				p.buffer.Reset()
				p.buffer.WriteString(ambiguous)
				if len(unambiguous) > 0 {
					events = append(events, gemma4EventThinkingContent{content: unambiguous})
				}
				return events, false
			}
		}

		// No close tag, emit thinking content (hold back trailing whitespace)
		if !done {
			whitespaceLen := trailingWhitespaceLen(bufStr)
			ambiguousStart := len(bufStr) - whitespaceLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, gemma4EventThinkingContent{content: unambiguous})
			}
		} else {
			p.buffer.Reset()
			if len(bufStr) > 0 {
				events = append(events, gemma4EventThinkingContent{content: bufStr})
			}
		}
		return events, false

	case Gemma4CollectingToolCall:
		if idx := strings.Index(bufStr, gemma4ToolCallCloseTag); idx != -1 {
			toolCallContent := bufStr[:idx]
			remaining := bufStr[idx+len(gemma4ToolCallCloseTag):]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Gemma4CollectingContent

			if toolCall, err := parseGemma4ToolCall(toolCallContent); err == nil {
				events = append(events, gemma4EventToolCall{toolCall: toolCall})
			} else {
				slog.Warn("gemma4 tool call parsing failed", "error", err, "content", toolCallContent)
			}
			return events, true
		}

		// If done, flush any accumulated tool call content even without closing tag.
		// The model may hit a stop token before emitting <tool_call|>.
		if done && len(bufStr) > 0 {
			p.buffer.Reset()
			p.state = Gemma4CollectingContent
			if toolCall, err := parseGemma4ToolCall(bufStr); err == nil {
				events = append(events, gemma4EventToolCall{toolCall: toolCall})
			} else {
				slog.Warn("gemma4 tool call flush on done failed", "error", err, "content", bufStr)
			}
			return events, false
		}

		// Wait for closing tag
		return events, false
	}

	return events, false
}

// parseGemma4ToolCall parses a tool call in Gemma 4 format:
// call:NAME{key:value,key:value}
func parseGemma4ToolCall(content string) (api.ToolCall, error) {
	// Expected format: call:NAME{args}
	if !strings.HasPrefix(content, "call:") {
		return api.ToolCall{}, errors.New("expected 'call:' prefix")
	}
	content = content[len("call:"):]

	// Find the opening brace for args
	braceIdx := strings.Index(content, "{")
	if braceIdx == -1 {
		return api.ToolCall{}, errors.New("expected '{' in tool call")
	}

	toolName := strings.TrimSpace(content[:braceIdx])
	argsStr := content[braceIdx:]

	// Convert Gemma 4 argument format to JSON
	jsonStr := gemma4ArgsToJSON(argsStr)

	var args api.ToolCallFunctionArguments
	if err := json.Unmarshal([]byte(jsonStr), &args); err != nil {
		return api.ToolCall{}, err
	}

	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      toolName,
			Arguments: args,
		},
	}, nil
}

// gemma4ArgsToJSON converts Gemma 4's custom argument format to valid JSON.
func gemma4ArgsToJSON(s string) string {
	s = strings.ReplaceAll(s, `<|"|>`, `"`)

	var buf strings.Builder
	buf.Grow(len(s) + 32)
	inString := false
	hex := "0123456789abcdef"
	i := 0
	for i < len(s) {
		ch := s[i]

		if ch == '"' {
			inString = !inString
			buf.WriteByte('"')
			i++
			continue
		}

		if inString {
			switch ch {
			case '\\':
				buf.WriteString(`\\`)
			case '\n':
				buf.WriteString(`\n`)
			case '\r':
				buf.WriteString(`\r`)
			case '\t':
				buf.WriteString(`\t`)
			case '\b':
				buf.WriteString(`\b`)
			case '\f':
				buf.WriteString(`\f`)
			default:
				if ch < 0x20 {
					buf.WriteString(`\u00`)
					buf.WriteByte(hex[ch>>4])
					buf.WriteByte(hex[ch&0x0f])
				} else {
					buf.WriteByte(ch)
				}
			}
			i++
			continue
		}

		if !inString && isIdentStart(ch) {
			j := i + 1
			for j < len(s) && isIdentPart(s[j]) {
				j++
			}
			word := s[i:j]
			if j < len(s) && s[j] == ':' {
				buf.WriteByte('"')
				buf.WriteString(word)
				buf.WriteByte('"')
			} else {
				buf.WriteString(word)
			}
			i = j
		} else {
			buf.WriteByte(ch)
			i++
		}
	}
	return buf.String()
}
