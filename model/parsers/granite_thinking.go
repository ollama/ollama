package parsers

import (
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

// GraniteThinkingParser implements the parser for the "granite thinking" chat
// template — the ChatML-with-<think> template used starting with Granite 4.2.
// It is not tied to a specific Granite version: older Granite models (4.1 and
// earlier) use a different, non-thinking template and must not be routed here
// (see create/metadata.go's graniteThinkingTemplateName, which gates on the
// template actually containing <think>).
type GraniteThinkingParser struct {
	state              graniteThinkingParserState
	buffer             strings.Builder
	tools              []api.Tool
	callIndex          int
	hasThinkingSupport bool
}

type graniteThinkingParserState int

const (
	graniteThinkingStateThinking graniteThinkingParserState = iota
	graniteThinkingStateContent
	graniteThinkingStateToolContent
)

const graniteThinkingCloseTag = "</think>"

func (p *GraniteThinkingParser) HasToolSupport() bool {
	return true
}

func (p *GraniteThinkingParser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *GraniteThinkingParser) ThinkingClose() []string {
	if p.state == graniteThinkingStateThinking {
		return []string{graniteThinkingCloseTag}
	}
	return nil
}

func (p *GraniteThinkingParser) PreservedTokens() []string {
	return []string{
		graniteThinkingCloseTag,
		toolOpenTag,
		toolCloseTag,
	}
}

func (p *GraniteThinkingParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.callIndex = 0
	p.buffer.Reset()

	// Granite's template never leaves a trailing assistant turn open for
	// continuation (prefill): every assistant message is closed with
	// <|im_end|>, and a fresh "<|im_start|>assistant\n<think>..." preamble is
	// always appended for the turn being generated. So the initial parse
	// state depends only on whether thinking is enabled for this request, not
	// on the role of the last input message.
	thinkingEnabled := p.HasThinkingSupport() && thinkValue != nil && thinkValue.Bool()
	if thinkingEnabled {
		p.state = graniteThinkingStateThinking
	} else {
		p.state = graniteThinkingStateContent
	}

	return tools
}

type graniteThinkingEvent interface{ isGraniteThinkingEvent() }

type graniteThinkingEventThinking struct{ content string }
type graniteThinkingEventContent struct{ content string }
type graniteThinkingEventToolCall struct{ toolCall api.ToolCall }

func (graniteThinkingEventThinking) isGraniteThinkingEvent() {}
func (graniteThinkingEventContent) isGraniteThinkingEvent()  {}
func (graniteThinkingEventToolCall) isGraniteThinkingEvent() {}

func (p *GraniteThinkingParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)

	events := p.parseEvents()

	if done && p.buffer.Len() > 0 {
		switch p.state {
		case graniteThinkingStateToolContent:
			// An unterminated tool call at end-of-stream: surface it as content
			// (matching Qwen3-Coder's behavior) rather than dropping it silently.
			events = append(events, graniteThinkingEventContent{content: toolOpenTag + p.buffer.String()})
		default:
			events = append(events, graniteThinkingEventContent{content: p.buffer.String()})
		}
		p.buffer.Reset()
		p.state = graniteThinkingStateContent
	}

	var toolCalls []api.ToolCall
	var contentSb, thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case graniteThinkingEventThinking:
			thinkingSb.WriteString(event.content)
		case graniteThinkingEventContent:
			contentSb.WriteString(event.content)
		case graniteThinkingEventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		}
	}

	for i := range toolCalls {
		toolCalls[i].Function.Index = p.callIndex
		p.callIndex++
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *GraniteThinkingParser) parseEvents() []graniteThinkingEvent {
	var all []graniteThinkingEvent
	keepLooping := true
	for keepLooping {
		var events []graniteThinkingEvent
		events, keepLooping = p.eat()
		all = append(all, events...)
	}
	return all
}

func (p *GraniteThinkingParser) eat() ([]graniteThinkingEvent, bool) {
	var events []graniteThinkingEvent
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case graniteThinkingStateThinking:
		if strings.Contains(bufStr, graniteThinkingCloseTag) {
			split := strings.SplitN(bufStr, graniteThinkingCloseTag, 2)
			thinking := split[0]
			remaining := strings.TrimLeftFunc(split[1], unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = graniteThinkingStateContent
			if len(thinking) > 0 {
				events = append(events, graniteThinkingEventThinking{content: thinking})
			}
			return events, true
		}
		if n := overlap(bufStr, graniteThinkingCloseTag); n > 0 {
			unambiguous := bufStr[:len(bufStr)-n]
			ambiguous := bufStr[len(bufStr)-n:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, graniteThinkingEventThinking{content: unambiguous})
			}
			return events, false
		}
		// withhold trailing whitespace in case it precedes the close tag
		wsLen := trailingWhitespaceLen(bufStr)
		unambiguous := bufStr[:len(bufStr)-wsLen]
		ambiguous := bufStr[len(bufStr)-wsLen:]
		p.buffer.Reset()
		p.buffer.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, graniteThinkingEventThinking{content: unambiguous})
		}
		return events, false

	case graniteThinkingStateContent:
		if strings.Contains(bufStr, toolOpenTag) {
			split := strings.SplitN(bufStr, toolOpenTag, 2)
			before := strings.TrimRightFunc(split[0], unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(split[1])
			p.state = graniteThinkingStateToolContent
			if len(before) > 0 {
				events = append(events, graniteThinkingEventContent{content: before})
			}
			return events, true
		}
		if n := overlap(bufStr, toolOpenTag); n > 0 {
			beforePartial := bufStr[:len(bufStr)-n]
			trailingLen := trailingWhitespaceLen(beforePartial)
			ambiguousStart := len(beforePartial) - trailingLen
			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, graniteThinkingEventContent{content: unambiguous})
			}
			return events, false
		}
		wsLen := trailingWhitespaceLen(bufStr)
		ambiguousStart := len(bufStr) - wsLen
		unambiguous := bufStr[:ambiguousStart]
		ambiguous := bufStr[ambiguousStart:]
		p.buffer.Reset()
		p.buffer.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, graniteThinkingEventContent{content: unambiguous})
		}
		return events, false

	case graniteThinkingStateToolContent:
		if strings.Contains(bufStr, toolCloseTag) {
			split := strings.SplitN(bufStr, toolCloseTag, 2)
			raw := split[0]
			remaining := strings.TrimLeftFunc(split[1], unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = graniteThinkingStateContent

			toolCall, err := parseToolCall(qwenEventRawToolCall{raw: raw}, p.tools)
			if err != nil {
				slog.Warn("granite-thinking tool call parsing failed", "error", err)
				return events, true
			}
			events = append(events, graniteThinkingEventToolCall{toolCall: toolCall})
			return events, true
		}
		return events, false
	}

	return events, false
}
