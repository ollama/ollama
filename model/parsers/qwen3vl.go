package parsers

import (
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

// TODO: call the init function
const (
	CollectingThinkingContent qwenParserState = iota
	CollectingContent
	CollectingToolContent
)

const (
	thinkingCloseTag = "</think>"
)

type Qwen3VLParser struct {
	state              qwenParserState
	buffer             strings.Builder
	tools              []api.Tool
	hasThinkingSupport bool
}

func (p *Qwen3VLParser) HasToolSupport() bool {
	return true
}

func (p *Qwen3VLParser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *Qwen3VLParser) setInitialState(lastMessage *api.Message) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"
	if !p.HasThinkingSupport() {
		p.state = CollectingContent
		return
	}

	if prefill && lastMessage.Content != "" {
		p.state = CollectingContent
		return
	}

	p.state = CollectingThinkingContent
}

func (p *Qwen3VLParser) Init(tools []api.Tool, lastMessage *api.Message) []api.Tool {
	p.tools = tools
	p.setInitialState(lastMessage)
	return tools
}

type qwenEventThinkingContent struct {
	content string
}

func (qwenEventThinkingContent) isQwenEvent() {}

func (p *Qwen3VLParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case qwenEventRawToolCall:
			toolCall, err := parseJSONToolCall(event, p.tools)
			if err != nil {
				slog.Warn("qwen tool call parsing failed", "error", err)
				return "", "", nil, err
			}
			toolCalls = append(toolCalls, toolCall)
		case qwenEventThinkingContent:
			thinkingSb.WriteString(event.content)
		case qwenEventContent:
			// TODO(drifkin): if the same turn contains multiple interleaved content
			// events, we naively append them together here.
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *Qwen3VLParser) parseEvents() []qwenEvent {
	var all []qwenEvent

	keepLooping := true
	for keepLooping {
		var events []qwenEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "qwen events parsed", "events", all, "state", p.state, "buffer", p.buffer.String())
	}

	return all
}

func emitContentBeforeTag(p *Qwen3VLParser, events []qwenEvent, tag string) []qwenEvent {
	split := strings.SplitN(p.buffer.String(), tag, 2)
	before := split[0]
	before = strings.TrimRightFunc(before, unicode.IsSpace)
	if len(before) > 0 {
		events = append(events, qwenEventContent{content: before})
	}
	after := split[1]
	p.buffer.Reset()
	p.buffer.WriteString(after)
	return events
}

func (p *Qwen3VLParser) eat() ([]qwenEvent, bool) {
	var events []qwenEvent

	switch p.state {
	case CollectingContent:
		if strings.Contains(p.buffer.String(), toolOpenTag) {
			events = emitContentBeforeTag(p, events, toolOpenTag)
			p.state = CollectingToolContent
			return events, true
		} else if overlapLen := overlap(p.buffer.String(), toolOpenTag); overlapLen > 0 {
			beforePartialTag := p.buffer.String()[:len(p.buffer.String())-overlapLen]
			trailingWhitespaceLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen

			unambiguous := p.buffer.String()[:ambiguousStart]
			ambiguous := p.buffer.String()[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwenEventContent{content: unambiguous})
			}
			return events, false
		} else {
			whitespaceLen := trailingWhitespaceLen(p.buffer.String())
			ambiguousStart := len(p.buffer.String()) - whitespaceLen

			unambiguous := p.buffer.String()[:ambiguousStart]
			ambiguous := p.buffer.String()[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwenEventContent{content: unambiguous})
			}
			return events, false
		}
	case CollectingToolContent:
		if strings.Contains(p.buffer.String(), toolCloseTag) {
			split := strings.SplitN(p.buffer.String(), toolCloseTag, 2)
			before := split[0] // do we also need to do it to tool calls?
			if len(before) == 0 {
				slog.Warn("qwen tool call closing tag found but no content before it")
			}

			after := strings.TrimLeftFunc(split[1], unicode.IsSpace)
			events = append(events, qwenEventRawToolCall{raw: before})
			p.buffer.Reset()
			p.buffer.WriteString(after)
			p.state = CollectingContent
			return events, true
		} else {
			return events, false
		}
	case CollectingThinkingContent:
		if strings.Contains(p.buffer.String(), thinkingCloseTag) {
			split := strings.SplitN(p.buffer.String(), thinkingCloseTag, 2)
			// before := split[0]
			before := strings.TrimRightFunc(split[0], unicode.IsSpace)
			after := strings.TrimLeftFunc(split[1], unicode.IsSpace)
			if len(before) > 0 {
				events = append(events, qwenEventThinkingContent{content: before})
			}
			p.buffer.Reset()
			p.buffer.WriteString(after)
			p.state = CollectingContent
			return events, true
		} else if overlapLen := overlap(p.buffer.String(), thinkingCloseTag); overlapLen > 0 {
			beforePartialTag := p.buffer.String()[:len(p.buffer.String())-overlapLen]
			trailingWhitespaceLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen

			unambiguous := p.buffer.String()[:ambiguousStart]
			ambiguous := p.buffer.String()[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwenEventThinkingContent{content: unambiguous})
			}
			return events, false
		} else {
			whitespaceLen := trailingWhitespaceLen(p.buffer.String())
			ambiguousStart := len(p.buffer.String()) - whitespaceLen

			unambiguous := p.buffer.String()[:ambiguousStart]
			ambiguous := p.buffer.String()[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwenEventThinkingContent{content: unambiguous})
			}
			return events, false
		}
	default:
		panic("unreachable")
	}
}

func parseJSONToolCall(raw qwenEventRawToolCall, tools []api.Tool) (api.ToolCall, error) {
	var toolCallFunction api.ToolCallFunction
	if err := json.Unmarshal([]byte(raw.raw), &toolCallFunction); err != nil {
		return api.ToolCall{}, err
	}

	toolCall := api.ToolCall{}
	toolCall.Function = toolCallFunction

	return toolCall, nil
}
