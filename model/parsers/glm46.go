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

const (
	glm46CollectingContent glm46ParserState = iota
	CollectingThinkingContent
	CollectingToolContent
)

const (
	thinkingCloseTag = "</think>"
)

// TODO(gguo): add a field for isThinking
type GLM46Parser struct {
	state  qwenParserState
	buffer strings.Builder
	tools  []api.Tool
}

func (p *GLM46Parser) HasToolSupport() bool {
	return true
}

// TODO(gguo): changes this to reference an objects param
func (p *GLM46Parser) HasThinkingSupport() bool {
	return true
}

func (p *GLM46Parser) Init(tools []api.Tool, lastMessage *api.Message) []api.Tool {
	p.tools = tools
	// p.state = p.initialState()
	return tools
}

type glm46EventThinkingContent struct {
	content string
}

func (glm46EventThinkingContent) isGLM46Event() {}

func (p *GLM46Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var sb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case glm46EventRawToolCall:
			toolCall, err := parseJSONToolCall(event, p.tools)
			if err != nil {
				slog.Warn("qwen tool call parsing failed", "error", err)
				return "", "", nil, err
			}
			toolCalls = append(toolCalls, toolCall)
		case glm46EventThinkingContent:
			sb.WriteString(event.content)
		case glm46EventContent:
			// TODO(drifkin): if the same turn contains multiple interleaved content
			// events, we naively append them together here.
			sb.WriteString(event.content)
		}
	}

	return sb.String(), "", toolCalls, nil
}

func (p *GLM46Parser) parseEvents() []glm46Event {
	var all []glm46Event

	keepLooping := true
	for keepLooping {
		var events []glm46Event
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

func emitContentBeforeTag(p *GLM46Parser, events []glm46Event, tag string) []glm46Event {
	split := strings.SplitN(p.buffer.String(), tag, 2)
	before := split[0]
	before = strings.TrimRightFunc(before, unicode.IsSpace)
	if len(before) > 0 {
		events = append(events, glm46EventContent{content: before})
	}
	after := split[1]
	p.buffer.Reset()
	p.buffer.WriteString(after)
	return events
}

func (p *GLM46Parser) eat() ([]glm46Event, bool) {
	var events []glm46Event

	switch p.state {
	case glm46CollectingContent:
		if strings.Contains(p.buffer.String(), toolOpenTag) {
			events = emitContentBeforeTag(p, events, toolOpenTag)
			p.state = glm46CollectingToolContent
			return events, true
		} else if overlapLen := overlap(p.buffer.String(), toolOpenTag); overlapLen > 0 {
			beforePartialTag := p.buffer.String()[:len(p.buffer.String())-overlapLen]
			trailingWhitespaceLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen

			unambiguous := p.buffer.String()[:ambiguousStart]
			ambiguous := p.buffer.String()[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 { // why does qwen3coder not have this here
				events = append(events, glm46EventContent{content: unambiguous})
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
				events = append(events, glm46EventContent{content: unambiguous})
			}
			return events, false
		}
	case CollectingToolContent:
		if strings.Contains(p.buffer.String(), glm46ToolCloseTag) {
			split := strings.SplitN(p.buffer.String(), toolCloseTag, 2)
			before := split[0]
			if len(before) == 0 {
				slog.Warn("qwen tool call closing tag found but no content before it")
			}

			after := strings.TrimLeftFunc(split[1], unicode.IsSpace)
			events = append(events, glm46EventRawToolCall{raw: before})
			p.buffer.Reset()
			p.buffer.WriteString(after)
			p.state = glm46CollectingContent
			return events, true
		} else {
			return events, false
		}
	case glm46CollectingThinkingContent: // so we want to hip the unambiguous stuff
		if strings.Contains(p.buffer.String(), thinkingCloseTag) {
			split := strings.SplitN(p.buffer.String(), thinkingCloseTag, 2)
			before := split[0]
			if len(before) == 0 {
				slog.Warn("qwen tool call closing tag found but no content before it")
			}
			after := strings.TrimLeftFunc(split[1], unicode.IsSpace)
			if len(before) > 0 {
				events = append(events, glm46EventThinkingContent{content: before})
			}
			p.buffer.Reset()
			p.buffer.WriteString(after)
			p.state = glm46CollectingContent
			return events, true
		} else if overlapLen := overlap(p.buffer.String(), thinkingCloseTag); overlapLen > 0 { // we see part of a close thinking tag
			beforePartialTag := p.buffer.String()[:len(p.buffer.String())-overlapLen]
			trailingWhitespaceLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen

			unambiguous := p.buffer.String()[:ambiguousStart]
			ambiguous := p.buffer.String()[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, glm46EventThinkingContent{content: unambiguous})
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
				events = append(events, glm46EventThinkingContent{content: unambiguous})
			}
			return events, false
		}
	default:
		panic("unreachable")
	}
}

func parseJSONToolCall(raw glm46EventRawToolCall, tools []api.Tool) (api.ToolCall, error) {
	var toolCallFunction api.ToolCallFunction
	if err := json.Unmarshal([]byte(raw.raw), &toolCallFunction); err != nil {
		return api.ToolCall{}, err
	}

	toolCall := api.ToolCall{}
	toolCall.Function = toolCallFunction

	return toolCall, nil
}
