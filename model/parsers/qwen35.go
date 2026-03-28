package parsers

import (
	"context"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

type qwen35ParserState int

const (
	qwen35ParserStateCollectingThinking qwen35ParserState = iota
	qwen35ParserStateThinkingDoneEatingWhitespace
	qwen35ParserStateCollectingContent
)

const (
	qwen35ThinkingOpenTag  = "<think>"
	qwen35ThinkingCloseTag = "</think>"
	qwen35ToolCallOpenTag  = "<tool_call>"
)

// Qwen35Parser handles qwen3.5 reasoning extraction and delegates post-thinking
// content (including XML tool calls) to Qwen3CoderParser.
type Qwen35Parser struct {
	toolParser Qwen3CoderParser

	state  qwen35ParserState
	buffer strings.Builder
	// Some checkpoints may emit an explicit leading <think> even when the
	// prompt already opened thinking. Strip at most one such tag.
	allowLeadingThinkOpenTag bool
}

func (p *Qwen35Parser) HasToolSupport() bool {
	return true
}

func (p *Qwen35Parser) HasThinkingSupport() bool {
	return true
}

func (p *Qwen35Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.buffer.Reset()
	p.toolParser = Qwen3CoderParser{}
	p.toolParser.Init(tools, nil, nil)

	thinkingEnabled := thinkValue != nil && thinkValue.Bool()
	if thinkValue == nil {
		thinkingEnabled = true
	}

	assistantPrefill := lastMessage != nil && lastMessage.Role == "assistant" && lastMessage.Content != ""
	if thinkingEnabled && !assistantPrefill {
		p.state = qwen35ParserStateCollectingThinking
		p.allowLeadingThinkOpenTag = true
	} else {
		p.state = qwen35ParserStateCollectingContent
		p.allowLeadingThinkOpenTag = false
	}

	return tools
}

type qwen35Event interface {
	isQwen35Event()
}

type qwen35EventContent struct {
	content string
}

func (qwen35EventContent) isQwen35Event() {}

type qwen35EventThinkingContent struct {
	content string
}

func (qwen35EventThinkingContent) isQwen35Event() {}

func (p *Qwen35Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case qwen35EventContent:
			parsedContent, _, parsedCalls, err := p.toolParser.Add(event.content, done)
			if err != nil {
				slog.Warn("qwen3.5 tool call parsing failed", "error", err)
				return "", "", nil, err
			}
			contentSb.WriteString(parsedContent)
			calls = append(calls, parsedCalls...)
		case qwen35EventThinkingContent:
			thinkingSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), calls, nil
}

func (p *Qwen35Parser) parseEvents() []qwen35Event {
	var all []qwen35Event

	keepLooping := true
	for keepLooping {
		var events []qwen35Event
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "qwen3.5 events parsed", "events", all, "state", p.state, "buffer", p.buffer.String())
	}

	return all
}

func (p *Qwen35Parser) splitAtTag(tag string, trimAfter bool) (string, string) {
	return splitAtTag(&p.buffer, tag, trimAfter)
}

func (p *Qwen35Parser) eatLeadingWhitespaceAndTransitionTo(nextState qwen35ParserState) ([]qwen35Event, bool) {
	trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
	p.buffer.Reset()
	if trimmed == "" {
		return nil, false
	}
	p.state = nextState
	p.buffer.WriteString(trimmed)
	return nil, true
}

// maybeConsumeLeadingThinkOpenTag handles a single optional leading <think> tag.
// Returns (handled, shouldContinueParsingNow).
func (p *Qwen35Parser) maybeConsumeLeadingThinkOpenTag(acc string) (bool, bool) {
	if !p.allowLeadingThinkOpenTag {
		return false, false
	}

	trimmed := strings.TrimLeftFunc(acc, unicode.IsSpace)
	if strings.HasPrefix(trimmed, qwen35ThinkingOpenTag) {
		after := strings.TrimPrefix(trimmed, qwen35ThinkingOpenTag)
		after = strings.TrimLeftFunc(after, unicode.IsSpace)
		p.buffer.Reset()
		p.buffer.WriteString(after)
		if after == "" {
			return true, false
		}
		p.allowLeadingThinkOpenTag = false
		return true, true
	}

	if strings.HasPrefix(qwen35ThinkingOpenTag, trimmed) {
		return true, false
	}

	p.allowLeadingThinkOpenTag = false
	return false, false
}

func (p *Qwen35Parser) eat() ([]qwen35Event, bool) {
	var events []qwen35Event

	switch p.state {
	case qwen35ParserStateCollectingThinking:
		acc := p.buffer.String()

		if handled, continueNow := p.maybeConsumeLeadingThinkOpenTag(acc); handled {
			return events, continueNow
		}

		if strings.Contains(acc, qwen35ThinkingCloseTag) {
			thinking, remaining := p.splitAtTag(qwen35ThinkingCloseTag, true)
			if len(thinking) > 0 {
				events = append(events, qwen35EventThinkingContent{content: thinking})
			}
			if remaining == "" {
				p.state = qwen35ParserStateThinkingDoneEatingWhitespace
			} else {
				p.state = qwen35ParserStateCollectingContent
			}
			return events, true
		} else if overlapLen := max(overlap(acc, qwen35ThinkingCloseTag), overlap(acc, qwen35ToolCallOpenTag)); overlapLen > 0 {
			beforePartialTag := acc[:len(acc)-overlapLen]
			trailingWsLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWsLen

			unambiguous := acc[:ambiguousStart]
			ambiguous := acc[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwen35EventThinkingContent{content: unambiguous})
			}
			return events, false
		} else if strings.Contains(acc, qwen35ToolCallOpenTag) {
			// qwen3.5:9b model forgets sometimes to use </think> tag before the <tool_call> block starts
			// this condition ends the Think block and continues with the <tool_call> when the tag
			// is found
			thinking, tooling := p.splitAtTag(qwen35ToolCallOpenTag, true)
			p.buffer.Reset()
			p.buffer.WriteString(thinking + qwen35ThinkingCloseTag + qwen35ToolCallOpenTag + tooling)
			return events, true
		}

		whitespaceLen := trailingWhitespaceLen(acc)
		ambiguousStart := len(acc) - whitespaceLen
		unambiguous := acc[:ambiguousStart]
		ambiguous := acc[ambiguousStart:]
		p.buffer.Reset()
		p.buffer.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, qwen35EventThinkingContent{content: unambiguous})
		}
		return events, false

	case qwen35ParserStateThinkingDoneEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(qwen35ParserStateCollectingContent)

	case qwen35ParserStateCollectingContent:
		if p.buffer.Len() == 0 {
			return events, false
		}

		content := p.buffer.String()
		p.buffer.Reset()
		if len(content) > 0 {
			events = append(events, qwen35EventContent{content: content})
		}
		return events, false

	default:
		slog.Warn("qwen3.5 parser entered unknown state; resetting to content mode", "state", p.state)
		p.state = qwen35ParserStateCollectingContent
		return events, false
	}
}
