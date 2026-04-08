package parsers

import (
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

type DeepSeek3ParserState int

const (
	DeepSeekCollectingThinking DeepSeek3ParserState = iota
	DeepSeekCollectingContent
	DeepSeekCollectingToolCalls
	DeepSeekCollectingToolOutput
)

const (
	deepseekThinkingCloseTag   = "</think>"
	deepseekToolCallsBeginTag  = "<｜tool▁calls▁begin｜>"
	deepseekToolCallsEndTag    = "<｜tool▁calls▁end｜>"
	deepseekToolCallBeginTag   = "<｜tool▁call▁begin｜>"
	deepseekToolCallEndTag     = "<｜tool▁call▁end｜>"
	deepseekToolSepTag         = "<｜tool▁sep｜>"
	deepseekToolOutputBeginTag = "<｜tool▁output▁begin｜>"
	deepseekToolOutputEndTag   = "<｜tool▁output▁end｜>"
)

type DeepSeek3Parser struct {
	state              DeepSeek3ParserState
	buffer             strings.Builder
	hasThinkingSupport bool
}

func (p *DeepSeek3Parser) HasToolSupport() bool {
	return true
}

func (p *DeepSeek3Parser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *DeepSeek3Parser) setInitialState(lastMessage *api.Message, tools []api.Tool, thinkValue *api.ThinkValue) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	// Check both model capability AND request preference
	thinkingEnabled := p.HasThinkingSupport() && (thinkValue != nil && thinkValue.Bool())

	if !thinkingEnabled {
		p.state = DeepSeekCollectingContent
		return
	}

	if prefill && lastMessage.Content != "" {
		p.state = DeepSeekCollectingContent
		return
	}

	p.state = DeepSeekCollectingThinking
}

func (p *DeepSeek3Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.setInitialState(lastMessage, tools, thinkValue)
	return tools
}

type deepseekEvent interface {
	isDeepSeekEvent()
}

type deepseekEventThinkingContent struct {
	content string
}

type deepseekEventContent struct {
	content string
}

type deepseekEventToolCall struct {
	toolCall api.ToolCall
}

func (deepseekEventThinkingContent) isDeepSeekEvent() {}
func (deepseekEventContent) isDeepSeekEvent()         {}
func (deepseekEventToolCall) isDeepSeekEvent()        {}

func (p *DeepSeek3Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case deepseekEventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		case deepseekEventThinkingContent:
			thinkingSb.WriteString(event.content)
		case deepseekEventContent:
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *DeepSeek3Parser) parseEvents() []deepseekEvent {
	var all []deepseekEvent

	keepLooping := true
	for keepLooping {
		var events []deepseekEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

func (p *DeepSeek3Parser) eat() ([]deepseekEvent, bool) {
	var events []deepseekEvent
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case DeepSeekCollectingThinking:
		if strings.Contains(bufStr, deepseekThinkingCloseTag) { // thinking[</think>] -> content
			split := strings.SplitN(bufStr, deepseekThinkingCloseTag, 2)
			thinking := split[0]
			thinking = strings.TrimRightFunc(thinking, unicode.IsSpace)

			remaining := split[1]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = DeepSeekCollectingContent

			if len(thinking) > 0 {
				events = append(events, deepseekEventThinkingContent{content: thinking})
			}
			return events, true
		} else if overlapLen := overlap(bufStr, deepseekThinkingCloseTag); overlapLen > 0 { // partial </think>
			beforePartialTag := bufStr[:len(bufStr)-overlapLen]
			trailingLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, deepseekEventThinkingContent{content: unambiguous})
			}
			return events, false
		} else { // otherwise its thinking content
			whitespaceLen := trailingWhitespaceLen(bufStr)
			ambiguousStart := len(bufStr) - whitespaceLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, deepseekEventThinkingContent{content: unambiguous})
			}
			return events, false
		}

	case DeepSeekCollectingContent:
		switch {
		case strings.Contains(bufStr, deepseekToolCallsBeginTag): // content[<｜tool▁calls▁begin｜>] -> tool calls
			split := strings.SplitN(bufStr, deepseekToolCallsBeginTag, 2)
			contentBefore := strings.TrimRightFunc(split[0], unicode.IsSpace)
			remaining := split[1]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = DeepSeekCollectingToolCalls

			if len(contentBefore) > 0 {
				events = append(events, deepseekEventContent{content: contentBefore})
			}
			return events, true
		case strings.Contains(bufStr, deepseekToolOutputBeginTag): // content[<｜tool▁output▁begin｜>] -> tool output
			split := strings.SplitN(bufStr, deepseekToolOutputBeginTag, 2)
			contentBefore := split[0] // Don't trim whitespace - preserve spaces
			remaining := split[1]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = DeepSeekCollectingToolOutput

			if len(contentBefore) > 0 {
				events = append(events, deepseekEventContent{content: contentBefore})
			}
			return events, true
		default: // otherwise its content
			p.buffer.Reset()
			if len(bufStr) > 0 {
				events = append(events, deepseekEventContent{content: bufStr})
			}
			return events, false
		}

	case DeepSeekCollectingToolCalls:
		if idx := strings.Index(bufStr, deepseekToolCallBeginTag); idx != -1 {
			startIdx := idx + len(deepseekToolCallBeginTag)
			if endIdx := strings.Index(bufStr[startIdx:], deepseekToolCallEndTag); endIdx != -1 {
				toolCallContent := bufStr[startIdx : startIdx+endIdx]

				if toolCall, err := p.parseToolCallContent(toolCallContent); err == nil {
					remaining := bufStr[startIdx+endIdx+len(deepseekToolCallEndTag):]
					remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

					p.buffer.Reset()
					p.buffer.WriteString(remaining)

					events = append(events, deepseekEventToolCall{toolCall: toolCall})
					return events, true
				} else {
					slog.Warn("deepseek tool call parsing failed", "error", err)
				}
			}
		}

		if idx := strings.Index(bufStr, deepseekToolCallsEndTag); idx != -1 {
			remaining := bufStr[idx+len(deepseekToolCallsEndTag):]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = DeepSeekCollectingContent

			return events, true
		}

		return events, false

	case DeepSeekCollectingToolOutput:
		if idx := strings.Index(bufStr, deepseekToolOutputEndTag); idx != -1 {
			toolOutputContent := bufStr[:idx]
			remaining := bufStr[idx+len(deepseekToolOutputEndTag):]
			// Don't trim whitespace - preserve spaces after tool output tags

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = DeepSeekCollectingContent

			if len(toolOutputContent) > 0 {
				events = append(events, deepseekEventContent{content: toolOutputContent})
			}
			return events, true
		}

		return events, false
	}

	return events, false
}

func (p *DeepSeek3Parser) parseToolCallContent(content string) (api.ToolCall, error) {
	// Expected format: tool_name<｜tool▁sep｜>{args}
	parts := strings.SplitN(content, deepseekToolSepTag, 2)
	if len(parts) < 2 {
		return api.ToolCall{}, errors.New("invalid format")
	}

	toolName := strings.TrimSpace(parts[0])
	argsJSON := strings.TrimSpace(parts[1])

	var args api.ToolCallFunctionArguments
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return api.ToolCall{}, err
	}

	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      toolName,
			Arguments: args,
		},
	}, nil
}
