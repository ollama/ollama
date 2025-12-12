package parsers

import (
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

type CogitoParserState int

const (
	CogitoCollectingThinking CogitoParserState = iota
	CogitoCollectingContent
	CogitoCollectingToolCalls
	CogitoCollectingToolOutput
)

const (
	cogitoThinkingCloseTag    = "</think>"
	cogitoToolCallsBeginTag   = "<｜tool▁calls▁begin｜>"
	cogitoToolCallsEndTag     = "<｜tool▁calls▁end｜>"
	cogitoToolCallBeginTag    = "<｜tool▁call▁begin｜>"
	cogitoToolCallEndTag      = "<｜tool▁call▁end｜>"
	cogitoToolSepTag          = "<｜tool▁sep｜>"
	cogitoToolOutputBeginTag  = "<｜tool▁output▁begin｜>"
	cogitoToolOutputEndTag    = "<｜tool▁output▁end｜>"
	cogitoToolOutputsBeginTag = "<｜tool▁outputs▁begin｜>"
	cogitoToolOutputsEndTag   = "<｜tool▁outputs▁end｜>"
)

type CogitoParser struct {
	state  CogitoParserState
	buffer strings.Builder
}

func (p *CogitoParser) HasToolSupport() bool {
	return true
}

func (p *CogitoParser) HasThinkingSupport() bool {
	return true
}

func (p *CogitoParser) setInitialState(lastMessage *api.Message, tools []api.Tool, thinkValue *api.ThinkValue) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	// Check both model capability AND request preference
	thinkingEnabled := thinkValue != nil && thinkValue.Bool()
	// thinkingEnabled should be set to false for tools

	if !thinkingEnabled {
		p.state = CogitoCollectingContent
		return
	}

	if prefill && lastMessage.Content != "" {
		p.state = CogitoCollectingContent
		return
	}

	// Note: for cogito, if there are tools, then we don't want to be thinking
	if len(tools) > 0 {
		p.state = CogitoCollectingContent
		return
	}

	p.state = CogitoCollectingThinking
}

func (p *CogitoParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.setInitialState(lastMessage, tools, thinkValue)
	return tools
}

type cogitoEvent interface {
	isCogitoEvent()
}

type cogitoEventThinkingContent struct {
	content string
}

type cogitoEventContent struct {
	content string
}

type cogitoEventToolCall struct {
	toolCall api.ToolCall
}

func (cogitoEventThinkingContent) isCogitoEvent() {}
func (cogitoEventContent) isCogitoEvent()         {}
func (cogitoEventToolCall) isCogitoEvent()        {}

func (p *CogitoParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case cogitoEventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		case cogitoEventThinkingContent:
			thinkingSb.WriteString(event.content)
		case cogitoEventContent:
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *CogitoParser) parseEvents() []cogitoEvent {
	var all []cogitoEvent

	keepLooping := true
	for keepLooping {
		var events []cogitoEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

func (p *CogitoParser) eat() ([]cogitoEvent, bool) {
	var events []cogitoEvent
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case CogitoCollectingThinking:
		if strings.Contains(bufStr, cogitoThinkingCloseTag) { // thinking[</think>] -> content
			split := strings.SplitN(bufStr, cogitoThinkingCloseTag, 2)
			thinking := split[0]
			thinking = strings.TrimRightFunc(thinking, unicode.IsSpace)

			remaining := split[1]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = CogitoCollectingContent

			if len(thinking) > 0 {
				events = append(events, cogitoEventThinkingContent{content: thinking})
			}
			return events, true
		} else if overlapLen := overlap(bufStr, cogitoThinkingCloseTag); overlapLen > 0 { // partial </think>
			beforePartialTag := bufStr[:len(bufStr)-overlapLen]
			trailingLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, cogitoEventThinkingContent{content: unambiguous})
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
				events = append(events, cogitoEventThinkingContent{content: unambiguous})
			}
			return events, false
		}

	case CogitoCollectingContent:
		switch {
		case strings.Contains(bufStr, cogitoToolCallsBeginTag): // content[<｜tool▁calls▁begin｜>] -> tool calls
			split := strings.SplitN(bufStr, cogitoToolCallsBeginTag, 2)
			contentBefore := strings.TrimRightFunc(split[0], unicode.IsSpace)
			remaining := split[1]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = CogitoCollectingToolCalls

			if len(contentBefore) > 0 {
				events = append(events, cogitoEventContent{content: contentBefore})
			}
			return events, true
		case strings.Contains(bufStr, cogitoToolOutputsBeginTag): // content[<｜tool▁outputs▁begin｜>] -> tool outputs
			split := strings.SplitN(bufStr, cogitoToolOutputsBeginTag, 2)
			contentBefore := strings.TrimRightFunc(split[0], unicode.IsSpace)
			remaining := split[1]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = CogitoCollectingToolOutput

			if len(contentBefore) > 0 {
				events = append(events, cogitoEventContent{content: contentBefore})
			}
			return events, true
		default: // otherwise its content
			p.buffer.Reset()
			if len(bufStr) > 0 {
				events = append(events, cogitoEventContent{content: bufStr})
			}
			return events, false
		}
	case CogitoCollectingToolCalls:
		if idx := strings.Index(bufStr, cogitoToolCallBeginTag); idx != -1 {
			startIdx := idx + len(cogitoToolCallBeginTag)
			if endIdx := strings.Index(bufStr[startIdx:], cogitoToolCallEndTag); endIdx != -1 {
				toolCallContent := bufStr[startIdx : startIdx+endIdx]

				if toolCall, err := p.parseToolCallContent(toolCallContent); err == nil {
					remaining := bufStr[startIdx+endIdx+len(cogitoToolCallEndTag):]
					remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

					p.buffer.Reset()
					p.buffer.WriteString(remaining)

					events = append(events, cogitoEventToolCall{toolCall: toolCall})
					return events, true
				} else {
					slog.Warn("cogito tool call parsing failed", "error", err)
				}
			}
		}

		if idx := strings.Index(bufStr, cogitoToolCallsEndTag); idx != -1 {
			remaining := bufStr[idx+len(cogitoToolCallsEndTag):]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = CogitoCollectingContent

			return events, true
		}

		return events, false

	case CogitoCollectingToolOutput:
		if idx := strings.Index(bufStr, cogitoToolOutputBeginTag); idx != -1 {
			startIdx := idx + len(cogitoToolOutputBeginTag)
			if endIdx := strings.Index(bufStr[startIdx:], cogitoToolOutputEndTag); endIdx != -1 {
				remaining := bufStr[startIdx+endIdx+len(cogitoToolOutputEndTag):]
				remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

				p.buffer.Reset()
				p.buffer.WriteString(remaining)

				return events, true
			}
		}

		if idx := strings.Index(bufStr, cogitoToolOutputsEndTag); idx != -1 {
			remaining := bufStr[idx+len(cogitoToolOutputsEndTag):]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = CogitoCollectingContent

			return events, true
		}

		return events, false
	}

	return events, false
}

func (p *CogitoParser) parseToolCallContent(content string) (api.ToolCall, error) {
	// Expected format: function<｜tool▁sep｜>tool_name\n```json\n{args}\n```
	parts := strings.SplitN(content, cogitoToolSepTag, 2)
	if len(parts) < 2 {
		return api.ToolCall{}, errors.New("invalid format")
	}
	nameAndArgs := parts[1]

	jsonStart := strings.Index(nameAndArgs, "\n```json\n")
	if jsonStart == -1 {
		return api.ToolCall{}, errors.New("invalid format")
	}
	toolName := strings.TrimSpace(nameAndArgs[:jsonStart])
	jsonContent := nameAndArgs[jsonStart+len("\n```json\n"):]

	jsonEnd := strings.Index(jsonContent, "\n```")
	if jsonEnd == -1 {
		return api.ToolCall{}, errors.New("invalid format")
	}
	argsJSON := jsonContent[:jsonEnd]

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
