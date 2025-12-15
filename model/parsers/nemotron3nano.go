package parsers

import (
	"regexp"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

type Nemotron3NanoParserState int

const (
	Nemotron3NanoCollectingThinking Nemotron3NanoParserState = iota
	Nemotron3NanoCollectingContent
	Nemotron3NanoCollectingToolCalls
)

const (
	nemotronThinkClose    = "</think>"
	nemotronToolCallOpen  = "<tool_call>"
	nemotronToolCallClose = "</tool_call>"
)

type Nemotron3NanoParser struct {
	state              Nemotron3NanoParserState
	buffer             strings.Builder
	tools              []api.Tool
	hasThinkingSupport bool
}

func (p *Nemotron3NanoParser) HasToolSupport() bool {
	return true
}

func (p *Nemotron3NanoParser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *Nemotron3NanoParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools

	// Check both model capability AND request preference
	thinkingEnabled := thinkValue != nil && thinkValue.Bool()

	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	if !thinkingEnabled {
		p.state = Nemotron3NanoCollectingContent
		return tools
	}

	if prefill && lastMessage.Content != "" {
		p.state = Nemotron3NanoCollectingContent
		return tools
	}

	p.state = Nemotron3NanoCollectingThinking
	return tools
}

type nemotronEvent interface {
	isNemotronEvent()
}

type nemotronEventThinkingContent struct {
	content string
}

type nemotronEventContent struct {
	content string
}

type nemotronEventToolCall struct {
	toolCall api.ToolCall
}

func (nemotronEventThinkingContent) isNemotronEvent() {}
func (nemotronEventContent) isNemotronEvent()         {}
func (nemotronEventToolCall) isNemotronEvent()        {}

func (p *Nemotron3NanoParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case nemotronEventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		case nemotronEventThinkingContent:
			thinkingSb.WriteString(event.content)
		case nemotronEventContent:
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *Nemotron3NanoParser) parseEvents() []nemotronEvent {
	var all []nemotronEvent

	keepLooping := true
	for keepLooping {
		var events []nemotronEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

func (p *Nemotron3NanoParser) eat() ([]nemotronEvent, bool) {
	var events []nemotronEvent
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case Nemotron3NanoCollectingThinking:
		if strings.Contains(bufStr, nemotronThinkClose) {
			split := strings.SplitN(bufStr, nemotronThinkClose, 2)
			thinking := split[0]
			thinking = strings.TrimRightFunc(thinking, unicode.IsSpace)

			remaining := split[1]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Nemotron3NanoCollectingContent

			if len(thinking) > 0 {
				events = append(events, nemotronEventThinkingContent{content: thinking})
			}
			return events, true
		} else if overlapLen := overlap(bufStr, nemotronThinkClose); overlapLen > 0 {
			beforePartialTag := bufStr[:len(bufStr)-overlapLen]
			trailingLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, nemotronEventThinkingContent{content: unambiguous})
			}
			return events, false
		} else {
			whitespaceLen := trailingWhitespaceLen(bufStr)
			ambiguousStart := len(bufStr) - whitespaceLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, nemotronEventThinkingContent{content: unambiguous})
			}
			return events, false
		}

	case Nemotron3NanoCollectingContent:
		switch {
		case strings.Contains(bufStr, nemotronToolCallOpen):
			split := strings.SplitN(bufStr, nemotronToolCallOpen, 2)
			contentBefore := strings.TrimRightFunc(split[0], unicode.IsSpace)
			remaining := split[1]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Nemotron3NanoCollectingToolCalls

			if len(contentBefore) > 0 {
				events = append(events, nemotronEventContent{content: contentBefore})
			}
			return events, true
		default:
			// Check for partial tool call tag
			if overlapLen := overlap(bufStr, nemotronToolCallOpen); overlapLen > 0 {
				beforePartialTag := bufStr[:len(bufStr)-overlapLen]
				trailingLen := trailingWhitespaceLen(beforePartialTag)
				ambiguousStart := len(beforePartialTag) - trailingLen

				unambiguous := bufStr[:ambiguousStart]
				ambiguous := bufStr[ambiguousStart:]
				p.buffer.Reset()
				p.buffer.WriteString(ambiguous)
				if len(unambiguous) > 0 {
					events = append(events, nemotronEventContent{content: unambiguous})
				}
				return events, false
			}

			// Otherwise emit content, withholding trailing whitespace
			whitespaceLen := trailingWhitespaceLen(bufStr)
			ambiguousStart := len(bufStr) - whitespaceLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, nemotronEventContent{content: unambiguous})
			}
			return events, false
		}

	case Nemotron3NanoCollectingToolCalls:
		// Look for complete tool call: <function=name>...</function>
		if strings.Contains(bufStr, nemotronToolCallClose) {
			// We have a complete tool call block
			split := strings.SplitN(bufStr, nemotronToolCallClose, 2)
			toolContent := split[0]
			remaining := strings.TrimLeftFunc(split[1], unicode.IsSpace)

			// Parse the tool call
			if toolCall, err := p.parseToolCall(toolContent); err == nil {
				events = append(events, nemotronEventToolCall{toolCall: toolCall})
			}

			p.buffer.Reset()
			p.buffer.WriteString(remaining)

			// Check if there are more tool calls
			if strings.Contains(remaining, nemotronToolCallOpen) {
				// Stay in tool call state
				return events, true
			}

			p.state = Nemotron3NanoCollectingContent
			return events, true
		}
		return events, false
	}

	return events, false
}

var (
	nemotronFunctionRegex  = regexp.MustCompile(`<function=([^>]+)>`)
	nemotronParameterRegex = regexp.MustCompile(`<parameter=([^>]+)>\n?([\s\S]*?)\n?</parameter>`)
)

func (p *Nemotron3NanoParser) parseToolCall(content string) (api.ToolCall, error) {
	toolCall := api.ToolCall{}

	// Extract function name
	fnMatch := nemotronFunctionRegex.FindStringSubmatch(content)
	if len(fnMatch) < 2 {
		return toolCall, nil
	}
	toolCall.Function.Name = fnMatch[1]

	// Extract parameters
	toolCall.Function.Arguments = make(api.ToolCallFunctionArguments)
	paramMatches := nemotronParameterRegex.FindAllStringSubmatch(content, -1)
	for _, match := range paramMatches {
		if len(match) >= 3 {
			paramName := match[1]
			paramValue := strings.TrimSpace(match[2])

			// Try to parse as typed value based on tool definition
			toolCall.Function.Arguments[paramName] = p.parseParamValue(paramName, paramValue)
		}
	}

	return toolCall, nil
}

func (p *Nemotron3NanoParser) parseParamValue(paramName string, raw string) any {
	// Find the matching tool to get parameter type
	var paramType api.PropertyType
	for _, tool := range p.tools {
		if prop, ok := tool.Function.Parameters.Properties[paramName]; ok {
			paramType = prop.Type
			break
		}
	}

	return parseValue(raw, paramType)
}
