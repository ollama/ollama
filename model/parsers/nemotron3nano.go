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
	Nemotron3NanoSkipWhitespaceAfterThinking
	Nemotron3NanoCollectingContent
	Nemotron3NanoCollectingToolCalls
)

const (
	nemotronThinkClose    = "</think>"
	nemotronToolCallOpen  = "<tool_call>"
	nemotronToolCallClose = "</tool_call>"
)

type Nemotron3NanoParser struct {
	state  Nemotron3NanoParserState
	buffer strings.Builder
	tools  []api.Tool
}

func (p *Nemotron3NanoParser) HasToolSupport() bool     { return true }
func (p *Nemotron3NanoParser) HasThinkingSupport() bool { return true }

func (p *Nemotron3NanoParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools

	// thinking is enabled if user requests it
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

// emitWithPartialCheck extracts unambiguous content before a potential partial tag
func (p *Nemotron3NanoParser) emitWithPartialCheck(bufStr, tag string) (unambiguous, ambiguous string) {
	if overlapLen := overlap(bufStr, tag); overlapLen > 0 {
		beforePartialTag := bufStr[:len(bufStr)-overlapLen]
		trailingLen := trailingWhitespaceLen(beforePartialTag)
		return bufStr[:len(beforePartialTag)-trailingLen], bufStr[len(beforePartialTag)-trailingLen:]
	}
	wsLen := trailingWhitespaceLen(bufStr)
	return bufStr[:len(bufStr)-wsLen], bufStr[len(bufStr)-wsLen:]
}

func (p *Nemotron3NanoParser) eat() ([]nemotronEvent, bool) {
	bufStr := p.buffer.String()
	if bufStr == "" {
		return nil, false
	}

	switch p.state {
	case Nemotron3NanoCollectingThinking:
		if strings.Contains(bufStr, nemotronThinkClose) {
			split := strings.SplitN(bufStr, nemotronThinkClose, 2)
			thinking := strings.TrimRightFunc(split[0], unicode.IsSpace)
			p.buffer.Reset()
			remainder := strings.TrimLeftFunc(split[1], unicode.IsSpace)
			p.buffer.WriteString(remainder)
			// Transition to whitespace-skipping state if buffer is empty,
			// otherwise go directly to content collection
			if remainder == "" {
				p.state = Nemotron3NanoSkipWhitespaceAfterThinking
			} else {
				p.state = Nemotron3NanoCollectingContent
			}
			if thinking != "" {
				return []nemotronEvent{nemotronEventThinkingContent{content: thinking}}, true
			}
			return nil, true
		}
		unambig, ambig := p.emitWithPartialCheck(bufStr, nemotronThinkClose)
		p.buffer.Reset()
		p.buffer.WriteString(ambig)
		if unambig != "" {
			return []nemotronEvent{nemotronEventThinkingContent{content: unambig}}, false
		}
		return nil, false

	// We only want to skip whitespace between thinking and content
	case Nemotron3NanoSkipWhitespaceAfterThinking:
		bufStr = strings.TrimLeftFunc(bufStr, unicode.IsSpace)
		p.buffer.Reset()
		p.buffer.WriteString(bufStr)
		if bufStr == "" {
			return nil, false
		}
		p.state = Nemotron3NanoCollectingContent
		return nil, true

	case Nemotron3NanoCollectingContent:
		if strings.Contains(bufStr, nemotronToolCallOpen) {
			split := strings.SplitN(bufStr, nemotronToolCallOpen, 2)
			content := strings.TrimRightFunc(split[0], unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(split[1])
			p.state = Nemotron3NanoCollectingToolCalls
			if content != "" {
				return []nemotronEvent{nemotronEventContent{content: content}}, true
			}
			return nil, true
		}
		unambig, ambig := p.emitWithPartialCheck(bufStr, nemotronToolCallOpen)
		p.buffer.Reset()
		p.buffer.WriteString(ambig)
		if unambig != "" {
			return []nemotronEvent{nemotronEventContent{content: unambig}}, false
		}
		return nil, false

	case Nemotron3NanoCollectingToolCalls:
		if strings.Contains(bufStr, nemotronToolCallClose) {
			split := strings.SplitN(bufStr, nemotronToolCallClose, 2)
			remaining := strings.TrimLeftFunc(split[1], unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(remaining)

			var events []nemotronEvent
			if tc, err := p.parseToolCall(split[0]); err == nil {
				events = append(events, nemotronEventToolCall{toolCall: tc})
			}

			if !strings.Contains(remaining, nemotronToolCallOpen) {
				p.state = Nemotron3NanoCollectingContent
			}
			return events, true
		}
		return nil, false
	}

	return nil, false
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
