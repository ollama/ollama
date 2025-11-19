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
	CogitoCollectingToolCall
	CogitoCollectingToolOutput
	CogitoThinkingDoneEatingWhitespace
	CogitoToolCallDoneEatingWhitespace
	CogitoToolOutputDoneEatingWhitespace
)

const (
	cogitoThinkingOpenTag    = "<think>"
	cogitoThinkingCloseTag   = "</think>"
	cogitoToolCallBeginTag   = "<｜tool▁call▁begin｜>"
	cogitoToolCallEndTag     = "<｜tool▁call▁end｜>"
	cogitoToolSepTag         = "<｜tool▁sep｜>"
	cogitoToolOutputBeginTag = "<｜tool▁output▁begin｜>"
	cogitoToolOutputEndTag   = "<｜tool▁output▁end｜>"
)

type CogitoParser struct {
	state  CogitoParserState
	buffer strings.Builder
	tools  []api.Tool
}

func (p *CogitoParser) HasToolSupport() bool {
	return true
}

func (p *CogitoParser) HasThinkingSupport() bool {
	return true
}

func (p *CogitoParser) setInitialState(lastMessage *api.Message) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"
	if prefill && lastMessage.Content != "" {
		p.state = CogitoCollectingContent
		return
	}
	p.state = CogitoCollectingThinking
}

func (p *CogitoParser) Init(tools []api.Tool, lastMessage *api.Message) []api.Tool {
	p.tools = tools
	p.setInitialState(lastMessage)
	return tools
}

type cogitoEvent interface {
	isCogitoEvent()
}

type cogitoEventThinkingContent struct {
	content string
}

func (cogitoEventThinkingContent) isCogitoEvent() {}

type cogitoEventContent struct {
	content string
}

func (cogitoEventContent) isCogitoEvent() {}

type cogitoEventToolCall struct {
	name      string
	arguments string
}

func (cogitoEventToolCall) isCogitoEvent() {}

type cogitoEventToolOutput struct {
	content string
}

func (cogitoEventToolOutput) isCogitoEvent() {}

func (p *CogitoParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder

	for _, event := range events {
		switch event := event.(type) {
		case cogitoEventToolCall:
			toolCall, err := p.parseToolCall(event)
			if err != nil {
				slog.Warn("cogito tool call parsing failed", "error", err)
				return "", "", nil, err
			}
			toolCalls = append(toolCalls, toolCall)
		case cogitoEventThinkingContent:
			thinkingSb.WriteString(event.content)
		case cogitoEventContent:
			contentSb.WriteString(event.content)
		case cogitoEventToolOutput:
			// Tool outputs are handled separately and not included in content/thinking
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

	switch p.state {
	case CogitoCollectingThinking:
		if strings.Contains(p.buffer.String(), cogitoThinkingCloseTag) {
			before, after := p.splitAtTag(cogitoThinkingCloseTag, true)
			events = append(events, cogitoEventThinkingContent{content: before})
			p.state = CogitoThinkingDoneEatingWhitespace
			p.buffer.WriteString(after)
			return events, true
		}
		// Check for tool calls in thinking
		if strings.Contains(p.buffer.String(), cogitoToolCallBeginTag) {
			before, after := p.splitAtTag(cogitoToolCallBeginTag, false)
			if len(before) > 0 {
				events = append(events, cogitoEventThinkingContent{content: before})
			}
			p.state = CogitoCollectingToolCall
			p.buffer.WriteString(cogitoToolCallBeginTag + after)
			return events, true
		}
		// Check for tool outputs in thinking
		if strings.Contains(p.buffer.String(), cogitoToolOutputBeginTag) {
			before, after := p.splitAtTag(cogitoToolOutputBeginTag, false)
			if len(before) > 0 {
				events = append(events, cogitoEventThinkingContent{content: before})
			}
			p.state = CogitoCollectingToolOutput
			p.buffer.WriteString(cogitoToolOutputBeginTag + after)
			return events, true
		}
		return nil, false

	case CogitoCollectingContent:
		// Check for tool calls in content
		if strings.Contains(p.buffer.String(), cogitoToolCallBeginTag) {
			before, after := p.splitAtTag(cogitoToolCallBeginTag, false)
			if len(before) > 0 {
				events = append(events, cogitoEventContent{content: before})
			}
			p.state = CogitoCollectingToolCall
			p.buffer.WriteString(cogitoToolCallBeginTag + after)
			return events, true
		}
		// Check for tool outputs in content
		if strings.Contains(p.buffer.String(), cogitoToolOutputBeginTag) {
			before, after := p.splitAtTag(cogitoToolOutputBeginTag, false)
			if len(before) > 0 {
				events = append(events, cogitoEventContent{content: before})
			}
			p.state = CogitoCollectingToolOutput
			p.buffer.WriteString(cogitoToolOutputBeginTag + after)
			return events, true
		}
		// No special tags found, emit all content in buffer
		content := p.buffer.String()
		if content != "" {
			p.buffer.Reset()
			events = append(events, cogitoEventContent{content: content})
		}
		return events, false

	case CogitoCollectingToolCall:
		if strings.Contains(p.buffer.String(), cogitoToolCallEndTag) {
			before, after := p.splitAtTag(cogitoToolCallEndTag, true)
			toolCallContent := cogitoToolCallBeginTag + before + cogitoToolCallEndTag

			// Parse the tool call
			if event, err := p.parseRawToolCall(toolCallContent); err == nil {
				events = append(events, event)
			}

			p.state = CogitoToolCallDoneEatingWhitespace
			p.buffer.WriteString(after)
			return events, true
		}
		return nil, false

	case CogitoCollectingToolOutput:
		if strings.Contains(p.buffer.String(), cogitoToolOutputEndTag) {
			before, after := p.splitAtTag(cogitoToolOutputEndTag, true)
			// Extract content between begin and end tags
			if strings.HasPrefix(before, cogitoToolOutputBeginTag) {
				content := strings.TrimPrefix(before, cogitoToolOutputBeginTag)
				events = append(events, cogitoEventToolOutput{content: content})
			}

			p.state = CogitoToolOutputDoneEatingWhitespace
			p.buffer.WriteString(after)
			return events, true
		}
		return nil, false

	case CogitoThinkingDoneEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(CogitoCollectingContent)

	case CogitoToolCallDoneEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(CogitoCollectingContent)

	case CogitoToolOutputDoneEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(CogitoCollectingContent)
	}

	return nil, false
}

func (p *CogitoParser) eatLeadingWhitespaceAndTransitionTo(nextState CogitoParserState) ([]cogitoEvent, bool) {
	trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
	p.buffer.Reset()
	if trimmed == "" {
		return nil, false
	}
	p.state = nextState
	p.buffer.WriteString(trimmed)
	return nil, true
}

func (p *CogitoParser) splitAtTag(tag string, trimAfter bool) (string, string) {
	split := strings.SplitN(p.buffer.String(), tag, 2)
	before := split[0]
	before = strings.TrimRightFunc(before, unicode.IsSpace)
	after := split[1]
	if trimAfter {
		after = strings.TrimLeftFunc(after, unicode.IsSpace)
	}
	p.buffer.Reset()
	p.buffer.WriteString(after)
	return before, after
}

func (p *CogitoParser) parseRawToolCall(raw string) (cogitoEventToolCall, error) {
	// Expected format: <｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n```json\n{args}\n```<｜tool▁call▁end｜>
	content := strings.TrimPrefix(raw, cogitoToolCallBeginTag)
	content = strings.TrimSuffix(content, cogitoToolCallEndTag)

	// Split by separator to get function type and name
	parts := strings.SplitN(content, cogitoToolSepTag, 2)
	if len(parts) != 2 {
		return cogitoEventToolCall{}, errors.New("invalid tool call format: missing separator")
	}

	// Verify function type
	functionType := strings.TrimSpace(parts[0])
	if functionType != "function" {
		return cogitoEventToolCall{}, errors.New("invalid tool call format: expected 'function'")
	}

	name := strings.TrimSpace(parts[1])

	// Find JSON block
	jsonStart := strings.Index(name, "```json\n")
	if jsonStart == -1 {
		return cogitoEventToolCall{}, errors.New("invalid tool call format: missing JSON block start")
	}

	jsonEnd := strings.Index(name[jsonStart+8:], "\n```")
	if jsonEnd == -1 {
		return cogitoEventToolCall{}, errors.New("invalid tool call format: missing JSON block end")
	}

	args := name[jsonStart+8 : jsonStart+8+jsonEnd]
	name = name[:jsonStart]

	return cogitoEventToolCall{
		name:      strings.TrimSpace(name),
		arguments: args,
	}, nil
}

func (p *CogitoParser) parseToolCall(event cogitoEventToolCall) (api.ToolCall, error) {
	var args api.ToolCallFunctionArguments
	if err := json.Unmarshal([]byte(event.arguments), &args); err != nil {
		return api.ToolCall{}, err
	}

	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      event.name,
			Arguments: args,
		},
	}, nil
}
