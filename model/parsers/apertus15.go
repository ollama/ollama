package parsers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

const (
	apertus15InnerStart     = "<|inner_prefix|>"
	apertus15InnerEnd       = "<|inner_suffix|>"
	apertus15ToolsStart     = "<|tools_prefix|>"
	apertus15ToolsEnd       = "<|tools_suffix|>"
	apertus15AssistantStart = "<|assistant_start|>"
	apertus15AssistantEnd   = "<|assistant_end|>"
)

type apertus15ParserState int

const (
	apertus15ParserContent apertus15ParserState = iota
	apertus15ParserThinking
	apertus15ParserTools
)

type Apertus15Parser struct {
	state           apertus15ParserState
	buffer          strings.Builder
	thinkingEnabled bool
	callIndex       int
}

func (p *Apertus15Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.state = apertus15ParserContent
	p.buffer.Reset()
	p.thinkingEnabled = thinkValue != nil && thinkValue.Bool()
	p.callIndex = 0

	if lastMessage != nil && lastMessage.Role == "assistant" && lastMessage.Thinking != "" && lastMessage.Content == "" {
		p.state = apertus15ParserThinking
	}

	return tools
}

func (p *Apertus15Parser) HasToolSupport() bool {
	return true
}

func (p *Apertus15Parser) HasThinkingSupport() bool {
	return true
}

func (p *Apertus15Parser) PreservedTokens() []string {
	return []string{
		apertus15InnerStart,
		apertus15InnerEnd,
		apertus15ToolsStart,
		apertus15ToolsEnd,
		apertus15AssistantStart,
		apertus15AssistantEnd,
	}
}

func (p *Apertus15Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	var contentBuilder, thinkingBuilder strings.Builder

	for {
		switch p.state {
		case apertus15ParserContent, apertus15ParserThinking:
			sourceState := p.state
			progress, text, err := p.consumeText(done)
			if err != nil {
				return "", "", nil, err
			}
			if sourceState == apertus15ParserThinking {
				if p.thinkingEnabled {
					thinkingBuilder.WriteString(text)
				}
			} else {
				contentBuilder.WriteString(text)
			}
			if !progress {
				return contentBuilder.String(), thinkingBuilder.String(), calls, nil
			}
		case apertus15ParserTools:
			progress, parsedCalls, err := p.consumeTools(done)
			if err != nil {
				return "", "", nil, err
			}
			calls = append(calls, parsedCalls...)
			if !progress {
				return contentBuilder.String(), thinkingBuilder.String(), calls, nil
			}
		default:
			return "", "", nil, fmt.Errorf("invalid apertus 1.5 parser state %d", p.state)
		}
	}
}

func (p *Apertus15Parser) consumeText(done bool) (bool, string, error) {
	acc := p.buffer.String()
	if acc == "" {
		return false, "", nil
	}

	tag, index := earliestApertus15Tag(acc)
	if index >= 0 {
		text := acc[:index]
		p.buffer.Reset()
		p.buffer.WriteString(acc[index+len(tag):])
		switch tag {
		case apertus15InnerStart:
			p.state = apertus15ParserThinking
		case apertus15InnerEnd:
			p.state = apertus15ParserContent
		case apertus15ToolsStart:
			p.state = apertus15ParserTools
		case apertus15ToolsEnd, apertus15AssistantStart, apertus15AssistantEnd:
			p.state = apertus15ParserContent
		}
		return true, text, nil
	}

	if done {
		p.buffer.Reset()
		return acc != "", acc, nil
	}

	keep := apertus15TagOverlap(acc)
	emitLen := len(acc) - keep
	if emitLen == 0 {
		return false, "", nil
	}
	text := acc[:emitLen]
	p.buffer.Reset()
	p.buffer.WriteString(acc[emitLen:])
	return true, text, nil
}

func (p *Apertus15Parser) consumeTools(done bool) (bool, []api.ToolCall, error) {
	acc := p.buffer.String()
	if index := strings.Index(acc, apertus15ToolsEnd); index >= 0 {
		payload := acc[:index]
		p.buffer.Reset()
		p.buffer.WriteString(acc[index+len(apertus15ToolsEnd):])
		p.state = apertus15ParserContent
		calls, err := p.parseToolCalls(payload)
		return true, calls, err
	}
	if !done {
		return false, nil, nil
	}

	p.buffer.Reset()
	p.state = apertus15ParserContent
	calls, err := p.parseToolCalls(acc)
	return true, calls, err
}

func (p *Apertus15Parser) parseToolCalls(payload string) ([]api.ToolCall, error) {
	var entries []json.RawMessage
	if err := json.Unmarshal([]byte(strings.TrimSpace(payload)), &entries); err != nil {
		return nil, fmt.Errorf("parse apertus 1.5 tool calls: %w", err)
	}

	calls := make([]api.ToolCall, 0, len(entries))
	for _, entry := range entries {
		var callObject map[string]json.RawMessage
		if err := json.Unmarshal(entry, &callObject); err != nil {
			return nil, fmt.Errorf("parse apertus 1.5 tool call: %w", err)
		}
		if len(callObject) != 1 {
			return nil, fmt.Errorf("parse apertus 1.5 tool call: expected one function, got %d", len(callObject))
		}
		for name, rawArguments := range callObject {
			if name == "" {
				return nil, fmt.Errorf("parse apertus 1.5 tool call: empty function name")
			}
			argumentJSON := strings.TrimSpace(string(rawArguments))
			if !strings.HasPrefix(argumentJSON, "{") || !strings.HasSuffix(argumentJSON, "}") {
				return nil, fmt.Errorf("parse apertus 1.5 tool call %q arguments: expected object", name)
			}
			var arguments api.ToolCallFunctionArguments
			if err := json.Unmarshal(rawArguments, &arguments); err != nil {
				return nil, fmt.Errorf("parse apertus 1.5 tool call %q arguments: %w", name, err)
			}
			calls = append(calls, api.ToolCall{
				Function: api.ToolCallFunction{
					Index:     p.callIndex,
					Name:      name,
					Arguments: arguments,
				},
			})
			p.callIndex++
		}
	}
	return calls, nil
}

func earliestApertus15Tag(s string) (string, int) {
	tag := ""
	index := -1
	for _, candidate := range apertus15ParserTags {
		candidateIndex := strings.Index(s, candidate)
		if candidateIndex >= 0 && (index < 0 || candidateIndex < index) {
			tag = candidate
			index = candidateIndex
		}
	}
	return tag, index
}

func apertus15TagOverlap(s string) int {
	keep := 0
	for _, tag := range apertus15ParserTags {
		keep = max(keep, overlap(s, tag))
	}
	return keep
}

var apertus15ParserTags = []string{
	apertus15InnerStart,
	apertus15InnerEnd,
	apertus15ToolsStart,
	apertus15ToolsEnd,
	apertus15AssistantStart,
	apertus15AssistantEnd,
}
