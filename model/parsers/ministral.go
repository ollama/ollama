package parsers

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

type ministralParserState int

const (
	ministralCollectingContent = iota
	ministralCollectingThinkingContent
	ministralCollectingToolName
	ministralCollectingToolArgs
)

// ministralEvent represents an event emitted during parsing
type ministralEvent interface {
	isMinistralEvent()
}

type ministralEventContent struct {
	content string
}

type ministralEventThinking struct {
	thinking string
}

type ministralEventToolCall struct {
	name string
	args string // raw JSON string
}

func (ministralEventContent) isMinistralEvent()  {}
func (ministralEventThinking) isMinistralEvent() {}
func (ministralEventToolCall) isMinistralEvent() {}

type MinistralParser struct {
	state              ministralParserState
	buffer             strings.Builder
	tools              []api.Tool
	hasThinkingSupport bool
	pendingToolName    string // stores tool name while collecting args
}

func (p *MinistralParser) HasToolSupport() bool {
	return true
}

func (p *MinistralParser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *MinistralParser) setInitialState(lastMessage *api.Message) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"
	if !p.HasThinkingSupport() {
		p.state = ministralCollectingContent
		return
	}

	if prefill && lastMessage.Content != "" {
		p.state = ministralCollectingContent
		return
	}

	p.state = ministralCollectingThinkingContent
}

func (p *MinistralParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.setInitialState(lastMessage)
	return tools
}

func toolByName(tools []api.Tool, n string) (*api.Tool, error) {
	for i := range tools {
		if tools[i].Function.Name == n {
			return &tools[i], nil
		}
	}
	return nil, fmt.Errorf("tool '%s' not found", n)
}

const (
	ministralToolCallsTag = "[TOOL_CALLS]"
	ministralThinkTag     = "[THINK]"
	ministralThinkEndTag  = "[/THINK]"
	ministralArgsTag      = "[ARGS]"
)

// eat consumes the parser's buffer, and returns a list of any unambiguous
// events from the current parser state. The second return value indicates
// whether to keep looping (true when state transitions, false when waiting
// for more data).
func (p *MinistralParser) eat() ([]ministralEvent, bool) {
	var events []ministralEvent

	switch p.state {
	case ministralCollectingContent:
		bufStr := p.buffer.String()

		// Check for [TOOL_CALLS] tag
		if strings.Contains(bufStr, ministralToolCallsTag) {
			split := strings.SplitN(bufStr, ministralToolCallsTag, 2)
			before := strings.TrimRightFunc(split[0], unicode.IsSpace)
			if len(before) > 0 {
				events = append(events, ministralEventContent{content: before})
			}
			after := split[1]
			p.buffer.Reset()
			p.buffer.WriteString(after)
			p.state = ministralCollectingToolName
			return events, true
		}

		// Check for [THINK] tag
		if strings.Contains(bufStr, ministralThinkTag) {
			split := strings.SplitN(bufStr, ministralThinkTag, 2)
			before := strings.TrimRightFunc(split[0], unicode.IsSpace)
			if len(before) > 0 {
				events = append(events, ministralEventContent{content: before})
			}
			after := split[1]
			p.buffer.Reset()
			p.buffer.WriteString(after)
			p.state = ministralCollectingThinkingContent
			return events, true
		}

		// Check for partial tag overlap with [TOOL_CALLS] or [THINK]
		overlapToolCalls := overlap(bufStr, ministralToolCallsTag)
		overlapThink := overlap(bufStr, ministralThinkTag)
		maxOverlap := max(overlapToolCalls, overlapThink)

		if maxOverlap > 0 {
			// Withhold the potential partial tag
			beforePartialTag := bufStr[:len(bufStr)-maxOverlap]
			trailingWS := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWS
			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, ministralEventContent{content: unambiguous})
			}
			return events, false
		}

		// No tag found: emit content but withhold trailing whitespace
		whitespaceLen := trailingWhitespaceLen(bufStr)
		ambiguousStart := len(bufStr) - whitespaceLen
		unambiguous := bufStr[:ambiguousStart]
		ambiguous := bufStr[ambiguousStart:]
		p.buffer.Reset()
		p.buffer.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, ministralEventContent{content: unambiguous})
		}
		return events, false

	case ministralCollectingThinkingContent:
		bufStr := p.buffer.String()

		if strings.Contains(bufStr, ministralThinkEndTag) {
			split := strings.SplitN(bufStr, ministralThinkEndTag, 2)
			thinkingContent := split[0]
			after := strings.TrimLeftFunc(split[1], unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(after)
			if len(thinkingContent) > 0 {
				events = append(events, ministralEventThinking{thinking: thinkingContent})
			}
			p.state = ministralCollectingContent
			return events, true
		}

		// Check for partial overlap with [/THINK]
		if overlapLen := overlap(bufStr, ministralThinkEndTag); overlapLen > 0 {
			unambiguous := bufStr[:len(bufStr)-overlapLen]
			ambiguous := bufStr[len(bufStr)-overlapLen:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, ministralEventThinking{thinking: unambiguous})
			}
			return events, false
		}

		// No tag found: emit all thinking content
		p.buffer.Reset()
		if len(bufStr) > 0 {
			events = append(events, ministralEventThinking{thinking: bufStr})
		}
		return events, false

	case ministralCollectingToolName:
		bufStr := p.buffer.String()

		if strings.Contains(bufStr, ministralArgsTag) {
			split := strings.SplitN(bufStr, ministralArgsTag, 2)
			toolName := split[0]
			after := split[1]
			p.pendingToolName = toolName
			p.buffer.Reset()
			p.buffer.WriteString(after)
			p.state = ministralCollectingToolArgs
			return events, true
		}
		// Wait for more data
		return events, false

	case ministralCollectingToolArgs:
		bufStr := p.buffer.String()
		jsonEnd := findJSONEnd(bufStr)

		if jsonEnd != -1 {
			jsonStr := bufStr[:jsonEnd+1]
			remaining := bufStr[jsonEnd+1:]

			events = append(events, ministralEventToolCall{
				name: p.pendingToolName,
				args: jsonStr,
			})

			p.pendingToolName = ""
			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = ministralCollectingContent
			return events, true
		}
		// Wait for more data
		return events, false

	default:
		panic("unexpected ministral event")
	}
}

// parseEvents loops calling eat() until it returns false
func (p *MinistralParser) parseEvents() []ministralEvent {
	var all []ministralEvent
	keepLooping := true
	for keepLooping {
		var events []ministralEvent
		events, keepLooping = p.eat()
		all = append(all, events...)
	}
	return all
}

func (p *MinistralParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)

	events := p.parseEvents()

	var contentBuilder, thinkingBuilder strings.Builder
	var toolCalls []api.ToolCall

	for _, event := range events {
		switch e := event.(type) {
		case ministralEventContent:
			contentBuilder.WriteString(e.content)
		case ministralEventThinking:
			thinkingBuilder.WriteString(e.thinking)
		case ministralEventToolCall:
			// Validate tool exists
			tool, toolErr := toolByName(p.tools, e.name)
			if toolErr != nil {
				return contentBuilder.String(), thinkingBuilder.String(), toolCalls, toolErr
			}
			// Parse JSON arguments
			var args api.ToolCallFunctionArguments
			if jsonErr := json.Unmarshal([]byte(e.args), &args); jsonErr != nil {
				return contentBuilder.String(), thinkingBuilder.String(), toolCalls, jsonErr
			}
			toolCalls = append(toolCalls, api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      tool.Function.Name,
					Arguments: args,
				},
			})
		}
	}

	return contentBuilder.String(), thinkingBuilder.String(), toolCalls, nil
}

// findJSONEnd finds the index of the closing brace that completes a JSON object.
// It properly handles nested objects, arrays, and strings (including escaped characters).
// Returns -1 if the JSON is not yet complete.
func findJSONEnd(s string) int {
	depth := 0
	inString := false
	escaped := false

	for i, r := range s {
		if inString {
			switch {
			case escaped:
				// If the previous character was a backslash, skip this character
				escaped = false
			case r == '\\':
				// Mark the next character as escaped
				escaped = true
			case r == '"':
				// End of string literal
				inString = false
			}
			continue
		}

		switch r {
		case '"':
			// Start of string literal
			inString = true
		case '{', '[':
			// Increase nesting level for objects and arrays
			depth++
		case '}', ']':
			// Decrease nesting level
			depth--
			if depth == 0 {
				// Reached the end of the root JSON structure
				return i
			}
		}
	}

	return -1
}
