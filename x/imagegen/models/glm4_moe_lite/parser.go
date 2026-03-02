//go:build mlx

package glm4_moe_lite

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

type parserState int

const (
	parserState_LookingForThinkingOpen parserState = iota
	parserState_ThinkingStartedEatingWhitespace
	parserState_CollectingThinking
	parserState_ThinkingDoneEatingWhitespace
	parserState_CollectingContent
	parserState_ToolStartedEatingWhitespace
	parserState_CollectingToolContent
)

const (
	thinkingOpenTag  = "<think>"
	thinkingCloseTag = "</think>"
	toolOpenTag      = "<tool_call>"
	toolCloseTag     = "</tool_call>"
)

// Parser parses GLM4-MoE-Lite model output to extract thinking and tool calls.
// GLM-4's prompt ends with <think> when thinking is enabled, so the parser
// must start in CollectingThinking state (the model outputs thinking content directly).
type Parser struct {
	state  parserState
	buffer strings.Builder
	tools  []api.Tool
}

// HasToolSupport returns true as GLM4 supports tool calling.
func (p *Parser) HasToolSupport() bool {
	return true
}

// HasThinkingSupport returns true as GLM4 supports thinking mode.
func (p *Parser) HasThinkingSupport() bool {
	return true
}

// Init initializes the parser with tools and thinking configuration.
func (p *Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	// When thinking is enabled (nil or true), the prompt ends with <think>,
	// so model output starts directly with thinking content (no opening tag).
	if thinkValue == nil || thinkValue.Bool() {
		p.state = parserState_CollectingThinking
	}
	return tools
}

type parserEvent interface {
	isParserEvent()
}

type eventContent struct {
	content string
}

func (eventContent) isParserEvent() {}

type eventRawToolCall struct {
	raw string
}

func (eventRawToolCall) isParserEvent() {}

type eventThinkingContent struct {
	content string
}

func (eventThinkingContent) isParserEvent() {}

// Add processes new output text and returns parsed content, thinking, and tool calls.
func (p *Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder

	for _, event := range events {
		switch event := event.(type) {
		case eventRawToolCall:
			toolCall, err := parseToolCall(event, p.tools)
			if err != nil {
				slog.Warn("glm-4 tool call parsing failed", "error", err)
				return "", "", nil, err
			}
			toolCalls = append(toolCalls, toolCall)
		case eventThinkingContent:
			thinkingSb.WriteString(event.content)
		case eventContent:
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *Parser) parseEvents() []parserEvent {
	var all []parserEvent

	keepLooping := true
	for keepLooping {
		var events []parserEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "glm-4 events parsed", "events", all, "state", p.state, "buffer", p.buffer.String())
	}

	return all
}

// eatLeadingWhitespaceAndTransitionTo consumes leading whitespace from the buffer
// and transitions to the next state. Returns (nil, false) if only whitespace remains
// in the buffer (needs more input), or (nil, true) if we successfully transitioned.
func (p *Parser) eatLeadingWhitespaceAndTransitionTo(nextState parserState) ([]parserEvent, bool) {
	trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
	p.buffer.Reset()
	if trimmed == "" {
		return nil, false // Still only whitespace, keep waiting for more input
	}
	p.state = nextState
	p.buffer.WriteString(trimmed)
	return nil, true // Successfully transitioned
}

// splitAtTag splits the buffer at the given tag, returns the content before (trimmed of trailing whitespace),
// the content after (optionally trimmed of leading whitespace), and updates the buffer
func (p *Parser) splitAtTag(tag string, trimAfter bool) (string, string) {
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

func (p *Parser) eat() ([]parserEvent, bool) {
	var events []parserEvent

	switch p.state {
	case parserState_LookingForThinkingOpen:
		trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
		if strings.HasPrefix(trimmed, thinkingOpenTag) {
			// Found <think> opening tag
			after := strings.TrimPrefix(trimmed, thinkingOpenTag)
			after = strings.TrimLeftFunc(after, unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(after)
			if after == "" {
				p.state = parserState_ThinkingStartedEatingWhitespace
			} else {
				p.state = parserState_CollectingThinking
			}
			return events, true
		} else if strings.HasPrefix(thinkingOpenTag, trimmed) {
			// Partial opening tag seen, keep accumulating
			return events, false
		} else if trimmed == "" {
			// Only whitespace, keep accumulating
			return events, false
		} else {
			// No thinking tag found, skip to content collection
			p.state = parserState_CollectingContent
			// Don't trim - we want to keep the original content
			return events, true
		}

	case parserState_ThinkingStartedEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(parserState_CollectingThinking)

	case parserState_CollectingThinking:
		acc := p.buffer.String()
		if strings.Contains(acc, thinkingCloseTag) {
			thinking, remaining := p.splitAtTag(thinkingCloseTag, true)
			if len(thinking) > 0 {
				events = append(events, eventThinkingContent{content: thinking})
			}
			if remaining == "" {
				p.state = parserState_ThinkingDoneEatingWhitespace
			} else {
				p.state = parserState_CollectingContent
			}
			return events, true
		} else if overlapLen := overlap(acc, thinkingCloseTag); overlapLen > 0 {
			// Partial closing tag - withhold it along with any trailing whitespace before it
			beforePartialTag := acc[:len(acc)-overlapLen]
			trailingWsLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWsLen

			unambiguous := acc[:ambiguousStart]
			ambiguous := acc[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, eventThinkingContent{content: unambiguous})
			}
			return events, false
		} else {
			// Pure thinking content - withhold trailing whitespace (might precede closing tag)
			whitespaceLen := trailingWhitespaceLen(acc)
			ambiguousStart := len(acc) - whitespaceLen

			unambiguous := acc[:ambiguousStart]
			ambiguous := acc[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, eventThinkingContent{content: unambiguous})
			}
			return events, false
		}

	case parserState_ThinkingDoneEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(parserState_CollectingContent)

	case parserState_CollectingContent:
		if strings.Contains(p.buffer.String(), toolOpenTag) {
			before, after := p.splitAtTag(toolOpenTag, true)
			if len(before) > 0 {
				events = append(events, eventContent{content: before})
			}
			if after == "" {
				p.state = parserState_ToolStartedEatingWhitespace
			} else {
				p.state = parserState_CollectingToolContent
			}
			return events, true
		} else if overlapLen := overlap(p.buffer.String(), toolOpenTag); overlapLen > 0 {
			beforePartialTag := p.buffer.String()[:len(p.buffer.String())-overlapLen]
			trailingWsLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWsLen

			unambiguous := p.buffer.String()[:ambiguousStart]
			ambiguous := p.buffer.String()[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, eventContent{content: unambiguous})
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
				events = append(events, eventContent{content: unambiguous})
			}
			return events, false
		}

	case parserState_ToolStartedEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(parserState_CollectingToolContent)

	case parserState_CollectingToolContent:
		acc := p.buffer.String()
		if strings.Contains(acc, toolCloseTag) {
			toolContent, _ := p.splitAtTag(toolCloseTag, true)
			if len(toolContent) == 0 {
				slog.Warn("glm4 tool call closing tag found but no content before it")
			}
			events = append(events, eventRawToolCall{raw: toolContent})
			p.state = parserState_CollectingContent
			return events, true
		} else {
			// Keep accumulating - tool calls are not streamed
			// We just wait for the closing tag
			return events, false
		}

	default:
		panic("unreachable")
	}
}

// overlap returns the length of the overlap between the end of s and the start of tag.
func overlap(s, tag string) int {
	for i := 1; i <= len(tag) && i <= len(s); i++ {
		if strings.HasSuffix(s, tag[:i]) {
			return i
		}
	}
	return 0
}

// trailingWhitespaceLen returns the length of trailing whitespace in s.
func trailingWhitespaceLen(s string) int {
	trimmed := strings.TrimRightFunc(s, unicode.IsSpace)
	return len(s) - len(trimmed)
}

// ToolCallXML represents the structure of a GLM-4 tool call for XML parsing
type ToolCallXML struct {
	XMLName xml.Name `xml:"tool_call"`
	Content string   `xml:",chardata"` // Function name (text nodes between tags)
	Keys    []string `xml:"arg_key"`   // All arg_key elements in document order
	Values  []string `xml:"arg_value"` // All arg_value elements in document order
}

// escapeContent escapes XML entities in text content while preserving arg_key/arg_value tags
func escapeContent(s string) string {
	var result strings.Builder
	inTag := false

	for i := range len(s) {
		ch := s[i]

		if ch == '<' {
			// Check if this is a known tag
			if strings.HasPrefix(s[i:], "<arg_key>") ||
				strings.HasPrefix(s[i:], "</arg_key>") ||
				strings.HasPrefix(s[i:], "<arg_value>") ||
				strings.HasPrefix(s[i:], "</arg_value>") {
				inTag = true
			}
		}

		if inTag {
			result.WriteByte(ch)
			if ch == '>' {
				inTag = false
			}
		} else {
			// Escape special characters in text content
			switch ch {
			case '&':
				result.WriteString("&amp;")
			case '<':
				result.WriteString("&lt;")
			case '>':
				result.WriteString("&gt;")
			default:
				result.WriteByte(ch)
			}
		}
	}

	return result.String()
}

func parseToolCall(raw eventRawToolCall, tools []api.Tool) (api.ToolCall, error) {
	// Escape any unescaped entities in text content
	escaped := escapeContent(raw.raw)

	// Wrap the content in a root element to make it valid XML
	xmlString := "<tool_call>" + escaped + "</tool_call>"

	// Parse XML into struct
	var parsed ToolCallXML
	if err := xml.Unmarshal([]byte(xmlString), &parsed); err != nil {
		return api.ToolCall{}, fmt.Errorf("failed to parse XML: %w", err)
	}

	// Extract and trim function name
	functionName := strings.TrimSpace(parsed.Content)
	if functionName == "" {
		return api.ToolCall{}, fmt.Errorf("empty function name")
	}

	// Verify keys and values are paired correctly
	if len(parsed.Keys) != len(parsed.Values) {
		return api.ToolCall{}, fmt.Errorf("mismatched arg_key and arg_value counts: %d keys, %d values", len(parsed.Keys), len(parsed.Values))
	}

	// Find the matching tool to get parameter types
	var matchedTool *api.Tool
	for i := range tools {
		if tools[i].Function.Name == functionName {
			matchedTool = &tools[i]
			break
		}
	}

	// Build arguments map by pairing keys and values
	toolCall := api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      functionName,
			Arguments: api.NewToolCallFunctionArguments(),
		},
	}

	for i := range parsed.Keys {
		key := strings.TrimSpace(parsed.Keys[i])
		value := parsed.Values[i] // Don't trim here - parseValue handles it

		// Look up parameter type
		var paramType api.PropertyType
		if matchedTool != nil && matchedTool.Function.Parameters.Properties != nil {
			if prop, ok := matchedTool.Function.Parameters.Properties.Get(key); ok {
				// Handle anyOf by collecting all types from the union
				if len(prop.AnyOf) > 0 {
					for _, anyOfProp := range prop.AnyOf {
						paramType = append(paramType, anyOfProp.Type...)
					}
				} else {
					paramType = prop.Type
				}
			}
		}

		// Parse value with type coercion
		toolCall.Function.Arguments.Set(key, parseValue(value, paramType))
	}

	return toolCall, nil
}

// parseValue parses a string value and coerces it to the appropriate type based on paramType.
func parseValue(value string, paramType api.PropertyType) any {
	value = strings.TrimSpace(value)

	// If no type specified, return as string
	if len(paramType) == 0 {
		return value
	}

	// Try to parse based on specified types
	for _, t := range paramType {
		switch t {
		case "boolean":
			if value == "true" {
				return true
			}
			if value == "false" {
				return false
			}
		case "integer":
			var i int64
			if _, err := fmt.Sscanf(value, "%d", &i); err == nil {
				return i
			}
		case "number":
			var f float64
			if _, err := fmt.Sscanf(value, "%f", &f); err == nil {
				return f
			}
		case "array", "object":
			// Try to parse as JSON
			var result any
			if err := json.Unmarshal([]byte(value), &result); err == nil {
				return result
			}
		}
	}

	// Default to string
	return value
}
