package parsers

import (
	"context"
	"encoding/xml"
	"fmt"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

type glm46ParserState int

const (
	glm46ParserState_LookingForThinkingOpen glm46ParserState = iota
	glm46ParserState_ThinkingStartedEatingWhitespace
	glm46ParserState_CollectingThinking
	glm46ParserState_ThinkingDoneEatingWhitespace
	glm46ParserState_CollectingContent
	glm46ParserState_ToolStartedEatingWhitespace
	glm46ParserState_CollectingToolContent
)

const (
	glm46ThinkingOpenTag  = "<think>"
	glm46ThinkingCloseTag = "</think>"
	glm46ToolOpenTag      = "<tool_call>"
	glm46ToolCloseTag     = "</tool_call>"
)

type GLM46Parser struct {
	state     glm46ParserState
	buffer    strings.Builder
	tools     []api.Tool
	callIndex int
}

func (p *GLM46Parser) HasToolSupport() bool {
	return true
}

func (p *GLM46Parser) HasThinkingSupport() bool {
	return true
}

// func (p *GLM46Parser) Init(tools []api.Tool, lastMessage *api.Message) []api.Tool {
func (p *GLM46Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.callIndex = 0
	return tools
}

type glm46Event interface {
	isGLM46Event()
}

type glm46EventContent struct {
	content string
}

func (glm46EventContent) isGLM46Event() {}

type glm46EventRawToolCall struct {
	raw string
}

func (glm46EventRawToolCall) isGLM46Event() {}

type glm46EventThinkingContent struct {
	content string
}

func (glm46EventThinkingContent) isGLM46Event() {}

func (p *GLM46Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder

	for _, event := range events {
		switch event := event.(type) {
		case glm46EventRawToolCall:
			toolCall, err := parseGLM46ToolCall(event, p.tools)
			if err != nil {
				slog.Warn("glm-4.6 tool call parsing failed", "error", err)
				return "", "", nil, err
			}
			toolCall.Function.Index = p.callIndex
			p.callIndex++
			toolCalls = append(toolCalls, toolCall)
		case glm46EventThinkingContent:
			thinkingSb.WriteString(event.content)
		case glm46EventContent:
			// TODO(drifkin): if the same turn contains multiple interleaved content
			// events, we naively append them together here.
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
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
		slog.Log(context.TODO(), logutil.LevelTrace, "glm-4.6 events parsed", "events", all, "state", p.state, "buffer", p.buffer.String())
	}

	return all
}

// eatLeadingWhitespaceAndTransitionTo consumes leading whitespace from the buffer
// and transitions to the next state. Returns (nil, false) if only whitespace remains
// in the buffer (needs more input), or (nil, true) if we successfully transitioned.
func (p *GLM46Parser) eatLeadingWhitespaceAndTransitionTo(nextState glm46ParserState) ([]glm46Event, bool) {
	trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
	p.buffer.Reset()
	if trimmed == "" {
		return nil, false // Still only whitespace, keep waiting for more input
	}
	p.state = nextState
	p.buffer.WriteString(trimmed)
	return nil, true // Successfully transitioned
}

// glm46SplitAtTag splits the buffer at the given tag, returns the content before (trimmed of trailing whitespace),
// the content after (optionally trimmed of leading whitespace), and updates the buffer
func glm46SplitAtTag(p *GLM46Parser, tag string, trimAfter bool) (string, string) {
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

func (p *GLM46Parser) eat() ([]glm46Event, bool) {
	var events []glm46Event

	switch p.state {
	case glm46ParserState_LookingForThinkingOpen:
		trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
		if strings.HasPrefix(trimmed, glm46ThinkingOpenTag) {
			// Found <think> opening tag
			after := strings.TrimPrefix(trimmed, glm46ThinkingOpenTag)
			after = strings.TrimLeftFunc(after, unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(after)
			if after == "" {
				p.state = glm46ParserState_ThinkingStartedEatingWhitespace
			} else {
				p.state = glm46ParserState_CollectingThinking
			}
			return events, true
		} else if strings.HasPrefix(glm46ThinkingOpenTag, trimmed) {
			// Partial opening tag seen, keep accumulating
			return events, false
		} else if trimmed == "" {
			// Only whitespace, keep accumulating
			return events, false
		} else {
			// No thinking tag found, skip to content collection
			p.state = glm46ParserState_CollectingContent
			// Don't trim - we want to keep the original content
			return events, true
		}

	case glm46ParserState_ThinkingStartedEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(glm46ParserState_CollectingThinking)

	case glm46ParserState_CollectingThinking:
		acc := p.buffer.String()
		if strings.Contains(acc, glm46ThinkingCloseTag) {
			thinking, remaining := glm46SplitAtTag(p, glm46ThinkingCloseTag, true)
			if len(thinking) > 0 {
				events = append(events, glm46EventThinkingContent{content: thinking})
			}
			if remaining == "" {
				p.state = glm46ParserState_ThinkingDoneEatingWhitespace
			} else {
				p.state = glm46ParserState_CollectingContent
			}
			return events, true
		} else if overlapLen := overlap(acc, glm46ThinkingCloseTag); overlapLen > 0 {
			// Partial closing tag - withhold it along with any trailing whitespace before it
			beforePartialTag := acc[:len(acc)-overlapLen]
			trailingWhitespaceLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen

			unambiguous := acc[:ambiguousStart]
			ambiguous := acc[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, glm46EventThinkingContent{content: unambiguous})
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
				events = append(events, glm46EventThinkingContent{content: unambiguous})
			}
			return events, false
		}

	case glm46ParserState_ThinkingDoneEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(glm46ParserState_CollectingContent)

	case glm46ParserState_CollectingContent:
		if strings.Contains(p.buffer.String(), glm46ToolOpenTag) {
			before, after := glm46SplitAtTag(p, glm46ToolOpenTag, true)
			if len(before) > 0 {
				events = append(events, glm46EventContent{content: before})
			}
			if after == "" {
				p.state = glm46ParserState_ToolStartedEatingWhitespace
			} else {
				p.state = glm46ParserState_CollectingToolContent
			}
			return events, true
		} else if overlapLen := overlap(p.buffer.String(), glm46ToolOpenTag); overlapLen > 0 {
			beforePartialTag := p.buffer.String()[:len(p.buffer.String())-overlapLen]
			trailingWhitespaceLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen

			unambiguous := p.buffer.String()[:ambiguousStart]
			ambiguous := p.buffer.String()[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
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

	case glm46ParserState_ToolStartedEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(glm46ParserState_CollectingToolContent)

	case glm46ParserState_CollectingToolContent:
		acc := p.buffer.String()
		if strings.Contains(acc, glm46ToolCloseTag) {
			toolContent, _ := glm46SplitAtTag(p, glm46ToolCloseTag, true)
			if len(toolContent) == 0 {
				slog.Warn("glm46 tool call closing tag found but no content before it")
			}
			events = append(events, glm46EventRawToolCall{raw: toolContent})
			p.state = glm46ParserState_CollectingContent
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

// GLMToolCallXML represents the structure of a GLM-4.6 tool call for XML parsing
type GLMToolCallXML struct {
	XMLName xml.Name `xml:"tool_call"`
	Content string   `xml:",chardata"` // Function name (text nodes between tags)
	Keys    []string `xml:"arg_key"`   // All arg_key elements in document order
	Values  []string `xml:"arg_value"` // All arg_value elements in document order
}

// escapeGLM46Content escapes XML entities in text content while preserving arg_key/arg_value tags
func escapeGLM46Content(s string) string {
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

// repairPhase represents the expected next tag in the repair cycle.
type repairPhase int

const (
	phaseArgKeyOpen  repairPhase = iota // expecting <arg_key>
	phaseArgKeyClose                    // expecting </arg_key>
	phaseArgValOpen                     // expecting <arg_value>
	phaseArgValClose                    // expecting </arg_value>
	phaseCount                          // number of phases
)

// repairGLM46XML reconstructs well-formed XML from GLM model output that may
// have missing or mismatched tags. The expected structure is:
//
//	func_name
//	<arg_key>key</arg_key>
//	<arg_value>value</arg_value>
//	...
//
// GLM models frequently omit opening or closing tags. This function follows
// the expected tag cycle, scanning forward for each expected tag in sequence.
// When a tag is missing, it inserts the tag and consumes any text in between.
func repairGLM46XML(s string) string {
	// tagCycle is the repeating sequence of tags after the function name.
	tagCycle := [phaseCount]string{"<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"}

	// findNextTag returns the index and identity of the earliest known tag in s.
	findNextTag := func(s string) (int, string) {
		bestIdx := -1
		bestTag := ""
		for _, tag := range tagCycle {
			if idx := strings.Index(s, tag); idx != -1 && (bestIdx == -1 || idx < bestIdx) {
				bestIdx = idx
				bestTag = tag
			}
		}
		return bestIdx, bestTag
	}

	// tagIndex returns the phase corresponding to the given tag.
	tagIndex := func(tag string) repairPhase {
		for i, t := range tagCycle {
			if t == tag {
				return repairPhase(i)
			}
		}
		return -1
	}

	var result strings.Builder

	idx, firstTag := findNextTag(s)
	if idx == -1 {
		return s
	}
	prefix := s[:idx]
	s = s[idx:]

	// If the first tag is not <arg_key>, the text before it may contain both
	// the function name and key content (e.g. "weather city</arg_key>").
	// Function names cannot contain space, so split at the first space.
	phase := phaseArgKeyOpen
	if firstTag != "<arg_key>" {
		if spIdx := strings.IndexFunc(prefix, unicode.IsSpace); spIdx != -1 {
			result.WriteString(prefix[:spIdx])
			keyContent := strings.TrimLeftFunc(prefix[spIdx:], unicode.IsSpace)
			result.WriteString("<arg_key>")
			result.WriteString(keyContent)
			phase = phaseArgKeyClose
		} else {
			result.WriteString(prefix)
		}
	} else {
		result.WriteString(prefix)
	}

	// Walk through the expected tag cycle. At each step, look for the
	// expected tag. If a different tag appears first, emit the missing
	// tags to catch up, then continue.
	for len(s) > 0 {
		idx, found := findNextTag(s)
		expected := tagCycle[phase]
		isOpen := phase%2 == 0 // even phases are opening tags

		if idx == -1 {
			// No more tags — emit remaining text with fixups
			if isOpen {
				// Expecting an opening tag but nothing left — we're done
				break
			}
			// Expecting a closing tag — emit text then close
			result.WriteString(s)
			result.WriteString(expected)
			phase = (phase + 1) % phaseCount
			break
		}

		if found == expected {
			// Found the expected tag — emit any text before it, then the tag
			result.WriteString(s[:idx])
			result.WriteString(expected)
			s = s[idx+len(expected):]
			phase = (phase + 1) % phaseCount
			continue
		}

		// Found a different tag. Insert missing tags to catch up.
		foundIdx := tagIndex(found)

		if isOpen && idx > 0 {
			// Text before the found tag while expecting an opening tag —
			// the opening tag was omitted. Emit it before the text.
			result.WriteString(expected)
			// Advance to the next phase (text content) and then look
			// for the closing tag — but the found tag might be that
			// closing tag or something further ahead. Emit text up to
			// the found tag and insert any missing tags between.
			result.WriteString(s[:idx])
			phase = (phase + 1) % phaseCount // now expecting closing
			s = s[idx:]
			// Fall through to re-evaluate with the closing tag expected
			continue
		}

		// Emit missing tags to advance from current phase to the found tag's phase
		for phase != foundIdx {
			tag := tagCycle[phase]
			if phase%2 == 0 {
				result.WriteString(tag)
			} else {
				// Closing tag — emit any text before the found tag first,
				// but only if we're one step before the found tag
				if (phase+1)%phaseCount == foundIdx && idx > 0 {
					result.WriteString(s[:idx])
					s = s[idx:]
					idx = 0
				}
				result.WriteString(tag)
			}
			phase = (phase + 1) % phaseCount
		}
		// Now phase == foundIdx, re-process without advancing s
	}

	// If we stopped mid-pair (after an opening tag), close it
	switch phase {
	case phaseArgKeyClose: // after <arg_key>, expecting text/</arg_key>
		result.WriteString("</arg_key>")
		result.WriteString("<arg_value>")
		result.WriteString("</arg_value>")
	case phaseArgValOpen: // after </arg_key>, expecting <arg_value>
		result.WriteString("<arg_value>")
		result.WriteString("</arg_value>")
	case phaseArgValClose: // after <arg_value>, expecting text/</arg_value>
		result.WriteString("</arg_value>")
	}

	return result.String()
}

func parseGLM46ToolCall(raw glm46EventRawToolCall, tools []api.Tool) (api.ToolCall, error) {
	// Escape any unescaped entities in text content
	// We need to escape text between tags, but not the tags themselves
	escaped := escapeGLM46Content(raw.raw)

	// Wrap the content in a root element to make it valid XML
	xmlString := "<tool_call>" + escaped + "</tool_call>"

	// Parse XML into struct, retrying once with repaired XML if it fails
	var parsed GLMToolCallXML
	if err := xml.Unmarshal([]byte(xmlString), &parsed); err != nil {
		parsed = GLMToolCallXML{}
		repaired := "<tool_call>" + repairGLM46XML(escaped) + "</tool_call>"
		if err2 := xml.Unmarshal([]byte(repaired), &parsed); err2 != nil {
			return api.ToolCall{}, fmt.Errorf("failed to parse XML: %w", err)
		}
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
