package parsers

import (
	"context"
	"encoding/json"
	"log/slog"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

type glm46ParserState int

const (
	thinkOpenTag     = "<think>"
	thinkCloseTag    = "</think>"
	glmToolOpenTag   = "<tool_call>"
	glmToolCloseTag  = "</tool_call>"
	argKeyOpenTag    = "<arg_key>"
	argKeyCloseTag   = "</arg_key>"
	argValueOpenTag  = "<arg_value>"
	argValueCloseTag = "</arg_value>"
)

const (
	glm46ParserState_LookingForTags glm46ParserState = iota
	glm46ParserState_CollectingThinking
	glm46ParserState_CollectingToolCall
)

type GLM46Parser struct {
	state glm46ParserState
	acc   strings.Builder
	tools []api.Tool
}

func (p *GLM46Parser) HasToolSupport() bool {
	return true
}

func (p *GLM46Parser) HasThinkingSupport() bool {
	return true
}

func (p *GLM46Parser) Init(tools []api.Tool, lastMessage *api.Message) []api.Tool {
	p.tools = tools
	return tools
}

func (p *GLM46Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.acc.WriteString(s)

	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var contentBuilder strings.Builder
	var thinkingBuilder strings.Builder

	for _, event := range events {
		switch event := event.(type) {
		case glm46EventRawToolCall:
			toolCall, err := parseGLMToolCall(event, p.tools)
			if err != nil {
				slog.Warn("glm46 tool call parsing failed", "error", err)
				return "", "", nil, err
			}
			toolCalls = append(toolCalls, toolCall)
		case glm46EventContent:
			contentBuilder.WriteString(event.content)
		case glm46EventThinking:
			thinkingBuilder.WriteString(event.thinking)
		}
	}

	return contentBuilder.String(), thinkingBuilder.String(), toolCalls, nil
}

func (p *GLM46Parser) parseEvents() []glm46Event {
	var all []glm46Event

	keepLooping := true
	for keepLooping {
		var events []glm46Event
		events, keepLooping = eatGLM(p)
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "glm46 events parsed", "events", all, "state", p.state, "acc", p.acc.String())
	}

	return all
}

type glm46Event interface {
	isGLM46Event()
}

type glm46EventRawToolCall struct {
	raw string
}

type glm46EventContent struct {
	content string
}

type glm46EventThinking struct {
	thinking string
}

func (glm46EventContent) isGLM46Event()     {}
func (glm46EventRawToolCall) isGLM46Event() {}
func (glm46EventThinking) isGLM46Event()    {}

func eatGLM(p *GLM46Parser) ([]glm46Event, bool) {
	var events []glm46Event

	switch p.state {
	case glm46ParserState_LookingForTags:
		buf := p.acc.String()

		// Check for thinking open tag first
		if strings.Contains(buf, thinkOpenTag) {
			split := strings.SplitN(buf, thinkOpenTag, 2)
			before := split[0]
			before = strings.TrimRightFunc(before, unicode.IsSpace)
			if len(before) > 0 {
				events = append(events, glm46EventContent{content: before})
			}
			after := split[1]
			p.acc.Reset()
			p.acc.WriteString(after)
			p.state = glm46ParserState_CollectingThinking
			return events, true
		}

		// Check for tool call open tag
		if strings.Contains(buf, glmToolOpenTag) {
			split := strings.SplitN(buf, glmToolOpenTag, 2)
			before := split[0]
			before = strings.TrimRightFunc(before, unicode.IsSpace)
			if len(before) > 0 {
				events = append(events, glm46EventContent{content: before})
			}
			after := split[1]
			p.acc.Reset()
			p.acc.WriteString(after)
			p.state = glm46ParserState_CollectingToolCall
			return events, true
		}

		// Check for partial tags
		if overlap := glmOverlap(buf, thinkOpenTag); overlap > 0 {
			beforePartialTag := buf[:len(buf)-overlap]
			trailingWhitespaceLen := glmTrailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen
			unambiguous := buf[:ambiguousStart]
			ambiguous := buf[ambiguousStart:]
			p.acc.Reset()
			p.acc.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, glm46EventContent{content: unambiguous})
			}
			return events, false
		}

		if overlap := glmOverlap(buf, glmToolOpenTag); overlap > 0 {
			beforePartialTag := buf[:len(buf)-overlap]
			trailingWhitespaceLen := glmTrailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen
			unambiguous := buf[:ambiguousStart]
			ambiguous := buf[ambiguousStart:]
			p.acc.Reset()
			p.acc.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, glm46EventContent{content: unambiguous})
			}
			return events, false
		}

		// No tags found, emit content but withhold trailing whitespace
		whitespaceLen := glmTrailingWhitespaceLen(buf)
		ambiguousStart := len(buf) - whitespaceLen
		unambiguous := buf[:ambiguousStart]
		ambiguous := buf[ambiguousStart:]
		p.acc.Reset()
		p.acc.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, glm46EventContent{content: unambiguous})
		}
		return events, false

	case glm46ParserState_CollectingThinking:
		if strings.Contains(p.acc.String(), thinkCloseTag) {
			split := strings.SplitN(p.acc.String(), thinkCloseTag, 2)
			thinkingContent := split[0]
			after := strings.TrimLeftFunc(split[1], unicode.IsSpace)
			p.acc.Reset()
			p.acc.WriteString(after)
			events = append(events, glm46EventThinking{thinking: thinkingContent})
			p.state = glm46ParserState_LookingForTags
			return events, true
		}
		return events, false

	case glm46ParserState_CollectingToolCall:
		if strings.Contains(p.acc.String(), glmToolCloseTag) {
			split := strings.SplitN(p.acc.String(), glmToolCloseTag, 2)
			toolCallContent := split[0]
			if len(toolCallContent) == 0 {
				slog.Warn("glm46 tool call closing tag found but no content before it")
			}
			after := strings.TrimLeftFunc(split[1], unicode.IsSpace)
			p.acc.Reset()
			p.acc.WriteString(after)
			events = append(events, glm46EventRawToolCall{raw: toolCallContent})
			p.state = glm46ParserState_LookingForTags
			return events, true
		}
		return events, false

	default:
		panic("unreachable")
	}
}

var (
	glmFunctionNameRegex = regexp.MustCompile(`^([^\n<]+)`)
	glmArgKeyRegex       = regexp.MustCompile(`<arg_key>(.*?)</arg_key>`)
	glmArgValueRegex     = regexp.MustCompile(`<arg_value>(.*?)</arg_value>`)
)

// parseGLMToolCall parses a raw GLM tool call string into an api.ToolCall.
// The raw string has the format:
// {function-name}
// <arg_key>{arg-key-1}</arg_key>
// <arg_value>{arg-value-1}</arg_value>
// <arg_key>{arg-key-2}</arg_key>
// <arg_value>{arg-value-2}</arg_value>
// ...
func parseGLMToolCall(raw glm46EventRawToolCall, tools []api.Tool) (api.ToolCall, error) {
	toolCall := api.ToolCall{}

	// Extract function name (first line or until first <)
	functionNameMatch := glmFunctionNameRegex.FindStringSubmatch(raw.raw)
	if len(functionNameMatch) < 2 {
		return api.ToolCall{}, nil
	}

	functionName := strings.TrimSpace(functionNameMatch[1])
	toolCall.Function = api.ToolCallFunction{
		Name: functionName,
	}

	// Find the matching tool to get parameter types
	var matchedTool *api.Tool
	for i := range tools {
		if tools[i].Function.Name == functionName {
			matchedTool = &tools[i]
			break
		}
	}

	// Extract all arg_key and arg_value pairs
	argKeys := glmArgKeyRegex.FindAllStringSubmatch(raw.raw, -1)
	argValues := glmArgValueRegex.FindAllStringSubmatch(raw.raw, -1)

	if len(argKeys) != len(argValues) {
		slog.Warn("glm46 tool call has mismatched arg_key and arg_value counts", "keys", len(argKeys), "values", len(argValues))
	}

	toolCall.Function.Arguments = make(api.ToolCallFunctionArguments)
	minLen := min(len(argKeys), len(argValues))

	for i := 0; i < minLen; i++ {
		if len(argKeys[i]) < 2 || len(argValues[i]) < 2 {
			continue
		}

		key := strings.TrimSpace(argKeys[i][1])
		value := argValues[i][1]

		// Trim leading and trailing newlines from value (following reference implementation)
		value = strings.TrimPrefix(value, "\n")
		value = strings.TrimSuffix(value, "\n")

		// Look up the parameter type if we found the tool
		var paramType api.PropertyType
		if matchedTool != nil && matchedTool.Function.Parameters.Properties != nil {
			if prop, ok := matchedTool.Function.Parameters.Properties[key]; ok {
				paramType = prop.Type
			}
		}

		// Parse the value according to its type
		toolCall.Function.Arguments[key] = parseGLMValue(value, paramType)
	}

	return toolCall, nil
}

// longest overlap between suffix of s and prefix of delim
func glmOverlap(s, delim string) int {
	max := min(len(delim), len(s))
	for i := max; i > 0; i-- {
		if strings.HasSuffix(s, delim[:i]) {
			return i
		}
	}
	return 0
}

func glmTrailingWhitespaceLen(s string) int {
	remaining := s
	total := 0
	for len(remaining) > 0 {
		r, size := utf8.DecodeLastRuneInString(remaining)
		// if it's an invalid utf8 rune, assume it isn't whitespace
		if r == utf8.RuneError && size == 1 {
			break
		}
		if !unicode.IsSpace(r) {
			break
		}
		total += size
		remaining = remaining[:len(remaining)-size]
	}
	return total
}

func parseGLMValue(raw string, paramType api.PropertyType) any {
	// Check for null first (case-insensitive) - this takes precedence over any type
	if strings.ToLower(raw) == "null" {
		return nil
	}

	// If no type is specified, try to parse as JSON, otherwise return as string
	if len(paramType) == 0 {
		var val any
		if err := json.Unmarshal([]byte(raw), &val); err == nil {
			return val
		}
		return raw
	}

	// Check if any of the specified types match, using type precedence
	// Order: boolean -> integer -> number -> array -> object -> string
	typeSet := make(map[string]bool)
	for _, t := range paramType {
		typeSet[t] = true
	}

	// Try boolean first (most restrictive)
	if typeSet["boolean"] {
		lower := strings.ToLower(raw)
		switch lower {
		case "true":
			return true
		case "false":
			return false
		}
		// If not a valid boolean but boolean is the only type, return false
		if len(paramType) == 1 {
			return false
		}
	}

	// Try parsing as JSON for complex types
	var jsonVal any
	if err := json.Unmarshal([]byte(raw), &jsonVal); err == nil {
		// Check if the parsed type matches any of the expected types
		switch v := jsonVal.(type) {
		case float64:
			if typeSet["number"] {
				return v
			}
			if typeSet["integer"] && v == float64(int64(v)) {
				return int64(v)
			}
		case bool:
			if typeSet["boolean"] {
				return v
			}
		case []any:
			if typeSet["array"] {
				return v
			}
		case map[string]any:
			if typeSet["object"] {
				return v
			}
		case string:
			if typeSet["string"] {
				return v
			}
		case nil:
			return nil
		}

		// If JSON parsed but type doesn't match, check if string is valid
		if typeSet["string"] {
			return raw
		}

		// Return the parsed JSON value as fallback
		return jsonVal
	}

	// If JSON parsing failed but string is valid, return as string
	if typeSet["string"] {
		return raw
	}

	// Fallback to string
	return raw
}
