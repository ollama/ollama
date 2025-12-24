package parsers

import (
	"context"
	"fmt"
	"log/slog"
	"regexp"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

type olmo3ParserState int

const (
	olmo3StateContent olmo3ParserState = iota
	olmo3StateToolCalls
	olmo3StateToolCallsDone
)

const (
	olmo3FuncCallsOpenTag  = "<function_calls>"
	olmo3FuncCallsCloseTag = "</function_calls>"
)

type Olmo3Parser struct {
	state  olmo3ParserState
	buffer strings.Builder
}

func (p *Olmo3Parser) HasToolSupport() bool {
	return true
}

func (p *Olmo3Parser) HasThinkingSupport() bool {
	return false
}

func (p *Olmo3Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.state = olmo3StateContent
	return tools
}

type olmo3ParserEvent interface {
	isOlmo3ParserEvent()
}

type olmo3ParserEventContent struct {
	content string
}

type olmo3ParserEventToolCalls struct {
	calls []api.ToolCall
}

func (olmo3ParserEventContent) isOlmo3ParserEvent()   {}
func (olmo3ParserEventToolCalls) isOlmo3ParserEvent() {}

func (p *Olmo3Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)

	if done {
		// Drain any remaining content
		bufStr := p.buffer.String()
		p.buffer.Reset()
		if p.state == olmo3StateContent && len(bufStr) > 0 {
			return bufStr, "", nil, nil
		}
		return "", "", nil, nil
	}

	events := p.parseEvents()

	var contentSb strings.Builder
	var allCalls []api.ToolCall
	for _, event := range events {
		switch event := event.(type) {
		case olmo3ParserEventContent:
			contentSb.WriteString(event.content)
		case olmo3ParserEventToolCalls:
			allCalls = append(allCalls, event.calls...)
		}
	}

	return contentSb.String(), "", allCalls, nil
}

func (p *Olmo3Parser) parseEvents() []olmo3ParserEvent {
	var all []olmo3ParserEvent

	keepLooping := true
	for keepLooping {
		var events []olmo3ParserEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "olmo3 events parsed", "events", all, "state", p.state, "buffer", p.buffer.String())
	}

	return all
}

func (p *Olmo3Parser) eat() ([]olmo3ParserEvent, bool) {
	var events []olmo3ParserEvent
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case olmo3StateContent:
		if strings.Contains(bufStr, olmo3FuncCallsOpenTag) {
			// Found <function_calls> tag
			split := strings.SplitN(bufStr, olmo3FuncCallsOpenTag, 2)
			content := split[0]
			remaining := split[1]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = olmo3StateToolCalls

			if len(content) > 0 {
				events = append(events, olmo3ParserEventContent{content: content})
			}
			return events, true
		} else if overlapLen := overlap(bufStr, olmo3FuncCallsOpenTag); overlapLen > 0 {
			// Partial <function_calls> tag - withhold ambiguous content
			unambiguous := bufStr[:len(bufStr)-overlapLen]
			ambiguous := bufStr[len(bufStr)-overlapLen:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, olmo3ParserEventContent{content: unambiguous})
			}
			return events, false
		} else {
			// Regular content - emit all
			p.buffer.Reset()
			if len(bufStr) > 0 {
				events = append(events, olmo3ParserEventContent{content: bufStr})
			}
			return events, false
		}

	case olmo3StateToolCalls:
		if strings.Contains(bufStr, olmo3FuncCallsCloseTag) {
			// Found </function_calls> tag
			split := strings.SplitN(bufStr, olmo3FuncCallsCloseTag, 2)
			toolCallsStr := split[0]
			remaining := split[1]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = olmo3StateToolCallsDone

			// Parse the function calls
			calls, err := parseOlmo3FunctionCalls(toolCallsStr)
			if err != nil {
				slog.Log(context.TODO(), logutil.LevelTrace, "failed to parse olmo3 function calls", "error", err, "content", toolCallsStr)
			} else if len(calls) > 0 {
				events = append(events, olmo3ParserEventToolCalls{calls: calls})
			}
			return events, true
		} else if overlapLen := overlap(bufStr, olmo3FuncCallsCloseTag); overlapLen > 0 {
			// Partial </function_calls> tag - wait for more
			return events, false
		}
		// Still collecting tool calls, wait for close tag
		return events, false

	case olmo3StateToolCallsDone:
		// After tool calls, emit remaining content
		p.buffer.Reset()
		p.state = olmo3StateContent
		if len(bufStr) > 0 {
			events = append(events, olmo3ParserEventContent{content: bufStr})
		}
		return events, false
	}

	return events, false
}

// parseOlmo3FunctionCalls parses function calls in Python-esque format:
// func_name(arg1="value1", arg2=123)
// Multiple calls are separated by newlines
func parseOlmo3FunctionCalls(s string) ([]api.ToolCall, error) {
	var calls []api.ToolCall
	s = strings.TrimSpace(s)
	if s == "" {
		return calls, nil
	}

	// Split by newlines for multiple function calls
	lines := strings.Split(s, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		call, err := parseOlmo3SingleFunctionCall(line)
		if err != nil {
			return nil, fmt.Errorf("failed to parse function call %q: %w", line, err)
		}
		calls = append(calls, call)
	}

	return calls, nil
}

// Regex to match function call: func_name(args)
var funcCallRegex = regexp.MustCompile(`^(\w+)\((.*)\)$`)

func parseOlmo3SingleFunctionCall(s string) (api.ToolCall, error) {
	matches := funcCallRegex.FindStringSubmatch(s)
	if matches == nil {
		return api.ToolCall{}, fmt.Errorf("invalid function call format")
	}

	funcName := matches[1]
	argsStr := matches[2]

	args, err := parseOlmo3Arguments(argsStr)
	if err != nil {
		return api.ToolCall{}, fmt.Errorf("failed to parse arguments: %w", err)
	}

	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      funcName,
			Arguments: args,
		},
	}, nil
}

// parseOlmo3Arguments parses comma-separated key=value pairs
// Handles nested parentheses, brackets, braces, and quoted strings
func parseOlmo3Arguments(s string) (api.ToolCallFunctionArguments, error) {
	args := api.NewToolCallFunctionArguments()
	s = strings.TrimSpace(s)
	if s == "" {
		return args, nil
	}

	// Split by commas, but respect nested structures and quotes
	parts := splitArguments(s)

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		// Find the first = sign
		eqIdx := strings.Index(part, "=")
		if eqIdx == -1 {
			return api.ToolCallFunctionArguments{}, fmt.Errorf("invalid argument format: %s", part)
		}

		key := strings.TrimSpace(part[:eqIdx])
		valueStr := strings.TrimSpace(part[eqIdx+1:])

		value, err := parseOlmo3Value(valueStr)
		if err != nil {
			return api.ToolCallFunctionArguments{}, fmt.Errorf("failed to parse value for %s: %w", key, err)
		}

		args.Set(key, value)
	}

	return args, nil
}

// splitArguments splits arguments by commas, respecting quotes and nested structures
func splitArguments(s string) []string {
	var parts []string
	var current strings.Builder
	depth := 0
	inString := false
	stringChar := byte(0)
	escaped := false

	for i := range s {
		c := s[i]

		if escaped {
			current.WriteByte(c)
			escaped = false
			continue
		}

		if c == '\\' && inString {
			current.WriteByte(c)
			escaped = true
			continue
		}

		if (c == '"' || c == '\'') && !inString {
			inString = true
			stringChar = c
			current.WriteByte(c)
			continue
		}

		if c == stringChar && inString {
			inString = false
			stringChar = 0
			current.WriteByte(c)
			continue
		}

		if !inString {
			switch c {
			case '(', '[', '{':
				depth++
				current.WriteByte(c)
			case ')', ']', '}':
				depth--
				current.WriteByte(c)
			case ',':
				if depth == 0 {
					parts = append(parts, current.String())
					current.Reset()
					continue
				}
				current.WriteByte(c)
			default:
				current.WriteByte(c)
			}
		} else {
			current.WriteByte(c)
		}
	}

	if current.Len() > 0 {
		parts = append(parts, current.String())
	}

	return parts
}

// parseOlmo3Value parses a value which can be a string, number, boolean, null, array, or object
func parseOlmo3Value(s string) (any, error) {
	s = strings.TrimSpace(s)

	// Check for quoted string
	if (strings.HasPrefix(s, `"`) && strings.HasSuffix(s, `"`)) ||
		(strings.HasPrefix(s, `'`) && strings.HasSuffix(s, `'`)) {
		// Remove quotes and unescape
		inner := s[1 : len(s)-1]
		return unescapeString(inner), nil
	}

	// Check for boolean
	if s == "true" || s == "True" {
		return true, nil
	}
	if s == "false" || s == "False" {
		return false, nil
	}

	// Check for null/None
	if s == "null" || s == "None" || s == "nil" {
		return nil, nil
	}

	// Check for number
	if i, err := strconv.ParseInt(s, 10, 64); err == nil {
		return i, nil
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f, nil
	}

	// Check for array [...]
	if strings.HasPrefix(s, "[") && strings.HasSuffix(s, "]") {
		return parseOlmo3Array(s[1 : len(s)-1])
	}

	// Check for object {...}
	if strings.HasPrefix(s, "{") && strings.HasSuffix(s, "}") {
		return parseOlmo3Object(s[1 : len(s)-1])
	}

	// Default to string without quotes
	return s, nil
}

func parseOlmo3Array(s string) ([]any, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return []any{}, nil
	}

	parts := splitArguments(s)
	var arr []any
	for _, part := range parts {
		val, err := parseOlmo3Value(part)
		if err != nil {
			return nil, err
		}
		arr = append(arr, val)
	}
	return arr, nil
}

func parseOlmo3Object(s string) (map[string]any, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return map[string]any{}, nil
	}

	// Objects use key: value or "key": value format
	obj := make(map[string]any)
	parts := splitArguments(s)
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		// Find colon separator
		colonIdx := strings.Index(part, ":")
		if colonIdx == -1 {
			return nil, fmt.Errorf("invalid object entry: %s", part)
		}

		keyStr := strings.TrimSpace(part[:colonIdx])
		valueStr := strings.TrimSpace(part[colonIdx+1:])

		// Remove quotes from key if present
		if (strings.HasPrefix(keyStr, `"`) && strings.HasSuffix(keyStr, `"`)) ||
			(strings.HasPrefix(keyStr, `'`) && strings.HasSuffix(keyStr, `'`)) {
			keyStr = keyStr[1 : len(keyStr)-1]
		}

		val, err := parseOlmo3Value(valueStr)
		if err != nil {
			return nil, fmt.Errorf("failed to parse value for key %s: %w", keyStr, err)
		}

		obj[keyStr] = val
	}

	return obj, nil
}

func unescapeString(s string) string {
	// Handle common escape sequences
	s = strings.ReplaceAll(s, `\\`, "\x00") // Placeholder for backslash
	s = strings.ReplaceAll(s, `\"`, `"`)
	s = strings.ReplaceAll(s, `\'`, `'`)
	s = strings.ReplaceAll(s, `\n`, "\n")
	s = strings.ReplaceAll(s, `\t`, "\t")
	s = strings.ReplaceAll(s, `\r`, "\r")
	s = strings.ReplaceAll(s, "\x00", `\`) // Restore backslash
	return s
}
