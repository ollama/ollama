package parsers

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"log/slog"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

type qwenParserState int

const (
	toolOpenTag  = "<tool_call>"
	toolCloseTag = "</tool_call>"
)

const (
	qwenParserState_LookingForToolStart qwenParserState = iota
	qwenParserState_CollectingToolContent
)

type Qwen3CoderParser struct {
	state qwenParserState
	acc   strings.Builder
	tools []api.Tool
}

func (p *Qwen3CoderParser) HasToolSupport() bool {
	return true
}

func (p *Qwen3CoderParser) HasThinkingSupport() bool {
	return false
}

func (p *Qwen3CoderParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	return tools // Qwen doesn't modify tools
}

func (p *Qwen3CoderParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.acc.WriteString(s)

	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var sb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case qwenEventRawToolCall:
			toolCall, err := parseToolCall(event, p.tools)
			if err != nil {
				slog.Warn("qwen tool call parsing failed", "error", err)
				return "", "", nil, err
			}
			toolCalls = append(toolCalls, toolCall)
		case qwenEventContent:
			// TODO(drifkin): if the same turn contains multiple interleaved content
			// events, we naively append them together here. See the note below about
			// `qwenEvent`s for more details
			sb.WriteString(event.content)
		}
	}

	return sb.String(), "", toolCalls, nil
}

func (p *Qwen3CoderParser) parseEvents() []qwenEvent {
	var all []qwenEvent

	keepLooping := true
	for keepLooping {
		var events []qwenEvent
		events, keepLooping = eat(p)
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "qwen events parsed", "events", all, "state", p.state, "acc", p.acc.String())
	}

	return all
}

// we use some internal event types in order to communicate between `Add` and
// `eat`. We do this to support interleaving content and parallel tool calls in
// the parser, even though qwen3-coder isn't supposed to do this. Our API
// doesn't currently support models outputting multiple messages in a turn, so
// we wouldn't be able to represent it yet, but there's no reason to prevent the
// parser from supporting it, especially for future models if they end up using
// a similar format.
type qwenEvent interface {
	isQwenEvent()
}

type qwenEventRawToolCall struct {
	raw string
}

type qwenEventContent struct {
	content string
}

func (qwenEventContent) isQwenEvent()     {}
func (qwenEventRawToolCall) isQwenEvent() {}

// eat consumes the parser's buffer, and returns a list of any unambiguous
// events from the current parser state. If the parser transitions to another
// state, it may have additional events to emit on the next call, which is what
// the second return value indicates
func eat(p *Qwen3CoderParser) ([]qwenEvent, bool) {
	var events []qwenEvent

	switch p.state {
	case qwenParserState_LookingForToolStart:
		if strings.Contains(p.acc.String(), toolOpenTag) {
			// we found a full tool open tag, so we can emit the content before the
			// tag, being sure to trim any trailing whitespace
			split := strings.SplitN(p.acc.String(), toolOpenTag, 2)
			before := split[0]
			before = strings.TrimRightFunc(before, unicode.IsSpace)
			if len(before) > 0 {
				events = append(events, qwenEventContent{content: before})
			}
			after := split[1]
			p.acc.Reset()
			p.acc.WriteString(after)
			p.state = qwenParserState_CollectingToolContent
			return events, true
		} else if overlap := overlap(p.acc.String(), toolOpenTag); overlap > 0 {
			// we found a partial tool open tag, so we can emit the unambiguous part,
			// which is the (trailing-whitespace trimmed) content before the partial
			// tool open tag
			beforePartialTag := p.acc.String()[:len(p.acc.String())-overlap]
			trailingWhitespaceLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen
			unambiguous := p.acc.String()[:ambiguousStart]
			ambiguous := p.acc.String()[ambiguousStart:]
			p.acc.Reset()
			p.acc.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwenEventContent{content: unambiguous})
			}
			return events, false
		} else {
			// we found content that is entirely not a tool call. We should withhold
			// any trailing whitespace in case this is the end of the content
			whitespaceLen := trailingWhitespaceLen(p.acc.String())
			ambiguousStart := len(p.acc.String()) - whitespaceLen
			unambiguous := p.acc.String()[:ambiguousStart]
			ambiguous := p.acc.String()[ambiguousStart:]
			p.acc.Reset()
			p.acc.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwenEventContent{content: unambiguous})
			}
			return events, false
		}
	case qwenParserState_CollectingToolContent:
		if strings.Contains(p.acc.String(), toolCloseTag) {
			split := strings.SplitN(p.acc.String(), toolCloseTag, 2)
			before := split[0]
			if len(before) == 0 {
				slog.Warn("qwen tool call closing tag found but no content before it")
			}
			// remove any whitespace between the tool call and any content after it
			after := strings.TrimLeftFunc(split[1], unicode.IsSpace)
			p.acc.Reset()
			p.acc.WriteString(after)
			events = append(events, qwenEventRawToolCall{raw: before})
			p.state = qwenParserState_LookingForToolStart
			return events, true
		} else {
			// note that we don't need to check the overlap here because we only plan
			// on parsing the tool call once we see the full closing tag. We don't
			// stream back the unparsed tool content, so there's no need to be eager
			// here
			return events, false
		}
	default:
		panic("unreachable")
	}
}

// TODO(drifkin): move this to a shared location
// longest overlap between suffix of s and prefix of delim
func overlap(s, delim string) int {
	max := min(len(delim), len(s))
	for i := max; i > 0; i-- {
		if strings.HasSuffix(s, delim[:i]) {
			return i
		}
	}
	return 0
}

func trailingWhitespaceLen(s string) int {
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

type XMLFunctionCall struct {
	XMLName    xml.Name       `xml:"function"`
	Name       string         `xml:"name,attr"`
	Parameters []XMLParameter `xml:"parameter"`
}

type XMLParameter struct {
	Name  string `xml:"name,attr"`
	Value string `xml:",chardata"`
}

// parseToolCall parses a raw tool call string into an api.ToolCall.
// The raw string follows an xml-like format, here's an example:
//
// <function=get_current_temperature>
// <parameter=location>
// San Francisco
// </parameter>
// <parameter=unit>
// celsius
// </parameter>
// </function>
func parseToolCall(raw qwenEventRawToolCall, tools []api.Tool) (api.ToolCall, error) {
	toolCall := api.ToolCall{}

	xmlString := transformToXML(raw.raw)

	var functionCall XMLFunctionCall
	err := xml.Unmarshal([]byte(xmlString), &functionCall)
	if err != nil {
		return api.ToolCall{}, err
	}

	toolCall.Function = api.ToolCallFunction{
		Name: functionCall.Name,
	}

	// Find the matching tool to get parameter types
	var matchedTool *api.Tool
	for i := range tools {
		if tools[i].Function.Name == functionCall.Name {
			matchedTool = &tools[i]
			break
		}
	}

	toolCall.Function.Arguments = api.NewToolCallFunctionArguments()
	for _, parameter := range functionCall.Parameters {
		// Look up the parameter type if we found the tool
		var paramType api.PropertyType
		if matchedTool != nil && matchedTool.Function.Parameters.Properties != nil {
			if prop, ok := matchedTool.Function.Parameters.Properties.Get(parameter.Name); ok {
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

		toolCall.Function.Arguments.Set(parameter.Name, parseValue(parameter.Value, paramType))
	}

	return toolCall, nil
}

// parseValue converts a raw string value to the appropriate type based on the parameter type specification.
//
// For union types (multiple types in PropertyType, which we support but doesn't
// seem as though the reference parser does type coercion with those types in
// mind) we use a type precedence approach:
// 1. null - checked first regardless of declared types (matches reference implementation)
// 2. boolean - only "true"/"false" are valid booleans
// 3. integer - must parse as a whole number
// 4. number - must parse as numeric (returns int if no decimal part)
// 5. array - must parse as valid JSON array
// 6. object - must parse as valid JSON object
// 7. string - always succeeds (least specific type)
//
// This precedence ensures we return the most specific type that successfully parses,
// following the principle of least surprise. For example, with PropertyType{"string", "number"},
// "123" becomes 123 (number), while "hello" becomes "hello" (string).
func parseValue(raw string, paramType api.PropertyType) any {
	// first remove a single leading newlines, and a single trailing newline (if
	// they exist). This follows the reference implementation
	raw = strings.TrimPrefix(raw, "\n")
	raw = strings.TrimSuffix(raw, "\n")

	// Check for null first (case-insensitive) - this takes precedence over any type
	if strings.ToLower(raw) == "null" {
		return nil
	}

	// If no type is specified, default to string
	if len(paramType) == 0 {
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
		// If not a valid boolean but boolean is the only type, return false (matching reference)
		if len(paramType) == 1 {
			return false
		}
		// Otherwise try other types
	}

	// Try integer
	if typeSet["integer"] {
		if i, err := strconv.ParseInt(raw, 10, 64); err == nil {
			// Return as int if it fits in int32, otherwise int64
			if i >= math.MinInt32 && i <= math.MaxInt32 {
				return int(i)
			}
			return i
		}
		// If integer is the only type and parsing failed, fall back to string
		if len(paramType) == 1 {
			return raw
		}
	}

	// Try number (float)
	if typeSet["number"] {
		if f, err := strconv.ParseFloat(raw, 64); err == nil {
			// If the number has no decimal part, return as int (matching reference)
			if f == math.Trunc(f) {
				i := int64(f)
				if i >= math.MinInt32 && i <= math.MaxInt32 {
					return int(i)
				}
				return i
			}
			return f
		}
		// If number is the only type and parsing failed, fall back to string
		if len(paramType) == 1 {
			return raw
		}
	}

	// Try array
	if typeSet["array"] {
		var arr []any
		if err := json.Unmarshal([]byte(raw), &arr); err == nil {
			return arr
		}
		// If array is the only type and parsing failed, fall back to string
		if len(paramType) == 1 {
			return raw
		}
	}

	// Try object
	if typeSet["object"] {
		var obj map[string]any
		if err := json.Unmarshal([]byte(raw), &obj); err == nil {
			return obj
		}
		// If object is the only type and parsing failed, fall back to string
		if len(paramType) == 1 {
			return raw
		}
	}

	// String always succeeds (or if "string" is in the type set)
	if typeSet["string"] {
		return raw
	}

	// If we get here, none of the types matched and string wasn't an option
	// We return string as a fallback. The reference implementation will attempt
	// to parse the value as a python literal, but we purposefully don't support
	// that
	return raw
}

var (
	qwenTagRegex    = regexp.MustCompile(`<(\w+)=([^>]+)>`)
	qwenXMLTagRegex = regexp.MustCompile(`</?(?:function|parameter)(?:\s+name="[^"]*")?>`)
)

// transformToXML transforms a raw qwen tool call with xml-like tags into valid
// xml so that it can be parsed by any xml parser
func transformToXML(raw string) string {
	// take the form `<tag=abc>` and transform it to `<tag name="abc">`, taking
	// care to properly escape the string that becomes the attribute value
	transformed := qwenTagRegex.ReplaceAllStringFunc(raw, func(match string) string {
		groups := qwenTagRegex.FindStringSubmatch(match)
		tag := groups[1]
		var escapedValue strings.Builder
		_ = xml.EscapeText(&escapedValue, []byte(groups[2])) // error is always nil for strings.Builder
		return fmt.Sprintf(`<%s name="%s">`, tag, escapedValue.String())
	})

	// Walk the resulting string, escaping any character data that sits between the
	// xml tags we just emitted
	var out strings.Builder
	lastIdx := 0
	for _, loc := range qwenXMLTagRegex.FindAllStringIndex(transformed, -1) {
		if loc[0] > lastIdx {
			escapeTextNode(&out, transformed[lastIdx:loc[0]])
		}
		out.WriteString(transformed[loc[0]:loc[1]])
		lastIdx = loc[1]
	}
	if lastIdx < len(transformed) {
		escapeTextNode(&out, transformed[lastIdx:])
	}

	return out.String()
}

// escapeTextNode escapes XML character data without altering other characters
// like newlines or tabs (which is why we don't use xml.EscapeText for this)
func escapeTextNode(sb *strings.Builder, s string) {
	for _, r := range s {
		switch r {
		case '&':
			sb.WriteString("&amp;")
		case '<':
			sb.WriteString("&lt;")
		case '>':
			sb.WriteString("&gt;")
		default:
			sb.WriteRune(r)
		}
	}
}
