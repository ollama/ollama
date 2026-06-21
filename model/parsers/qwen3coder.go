package parsers

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

// errEmptyToolCall is returned by parseToolCall when the model emitted a
// <tool_call>…</tool_call> envelope that contains no parseable function
// block. qwen3.6 occasionally does this when it changes its mind mid-stream
// or drifts to an older training format whose tags don't match. Callers
// should treat the envelope as a no-op and let the agent loop decide
// whether to retry or finalize, rather than returning 500 from /api/chat.
var errEmptyToolCall = errors.New("qwen tool call envelope contained no function block")

type qwenParserState int

const (
	toolOpenTag  = "<tool_call>"
	toolCloseTag = "</tool_call>"
	// funcOpenTag / funcCloseTag are the inner function-block tags. qwen3-coder
	// (and qwen3.6) sometimes emit a bare <function=…></function> block with no
	// enclosing <tool_call> opening tag — often leaving a stray </tool_call>
	// behind. We treat <function= as an alternate tool-call trigger so the call
	// isn't dropped into content. See https://github.com/ollama/ollama/issues/16686.
	funcOpenTag  = "<function="
	funcCloseTag = "</function>"
)

const (
	qwenParserState_LookingForToolStart qwenParserState = iota
	qwenParserState_CollectingToolContent
	// CollectingBareFunction handles a <function=…> block that appeared without
	// an opening <tool_call>; it collects up to and including </function>.
	qwenParserState_CollectingBareFunction
)

type Qwen3CoderParser struct {
	state     qwenParserState
	acc       strings.Builder
	tools     []api.Tool
	callIndex int
}

func (p *Qwen3CoderParser) HasToolSupport() bool {
	return true
}

func (p *Qwen3CoderParser) HasThinkingSupport() bool {
	return false
}

func (p *Qwen3CoderParser) PreservedTokens() []string {
	return []string{
		toolOpenTag,
		toolCloseTag,
	}
}

func (p *Qwen3CoderParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.callIndex = 0
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
			if errors.Is(err, errEmptyToolCall) {
				// Model emitted an empty or non-tool <tool_call> envelope.
				// Skip silently — returning an error here would 500 the chat
				// request even though the rest of the turn is fine.
				slog.Warn("qwen tool call envelope was empty; skipping", "raw", event.raw)
				continue
			}
			if err != nil {
				slog.Warn("qwen tool call parsing failed", "error", err)
				return "", "", nil, err
			}
			toolCall.Function.Index = p.callIndex
			p.callIndex++
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
		acc := p.acc.String()
		// Dispatch on the earliest of three triggers:
		//   <tool_call>   — normal envelope open
		//   <function=    — a bare function block with no opening <tool_call> (#16686)
		//   </tool_call>  — a stray close tag with no opener (the same malformation
		//                   leaves one behind); drop it so it doesn't leak as content
		best, kind := -1, 0
		for _, t := range []struct {
			idx, kind int
		}{
			{strings.Index(acc, toolOpenTag), 1},
			{strings.Index(acc, funcOpenTag), 2},
			{strings.Index(acc, toolCloseTag), 3},
		} {
			if t.idx >= 0 && (best < 0 || t.idx < best) {
				best, kind = t.idx, t.kind
			}
		}

		switch kind {
		case 1: // <tool_call>: emit content before it, drop the tag, collect the envelope
			before := strings.TrimRightFunc(acc[:best], unicode.IsSpace)
			if len(before) > 0 {
				events = append(events, qwenEventContent{content: before})
			}
			p.acc.Reset()
			p.acc.WriteString(acc[best+len(toolOpenTag):])
			p.state = qwenParserState_CollectingToolContent
			return events, true
		case 2: // bare <function=>: emit content before it, KEEP the tag, collect to </function>
			before := strings.TrimRightFunc(acc[:best], unicode.IsSpace)
			if len(before) > 0 {
				events = append(events, qwenEventContent{content: before})
			}
			p.acc.Reset()
			p.acc.WriteString(acc[best:])
			p.state = qwenParserState_CollectingBareFunction
			return events, true
		case 3: // stray </tool_call>: emit content before it, drop the tag + trailing space
			before := strings.TrimRightFunc(acc[:best], unicode.IsSpace)
			if len(before) > 0 {
				events = append(events, qwenEventContent{content: before})
			}
			rest := strings.TrimLeftFunc(acc[best+len(toolCloseTag):], unicode.IsSpace)
			p.acc.Reset()
			p.acc.WriteString(rest)
			return events, true
		}

		// No complete trigger. Withhold a trailing partial of any of the three
		// tags (and the whitespace before it, which a real tag would trim), so we
		// never stream out part of a tag or whitespace that may precede one.
		o := overlap(acc, toolOpenTag)
		if of := overlap(acc, funcOpenTag); of > o {
			o = of
		}
		if oc := overlap(acc, toolCloseTag); oc > o {
			o = oc
		}
		beforePartial := acc[:len(acc)-o]
		ambiguousStart := len(beforePartial) - trailingWhitespaceLen(beforePartial)
		unambiguous := acc[:ambiguousStart]
		ambiguous := acc[ambiguousStart:]
		p.acc.Reset()
		p.acc.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, qwenEventContent{content: unambiguous})
		}
		return events, false
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
	case qwenParserState_CollectingBareFunction:
		// We're inside a bare <function=…> block (no enclosing <tool_call>).
		// Collect up to and including </function>, then emit it as a raw tool
		// call. We don't stream partial tool content, so we just wait for the
		// closing tag.
		acc := p.acc.String()
		if idx := strings.Index(acc, funcCloseTag); idx >= 0 {
			raw := acc[:idx+len(funcCloseTag)]
			// Hand the remainder back to LookingForToolStart. A stray </tool_call>
			// the model often leaves after the bare block (the #16686 malformation)
			// is dropped there along with any other trailing tags.
			p.acc.Reset()
			p.acc.WriteString(acc[idx+len(funcCloseTag):])
			events = append(events, qwenEventRawToolCall{raw: raw})
			p.state = qwenParserState_LookingForToolStart
			return events, true
		}
		return events, false
	default:
		panic("unreachable")
	}
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

// extractFunctionBlock narrows the tool-call payload to the first
// <function=...>...</function> block in the input. Some models occasionally
// drift from the documented chat_template format (e.g. qwen3.6, which is
// trained on a wrapper from an earlier generation) and emit stray closing
// tags or unrelated elements alongside an otherwise valid function block.
// Anchoring on the function block lets the XML unmarshaler handle the
// well-formed portion and discard the noise.
//
// Returns the extracted slice and ok=true on success. Returns ("", false)
// when the envelope is empty, whitespace-only, or contains no usable
// <function=...>...</function> pair — callers should treat that as a
// silent no-op rather than an unmarshaler error.
func extractFunctionBlock(raw string) (string, bool) {
	open := strings.Index(raw, "<function=")
	if open < 0 {
		return "", false
	}
	// The first </function> after the opener is the matching close tag.
	// Using LastIndex would accidentally absorb stray duplicates (e.g.
	// `</function></function>`) into the slice and re-introduce the same
	// xml.Unmarshal error this helper is meant to prevent.
	end := strings.Index(raw[open:], "</function>")
	if end < 0 {
		return "", false
	}
	return raw[open : open+end+len("</function>")], true
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

	extracted, ok := extractFunctionBlock(raw.raw)
	if !ok {
		// Empty / non-tool envelope. Surface as a sentinel so the streaming
		// caller can skip silently instead of failing the whole request.
		return api.ToolCall{}, errEmptyToolCall
	}
	xmlString := transformToXML(extracted)

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
