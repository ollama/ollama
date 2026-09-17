package parsers

import (
	"log/slog"
	"regexp"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

// MiniCPM5 emits native tool calls as XML using special vocabulary tokens, e.g.
// <function name="get_weather"><param name="city">Paris</param></function>.
// Because the tags are special tokens in the MiniCPM5 tokenizer, the runner
// strips them from detokenized output unless they are listed as preserved
// tokens, which leaves only mangled fragments like ` name="get_weather">` in
// the content and makes tool calls unparseable. This parser preserves those
// tokens and parses the XML tool calls into structured tool calls.

type minicpm5State int

const (
	minicpm5CollectingThinking minicpm5State = iota
	minicpm5CollectingContent
	minicpm5CollectingToolCall
)

const (
	minicpm5ThinkOpenTag  = "<think>"
	minicpm5ThinkCloseTag = "</think>"

	minicpm5ToolCallOpenTag  = "<tool_call>"
	minicpm5ToolCallCloseTag = "</tool_call>"

	minicpm5FunctionOpenTag   = "<function"
	minicpm5FunctionCloseTag  = "</function>"
	minicpm5ParamOpenTag      = "<param"
	minicpm5ParamCloseTag     = "</param>"
	minicpm5ArgumentsOpenTag  = "<arguments>"
	minicpm5ArgumentsCloseTag = "</arguments>"
)

var (
	// matches name="get_weather" (or name='get_weather') on <function>/<param> tags
	minicpm5NameAttrRe = regexp.MustCompile(`name\s*=\s*["']([^"']+)["']`)
	// matches <param name="city">Paris</param>, values may span lines
	minicpm5ParamRe = regexp.MustCompile(`(?s)<param\s+name\s*=\s*["']([^"']+)["']\s*>(.*?)</param>`)
)

// minicpm5Event represents an event emitted during parsing
type minicpm5Event interface {
	isMiniCPM5Event()
}

type minicpm5EventThinking struct {
	content string
}

type minicpm5EventContent struct {
	content string
}

type minicpm5EventToolCall struct {
	toolCall api.ToolCall
}

func (minicpm5EventThinking) isMiniCPM5Event() {}
func (minicpm5EventContent) isMiniCPM5Event()  {}
func (minicpm5EventToolCall) isMiniCPM5Event() {}

type MiniCPM5Parser struct {
	state              minicpm5State
	buffer             strings.Builder
	tools              []api.Tool
	callIndex          int
	hasThinkingSupport bool
	// thinkTagHandled tracks whether a leading <think> open tag (emitted by
	// the model when the prompt did not prefill it) has been stripped.
	thinkTagHandled bool
}

func (p *MiniCPM5Parser) HasToolSupport() bool {
	return true
}

func (p *MiniCPM5Parser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

// PreservedTokens returns the special XML tool-call tokens that must remain
// visible in detokenized output for this parser to recognize tool calls.
func (p *MiniCPM5Parser) PreservedTokens() []string {
	return []string{
		minicpm5ToolCallOpenTag,
		minicpm5ToolCallCloseTag,
		minicpm5FunctionOpenTag,
		minicpm5FunctionCloseTag,
		minicpm5ParamOpenTag,
		minicpm5ParamCloseTag,
		minicpm5ArgumentsOpenTag,
		minicpm5ArgumentsCloseTag,
	}
}

func (p *MiniCPM5Parser) setInitialState(lastMessage *api.Message, thinkValue *api.ThinkValue) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	thinkingEnabled := p.HasThinkingSupport() && thinkValue != nil && thinkValue.Bool()
	if !thinkingEnabled {
		p.state = minicpm5CollectingContent
		return
	}

	if prefill && lastMessage.Content != "" {
		p.state = minicpm5CollectingContent
		return
	}

	p.state = minicpm5CollectingThinking
}

func (p *MiniCPM5Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.callIndex = 0
	p.thinkTagHandled = false
	p.setInitialState(lastMessage, thinkValue)
	return tools
}

func (p *MiniCPM5Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)

	events := p.parseEvents()

	var contentSb, thinkingSb strings.Builder
	var toolCalls []api.ToolCall
	for _, event := range events {
		switch event := event.(type) {
		case minicpm5EventThinking:
			thinkingSb.WriteString(event.content)
		case minicpm5EventContent:
			contentSb.WriteString(event.content)
		case minicpm5EventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		}
	}

	// On the final chunk, flush whatever is left in the buffer as visible
	// text instead of silently dropping it.
	if done && p.buffer.Len() > 0 {
		remaining := p.buffer.String()
		p.buffer.Reset()
		switch p.state {
		case minicpm5CollectingThinking:
			thinkingSb.WriteString(remaining)
		default:
			contentSb.WriteString(remaining)
		}
	}

	for i := range toolCalls {
		toolCalls[i].Function.Index = p.callIndex
		p.callIndex++
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *MiniCPM5Parser) parseEvents() []minicpm5Event {
	var all []minicpm5Event

	keepLooping := true
	for keepLooping {
		var events []minicpm5Event
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

func (p *MiniCPM5Parser) eat() ([]minicpm5Event, bool) {
	var events []minicpm5Event
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case minicpm5CollectingThinking:
		// Strip a leading <think> open tag. The model emits it itself when
		// the prompt did not prefill it; it is only valid at the very start
		// of the thinking content.
		if !p.thinkTagHandled {
			trimmed := strings.TrimLeftFunc(bufStr, unicode.IsSpace)
			if rest, ok := strings.CutPrefix(trimmed, minicpm5ThinkOpenTag); ok {
				bufStr = strings.TrimLeftFunc(rest, unicode.IsSpace)
				p.buffer.Reset()
				p.buffer.WriteString(bufStr)
				p.thinkTagHandled = true
			} else if n := overlap(trimmed, minicpm5ThinkOpenTag); n > 0 && n == len(trimmed) {
				// Buffer holds only a partial <think> so far; wait for more.
				return events, false
			} else {
				p.thinkTagHandled = true
			}
		}
		if bufStr == "" {
			return events, false
		}

		if idx := strings.Index(bufStr, minicpm5ThinkCloseTag); idx != -1 {
			thinking := strings.TrimRightFunc(bufStr[:idx], unicode.IsSpace)
			remaining := strings.TrimLeftFunc(bufStr[idx+len(minicpm5ThinkCloseTag):], unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = minicpm5CollectingContent

			if thinking != "" {
				events = append(events, minicpm5EventThinking{content: thinking})
			}
			return events, true
		}

		// Hold back a trailing partial </think> (and any whitespace before
		// it) so a split tag is not emitted as thinking content.
		if n := overlap(bufStr, minicpm5ThinkCloseTag); n > 0 {
			beforePartial := bufStr[:len(bufStr)-n]
			trailingLen := trailingWhitespaceLen(beforePartial)
			ambiguousStart := len(beforePartial) - trailingLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if unambiguous != "" {
				events = append(events, minicpm5EventThinking{content: unambiguous})
			}
			return events, false
		}

		whitespaceLen := trailingWhitespaceLen(bufStr)
		ambiguousStart := len(bufStr) - whitespaceLen
		unambiguous := bufStr[:ambiguousStart]
		ambiguous := bufStr[ambiguousStart:]
		p.buffer.Reset()
		p.buffer.WriteString(ambiguous)
		if unambiguous != "" {
			events = append(events, minicpm5EventThinking{content: unambiguous})
		}
		return events, false

	case minicpm5CollectingContent:
		// <tool_call> wrappers are transparent: emit preceding content and drop the tag
		if idx := strings.Index(bufStr, minicpm5ToolCallOpenTag); idx != -1 {
			before := strings.TrimRightFunc(bufStr[:idx], unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(bufStr[idx+len(minicpm5ToolCallOpenTag):])
			if before != "" {
				events = append(events, minicpm5EventContent{content: before})
			}
			return events, true
		}
		if idx := strings.Index(bufStr, minicpm5ToolCallCloseTag); idx != -1 {
			p.buffer.Reset()
			p.buffer.WriteString(bufStr[:idx] + bufStr[idx+len(minicpm5ToolCallCloseTag):])
			return events, true
		}

		if idx := strings.Index(bufStr, minicpm5FunctionOpenTag); idx != -1 {
			before := strings.TrimRightFunc(bufStr[:idx], unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(bufStr[idx:])
			p.state = minicpm5CollectingToolCall
			if before != "" {
				events = append(events, minicpm5EventContent{content: before})
			}
			return events, true
		}

		// Hold back a trailing partial tag so it is not emitted as content.
		if n := maxTagOverlap(bufStr); n > 0 {
			unambiguous := bufStr[:len(bufStr)-n]
			ambiguous := bufStr[len(bufStr)-n:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if unambiguous != "" {
				events = append(events, minicpm5EventContent{content: unambiguous})
			}
			return events, false
		}

		p.buffer.Reset()
		events = append(events, minicpm5EventContent{content: bufStr})
		return events, false

	case minicpm5CollectingToolCall:
		if idx := strings.Index(bufStr, minicpm5FunctionCloseTag); idx != -1 {
			block := bufStr[:idx+len(minicpm5FunctionCloseTag)]
			remaining := strings.TrimLeftFunc(bufStr[idx+len(minicpm5FunctionCloseTag):], unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = minicpm5CollectingContent

			if call, ok := parseMiniCPM5ToolCall(block, p.tools); ok {
				events = append(events, minicpm5EventToolCall{toolCall: call})
			} else {
				slog.Warn("minicpm5 tool call parsing failed, emitting as content", "block", block)
				events = append(events, minicpm5EventContent{content: block})
			}
			return events, true
		}

		return events, false
	}

	return events, false
}

// maxTagOverlap returns the longest overlap between the suffix of s and the
// prefix of any tag the content state may be waiting on.
func maxTagOverlap(s string) int {
	n := 0
	for _, tag := range []string{minicpm5FunctionOpenTag, minicpm5ToolCallOpenTag, minicpm5ToolCallCloseTag} {
		if m := overlap(s, tag); m > n {
			n = m
		}
	}
	return n
}

// parseMiniCPM5ToolCall parses one <function name="...">...</function> block.
// It returns false when the block does not name a known tool, in which case
// the caller emits the block as regular content.
func parseMiniCPM5ToolCall(block string, tools []api.Tool) (api.ToolCall, bool) {
	openEnd := strings.Index(block, ">")
	if openEnd == -1 {
		return api.ToolCall{}, false
	}

	m := minicpm5NameAttrRe.FindStringSubmatch(block[:openEnd+1])
	if m == nil {
		return api.ToolCall{}, false
	}
	name := m[1]

	if _, err := toolByName(tools, name); err != nil {
		return api.ToolCall{}, false
	}

	inner := block[openEnd+1 : len(block)-len(minicpm5FunctionCloseTag)]
	args := api.NewToolCallFunctionArguments()
	for _, pm := range minicpm5ParamRe.FindAllStringSubmatch(inner, -1) {
		value := pm[2]
		if strings.HasPrefix(value, "<![CDATA[") && strings.HasSuffix(value, "]]>") {
			value = value[len("<![CDATA[") : len(value)-len("]]>")]
		} else {
			value = strings.TrimSpace(value)
		}
		args.Set(pm[1], value)
	}

	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      name,
			Arguments: args,
		},
	}, true
}
