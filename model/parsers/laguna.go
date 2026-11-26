package parsers

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

const (
	lagunaThinkingOpenTag  = "<think>"
	lagunaThinkingCloseTag = "</think>"
	lagunaToolCallOpenTag  = "<tool_call>"
	lagunaToolCallCloseTag = "</tool_call>"
	lagunaUserOpenTag      = "<user>"
	lagunaUserCloseTag     = "</user>"
	lagunaArgKeyOpenTag    = "<arg_key>"
	lagunaArgKeyCloseTag   = "</arg_key>"
	lagunaArgValueOpenTag  = "<arg_value>"
	lagunaArgValueCloseTag = "</arg_value>"
)

type lagunaParserState int

const (
	lagunaParserStateThinking lagunaParserState = iota
	lagunaParserStateContent
	lagunaParserStateTool
)

type LagunaParser struct {
	state                 lagunaParserState
	buffer                strings.Builder
	tools                 []api.Tool
	callIndex             int
	thinkingEnabled       bool
	thinkingSuppressed    bool
	allowLeadingThinkOpen bool
	atContentStart        bool
}

func (p *LagunaParser) HasToolSupport() bool {
	return true
}

func (p *LagunaParser) HasThinkingSupport() bool {
	return true
}

func (p *LagunaParser) ThinkingClose() []string {
	if p.state == lagunaParserStateThinking {
		return []string{lagunaThinkingCloseTag}
	}
	return nil
}

func (p *LagunaParser) PreservedTokens() []string {
	return []string{
		lagunaThinkingOpenTag,
		lagunaThinkingCloseTag,
		lagunaToolCallOpenTag,
		lagunaToolCallCloseTag,
		lagunaUserOpenTag,
		lagunaUserCloseTag,
		lagunaArgKeyOpenTag,
		lagunaArgKeyCloseTag,
		lagunaArgValueOpenTag,
		lagunaArgValueCloseTag,
	}
}

func (p *LagunaParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.callIndex = 0
	p.buffer.Reset()
	// The prompt primes the reasoning mode: <think> when thinking is enabled
	// (so the model's output begins with reasoning and no opening tag), or
	// </think> otherwise. Thinking defaults off, matching the chat template.
	p.thinkingEnabled = thinkValue != nil && thinkValue.Bool()
	p.thinkingSuppressed = !p.thinkingEnabled
	p.atContentStart = true

	// When the request ends with an assistant prefill, the renderer continues
	// that turn in place instead of emitting a fresh generation prompt: it has
	// already written the closing </think>, so the model resumes with content
	// (or a tool call), not reasoning. Start in the content state even when
	// thinking is enabled, otherwise the continuation would be reported as
	// thinking until done and clients would receive an empty answer.
	assistantPrefill := lastMessage != nil && lastMessage.Role == "assistant"

	if p.thinkingEnabled && !assistantPrefill {
		p.state = lagunaParserStateThinking
		p.allowLeadingThinkOpen = true
	} else {
		p.state = lagunaParserStateContent
		p.allowLeadingThinkOpen = false
	}
	return tools
}

// LagunaV8Parser matches the v8 renderer, which closes any assistant history
// turn and emits a fresh assistant generation prompt instead of continuing the
// final assistant message in place.
type LagunaV8Parser struct {
	LagunaParser
}

func (p *LagunaV8Parser) Init(tools []api.Tool, _ *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	return p.LagunaParser.Init(tools, nil, thinkValue)
}

func (p *LagunaParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	var contentSB, thinkingSB strings.Builder

	for {
		progress := false
		switch p.state {
		case lagunaParserStateThinking:
			progress, thinking = p.consumeThinking(done)
			if p.thinkingEnabled {
				thinkingSB.WriteString(thinking)
			}
		case lagunaParserStateContent:
			var parsedCalls []api.ToolCall
			progress, content, parsedCalls, err = p.consumeContent(done)
			if err != nil {
				return "", "", nil, err
			}
			contentSB.WriteString(content)
			calls = append(calls, parsedCalls...)
		case lagunaParserStateTool:
			var call api.ToolCall
			progress, call, err = p.consumeTool(done)
			if err != nil {
				return "", "", nil, err
			}
			if progress {
				calls = append(calls, call)
			}
		}
		if !progress {
			break
		}
	}

	return contentSB.String(), thinkingSB.String(), calls, nil
}

func (p *LagunaParser) consumeThinking(done bool) (bool, string) {
	acc := p.buffer.String()
	if p.allowLeadingThinkOpen {
		trimmed := strings.TrimLeftFunc(acc, unicode.IsSpace)
		if strings.HasPrefix(trimmed, lagunaThinkingOpenTag) {
			// the model echoed the primed <think>; drop it
			p.buffer.Reset()
			p.buffer.WriteString(strings.TrimLeftFunc(strings.TrimPrefix(trimmed, lagunaThinkingOpenTag), unicode.IsSpace))
			p.allowLeadingThinkOpen = false
			return true, ""
		}
		if strings.HasPrefix(lagunaThinkingOpenTag, trimmed) && !done {
			// possibly a partial opening tag; keep it (minus leading space) and wait
			p.buffer.Reset()
			p.buffer.WriteString(trimmed)
			return false, ""
		}
		// reasoning begins here: drop the leading whitespace the model emits
		// after the primed <think> (its trained format is "<think>\n…").
		p.buffer.Reset()
		p.buffer.WriteString(trimmed)
		p.allowLeadingThinkOpen = false
		acc = trimmed
	}

	if idx := strings.Index(acc, lagunaThinkingCloseTag); idx != -1 {
		thinking := strings.TrimRightFunc(acc[:idx], unicode.IsSpace)
		after := strings.TrimLeftFunc(acc[idx+len(lagunaThinkingCloseTag):], unicode.IsSpace)
		p.buffer.Reset()
		p.buffer.WriteString(after)
		p.state = lagunaParserStateContent
		return true, thinking
	}
	if idx := strings.Index(acc, lagunaToolCallOpenTag); idx != -1 {
		thinking := strings.TrimRightFunc(acc[:idx], unicode.IsSpace)
		after := acc[idx+len(lagunaToolCallOpenTag):]
		p.buffer.Reset()
		p.buffer.WriteString(after)
		p.state = lagunaParserStateTool
		return true, thinking
	}
	if done {
		p.buffer.Reset()
		p.state = lagunaParserStateContent
		acc = strings.TrimRightFunc(acc, unicode.IsSpace)
		return acc != "", acc
	}

	overlapLen := max(overlap(acc, lagunaThinkingCloseTag), overlap(acc, lagunaToolCallOpenTag))
	trailingLen := trailingWhitespaceLen(acc)
	keep := max(overlapLen, trailingLen)
	if keep > 0 && keep < len(acc) {
		emit := acc[:len(acc)-keep]
		p.buffer.Reset()
		p.buffer.WriteString(acc[len(acc)-keep:])
		return emit != "", emit
	}
	return false, ""
}

func (p *LagunaParser) consumeContent(done bool) (bool, string, []api.ToolCall, error) {
	if p.atContentStart {
		// Drop the leading whitespace the model emits before content (its trained
		// format puts a newline after the primed/closed </think>).
		trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
		p.buffer.Reset()
		p.buffer.WriteString(trimmed)
		if trimmed == "" {
			return false, "", nil, nil
		}
		p.atContentStart = false
	}
	acc := p.buffer.String()
	if p.thinkingEnabled || p.thinkingSuppressed {
		if idx := strings.Index(acc, lagunaThinkingOpenTag); idx != -1 {
			content := acc[:idx]
			after := strings.TrimLeftFunc(acc[idx+len(lagunaThinkingOpenTag):], unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(after)
			p.state = lagunaParserStateThinking
			p.allowLeadingThinkOpen = false
			return true, content, nil, nil
		}
		if !done {
			overlapLen := overlap(acc, lagunaThinkingOpenTag)
			if overlapLen > 0 && overlapLen < len(acc) {
				content := acc[:len(acc)-overlapLen]
				p.buffer.Reset()
				p.buffer.WriteString(acc[len(acc)-overlapLen:])
				return content != "", content, nil, nil
			}
		}
	}
	if p.thinkingEnabled {
		trimmed := strings.TrimLeftFunc(acc, unicode.IsSpace)
		if strings.HasPrefix(trimmed, lagunaThinkingCloseTag) {
			p.buffer.Reset()
			p.buffer.WriteString(strings.TrimLeftFunc(strings.TrimPrefix(trimmed, lagunaThinkingCloseTag), unicode.IsSpace))
			return true, "", nil, nil
		}
		if strings.HasPrefix(lagunaThinkingCloseTag, trimmed) && !done {
			return false, "", nil, nil
		}
	}
	if p.thinkingSuppressed {
		trimmed := strings.TrimLeftFunc(acc, unicode.IsSpace)
		if strings.HasPrefix(trimmed, lagunaThinkingCloseTag) {
			p.buffer.Reset()
			p.buffer.WriteString(strings.TrimLeftFunc(strings.TrimPrefix(trimmed, lagunaThinkingCloseTag), unicode.IsSpace))
			return true, "", nil, nil
		}
		if strings.HasPrefix(lagunaThinkingCloseTag, trimmed) && !done {
			return false, "", nil, nil
		}
	}
	if idx := strings.Index(acc, lagunaToolCallOpenTag); idx != -1 {
		content := strings.TrimRightFunc(acc[:idx], unicode.IsSpace)
		after := acc[idx+len(lagunaToolCallOpenTag):]
		p.buffer.Reset()
		p.buffer.WriteString(after)
		p.state = lagunaParserStateTool
		return true, content, nil, nil
	}
	if idx := strings.Index(acc, lagunaUserOpenTag); idx != -1 && len(p.tools) > 0 {
		before := strings.TrimRightFunc(acc[:idx], unicode.IsSpace)
		afterOpen := acc[idx+len(lagunaUserOpenTag):]
		if closeIdx := strings.Index(afterOpen, lagunaUserCloseTag); closeIdx != -1 {
			raw := afterOpen[:closeIdx]
			if call, ok := p.parseToolAlias(raw); ok {
				after := strings.TrimLeftFunc(afterOpen[closeIdx+len(lagunaUserCloseTag):], unicode.IsSpace)
				p.buffer.Reset()
				p.buffer.WriteString(after)
				return true, before, []api.ToolCall{call}, nil
			}
		} else if !done {
			if idx > 0 {
				p.buffer.Reset()
				p.buffer.WriteString(acc[idx:])
				return true, before, nil, nil
			}
			return false, "", nil, nil
		}
	}
	if len(p.tools) > 0 {
		if progress, content, call, ok, err := p.consumeStandaloneJSONTool(done); ok || err != nil {
			if err != nil {
				return false, "", nil, err
			}
			if progress {
				return true, content, []api.ToolCall{call}, nil
			}
			return false, "", nil, nil
		}
	}
	if done {
		p.buffer.Reset()
		acc = strings.TrimRightFunc(acc, unicode.IsSpace)
		return acc != "", acc, nil, nil
	}
	overlapLen := max(overlap(acc, lagunaToolCallOpenTag), overlap(acc, lagunaUserOpenTag))
	if p.thinkingEnabled || p.thinkingSuppressed {
		overlapLen = max(overlapLen, overlap(acc, lagunaThinkingOpenTag))
	}
	if p.thinkingSuppressed {
		overlapLen = max(overlapLen, overlap(acc, lagunaThinkingCloseTag))
	}
	trailingLen := trailingWhitespaceLen(acc)
	keep := max(overlapLen, trailingLen)
	if keep > 0 && keep < len(acc) {
		emit := acc[:len(acc)-keep]
		p.buffer.Reset()
		p.buffer.WriteString(acc[len(acc)-keep:])
		return emit != "", emit, nil, nil
	}
	if keep == 0 && acc != "" {
		p.buffer.Reset()
		return true, acc, nil, nil
	}
	return false, "", nil, nil
}

func (p *LagunaParser) consumeStandaloneJSONTool(done bool) (progress bool, content string, call api.ToolCall, ok bool, err error) {
	acc := p.buffer.String()
	jsonIdx := strings.Index(acc, "{")
	if jsonIdx == -1 {
		return false, "", api.ToolCall{}, false, nil
	}

	before := strings.TrimRightFunc(acc[:jsonIdx], unicode.IsSpace)
	raw := strings.TrimLeftFunc(acc[jsonIdx:], unicode.IsSpace)
	if !lagunaLooksLikeJSONToolCall(raw, done) {
		return false, "", api.ToolCall{}, false, nil
	}

	if !done && !json.Valid([]byte(strings.TrimSpace(raw))) {
		if before != "" {
			p.buffer.Reset()
			p.buffer.WriteString(acc[jsonIdx:])
			return true, before, api.ToolCall{}, true, nil
		}
		return false, "", api.ToolCall{}, true, nil
	}

	call, err = parseLagunaToolCall(raw, p.tools)
	if err != nil {
		return false, "", api.ToolCall{}, true, err
	}
	call.Function.Index = p.callIndex
	p.callIndex++
	p.buffer.Reset()
	p.state = lagunaParserStateContent
	return true, before, call, true, nil
}

func lagunaLooksLikeJSONToolCall(raw string, done bool) bool {
	trimmed := strings.TrimLeftFunc(raw, unicode.IsSpace)
	if !strings.HasPrefix(trimmed, "{") {
		return false
	}
	if strings.Contains(trimmed, `"name"`) || strings.Contains(trimmed, `"arguments"`) {
		return true
	}
	if done {
		return false
	}
	return strings.HasPrefix(trimmed, `{"`) || strings.HasPrefix(trimmed, "{\n") || strings.HasPrefix(trimmed, "{\r\n")
}

func (p *LagunaParser) parseToolAlias(raw string) (api.ToolCall, bool) {
	raw = cleanLagunaToolCallRaw(raw)
	name, ok := lagunaToolCallName(raw)
	if !ok {
		return api.ToolCall{}, false
	}
	if _, ok := lagunaResolveToolName(name, p.tools); !ok {
		return api.ToolCall{}, false
	}
	call, err := parseLagunaToolCall(raw, p.tools)
	if err != nil {
		return api.ToolCall{}, false
	}
	call.Function.Index = p.callIndex
	p.callIndex++
	return call, true
}

func lagunaResolveToolName(name string, tools []api.Tool) (string, bool) {
	for i := range tools {
		if tools[i].Function.Name == name {
			return name, true
		}
	}

	aliases := map[string]string{
		"read_file":  "read",
		"write_file": "write",
		"edit_file":  "edit",
		"web_fetch":  "webfetch",
	}
	if alias, ok := aliases[name]; ok {
		for i := range tools {
			if tools[i].Function.Name == alias {
				return alias, true
			}
		}
	}
	return name, false
}

func cleanLagunaToolCallRaw(raw string) string {
	raw = strings.TrimSpace(raw)
	for strings.HasPrefix(raw, lagunaToolCallOpenTag) {
		raw = strings.TrimSpace(strings.TrimPrefix(raw, lagunaToolCallOpenTag))
	}
	if idx := strings.Index(raw, lagunaToolCallCloseTag); idx != -1 {
		raw = strings.TrimSpace(raw[:idx])
	}
	// A "<tool_call>" opener inside an <arg_value> block is part of the
	// argument's value, not the start of a new call. The call boundary is
	// the first opener that lies outside every <arg_value>…</arg_value>
	// span: a model may also open a second call before closing the first,
	// and that second opener (outside any arg payload) still ends the raw
	// body here.
	nestedIdx := -1
	inValue := false
	for i := 0; i < len(raw); {
		if inValue {
			if close := strings.Index(raw[i:], lagunaArgValueCloseTag); close != -1 {
				i += close + len(lagunaArgValueCloseTag)
				inValue = false
			} else {
				break
			}
			continue
		}
		openers := []string{lagunaToolCallOpenTag, lagunaArgValueOpenTag}
		best := -1
		bestTag := ""
		for _, tag := range openers {
			if idx := strings.Index(raw[i:], tag); idx != -1 && (best == -1 || idx < best) {
				best = idx
				bestTag = tag
			}
		}
		if best == -1 {
			break
		}
		if bestTag == lagunaToolCallOpenTag {
			nestedIdx = i + best
			break
		}
		// <arg_value> opened: skip to its own closing tag, but stop if the
		// payload itself ends the call without one (no closer at all).
		i += best + len(lagunaArgValueOpenTag)
		inValue = true
	}
	if nestedIdx != -1 {
		before := strings.TrimSpace(raw[:nestedIdx])
		if before != "" {
			return before
		}
		raw = strings.TrimSpace(raw[nestedIdx+len(lagunaToolCallOpenTag):])
	}
	return raw
}

func lagunaToolCallName(raw string) (string, bool) {
	raw = cleanLagunaToolCallRaw(raw)
	if strings.HasPrefix(raw, "{") {
		var parsed struct {
			Name string `json:"name"`
		}
		if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
			return "", false
		}
		name := strings.TrimSpace(parsed.Name)
		return name, name != ""
	}

	nameEnd := strings.Index(raw, "<arg_key>")
	if nameEnd < 0 {
		nameEnd = strings.Index(raw, "{")
	}
	if nameEnd < 0 {
		nameEnd = strings.IndexAny(raw, "\r\n")
	}
	if nameEnd < 0 {
		nameEnd = len(raw)
	}
	name := strings.TrimSpace(raw[:nameEnd])
	return name, name != ""
}

func (p *LagunaParser) consumeTool(done bool) (bool, api.ToolCall, error) {
	acc := p.buffer.String()
	if idx := strings.Index(acc, lagunaToolCallCloseTag); idx != -1 {
		raw := acc[:idx]
		after := strings.TrimLeftFunc(acc[idx+len(lagunaToolCallCloseTag):], unicode.IsSpace)
		p.buffer.Reset()
		p.buffer.WriteString(after)
		p.state = lagunaParserStateContent
		call, err := parseLagunaToolCall(raw, p.tools)
		if err != nil {
			return false, api.ToolCall{}, err
		}
		call.Function.Index = p.callIndex
		p.callIndex++
		return true, call, nil
	}
	if done && strings.TrimSpace(acc) != "" {
		p.buffer.Reset()
		p.state = lagunaParserStateContent
		call, err := parseLagunaToolCall(acc, p.tools)
		if err != nil {
			return false, api.ToolCall{}, err
		}
		call.Function.Index = p.callIndex
		p.callIndex++
		return true, call, nil
	}
	return false, api.ToolCall{}, nil
}

var lagunaArgRE = regexp.MustCompile(`(?s)<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>`)

// lagunaNestedArgValueRE matches one arg_key/arg_value pair whose value may
// itself contain "<tool_call>" openers, so a nested opener inside a value
// does not end the tool call early. Values with a nested opener are returned
// as raw strings via lagunaArgValue.
const lagunaArgValue = "\x00lagunaRawArgValue"

// lagunaSplitArgs finds each <arg_key>…</arg_key> <arg_value>…</arg_value>
// pair, scanning for the value's own closing tag instead of relying on a
// non-greedy regex that would stop at a nested "<tool_call>" opener.
func lagunaSplitArgs(s string) [][2]string {
	var pairs [][2]string
	for {
		keyStart := strings.Index(s, "<arg_key>")
		if keyStart < 0 {
			return pairs
		}
		keyRest := s[keyStart+len("<arg_key>"):]
		keyEnd := strings.Index(keyRest, "</arg_key>")
		if keyEnd < 0 {
			return pairs
		}
		key := keyRest[:keyEnd]
		afterKey := keyRest[keyEnd+len("</arg_key>"):]

		valueStart := strings.Index(afterKey, "<arg_value>")
		if valueStart < 0 {
			return pairs
		}
		afterValueOpen := afterKey[valueStart+len("<arg_value>"):]
		valueEnd := strings.Index(afterValueOpen, "</arg_value>")
		if valueEnd < 0 {
			return pairs
		}
		value := afterValueOpen[:valueEnd]
		pairs = append(pairs, [2]string{key, value})
		s = afterValueOpen[valueEnd+len("</arg_value>"):]
	}
}

func parseLagunaToolCall(raw string, tools []api.Tool) (api.ToolCall, error) {
	raw = cleanLagunaToolCallRaw(raw)
	if strings.HasPrefix(raw, "{") {
		var parsed struct {
			Name      string                        `json:"name"`
			Arguments api.ToolCallFunctionArguments `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
			return api.ToolCall{}, fmt.Errorf("failed to parse Laguna JSON tool call: %w", err)
		}
		if parsed.Name == "" {
			return api.ToolCall{}, fmt.Errorf("empty Laguna tool call name")
		}
		if name, ok := lagunaResolveToolName(parsed.Name, tools); ok {
			parsed.Name = name
		}
		return api.ToolCall{
			Function: api.ToolCallFunction{
				Name:      parsed.Name,
				Arguments: parsed.Arguments,
			},
		}, nil
	}

	nameEnd := strings.Index(raw, "<arg_key>")
	name := raw
	argsText := ""
	if nameEnd >= 0 {
		name = raw[:nameEnd]
		argsText = raw[nameEnd:]
	} else if jsonStart := strings.Index(raw, "{"); jsonStart >= 0 {
		name = raw[:jsonStart]
		argsText = raw[jsonStart:]
	}
	name = strings.TrimSpace(name)
	if resolved, ok := lagunaResolveToolName(name, tools); ok {
		name = resolved
	}

	var matchedTool *api.Tool
	for i := range tools {
		if tools[i].Function.Name == name {
			matchedTool = &tools[i]
			break
		}
	}

	call := api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      name,
			Arguments: api.NewToolCallFunctionArguments(),
		},
	}
	if strings.HasPrefix(strings.TrimSpace(argsText), "{") {
		if err := json.Unmarshal([]byte(strings.TrimSpace(argsText)), &call.Function.Arguments); err != nil {
			return api.ToolCall{}, fmt.Errorf("failed to parse Laguna JSON tool call arguments: %w", err)
		}
		return call, nil
	}
	for _, pair := range lagunaSplitArgs(argsText) {
		key := strings.TrimSpace(pair[0])
		value := pair[1]
		var paramType api.PropertyType
		if matchedTool != nil && matchedTool.Function.Parameters.Properties != nil {
			if prop, ok := matchedTool.Function.Parameters.Properties.Get(key); ok {
				if len(prop.AnyOf) > 0 {
					for _, anyOfProp := range prop.AnyOf {
						paramType = append(paramType, anyOfProp.Type...)
					}
				} else {
					paramType = prop.Type
				}
			}
		}
		call.Function.Arguments.Set(key, parseValue(value, paramType))
	}
	return call, nil
}
