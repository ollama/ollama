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
}

func (p *LagunaParser) HasToolSupport() bool {
	return true
}

func (p *LagunaParser) HasThinkingSupport() bool {
	return true
}

func (p *LagunaParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.callIndex = 0
	p.buffer.Reset()
	p.thinkingEnabled = thinkValue == nil || thinkValue.Bool()
	p.thinkingSuppressed = thinkValue != nil && !thinkValue.Bool()
	p.state = lagunaParserStateContent
	p.allowLeadingThinkOpen = false
	return tools
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
			p.buffer.Reset()
			p.buffer.WriteString(strings.TrimLeftFunc(strings.TrimPrefix(trimmed, lagunaThinkingOpenTag), unicode.IsSpace))
			p.allowLeadingThinkOpen = false
			return true, ""
		}
		if strings.HasPrefix(lagunaThinkingOpenTag, trimmed) && !done {
			return false, ""
		}
		p.allowLeadingThinkOpen = false
	}

	if idx := strings.Index(acc, lagunaThinkingCloseTag); idx != -1 {
		thinking := acc[:idx]
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
	if idx := strings.Index(raw, lagunaToolCallOpenTag); idx != -1 {
		before := strings.TrimSpace(raw[:idx])
		if before != "" {
			return before
		}
		raw = strings.TrimSpace(raw[idx+len(lagunaToolCallOpenTag):])
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
	for _, match := range lagunaArgRE.FindAllStringSubmatch(argsText, -1) {
		key := strings.TrimSpace(match[1])
		value := match[2]
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
