package parsers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

type rippleState int

const (
	rippleStart rippleState = iota
	rippleThinkLead
	rippleThinking
	rippleThinkEnd
	rippleContent
	rippleTool
)

const (
	rippleThinkOpen  = "<think>"
	rippleThinkClose = "</think>"
	rippleToolOpen   = "<tool_call>"
	rippleToolClose  = "</tool_call>"
)

type RippleParser struct {
	state      rippleState
	buffer     string
	callIndex  int
	toolScan   int
	toolString bool
	toolEscape bool
}

func (p *RippleParser) Init(tools []api.Tool, _ *api.Message, _ *api.ThinkValue) []api.Tool {
	*p = RippleParser{}
	return tools
}

func (p *RippleParser) HasToolSupport() bool     { return true }
func (p *RippleParser) HasThinkingSupport() bool { return true }

func (p *RippleParser) ThinkingClose() []string {
	if p.state == rippleThinkLead || p.state == rippleThinking {
		return []string{rippleThinkClose}
	}
	return nil
}

func (p *RippleParser) PreservedTokens() []string {
	return []string{rippleThinkOpen, rippleThinkClose, rippleToolOpen, rippleToolClose}
}

func (p *RippleParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer += s
parse:
	for p.buffer != "" {
		switch p.state {
		case rippleStart:
			if strings.HasPrefix(p.buffer, rippleThinkOpen) {
				p.buffer = p.buffer[len(rippleThinkOpen):]
				p.state = rippleThinkLead
			} else if !done && strings.HasPrefix(rippleThinkOpen, p.buffer) {
				return content, thinking, calls, nil
			} else {
				p.state = rippleContent
			}
		case rippleThinkLead:
			p.buffer = strings.TrimPrefix(p.buffer, "\n")
			p.state = rippleThinking
		case rippleThinking:
			if i := strings.Index(p.buffer, rippleThinkClose); i >= 0 {
				thinking += strings.TrimSuffix(p.buffer[:i], "\n")
				p.buffer = p.buffer[i+len(rippleThinkClose):]
				p.state = rippleThinkEnd
			} else {
				n := len(p.buffer)
				if !done {
					// The newline before a closing tag belongs to the template.
					n -= max(overlap(p.buffer, rippleThinkClose), overlap(p.buffer, "\n"+rippleThinkClose))
				}
				thinking += p.buffer[:n]
				p.buffer = p.buffer[n:]
				return content, thinking, calls, nil
			}
		case rippleThinkEnd:
			p.buffer = strings.TrimPrefix(p.buffer, "\n")
			p.state = rippleContent
		case rippleContent:
			if i := strings.Index(p.buffer, rippleToolOpen); i >= 0 {
				if p.callIndex == 0 {
					content += strings.TrimSuffix(p.buffer[:i], "\n")
				}
				p.buffer = p.buffer[i+len(rippleToolOpen):]
				p.state = rippleTool
				p.toolScan, p.toolString, p.toolEscape = 0, false, false
			} else {
				n := len(p.buffer)
				if !done {
					n -= max(overlap(p.buffer, rippleToolOpen), overlap(p.buffer, "\n"+rippleToolOpen))
				}
				if p.callIndex == 0 {
					content += p.buffer[:n]
				}
				p.buffer = p.buffer[n:]
				return content, thinking, calls, nil
			}
		case rippleTool:
			if i := p.toolEnd(); i >= 0 {
				call, parseErr := parseRippleTool(p.buffer[:i])
				p.buffer = p.buffer[i+len(rippleToolClose):]
				p.state = rippleContent
				if parseErr != nil {
					return content, thinking, calls, parseErr
				}
				call.Function.Index = p.callIndex
				p.callIndex++
				calls = append(calls, call)
			} else if !done {
				return content, thinking, calls, nil
			} else {
				break parse
			}
		}
	}
	if done && p.state == rippleTool {
		p.buffer = ""
		p.state = rippleContent
		return content, thinking, calls, fmt.Errorf("ripple: unterminated tool call")
	}
	return content, thinking, calls, nil
}

// toolEnd ignores closing tags inside JSON strings, including escaped quotes.
func (p *RippleParser) toolEnd() int {
	for p.toolScan < len(p.buffer) {
		i := p.toolScan
		c := p.buffer[i]
		if p.toolString {
			if p.toolEscape {
				p.toolEscape = false
			} else if c == '\\' {
				p.toolEscape = true
			} else if c == '"' {
				p.toolString = false
			}
		} else if c == '"' {
			p.toolString = true
		} else if c == '<' {
			if strings.HasPrefix(p.buffer[i:], rippleToolClose) {
				return i
			}
			if strings.HasPrefix(rippleToolClose, p.buffer[i:]) {
				return -1
			}
		}
		p.toolScan++
	}
	return -1
}

func parseRippleTool(raw string) (api.ToolCall, error) {
	var value struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(raw), &value); err != nil {
		return api.ToolCall{}, fmt.Errorf("ripple: invalid tool call JSON: %w", err)
	}
	if value.Name == "" {
		return api.ToolCall{}, fmt.Errorf("ripple: empty function name")
	}
	if !strings.HasPrefix(strings.TrimSpace(string(value.Arguments)), "{") {
		return api.ToolCall{}, fmt.Errorf("ripple: tool arguments must be a JSON object")
	}
	var arguments api.ToolCallFunctionArguments
	if err := json.Unmarshal(value.Arguments, &arguments); err != nil {
		return api.ToolCall{}, fmt.Errorf("ripple: invalid tool arguments: %w", err)
	}
	return api.ToolCall{Function: api.ToolCallFunction{Name: value.Name, Arguments: arguments}}, nil
}
