package parsers

import (
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
)

const (
	hy4ThinkStart    = "<think:opensource>"
	hy4ThinkEnd      = "</think:opensource>"
	hy4CallsStart    = "<tool_calls:opensource>"
	hy4CallsEnd      = "</tool_calls:opensource>"
	hy4CallStart     = "<tool_call:opensource>"
	hy4CallEnd       = "</tool_call:opensource>"
	hy4ArgKeyStart   = "<arg_key:opensource>"
	hy4ArgKeyEnd     = "</arg_key:opensource>"
	hy4ArgValueStart = "<arg_value:opensource>"
	hy4ArgValueEnd   = "</arg_value:opensource>"
)

func hy4ThinkingEnabled(think *api.ThinkValue) bool {
	if think != nil {
		switch think.Value {
		case false, "no_think":
			return false
		}
	}
	return true
}

func init() {
	Register("hy4-flash", func() Parser { return &hy4FlashParser{} })
}

type hy4FlashParser struct {
	pending         string
	thinking        bool
	atThinkingStart bool
	tools           []api.Tool
	callIndex       int
}

func (*hy4FlashParser) HasToolSupport() bool     { return true }
func (*hy4FlashParser) HasThinkingSupport() bool { return true }

func (p *hy4FlashParser) Init(tools []api.Tool, last *api.Message, think *api.ThinkValue) []api.Tool {
	thinking := hy4ThinkingEnabled(think) && (last == nil || last.Role != "assistant")
	*p = hy4FlashParser{tools: tools, thinking: thinking, atThinkingStart: thinking}
	return tools
}

func (p *hy4FlashParser) ThinkingClose() []string {
	if p.thinking {
		return []string{hy4ThinkEnd}
	}
	return nil
}

func (*hy4FlashParser) PreservedTokens() []string {
	return []string{
		hy4ThinkStart, hy4ThinkEnd, hy4CallsStart, hy4CallsEnd,
		hy4CallStart, hy4CallEnd, hy4ArgKeyStart, hy4ArgKeyEnd, hy4ArgValueStart, hy4ArgValueEnd,
	}
}

func (p *hy4FlashParser) Add(chunk string, done bool) (content, thinking string, calls []api.ToolCall, err error) {
	p.pending += chunk
	if p.thinking {
		if p.atThinkingStart {
			if !done && strings.HasPrefix(hy4ThinkStart, p.pending) {
				return
			}
			p.pending = strings.TrimPrefix(p.pending, hy4ThinkStart)
			p.atThinkingStart = false
		}
		if before, after, ok := strings.Cut(p.pending, hy4ThinkEnd); ok {
			thinking, p.pending, p.thinking = before, after, false
		} else {
			thinking, p.pending = hy4StreamPart(p.pending, done, hy4ThinkEnd)
			return
		}
	}
	var output strings.Builder
	for p.pending != "" {
		before, after, ok := strings.Cut(p.pending, hy4CallsStart)
		if !ok {
			var part string
			part, p.pending = hy4StreamPart(p.pending, done, hy4CallsStart)
			output.WriteString(part)
			break
		}
		output.WriteString(before)
		block, tail, complete := strings.Cut(after, hy4CallsEnd)
		if !complete {
			p.pending = hy4CallsStart + after
			if done {
				output.WriteString(p.pending)
				p.pending = ""
			}
			break
		}
		parsed, parseErr := p.parseCalls(block)
		if parseErr != nil {
			return output.String(), thinking, calls, parseErr
		}
		calls = append(calls, parsed...)
		p.pending = tail
	}
	return output.String(), thinking, calls, nil
}

func hy4StreamPart(text string, done bool, tag string) (string, string) {
	if done {
		return text, ""
	}
	end := len(text) - overlap(text, tag)
	return text[:end], text[end:]
}

func (p *hy4FlashParser) parseCalls(block string) ([]api.ToolCall, error) {
	var parsed []api.ToolCall
	for block = strings.TrimSpace(block); block != ""; block = strings.TrimSpace(block) {
		raw, ok := strings.CutPrefix(block, hy4CallStart)
		if !ok {
			return nil, fmt.Errorf("Hy4 tool block is missing a call opener")
		}
		call, rest, ok := strings.Cut(raw, hy4CallEnd)
		if !ok {
			return nil, fmt.Errorf("Hy4 tool call is missing its closing tag")
		}
		block = rest
		name, _, _ := strings.Cut(call, hy4ArgKeyStart)
		name = strings.TrimSpace(name)
		if name == "" {
			return nil, fmt.Errorf("Hy4 tool call is missing its name")
		}
		idx := slices.IndexFunc(p.tools, func(t api.Tool) bool { return t.Function.Name == name })
		var tool *api.Tool
		if idx >= 0 {
			tool = &p.tools[idx]
		}
		args := api.NewToolCallFunctionArguments()
		for {
			_, rest, ok := strings.Cut(call, hy4ArgKeyStart)
			if !ok {
				break
			}
			key, rest, ok := strings.Cut(rest, hy4ArgKeyEnd)
			if !ok {
				return nil, fmt.Errorf("Hy4 tool call %q has an unterminated argument key", name)
			}
			valueText, ok := strings.CutPrefix(strings.TrimLeft(rest, " \n\r\t"), hy4ArgValueStart)
			if !ok {
				return nil, fmt.Errorf("Hy4 tool call %q is missing the value for %q", name, key)
			}
			value, rest, ok := strings.Cut(valueText, hy4ArgValueEnd)
			if !ok {
				return nil, fmt.Errorf("Hy4 tool call %q has an unterminated value for %q", name, key)
			}
			args.Set(key, parseValue(value, toolArgumentTypes(tool, key)))
			call = rest
		}
		parsed = append(parsed, api.ToolCall{Function: api.ToolCallFunction{Index: p.callIndex + len(parsed), Name: name, Arguments: args}})
	}
	p.callIndex += len(parsed)
	return parsed, nil
}
