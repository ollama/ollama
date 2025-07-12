package tools

import (
	"bytes"
	"encoding/json"
	"strings"
	"text/template"

	"github.com/ollama/ollama/api"
)

type toolsState int

const (
	toolsState_LookingForTag toolsState = iota
	toolsState_ToolCalling
	toolsState_Done
)

type Parser struct {
	tag   string
	tools []api.Tool

	state  toolsState
	buffer []byte
	n      int
}

// NewParser creates a new tool call parser from a model's chat
// template and a list of provided tools.
func NewParser(tmpl *template.Template, tools []api.Tool) *Parser {
	return NewParserWithTag(tools, parseTag(tmpl))
}

func NewParserWithTag(tools []api.Tool, tag string) *Parser {
	return &Parser{
		tag:   tag,
		tools: tools,
	}
}

// Add processes a string input to parse tool calls and content that
// should be sent back to the user.
func (p *Parser) Add(s string) (calls []api.ToolCall, content string) {
	if p.state == toolsState_Done {
		return nil, s
	}

	p.buffer = append(p.buffer, s...)

	if p.state == toolsState_LookingForTag {
		i, found := p.findTag()
		if i == -1 {
			content = string(p.buffer)
			p.buffer = []byte{}
		} else {
			content = string(p.buffer[:i])
			p.buffer = p.buffer[i:]
		}

		// for models where { or [ are used as tool calling
		// tags, we only support parsing tools if the first non-
		// whitespace character is { or [
		if p.tag == "{" || p.tag == "[" {
			if strings.TrimSpace(content) != "" {
				p.state = toolsState_Done
				return nil, content + string(p.buffer)
			}
		}

		if !found {
			return nil, content
		}

		p.state = toolsState_ToolCalling
	}

	for {
		call := p.parseToolCall()
		if call == nil {
			break
		}

		calls = append(calls, *call)
	}

	if p.done() {
		p.state = toolsState_Done
		content = string(p.buffer)
		p.buffer = []byte{}
	}

	return calls, content
}

// findTag searches the buffer to find and handle a tool calling tag
// returning true if the tag was found and false otherwise, and
// a string content signaling any content that should be sent back to the user
func (p *Parser) findTag() (int, bool) {
	// First check for complete substring anywhere in s
	if i := bytes.Index(p.buffer, []byte(p.tag)); i > -1 {
		return i, true
	}

	// Then check for partial suffix overlap
	max := min(len(p.buffer), len(p.tag))
	for i := max; i > 0; i-- {
		if bytes.HasSuffix(p.buffer, []byte(p.tag[:i])) {
			return len(p.buffer) - i, false
		}
	}
	return -1, false
}

// parseToolCall finds the next complete tool call in the buffer
// incrementing n and advancing the buffer.
func (p *Parser) parseToolCall() *api.ToolCall {
	var tool *api.Tool
	var end int = len(p.buffer)
	var i int

	// find the earliest tool name
	start := end
	for _, t := range p.tools {
		n := t.Function.Name
		if i = bytes.Index(p.buffer, []byte(n)); i != -1 {
			if start > i {
				start = i
			}
		}
	}
	// find the longest tool name
	longest := 0
	for _, t := range p.tools {
		n := t.Function.Name
		if bytes.Index(p.buffer[start:], []byte(n)) == 0 {
			if longest < len(n) {
				longest = len(n)
				tool = &t
				end = start + len(n)
			}
		}
	}

	if tool == nil {
		return nil
	}

	// only look for arguments after the tool name if the tool has parameters
	// TODO (jmorganca): while probably uncommon, this doesn't support
	// parsing arguments before the tool name, which may be needed in the future
	args := map[string]any{}
	if len(tool.Function.Parameters.Properties) > 0 {
		if args, i = findArguments(*tool, p.buffer[end:]); args == nil {
			return nil
		}

		end += i
	}

	tc := &api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      tool.Function.Name,
			Arguments: args,
			Index:     p.n,
		},
	}

	p.n++
	p.buffer = p.buffer[end:]
	return tc
}

// findArguments returns the first object that appears to be
// arguments for the provided tool in the provided buffer,
// returning nil if no arguments are found.
// TODO (jmorganca): this does not support parsing omitted arguments
// objects for functions that have all-optional parameters
// e.g. `{"name": "get_conditions", "arguments": {}}` will work but
// `{"name": "get_conditions"}` will not currently work
func findArguments(tool api.Tool, buffer []byte) (map[string]any, int) {
	if len(buffer) == 0 {
		return nil, 0
	}

	var braces int
	var start int = -1
	var end int
	var object []byte

	// find any outer json object
	for i, c := range buffer {
		if c == '{' {
			braces++
			if start == -1 {
				start = i
			}
		}

		if c == '}' {
			if start != -1 {
				braces--
				if braces == 0 {
					end = i + 1
					object = buffer[start:end]
					break
				}
			}
		}
	}

	if braces > 0 {
		return nil, 0
	}

	var data map[string]any
	if err := json.Unmarshal(object, &data); err != nil {
		return nil, 0
	}

	var find func(obj any) map[string]any
	find = func(obj any) map[string]any {
		switch obj := obj.(type) {
		case map[string]any:
			valid := true
			// check if all keys in the object exist in the tool's parameters
			for key := range obj {
				if _, exists := tool.Function.Parameters.Properties[key]; !exists {
					valid = false
					break
				}
			}

			// check for required parameters
			// TODO (jmorganca): this should error instead of silently failing
			if valid {
				for _, required := range tool.Function.Parameters.Required {
					if _, exists := obj[required]; !exists {
						valid = false
						break
					}
				}
			}

			if valid {
				return obj
			}

			for _, value := range obj {
				if result := find(value); result != nil {
					return result
				}
			}
		case []any:
			for _, item := range obj {
				if result := find(item); result != nil {
					return result
				}
			}
		}

		return nil
	}

	result := find(data)
	if result != nil {
		return result, end
	}

	return nil, 0
}

// done checks if the parser is done parsing by looking
// for closing tag. currently only } and ] are supported
// for closing tags as {} or [] pairs may not always
// represent tool calls and we need to send the content back
func (p *Parser) done() bool {
	var open, close rune
	switch p.tag {
	case "{":
		open, close = '{', '}'
	case "[":
		open, close = '[', ']'
	default:
		return false
	}

	var count int
	for _, c := range p.buffer {
		if c == byte(open) {
			count++
		} else if c == byte(close) {
			count--
			if count == 0 {
				return true
			}
		}
	}

	return false
}

// Content returns any remaining content that
// should be sent to the user. This should be the empty string
// string unless the tag is { or [ and a tool call was not found
func (p *Parser) Content() string {
	if p.n > 0 {
		return ""
	}

	if p.tag == "{" || p.tag == "[" {
		return string(p.buffer)
	}

	return ""
}
