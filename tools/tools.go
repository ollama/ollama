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

func (p *Parser) GetBuffer() []byte {
	return p.buffer
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
	tool, end := findTool(p.tools, p.buffer)
	if tool == nil {
		return nil
	}

	var argsMap map[string]any
	if found, i := findArguments(tool, p.buffer); found == nil {
		return nil
	} else {
		argsMap = found
		if i > end {
			end = i
		}
	}

	args := api.NewToolCallFunctionArguments()
	for k, v := range argsMap {
		args.Set(k, v)
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

// findTool finds the first tool name in the list that matches the
// beginning of the buffer, returning nil if no tool is found
// or if the buffer ends with a partial tool name since we need
// to wait for more data to disambiguate.
// The second return value is the end position of the tool name
// if one is found, otherwise 0.
func findTool(tools []api.Tool, buf []byte) (*api.Tool, int) {
	if len(buf) == 0 {
		return nil, 0
	}

	// check if buffer ends with a partial tool name
	// this prevents matching "get" when seeing "get_weather"
	var longest string
	for _, t := range tools {
		if len(t.Function.Name) > len(longest) {
			longest = t.Function.Name
		}
	}

	// Only check up to longest characters from the end
	for i := 1; i <= min(len(buf), len(longest)); i++ {
		tail := buf[len(buf)-i:]
		for _, t := range tools {
			name := []byte(t.Function.Name)
			if len(tail) < len(name) && bytes.HasPrefix(name, tail) {
				return nil, 0
			}
		}
	}

	// find first occurrence of the longest tool name
	var found *api.Tool
	start := -1
	end := -1

	for i := range tools {
		name := []byte(tools[i].Function.Name)
		pos := bytes.Index(buf, name)
		if pos == -1 {
			continue
		}

		// Skip if we have a better match already
		if start != -1 {
			if pos > start {
				continue
			}
			if pos == start && len(name) <= len(found.Function.Name) {
				continue
			}
		}

		found = &tools[i]
		start = pos
		end = pos + len(name)
	}

	if found != nil {
		return found, end
	}

	return nil, 0
}

// findArguments returns the first object that appears to be
// arguments for the provided tool in the provided buffer,
// returning nil if no arguments are found and the end position
// TODO (jmorganca): this does not support parsing omitted arguments
// objects for functions that have all-optional parameters
// e.g. `{"name": "get_conditions", "arguments": {}}` will work but
// `{"name": "get_conditions"}` will not currently work
func findArguments(tool *api.Tool, buffer []byte) (map[string]any, int) {
	if len(buffer) == 0 {
		return nil, 0
	}

	start := -1
	var braces int
	var inString, escaped bool

	for i := range buffer {
		c := buffer[i]

		if escaped {
			escaped = false
			continue
		}

		if c == '\\' {
			escaped = true
			continue
		}

		if c == '"' {
			inString = !inString
			continue
		}

		if inString {
			continue
		}

		if c == '{' {
			if braces == 0 {
				start = i
			}
			braces++
		} else if c == '}' {
			braces--
			if braces == 0 && start != -1 {
				object := buffer[start : i+1]

				var data map[string]any
				if err := json.Unmarshal(object, &data); err != nil {
					// not a valid object, keep looking
					start = -1
					continue
				}

				var findObject func(obj map[string]any) (map[string]any, bool)
				findObject = func(obj map[string]any) (map[string]any, bool) {
					findMap := func(name string, obj map[string]any) (map[string]any, bool) {
						if args, ok := obj[name].(map[string]any); ok {
							return args, true
						}
						if argsStr, ok := obj[name].(string); ok {
							var argsData map[string]interface{}
							if err := json.Unmarshal([]byte(argsStr), &argsData); err == nil {
								return argsData, ok
							}
						}
						return nil, false
					}
					if _, hasName := obj["name"]; hasName {
						if args, ok := findMap("arguments", obj); ok {
							return args, true
						}
						if args, ok := findMap("parameters", obj); ok {
							return args, true
						}
						return nil, true
					}
					if args, ok := findMap(tool.Function.Name, obj); ok {
						return args, true
					}

					for _, v := range obj {
						switch child := v.(type) {
						case map[string]any:
							if result, found := findObject(child); found {
								return result, true
							}
						case []any:
							for _, item := range child {
								if childObj, ok := item.(map[string]any); ok {
									if result, found := findObject(childObj); found {
										return result, true
									}
								}
							}
						}
					}

					return nil, false
				}

				if args, found := findObject(data); found {
					return args, i
				}

				return data, i
			}

			if braces < 0 {
				braces = 0
			}
		}
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
