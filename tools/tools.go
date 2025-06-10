package tools

import (
	"encoding/json"
	"strings"
	"text/template"

	"github.com/ollama/ollama/api"
)

type Parser struct {
	tag        string
	names      []string
	properties []string

	parsing bool
	buffer  string
	n       int
}

// NewParser creates a new tool call parser from a model's chat
// template and a list of provided tools.
func NewParser(tmpl *template.Template, tools []api.Tool) *Parser {
	tag := parseTag(tmpl)
	if tag == "" {
		tag = "{"
	}
	return NewParserWithTag(tools, tag)
}

func NewParserWithTag(tools []api.Tool, tag string) *Parser {
	var p Parser
	for _, t := range tools {
		p.names = append(p.names, t.Function.Name)
		for r := range t.Function.Parameters.Properties {
			p.properties = append(p.properties, r)
		}
	}
	p.tag = tag

	return &p
}

// Add processes a string input to parse tool calls and content.
// It handles prefix detection and JSON parsing to extract tool calls.
//
// Returns:
//   - tools: Any parsed tool calls
//   - content: Non-tool call content
func (p *Parser) Add(s string) (calls []api.ToolCall, content string) {
	p.buffer += s

	if !p.parsing {
		i := p.findTag()
		if i == -1 {
			p.parsing = false
			content = p.buffer
			p.buffer = ""
			return
		}

		content = p.buffer[:i]
		p.buffer = p.buffer[i:]

		if strings.Contains(p.buffer, p.tag) {
			p.parsing = true
		} else {
			return
		}
	}

	for {
		call, end := p.findToolCall()
		if call == nil {
			break
		}

		call.Function.Index = p.n
		p.n++
		calls = append(calls, *call)

		p.buffer = p.buffer[end:]
		if p.buffer == "" {
			break
		}
	}

	if len(calls) > 0 {
		return calls, content
	}

	// check if we should stop parsing and flush the content
	// e.g. tag is { or [ and a matching } or ] is found
	if p.shouldFlush() {
		content = p.buffer
		p.buffer = ""
		return nil, content
	}

	return nil, content
}

// findTag processes a string to find and handle a tag pattern
// returning true if the tag was found and false otherwise, and
// a string content signaling any content that should be sent back to the user
func (p *Parser) findTag() int {
	// First check for complete substring anywhere in s
	if i := strings.Index(p.buffer, p.tag); i > -1 {
		return i
	}

	// Then check for partial suffix overlap
	max := min(len(p.buffer), len(p.tag))
	for i := max; i > 0; i-- {
		if strings.HasSuffix(p.buffer, p.tag[:i]) {
			return len(p.buffer) - i
		}
	}
	return -1
}

// findToolCall finds the next complete tool call in the given string
// Returns the tool call, the number of characters consumed, and whether a tool call was found
func (p *Parser) findToolCall() (*api.ToolCall, int) {
	var name string
	var args map[string]any
	var end int = len(p.buffer)

	// find name
	var i int
	for _, n := range p.names {
		if i = strings.Index(p.buffer, n); i != -1 {
			if i+len(n) < end {
				name = n
				end = i + len(n)
			}
		}
	}

	if name == "" {
		return nil, -1
	}

	if args, i = p.findArguments(p.buffer); args == nil {
		return nil, -1
	}

	if i > end {
		end = i
	}

	tc := &api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      name,
			Arguments: args,
		},
	}

	return tc, end
}

// findArguments returns the first object that appears to be
// arguments and the position where the arguments end, returning nil and 0 if
// an invalid JSON object or non-arguments object is found first
func (p *Parser) findArguments(s string) (map[string]any, int) {
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return nil, 0
	}

	var braces int
	var start int = -1
	var end int
	var object string

	// find any outer json object
	for i, c := range s {
		if c == '{' {
			braces++
			if start == -1 {
				start = i
			}
		}

		if c == '}' {
			braces--
			if braces == 0 && start != -1 {
				end = i + 1
				object = s[start:end]
				break
			}
		}
	}

	if braces > 0 {
		return nil, 0
	}

	var data map[string]any

	// not valid json
	if err := json.Unmarshal([]byte(object), &data); err != nil {
		return nil, 0
	}

	var find func(obj any) map[string]any
	find = func(obj any) map[string]any {
		switch v := obj.(type) {
		case map[string]any:
			// check if the object keys are valid tool properties
			// TODO (jmorganca): check only sets of properties that
			// go together instead of the entire set
			for _, prop := range p.properties {
				if _, exists := v[prop]; exists {
					return v
				}
			}

			for _, value := range v {
				if result := find(value); result != nil {
					return result
				}
			}
		case []any:
			for _, item := range v {
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

// Content returns any remaining content that should be sent
// back to the user. If tools were called, an empty string
// is returned. Otherwise, content up until the tag is returned
// unless the tag is { or [ in which case the entire buffer is returned
func (p *Parser) Content() string {
	if p.n > 0 {
		return ""
	}

	i := strings.Index(p.buffer, p.tag)
	if i > 0 {
		if p.tag == "{" || p.tag == "[" {
			return p.buffer
		}
		return p.buffer[:i]
	}
	return p.buffer
}

// shouldFlush checks if the parser should stop parsing and flush the content
// e.g. tag is { and if matching } is found or tag is [ and if matching ] is found
func (p *Parser) shouldFlush() bool {
	if p.tag == "" || p.buffer == "" {
		return false
	}

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
		if c == open {
			count++
		} else if c == close {
			count--
			if count == 0 {
				return true
			}
		}
	}
	return false
}
