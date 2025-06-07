package tools

import (
	"encoding/json"
	"errors"
	"log/slog"
	"slices"
	"strings"
	gotmpl "text/template"
	"text/template/parse"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

var (
	errInvalidToolCall = errors.New("invalid tool call format")
	errAccumulateMore  = errors.New("need to accumulate more content")
)

type Parser struct {
	// prefix is an optional string, often the opening tag to signal
	// future content should be parsed as tool calls
	prefix string

	// tools is a list of tool names that are available to the model
	tools []string

	prefixFound    bool
	nameFound      bool
	argumentsFound bool

	sb    strings.Builder
	index int
}

// NewParser creates a new tool call parser from a model's chat
// template and a list of provided tools.
func NewParser(tmpl *template.Template, tools []api.Tool) (*Parser, error) {
	var p Parser
	for _, tool := range tools {
		p.tools = append(p.tools, tool.Function.Name)
	}

	p.prefix = parsePrefix(tmpl.Template)
	p.sb = strings.Builder{}

	return &p, nil
}

// Add processes a string input to parse tool calls and content.
// It handles prefix detection and JSON parsing to extract tool calls.
//
// Returns:
//   - tools: Any parsed tool calls
//   - content: Non-tool call content
func (p *Parser) Add(s string) (tools []api.ToolCall, content string) {
	p.sb.WriteString(s)
	s = p.sb.String()

	content, ok := p.scanPrefix(s)
	if !ok {
		return nil, content
	}

	// TODO (jmorganca): attempt to parse tool name if not yet found
	// TODO (jmorganca): attempt to parse tool arguments if not yet found

	// toolCalls, err := parseJSONToolCalls(s, p.name, p.arguments, p.prefix)
	// if err != nil {
	// 	if errors.Is(err, errAccumulateMore) {
	// 		return nil, ""
	// 	}
	// 	p.sb.Reset()
	// 	// Only do greedy JSON parsing if there is no prefix from template
	// 	if p.prefix != "" {
	// 		p.greedyParseJSON = false
	// 	}
	// 	if p.index != 0 && p.prefix == "" {
	// 		return nil, ""
	// 	}
	// 	if p.prefixFound {
	// 		// Drop tokens since prefix was found
	// 		return nil, ""
	// 	}
	// 	return nil, s
	// }

	var toolCalls []api.ToolCall
	for _, tc := range toolCalls {
		tc.Function.Index = p.index
		p.index++
	}

	p.sb.Reset()
	return toolCalls, content
}

// parseJSONToolCalls attempts to parse a JSON string into a slice of ToolCalls.
//
// Parameters:
//   - s: The string to parse
//   - name: The field name from template that identifies the tool call name
//   - arguments: The field name from template that identifies the tool call arguments
//
// Returns:
//   - []api.ToolCall: The parsed tool calls if successful
//   - error: ErrAccumulateMore if braces unbalanced, ErrInvalidToolCall if invalid, or nil if successful
func parseJSONToolCalls(s string, name, arguments string, prefix string) ([]api.ToolCall, error) {
	// Check for balanced braces before attempting to parse
	braceCount := 0
	squareCount := 0
	startIndex := -1
	var rawToolCalls []string
	s = strings.TrimSpace(s)

	// Only track these if we don't have a prefix as it will be cut off from the prefix. Also track in the parseLeadingJSON case.
	trackSquareBrackets := prefix == "" || !strings.HasSuffix(prefix, "[") || strings.HasPrefix(s, "[")
	for i, c := range s {
		switch c {
		case '{':
			braceCount++
			if startIndex == -1 {
				startIndex = i
			}
		case '}':
			braceCount--
			if braceCount == 0 {
				rawToolCalls = append(rawToolCalls, s[startIndex:i+1])
				startIndex = -1
			}
		case '[':
			if trackSquareBrackets {
				squareCount++
			}
		case ']':
			if trackSquareBrackets {
				squareCount--
			}
		}

		// Negative means we have an extra closing brace/bracket
		if braceCount < 0 || squareCount < 0 {
			return nil, errInvalidToolCall
		}
	}

	// If braces/brackets aren't balanced, need more input
	if braceCount > 0 || squareCount > 0 {
		return nil, errAccumulateMore
	}

	t := strings.TrimSpace(s)
	if len(t) == 0 {
		return nil, errAccumulateMore
	}
	// If the input is a single square bracket, it's not a valid tool call
	if t[0] == '[' && len(t) == 1 {
		return nil, errAccumulateMore
	}

	// Attempt full unmarshal of the JSON
	var toolCalls []api.ToolCall
	for _, rawToolCall := range rawToolCalls {
		var resp map[string]any
		if err := json.Unmarshal([]byte(rawToolCall), &resp); err != nil {
			continue
		}

		// Collect nested objects that could contain tool calls
		objs := collect(resp)
		if len(objs) == 0 {
			continue
		}

		// Extract tool calls from objects
		for _, kv := range objs {
			n, nok := kv[name].(string)
			a, aok := kv[arguments].(map[string]any)
			if nok && aok {
				toolCalls = append(toolCalls, api.ToolCall{
					Function: api.ToolCallFunction{
						Name:      n,
						Arguments: a,
					},
				})
			} else {
				slog.Debug("No valid tool call found in object.", "object", kv)
			}
		}
	}

	// Valid JSON, no tool calls found
	if len(toolCalls) == 0 {
		slog.Debug("No valid tool calls found in any raw tool calls.", "rawToolCalls", rawToolCalls)
		return nil, errInvalidToolCall
	}

	return toolCalls, nil
}

// scanPrefix processes a string to find and handle a prefix pattern
// returning true if the prefix was found and false otherwise, and
// a string content signaling any content that should be sent back to the user
func (p *Parser) scanPrefix(s string) (string, bool) {
	if s == "" {
		return "", false
	}

	if p.prefix == "" {
		// if the prefix is empty, we need to check for
		// 1. {
		// 2. tool call names such as get_weather, add, etc.
		return s, true
	}

	// Check for prefix at start of string
	if cut, ok := strings.CutPrefix(s, p.prefix); ok {
		return cut, true
	}

	// Check if prefix overlaps end of string
	if idx := suffixOverlap(s, p.prefix); idx != -1 {
		// Return everything except overlapping portion
		p.sb.Reset()
		p.sb.WriteString(s[idx:])
		return s[:idx], false
	}

	// Check if prefix appears in middle of string
	if idx := strings.Index(s, p.prefix); idx != -1 {
		// Save remainder starting at prefix for next pass
		p.sb.Reset()
		p.sb.WriteString(strings.TrimSpace(s[idx:]))
		// Return everything before prefix
		return s[:idx], false
	}

	// No partial prefix found
	return "", false
}

// parsePrefix finds the prefix text value in a Go template
// often <tool_call> [TOOL_CALL] or similar by finding the
// first text node after .ToolCalls and returning the content
// before [, { and (TODO: or the name of a tool call)
// parsePrefix finds the prefix text value in a Go template
// often <tool_call> [TOOL_CALL] or similar by finding the
// first text node after .ToolCalls and returning the content
// before [, { and (TODO: or the name of a tool call)
func parsePrefix(tmpl *gotmpl.Template) string {
	if tmpl == nil || tmpl.Tree == nil {
		slog.Debug("template or tree is nil")
		return ""
	}

	tc := findToolCallNode(tmpl.Tree.Root.Nodes)
	if tc == nil {
		return ""
	}

	text := findText(tc.List.Nodes)
	text = strings.TrimSpace(text)
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.Split(text, "{")[0]
	return text
}

// findToolCalls searches for an IfNode with .ToolCalls and returns its contents
func findToolCallNode(nodes []parse.Node) *parse.IfNode {
	isToolCallsNode := func(n *parse.IfNode) bool {
		for _, cmd := range n.Pipe.Cmds {
			for _, arg := range cmd.Args {
				if field, ok := arg.(*parse.FieldNode); ok {
					if slices.Contains(field.Ident, "ToolCalls") {
						return true
					}
				}
			}
		}
		return false
	}

	for _, node := range nodes {
		switch n := node.(type) {
		case *parse.IfNode:
			if isToolCallsNode(n) {
				return n
			}
			// Recursively search in nested IfNodes
			if result := findToolCallNode(n.List.Nodes); result != nil {
				return result
			}
			if n.ElseList != nil {
				if result := findToolCallNode(n.ElseList.Nodes); result != nil {
					return result
				}
			}
		case *parse.ListNode:
			if result := findToolCallNode(n.Nodes); result != nil {
				return result
			}
		case *parse.RangeNode:
			if result := findToolCallNode(n.List.Nodes); result != nil {
				return result
			}
			if n.ElseList != nil {
				if result := findToolCallNode(n.ElseList.Nodes); result != nil {
					return result
				}
			}
		case *parse.WithNode:
			if result := findToolCallNode(n.List.Nodes); result != nil {
				return result
			}
			if n.ElseList != nil {
				if result := findToolCallNode(n.ElseList.Nodes); result != nil {
					return result
				}
			}
		}
	}
	return nil
}

// findText does a depth-first search for the first text content in nodes
func findText(nodes []parse.Node) string {
	for _, node := range nodes {
		switch n := node.(type) {
		case *parse.TextNode:
			return string(n.Text)
		case *parse.IfNode:
			if text := findText(n.List.Nodes); text != "" {
				return text
			}
			if n.ElseList != nil {
				if text := findText(n.ElseList.Nodes); text != "" {
					return text
				}
			}
		case *parse.ListNode:
			if text := findText(n.Nodes); text != "" {
				return text
			}
		case *parse.RangeNode:
			if text := findText(n.List.Nodes); text != "" {
				return text
			}
			if n.ElseList != nil {
				if text := findText(n.ElseList.Nodes); text != "" {
					return text
				}
			}
		case *parse.WithNode:
			if text := findText(n.List.Nodes); text != "" {
				return text
			}
			if n.ElseList != nil {
				if text := findText(n.ElseList.Nodes); text != "" {
					return text
				}
			}
		}
	}
	return ""
}

// suffixOverlap returns the index in s where the longest suffix overlap with prefix begins
//
// Returns:
//   - int: The starting index in s where the suffix overlap begins
func suffixOverlap(s, prefix string) int {
	max := min(len(prefix), len(s))
	for i := max; i > 0; i-- {
		if strings.HasSuffix(s, prefix[:i]) {
			return len(s) - i
		}
	}
	return -1
}

// collect recursively traverses an object to collect all nested maps
//
// Returns:
//   - []map[string]any: A slice of all nested maps found in the object
func collect(obj any) []map[string]any {
	var all []map[string]any
	switch o := obj.(type) {
	case map[string]any:
		all = append(all, o)
		for _, v := range o {
			all = append(all, collect(v)...)
		}
	case []any:
		for _, v := range o {
			all = append(all, collect(v)...)
		}
	default:
		return nil
	}

	return all
}
