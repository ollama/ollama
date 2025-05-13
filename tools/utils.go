package tools

import (
	"bytes"
	"errors"
	"log/slog"
	"slices"
	"strings"
	gotmpl "text/template"
	"text/template/parse"

	jsonv2 "github.com/go-json-experiment/json"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

// extractToolCallsFormat traverses a template AST to find text that follows a ".ToolCalls" condition.
// It walks the template nodes looking for if-statements containing ".ToolCalls" and extracts any
// immediate text nodes that follow. This is used to identify tool call prefixes and formatting.
//
// Returns:
//   - string: The extracted text following the first ".ToolCalls" condition found
//   - bool: Whether a ".ToolCalls" condition was found in the template
func extractToolCallsFormat(tmpl *gotmpl.Template) (string, bool) {
	if tmpl == nil || tmpl.Tree == nil {
		slog.Debug("TextAfterToolCalls: template or tree is nil")
		return "", false
	}

	var result string
	var found bool

	var walk func(nodes []parse.Node)
	walk = func(nodes []parse.Node) {
		for _, node := range nodes {
			if found {
				return
			}

			switch n := node.(type) {
			case *parse.IfNode:
				if isToolCallsNode(n) {
					// Collect immediate TextNode(s) at start of IfNode's list
					var sb strings.Builder
					for _, innerNode := range n.List.Nodes {
						if tn, ok := innerNode.(*parse.TextNode); ok {
							sb.Write(tn.Text)
						} else {
							// Stop at first non-text node
							break
						}
					}
					result = sb.String()
					found = true
					return
				}
				// Recurse into child nodes
				walk(n.List.Nodes)
				if n.ElseList != nil {
					walk(n.ElseList.Nodes)
				}
			case *parse.ListNode:
				walk(n.Nodes)
			case *parse.RangeNode:
				walk(n.List.Nodes)
				if n.ElseList != nil {
					walk(n.ElseList.Nodes)
				}
			case *parse.WithNode:
				walk(n.List.Nodes)
				if n.ElseList != nil {
					walk(n.ElseList.Nodes)
				}
			default:
				// Continue to next node
				continue
			}

			if found {
				return
			}
		}
	}

	walk(tmpl.Tree.Root.Nodes)
	return result, found
}

// isToolCallsNode detects if a node's condition includes ".ToolCalls"
func isToolCallsNode(n *parse.IfNode) bool {
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

// TODO(parthsareen): get full prefix from the template instead of just the first token

// toolPrefix returns the prefix for the tool call if it exists from a template
func toolPrefix(tmpl *gotmpl.Template) string {
	tokenText, ok := extractToolCallsFormat(tmpl)
	if !ok {
		return ""
	}
	tokenText = strings.TrimSpace(tokenText)
	if tokenText == "" {
		return ""
	}
	first := strings.Fields(tokenText)[0]

	start := -1
	end := -1
	for i, r := range tokenText {
		if r == '<' || r == '[' {
			start = i
		}
		if (r == '>' || r == ']') && start != -1 {
			end = i
			break
		}
	}
	if start != -1 && end != -1 {
		// return the token including the [ or < and the ] or >
		return tokenText[start : end+1]
	} else if start != -1 {
		// get until the [ or < - in the case tag was not closed
		return tokenText[:start]
	} else if end != -1 {
		// get after the ] or > - in the case tag was not opened
		return tokenText[end+1:]
	}
	return first
}

// toolTemplate creates a subtree from the node that ranges over .ToolCalls
//
// Returns:
//   - *gotmpl.Template: The subtree containing the .ToolCalls range
//   - bool: Whether a .ToolCalls range was found in the template
func toolTemplate(t *template.Template) (*gotmpl.Template, bool) {
	tmpl := t.Subtree(func(n parse.Node) bool {
		if t, ok := n.(*parse.RangeNode); ok {
			return slices.Contains(template.Identifiers(t.Pipe), "ToolCalls")
		}

		return false
	})

	if tmpl == nil {
		return nil, false
	}

	return tmpl, true
}

// suffixOverlap returns the length of the longest suffix overlap between two strings
//
// Returns:
//   - int: The length of the longest suffix overlap
func suffixOverlap(s, delim string) int {
	max := min(len(delim), len(s))
	for i := max; i > 0; i-- {
		if strings.HasSuffix(s, delim[:i]) {
			return i
		}
	}
	return 0
}

// extractToolArgs executes a template with a known tool call format to extract the name and arguments
//
// Returns:
//   - string: The name of the tool call
//   - string: The arguments of the tool call
//   - error: Error if parsing failed
func extractToolArgs(tmpl *gotmpl.Template) (name, arguments string, err error) {
	var b bytes.Buffer
	if err := tmpl.Execute(&b, map[string][]api.ToolCall{
		"ToolCalls": {
			{
				Function: api.ToolCallFunction{
					Name: "@@name@@",
					Arguments: api.ToolCallFunctionArguments{
						"@@argument@@": 1,
					},
				},
			},
		},
	}); err != nil {
		return "", "", err
	}

	var obj any
	err = jsonv2.Unmarshal(b.Bytes(), &obj)
	if err != nil {
		return "", "", err
	}

	var objs []map[string]any
	switch v := obj.(type) {
	case map[string]any:
		objs = []map[string]any{v}
	case []map[string]any:
		objs = v
	case []any:
		objs = collect(v)
	}
	if len(objs) == 0 {
		return "", "", errors.New("no template objects found")
	}

	// find the keys that correspond to the name and arguments fields
	for k, v := range objs[0] {
		switch v.(type) {
		case string:
			name = k
		case map[string]any:
			arguments = k
		}
	}

	if name == "" || arguments == "" {
		slog.Debug("missing required fields in tool call template", "name", name, "arguments", arguments)
		return "", "", errors.New("missing required fields in tool call template")
	}

	return name, arguments, nil
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
