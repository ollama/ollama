package tools

import (
	"bytes"
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

// extractToolCallsFormat traverses a template AST to find text that follows a ".ToolCalls" condition.
// It walks the template nodes looking for if-statements containing ".ToolCalls" and extracts any
// immediate text nodes that follow. This is used to identify tool call prefixes and formatting.
//
// Returns:
//   - string: The extracted text following the first ".ToolCalls" condition found
//   - bool: Whether a ".ToolCalls" condition was found in the template
func extractToolCallsFormat(tmpl *gotmpl.Template) (string, bool) {
	if tmpl == nil || tmpl.Tree == nil {
		slog.Debug("template or tree is nil")
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

func toolPrefix(tmpl *gotmpl.Template) string {
	tokenText, ok := extractToolCallsFormat(tmpl)
	if !ok {
		return ""
	}
	tokenText = strings.TrimSpace(tokenText)
	tokenText = strings.ReplaceAll(tokenText, "\r", "")
	tokenText = strings.ReplaceAll(tokenText, "\n", " ")

	return tokenText
}

// toolTemplate creates a subtree from the node that ranges over .ToolCalls
//
// Returns:
//   - *gotmpl.Template: The subtree containing the .ToolCalls range
//   - error: Error if parsing failed
func toolTemplate(t *template.Template) (*gotmpl.Template, error) {
	tmpl := t.Subtree(func(n parse.Node) bool {
		if t, ok := n.(*parse.RangeNode); ok {
			return slices.Contains(template.Identifiers(t.Pipe), "ToolCalls")
		}

		return false
	})

	if tmpl == nil {
		return nil, errors.New("failed to find tool template")
	}

	return tmpl, nil
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
	err = json.Unmarshal(b.Bytes(), &obj)
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
