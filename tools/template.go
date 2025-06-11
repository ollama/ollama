package tools

import (
	"bytes"
	"log/slog"
	"slices"
	"strings"
	"text/template"
	"text/template/parse"
)

// parseTag finds the tool calling tag from a Go template
// often <tool_call> [TOOL_CALL] or similar by finding the
// first text node after .ToolCalls and returning the content
// if no tag is found, return "{" to indicate that json objects
// should be attempted to be parsed as tool calls
func parseTag(tmpl *template.Template) string {
	if tmpl == nil || tmpl.Tree == nil {
		slog.Debug("template or tree is nil")
		return "{"
	}

	tc := findToolCallNode(tmpl.Tree.Root.Nodes)
	if tc == nil {
		return "{"
	}

	tn := findTextNode(tc.List.Nodes)
	if tn == nil {
		return "{"
	}

	tag := string(tn.Text)
	tag = strings.ReplaceAll(tag, "\r\n", "\n")

	// avoid parsing { onwards as this may be a tool call
	// however keep '{' as a prefix if there is no tag
	// so that all json objects will be attempted to
	// be parsed as tool calls
	tag, _, _ = strings.Cut(tag, "{")
	tag = strings.TrimSpace(tag)
	if tag == "" {
		tag = "{"
	}

	return tag
}

// findToolCallNode searches for and returns an IfNode with .ToolCalls
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

// findTextNode does a depth-first search for the first text content in nodes,
// stopping at template constructs to avoid parsing text after the tool calls
func findTextNode(nodes []parse.Node) *parse.TextNode {
	for _, node := range nodes {
		switch n := node.(type) {
		case *parse.TextNode:
			// skip whitespace-only text nodes
			if len(bytes.TrimSpace(n.Text)) == 0 {
				continue
			}
			return n
		case *parse.IfNode:
			if text := findTextNode(n.List.Nodes); text != nil {
				return text
			}
			if n.ElseList != nil {
				if text := findTextNode(n.ElseList.Nodes); text != nil {
					return text
				}
			}
			return nil
		case *parse.ListNode:
			if text := findTextNode(n.Nodes); text != nil {
				return text
			}
		case *parse.RangeNode:
			if text := findTextNode(n.List.Nodes); text != nil {
				return text
			}
			if n.ElseList != nil {
				if text := findTextNode(n.ElseList.Nodes); text != nil {
					return text
				}
			}
			return nil
		case *parse.WithNode:
			if text := findTextNode(n.List.Nodes); text != nil {
				return text
			}
			if n.ElseList != nil {
				if text := findTextNode(n.ElseList.Nodes); text != nil {
					return text
				}
			}
			return nil
		case *parse.ActionNode:
			return nil
		}
	}
	return nil
}
