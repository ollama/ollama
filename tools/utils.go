package tools

import (
	"log/slog"
	"slices"
	"strings"
	gotmpl "text/template"
	"text/template/parse"

	"github.com/ollama/ollama/template"
)

// extractToolCallsTemplate finds the immediate following text after any IfNode containing ".ToolCalls"
func extractToolCallsTemplate(tmpl *gotmpl.Template) (string, bool) {
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
				if nodeContainsToolCalls(n) {
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

// Helper to detect if a node's condition includes ".ToolCalls"
func nodeContainsToolCalls(n *parse.IfNode) bool {
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

// ToolPrefix returns the prefix for the tool call if it exists
// TODO(parthsareen): get full prefix from the template instead of just the first token
func ToolPrefix(tmpl *gotmpl.Template) (string, bool) {
	tokenText, ok := extractToolCallsTemplate(tmpl)
	if !ok {
		return "", false
	}
	tokenText = strings.TrimSpace(tokenText)
	if tokenText == "" {
		return "", false
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
		return tokenText[start : end+1], true
	} else if start != -1 {
		// get until the [ or < - in the case tag was not closed
		return tokenText[:start], true
	} else if end != -1 {
		// get after the ] or > - in the case tag was not opened
		return tokenText[end+1:], true
	}
	return first, true
}

func toolTemplate(t *template.Template) (*gotmpl.Template, bool) {
	// create a subtree from the node that ranges over .ToolCalls
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

func suffixOverlap(s, delim string) int {
	max := min(len(delim), len(s))
	for i := max; i > 0; i-- {
		if strings.HasSuffix(s, delim[:i]) {
			return i
		}
	}
	return 0
}
