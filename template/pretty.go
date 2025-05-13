package template

import (
	"strings"
	texttmpl "text/template"
	"text/template/parse"
)

// Format returns a human-readable representation of the template.
// The formatting indents nested sections such as if/else blocks.
func Format(src string) (string, error) {
	tmpl, err := texttmpl.New("pretty").Parse(src)
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	printNodes(tmpl.Tree.Root, 0, &sb)
	return sb.String(), nil
}

func indent(sb *strings.Builder, level int) {
	for i := 0; i < level; i++ {
		sb.WriteString("  ")
	}
}

func printNodes(list *parse.ListNode, level int, sb *strings.Builder) {
	if list == nil {
		return
	}
	for _, n := range list.Nodes {
		printNode(n, level, sb)
	}
}

func printNode(n parse.Node, level int, sb *strings.Builder) {
	switch n := n.(type) {
	case *parse.TextNode:
		text := strings.TrimSpace(string(n.Text))
		if text == "" {
			return
		}
		indent(sb, level)
		sb.WriteString(text)
		sb.WriteByte('\n')
	case *parse.ActionNode:
		indent(sb, level)
		// sb.WriteString("ACTION {{ ")
		sb.WriteString(n.String())
		// sb.WriteString(" }}\n")
		sb.WriteByte('\n')
	case *parse.IfNode:
		indent(sb, level)
		sb.WriteString("{{ if ")
		sb.WriteString(n.Pipe.String())
		sb.WriteString(" }}\n")
		printNodes(n.List, level+1, sb)
		if n.ElseList != nil {
			indent(sb, level)
			sb.WriteString("{{ else }}\n")
			printNodes(n.ElseList, level+1, sb)
		}
		indent(sb, level)
		sb.WriteString("{{ end }}\n")
	case *parse.RangeNode:
		indent(sb, level)
		sb.WriteString("{{ range ")
		sb.WriteString(n.Pipe.String())
		sb.WriteString(" }}\n")
		printNodes(n.List, level+1, sb)
		if n.ElseList != nil {
			indent(sb, level)
			sb.WriteString("{{ else }}\n")
			printNodes(n.ElseList, level+1, sb)
		}
		indent(sb, level)
		sb.WriteString("{{ end }}\n")
	case *parse.WithNode:
		indent(sb, level)
		sb.WriteString("{{ with ")
		sb.WriteString(n.Pipe.String())
		sb.WriteString(" }}\n")
		printNodes(n.List, level+1, sb)
		if n.ElseList != nil {
			indent(sb, level)
			sb.WriteString("{{ else }}\n")
			printNodes(n.ElseList, level+1, sb)
		}
		indent(sb, level)
		sb.WriteString("{{ end }}\n")
	case *parse.TemplateNode:
		indent(sb, level)
		sb.WriteString("{{ template ")
		sb.WriteString(n.Name)
		sb.WriteString(" }}\n")
	default:
		indent(sb, level)
		sb.WriteString(n.String())
		sb.WriteByte('\n')
	}
}
