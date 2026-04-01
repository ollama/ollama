package thinking

import (
	"strings"
	"text/template"
	"text/template/parse"
)

func templateVisit(n parse.Node, enterFn func(parse.Node) bool, exitFn func(parse.Node)) {
	if n == nil {
		return
	}
	shouldContinue := enterFn(n)
	if !shouldContinue {
		return
	}
	switch x := n.(type) {
	case *parse.ListNode:
		for _, c := range x.Nodes {
			templateVisit(c, enterFn, exitFn)
		}
	case *parse.BranchNode:
		if x.Pipe != nil {
			templateVisit(x.Pipe, enterFn, exitFn)
		}
		if x.List != nil {
			templateVisit(x.List, enterFn, exitFn)
		}
		if x.ElseList != nil {
			templateVisit(x.ElseList, enterFn, exitFn)
		}
	case *parse.ActionNode:
		templateVisit(x.Pipe, enterFn, exitFn)
	case *parse.WithNode:
		templateVisit(&x.BranchNode, enterFn, exitFn)
	case *parse.RangeNode:
		templateVisit(&x.BranchNode, enterFn, exitFn)
	case *parse.IfNode:
		templateVisit(&x.BranchNode, enterFn, exitFn)
	case *parse.TemplateNode:
		templateVisit(x.Pipe, enterFn, exitFn)
	case *parse.PipeNode:
		for _, c := range x.Cmds {
			templateVisit(c, enterFn, exitFn)
		}
	case *parse.CommandNode:
		for _, a := range x.Args {
			templateVisit(a, enterFn, exitFn)
		}
		// text, field, number, etc. are leaves – nothing to recurse into
	}
	if exitFn != nil {
		exitFn(n)
	}
}

// InferTags uses a heuristic to infer the tags that surround thinking traces:
// We look for a range node that iterates over "Messages" and then look for a
// reference to "Thinking" like `{{.Thinking}}`. We then go up to the nearest
// ListNode and take the first and last TextNodes as the opening and closing
// tags.
func InferTags(t *template.Template) (string, string) {
	ancestors := []parse.Node{}

	openingTag := ""
	closingTag := ""

	enterFn := func(n parse.Node) bool {
		ancestors = append(ancestors, n)

		switch x := n.(type) {
		case *parse.FieldNode:
			if len(x.Ident) > 0 && x.Ident[0] == "Thinking" {
				var mostRecentRange *parse.RangeNode
				for i := len(ancestors) - 1; i >= 0; i-- {
					if r, ok := ancestors[i].(*parse.RangeNode); ok {
						mostRecentRange = r
						break
					}
				}
				if mostRecentRange == nil || !rangeUsesField(mostRecentRange, "Messages") {
					return true
				}

				// TODO(drifkin): to be more robust, check that it's in the action
				// part, not the `if`'s pipeline part. We do match on the nearest list
				// that starts and ends with text nodes, which makes this not strictly
				// necessary for our heuristic

				// go up to the nearest ancestor that is a *parse.ListNode
				for i := len(ancestors) - 1; i >= 0; i-- {
					if l, ok := ancestors[i].(*parse.ListNode); ok {
						firstNode := l.Nodes[0]
						if t, ok := firstNode.(*parse.TextNode); ok {
							openingTag = strings.TrimSpace(t.String())
						}
						lastNode := l.Nodes[len(l.Nodes)-1]
						if t, ok := lastNode.(*parse.TextNode); ok {
							closingTag = strings.TrimSpace(t.String())
						}

						break
					}
				}
			}
		}

		return true
	}

	exitFn := func(n parse.Node) {
		ancestors = ancestors[:len(ancestors)-1]
	}

	templateVisit(t.Root, enterFn, exitFn)

	return openingTag, closingTag
}

// checks to see if the given field name is present in the pipeline of the given range node
func rangeUsesField(rangeNode *parse.RangeNode, field string) bool {
	found := false
	enterFn := func(n parse.Node) bool {
		switch x := n.(type) {
		case *parse.FieldNode:
			if x.Ident[0] == field {
				found = true
			}
		}
		return true
	}
	templateVisit(rangeNode.BranchNode.Pipe, enterFn, nil)
	return found
}
