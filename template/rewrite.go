package template

import (
	"text/template"
	"text/template/parse"
)

// rewritePropertiesCheck walks the template AST and rewrites .Function.Parameters.Properties
// to .Function.Parameters.HasProperties in if/with conditions to fix truthiness checking.
// This maintains backward compatibility with templates that check if Properties exist.
func rewritePropertiesCheck(tmpl *template.Template) {
	walk(tmpl.Tree.Root)
}

func walk(n parse.Node) {
	if n == nil {
		return
	}

	switch node := n.(type) {
	case *parse.ListNode:
		for _, child := range node.Nodes {
			walk(child)
		}
	case *parse.ActionNode:
		// Rewrite len calls in action nodes
		rewritePipeProperties(node.Pipe)
	case *parse.IfNode:
		rewritePipeProperties(node.Pipe)
		walk(&node.BranchNode)
	case *parse.WithNode:
		rewritePipeProperties(node.Pipe)
		walk(&node.BranchNode)
	case *parse.RangeNode:
		// Don't rewrite the pipe for range nodes - they need .Properties for iteration
		walk(&node.BranchNode)
	case *parse.BranchNode:
		if node.List != nil {
			walk(node.List)
		}
		if node.ElseList != nil {
			walk(node.ElseList)
		}
	}
}

func rewritePipeProperties(pipe *parse.PipeNode) {
	if pipe == nil {
		return
	}

	for _, cmd := range pipe.Cmds {
		rewriteCommand(cmd)
	}
}

// rewriteCommand recursively rewrites a command and all its nested command arguments
func rewriteCommand(cmd *parse.CommandNode) {
	// Check if this is a "len .Function.Parameters.Properties" call
	if isLenPropertiesCall(cmd) {
		// Replace entire command with .Function.Parameters.Len field access
		replaceLenWithLenMethod(cmd)
		return
	}

	// Recursively process all arguments
	for i, arg := range cmd.Args {
		switch argNode := arg.(type) {
		case *parse.FieldNode:
			// Check for direct .Properties field access
			if isPropertiesField(argNode.Ident) {
				cmd.Args[i] = replaceWithHasProperties(argNode)
			}
		case *parse.CommandNode:
			// Recursively process nested commands (e.g., inside "and", "gt", etc.)
			rewriteCommand(argNode)
		case *parse.PipeNode:
			// Template function arguments can be wrapped in PipeNodes
			rewritePipeProperties(argNode)
		}
	}
}

// isLenPropertiesCall checks if a command is "len .Function.Parameters.Properties"
func isLenPropertiesCall(cmd *parse.CommandNode) bool {
	if len(cmd.Args) != 2 {
		return false
	}

	// First arg should be the "len" identifier
	if ident, ok := cmd.Args[0].(*parse.IdentifierNode); !ok || ident.Ident != "len" {
		return false
	}

	// Second arg should be .Function.Parameters.Properties field
	if field, ok := cmd.Args[1].(*parse.FieldNode); ok {
		return isPropertiesField(field.Ident)
	}

	return false
}

// replaceLenWithLenMethod replaces "len .Function.Parameters.Properties" with ".Function.Parameters.Len"
func replaceLenWithLenMethod(cmd *parse.CommandNode) {
	if len(cmd.Args) < 2 {
		return
	}

	field, ok := cmd.Args[1].(*parse.FieldNode)
	if !ok {
		return
	}

	// Create new field node with .Len instead of .Properties
	newIdent := make([]string, len(field.Ident))
	copy(newIdent, field.Ident)
	newIdent[len(newIdent)-1] = "Len"

	newField := &parse.FieldNode{
		NodeType: parse.NodeField,
		Ident:    newIdent,
		Pos:      field.Pos,
	}

	// Replace the command with just the field access (remove "len" function call)
	cmd.Args = []parse.Node{newField}
}

func isPropertiesField(ident []string) bool {
	// Match: .Function.Parameters.Properties
	// We only rewrite if it ends with Parameters.Properties to avoid false positives
	if len(ident) < 3 {
		return false
	}
	return ident[len(ident)-1] == "Properties" && ident[len(ident)-2] == "Parameters"
}

func replaceWithHasProperties(field *parse.FieldNode) *parse.FieldNode {
	// Clone the identifier slice and replace the last element
	newIdent := make([]string, len(field.Ident))
	copy(newIdent, field.Ident)
	newIdent[len(newIdent)-1] = "HasProperties"

	return &parse.FieldNode{
		NodeType: parse.NodeField,
		Ident:    newIdent,
		Pos:      field.Pos,
	}
}
