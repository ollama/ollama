package main

import (
	"embed"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"text/template"

	tree_sitter "github.com/tree-sitter/go-tree-sitter"
	tree_sitter_cpp "github.com/tree-sitter/tree-sitter-cpp/bindings/go"
)

//go:embed *.gotmpl
var fsys embed.FS

type Function struct {
	Type,
	Name,
	Parameters,
	Args string
}

func ParseFunction(node *tree_sitter.Node, tc *tree_sitter.TreeCursor, source []byte) Function {
	var fn Function
	fn.Name = node.ChildByFieldName("declarator").Utf8Text(source)
	if params := node.ChildByFieldName("parameters"); params != nil {
		fn.Parameters = params.Utf8Text(source)
		fn.Args = ParseParameters(params, tc, source)
	}

	var types []string
	for node.Parent() != nil && node.Parent().Kind() != "declaration" {
		if node.Parent().Kind() == "pointer_declarator" {
			types = append(types, "*")
		}
		node = node.Parent()
	}

	for sibling := node.PrevSibling(); sibling != nil; sibling = sibling.PrevSibling() {
		types = append(types, sibling.Utf8Text(source))
	}

	slices.Reverse(types)
	fn.Type = strings.Join(types, " ")
	return fn
}

func ParseParameters(node *tree_sitter.Node, tc *tree_sitter.TreeCursor, source []byte) string {
	var s []string
	for _, child := range node.Children(tc) {
		if child.IsNamed() {
			child := child.ChildByFieldName("declarator")
			for child != nil && child.Kind() != "identifier" {
				if child.Kind() == "parenthesized_declarator" {
					child = child.Child(1)
				} else {
					child = child.ChildByFieldName("declarator")
				}
			}

			if child != nil {
				s = append(s, child.Utf8Text(source))
			}
		}
	}
	return strings.Join(s, ", ")
}

func main() {
	var output string
	flag.StringVar(&output, "output", ".", "Output directory for generated files")
	flag.Parse()

	parser := tree_sitter.NewParser()
	defer parser.Close()

	language := tree_sitter.NewLanguage(tree_sitter_cpp.Language())
	parser.SetLanguage(language)

	query, _ := tree_sitter.NewQuery(language, `(function_declarator declarator: (identifier)) @func`)
	defer query.Close()

	qc := tree_sitter.NewQueryCursor()
	defer qc.Close()

	var funs []Function
	for _, arg := range flag.Args() {
		bts, err := os.ReadFile(arg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading file %s: %v\n", arg, err)
			continue
		}

		tree := parser.Parse(bts, nil)
		defer tree.Close()

		tc := tree.Walk()
		defer tc.Close()

		matches := qc.Matches(query, tree.RootNode(), bts)
		for match := matches.Next(); match != nil; match = matches.Next() {
			for _, capture := range match.Captures {
				funs = append(funs, ParseFunction(&capture.Node, tc, bts))
			}
		}
	}

	tmpl, err := template.New("").ParseFS(fsys, "*.gotmpl")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing template: %v\n", err)
		return
	}

	for _, tmpl := range tmpl.Templates() {
		name := filepath.Join(output, strings.TrimSuffix(tmpl.Name(), ".gotmpl"))

		fmt.Println("Generating", name)
		f, err := os.Create(name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating file %s: %v\n", name, err)
			continue
		}
		defer f.Close()

		if err := tmpl.Execute(f, map[string]any{
			"Functions": funs,
		}); err != nil {
			fmt.Fprintf(os.Stderr, "Error executing template %s: %v\n", tmpl.Name(), err)
		}
	}
}
