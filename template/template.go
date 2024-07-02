package template

import (
	"bytes"
	"embed"
	"encoding/json"
	"errors"
	"io"
	"math"
	"slices"
	"strings"
	"sync"
	"text/template"
	"text/template/parse"

	"github.com/agnivade/levenshtein"
	"golang.org/x/exp/maps"
)

//go:embed index.json
var indexBytes []byte

//go:embed *.gotmpl
var templatesFS embed.FS

var templatesOnce = sync.OnceValues(func() ([]*named, error) {
	var templates []*named
	if err := json.Unmarshal(indexBytes, &templates); err != nil {
		return nil, err
	}

	for _, t := range templates {
		bts, err := templatesFS.ReadFile(t.Name + ".gotmpl")
		if err != nil {
			return nil, err
		}

		// normalize line endings
		t.Bytes = bytes.ReplaceAll(bts, []byte("\r\n"), []byte("\n"))
	}

	return templates, nil
})

type named struct {
	Name     string `json:"name"`
	Template string `json:"template"`
	Bytes    []byte
}

func (t named) Reader() io.Reader {
	return bytes.NewReader(t.Bytes)
}

func Named(s string) (*named, error) {
	templates, err := templatesOnce()
	if err != nil {
		return nil, err
	}

	var template *named
	score := math.MaxInt
	for _, t := range templates {
		if s := levenshtein.ComputeDistance(s, t.Template); s < score {
			score = s
			template = t
		}
	}

	if score < 100 {
		return template, nil
	}

	return nil, errors.New("no matching template found")
}

type Template struct {
	*template.Template
	raw string
}

func (t *Template) String() string {
	return t.raw
}

var DefaultTemplate, _ = Parse("{{ .Prompt }}")

func Parse(s string) (*Template, error) {
	t, err := template.New("").Option("missingkey=zero").Parse(s)
	if err != nil {
		return nil, err
	}

	return &Template{Template: t, raw: s}, nil
}

func (t *Template) Vars() []string {
	var vars []string
	for _, n := range t.Tree.Root.Nodes {
		vars = append(vars, parseNode(n)...)
	}

	set := make(map[string]struct{})
	for _, n := range vars {
		set[strings.ToLower(n)] = struct{}{}
	}

	vars = maps.Keys(set)
	slices.Sort(vars)
	return vars
}

func parseNode(n parse.Node) []string {
	switch n := n.(type) {
	case *parse.ActionNode:
		return parseNode(n.Pipe)
	case *parse.IfNode:
		names := parseNode(n.Pipe)
		names = append(names, parseNode(n.List)...)
		if n.ElseList != nil {
			names = append(names, parseNode(n.ElseList)...)
		}
		return names
	case *parse.RangeNode:
		names := parseNode(n.Pipe)
		names = append(names, parseNode(n.List)...)
		if n.ElseList != nil {
			names = append(names, parseNode(n.ElseList)...)
		}
		return names
	case *parse.WithNode:
		names := parseNode(n.Pipe)
		names = append(names, parseNode(n.List)...)
		if n.ElseList != nil {
			names = append(names, parseNode(n.ElseList)...)
		}
		return names
	case *parse.PipeNode:
		var names []string
		for _, c := range n.Cmds {
			for _, a := range c.Args {
				names = append(names, parseNode(a)...)
			}
		}
		return names
	case *parse.ListNode:
		var names []string
		for _, n := range n.Nodes {
			names = append(names, parseNode(n)...)
		}

		return names
	case *parse.FieldNode:
		return n.Ident
	}

	return nil
}
