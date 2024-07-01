package template

import (
	"bytes"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
	"sync"
	"text/template"
	"text/template/parse"

	"github.com/agnivade/levenshtein"
	"github.com/ollama/ollama/api"
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

var DefaultTemplate, _ = Parse("{{ .Prompt }}")

type Template struct {
	*template.Template
	raw string
}

// response is a template node that can be added to templates that don't already have one
var response = parse.ActionNode{
	NodeType: parse.NodeAction,
	Pipe: &parse.PipeNode{
		NodeType: parse.NodePipe,
		Cmds: []*parse.CommandNode{
			{
				NodeType: parse.NodeCommand,
				Args: []parse.Node{
					&parse.FieldNode{
						NodeType: parse.NodeField,
						Ident:    []string{"Response"},
					},
				},
			},
		},
	},
}

var funcs = template.FuncMap{
	"toJson": func(v any) string {
		b, err := json.Marshal(v)
		if err != nil {
			return ""
		}

		return string(b)
	},
	"add": func(a, b int) int {
		return a + b
	},
	"sub": func(a, b int) int {
		return a - b
	},
}

func Parse(s string) (*Template, error) {
	tmpl := template.New("").Option("missingkey=zero").Funcs(funcs)

	tmpl, err := tmpl.Parse(s)
	if err != nil {
		return nil, err
	}

	t := Template{Template: tmpl, raw: s}
	if vars := t.Vars(); !slices.Contains(vars, "messages") && !slices.Contains(vars, "response") {
		// touch up the template and append {{ .Response }}
		tmpl.Tree.Root.Nodes = append(tmpl.Tree.Root.Nodes, &response)
	}

	return &t, nil
}

func (t *Template) String() string {
	return t.raw
}

func (t *Template) Vars() []string {
	var vars []string
	for _, tt := range t.Templates() {
		for _, n := range tt.Root.Nodes {
			vars = append(vars, Identifiers(n)...)
		}
	}

	set := make(map[string]struct{})
	for _, n := range vars {
		set[strings.ToLower(n)] = struct{}{}
	}

	vars = maps.Keys(set)
	slices.Sort(vars)
	return vars
}

type Values struct {
	Messages []api.Message
	Tools    []api.Tool
}

func (t *Template) Subtree(fn func(parse.Node) bool) *template.Template {
	var walk func(parse.Node) parse.Node
	walk = func(n parse.Node) parse.Node {
		if fn(n) {
			return n
		}

		switch t := n.(type) {
		case *parse.ListNode:
			for _, c := range t.Nodes {
				if n := walk(c); n != nil {
					return n
				}
			}
		case *parse.BranchNode:
			for _, n := range []*parse.ListNode{t.List, t.ElseList} {
				if n != nil {
					if n := walk(n); n != nil {
						return n
					}
				}
			}
		case *parse.IfNode:
			return walk(&t.BranchNode)
		case *parse.WithNode:
			return walk(&t.BranchNode)
		case *parse.RangeNode:
			return walk(&t.BranchNode)
		}

		return nil
	}

	if n := walk(t.Tree.Root); n != nil {
		return (&template.Template{
			Tree: &parse.Tree{
				Root: &parse.ListNode{
					Nodes: []parse.Node{n},
				},
			},
		}).Funcs(funcs)
	}

	return nil
}

func (t *Template) Execute(w io.Writer, v Values) error {
	system, collated := collate(v.Messages)
	if slices.Contains(t.Vars(), "messages") {
		return t.Template.Execute(w, map[string]any{
			"System":   system,
			"Messages": collated,
			"Tools":    v.Tools,
		})
	}

	var b bytes.Buffer
	var prompt, response string
	for i, m := range collated {
		if m.Role == "user" {
			prompt = m.Content
		} else {
			response = m.Content
		}

		if i != len(collated)-1 && prompt != "" && response != "" {
			if err := t.Template.Execute(&b, map[string]any{
				"System":   "",
				"Prompt":   prompt,
				"Response": response,
			}); err != nil {
				return err
			}

			prompt = ""
			response = ""
		}
	}

	var cut bool
	tree := t.Template.Copy()
	// for the last message, cut everything after "{{ .Response }}"
	tree.Root.Nodes = slices.DeleteFunc(tree.Root.Nodes, func(n parse.Node) bool {
		if slices.Contains(Identifiers(n), "Response") {
			cut = true
		}

		return cut
	})

	if err := template.Must(template.New("").AddParseTree("", tree)).Execute(&b, map[string]any{
		"System": system,
		"Prompt": prompt,
	}); err != nil {
		return err
	}

	_, err := io.Copy(w, &b)
	return err
}

type messages []*api.Message

// collate messages based on role. consecutive messages of the same role are merged
// into a single message. collate also pulls out and merges messages with Role == "system"
// which are templated separately. As a side effect, it mangles message content adding image
// tags ([img-%d]) as needed
func collate(msgs []api.Message) (system string, collated messages) {
	var n int
	for i := range msgs {
		msg := msgs[i]
		if msg.Role == "system" {
			if system != "" {
				system += "\n\n"
			}

			system += msg.Content
			continue
		}

		for range msg.Images {
			imageTag := fmt.Sprintf("[img-%d]", n)
			if !strings.Contains(msg.Content, "[img]") {
				msg.Content = strings.TrimSpace("[img] " + msg.Content)
			}

			msg.Content = strings.Replace(msg.Content, "[img]", imageTag, 1)
			n++
		}

		if len(collated) > 0 && collated[len(collated)-1].Role == msg.Role {
			collated[len(collated)-1].Content += "\n\n" + msg.Content
		} else {
			collated = append(collated, &msg)
		}
	}

	return
}

// Identifiers walks the node tree returning any identifiers it finds along the way
func Identifiers(n parse.Node) []string {
	switch n := n.(type) {
	case *parse.ListNode:
		var names []string
		for _, n := range n.Nodes {
			names = append(names, Identifiers(n)...)
		}

		return names
	case *parse.TemplateNode:
		return Identifiers(n.Pipe)
	case *parse.ActionNode:
		return Identifiers(n.Pipe)
	case *parse.BranchNode:
		names := Identifiers(n.Pipe)
		for _, n := range []*parse.ListNode{n.List, n.ElseList} {
			if n != nil {
				names = append(names, Identifiers(n)...)
			}
		}
		return names
	case *parse.IfNode:
		return Identifiers(&n.BranchNode)
	case *parse.RangeNode:
		return Identifiers(&n.BranchNode)
	case *parse.WithNode:
		return Identifiers(&n.BranchNode)
	case *parse.PipeNode:
		var names []string
		for _, c := range n.Cmds {
			for _, a := range c.Args {
				names = append(names, Identifiers(a)...)
			}
		}
		return names
	case *parse.FieldNode:
		return n.Ident
	case *parse.VariableNode:
		return n.Ident
	}

	return nil
}
