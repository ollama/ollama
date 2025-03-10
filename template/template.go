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

	"github.com/ollama/ollama/api"
)

//go:embed index.json
var indexBytes []byte

//go:embed *.gotmpl
//go:embed *.json
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

		params, err := templatesFS.ReadFile(t.Name + ".json")
		if err != nil {
			continue
		}

		if err := json.Unmarshal(params, &t.Parameters); err != nil {
			return nil, err
		}
	}

	return templates, nil
})

type named struct {
	Name     string `json:"name"`
	Template string `json:"template"`
	Bytes    []byte

	Parameters *struct {
		Stop []string `json:"stop"`
	}
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
	tree *parse.Tree
	raw  string
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
	"json": func(v any) string {
		b, _ := json.Marshal(v)
		return string(b)
	},
}

func Parse(s string) (*Template, error) {
	tree := parse.New("")
	tree.Mode = tree.Mode | parse.SkipFuncCheck

	tree, err := tree.Parse(s, "", "", map[string]*parse.Tree{})
	if err != nil {
		return nil, err
	}

	t := Template{tree, s}
	if vars := t.Vars(); !slices.Contains(vars, "messages") && !slices.Contains(vars, "response") {
		// touch up the template and append {{ .Response }}
		t.tree.Root.Nodes = append(t.tree.Root.Nodes, &response)
	}

	return &t, nil
}

func (t *Template) String() string {
	return t.raw
}

func (t *Template) Vars() []string {
	var vars []string
	for _, n := range t.tree.Root.Nodes {
		vars = append(vars, Identifiers(n)...)
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
	api.Tools
	Prompt string
	Suffix string

	// forceLegacy is a flag used to test compatibility with legacy templates
	forceLegacy bool
}

// Sub returns a new template with the subtree that matches the predicate
func (t *Template) Sub(fn func(parse.Node) bool) *Template {
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

	if n := walk(t.tree.Root); n != nil {
		return &Template{
			tree: &parse.Tree{
				Root: &parse.ListNode{
					Nodes: []parse.Node{n},
				},
			},
		}
	}

	return nil
}

func (t *Template) Template() *template.Template {
	return template.Must(template.New("").Option("missingkey=zero").Funcs(funcs).AddParseTree("", t.tree))
}

func (t *Template) Execute(w io.Writer, v Values) error {
	tmpl := t.Template()
	system, messages := collate(v.Messages)
	if v.Prompt != "" && v.Suffix != "" {
		return tmpl.Execute(w, map[string]any{
			"Prompt":   v.Prompt,
			"Suffix":   v.Suffix,
			"Response": "",
		})
	} else if !v.forceLegacy && slices.Contains(t.Vars(), "messages") {
		return tmpl.Execute(w, map[string]any{
			"System":   system,
			"Messages": messages,
			"Tools":    v.Tools,
			"Response": "",
		})
	}

	system = ""
	var b bytes.Buffer
	var prompt, response string
	for _, m := range messages {
		execute := func() error {
			if err := tmpl.Execute(&b, map[string]any{
				"System":   system,
				"Prompt":   prompt,
				"Response": response,
			}); err != nil {
				return err
			}

			system = ""
			prompt = ""
			response = ""
			return nil
		}

		switch m.Role {
		case "system":
			if prompt != "" || response != "" {
				if err := execute(); err != nil {
					return err
				}
			}
			system = m.Content
		case "user":
			if response != "" {
				if err := execute(); err != nil {
					return err
				}
			}
			prompt = m.Content
		case "assistant":
			response = m.Content
		}
	}

	var cut bool
	nodes := deleteNode(t.tree.Root.Copy(), func(n parse.Node) bool {
		if field, ok := n.(*parse.FieldNode); ok && slices.Contains(field.Ident, "Response") {
			cut = true
			return false
		}

		return cut
	})

	tree := parse.Tree{Root: nodes.(*parse.ListNode)}
	if err := template.Must(tmpl.AddParseTree("", &tree)).Execute(&b, map[string]any{
		"System":   system,
		"Prompt":   prompt,
		"Response": response,
	}); err != nil {
		return err
	}

	_, err := io.Copy(w, &b)
	return err
}

// collate messages based on role. consecutive messages of the same role are merged
// into a single message. collate also collects and returns all system messages.
// collate mutates message content adding image tags ([img-%d]) as needed
func collate(msgs []api.Message) (string, []*api.Message) {
	var system []string
	var collated []*api.Message
	for i := range msgs {
		msg := msgs[i]
		if msg.Role == "system" {
			system = append(system, msg.Content)
		}

		if len(collated) > 0 && collated[len(collated)-1].Role == msg.Role {
			collated[len(collated)-1].Content += "\n\n" + msg.Content
		} else {
			collated = append(collated, &msg)
		}
	}

	return strings.Join(system, "\n\n"), collated
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

// deleteNode walks the node list and deletes nodes that match the predicate
// this is currently to remove the {{ .Response }} node from templates
func deleteNode(n parse.Node, fn func(parse.Node) bool) parse.Node {
	var walk func(n parse.Node) parse.Node
	walk = func(n parse.Node) parse.Node {
		if fn(n) {
			return nil
		}

		switch t := n.(type) {
		case *parse.ListNode:
			var nodes []parse.Node
			for _, c := range t.Nodes {
				if n := walk(c); n != nil {
					nodes = append(nodes, n)
				}
			}

			t.Nodes = nodes
			return t
		case *parse.IfNode:
			t.BranchNode = *(walk(&t.BranchNode).(*parse.BranchNode))
		case *parse.WithNode:
			t.BranchNode = *(walk(&t.BranchNode).(*parse.BranchNode))
		case *parse.RangeNode:
			t.BranchNode = *(walk(&t.BranchNode).(*parse.BranchNode))
		case *parse.BranchNode:
			t.List = walk(t.List).(*parse.ListNode)
			if t.ElseList != nil {
				t.ElseList = walk(t.ElseList).(*parse.ListNode)
			}
		case *parse.ActionNode:
			n := walk(t.Pipe)
			if n == nil {
				return nil
			}

			t.Pipe = n.(*parse.PipeNode)
		case *parse.PipeNode:
			var commands []*parse.CommandNode
			for _, c := range t.Cmds {
				var args []parse.Node
				for _, a := range c.Args {
					if n := walk(a); n != nil {
						args = append(args, n)
					}
				}

				if len(args) == 0 {
					return nil
				}

				c.Args = args
				commands = append(commands, c)
			}

			if len(commands) == 0 {
				return nil
			}

			t.Cmds = commands
		}

		return n
	}

	return walk(n)
}
