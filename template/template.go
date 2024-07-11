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
	"aggregate": func(v []*api.Message, role string) string {
		var aggregated []string
		for _, m := range v {
			if m.Role == role {
				aggregated = append(aggregated, m.Content)
			}
		}

		return strings.Join(aggregated, "\n\n")
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
			vars = append(vars, parseNode(n)...)
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

	// forceLegacy is a flag used to test compatibility with legacy templates
	forceLegacy bool
}

func (t *Template) Execute(w io.Writer, v Values) error {
	collated := collate(v.Messages)
	if !v.forceLegacy && slices.Contains(t.Vars(), "messages") {
		return t.Template.Execute(w, map[string]any{
			"Messages": collated,
		})
	}

	var b bytes.Buffer
	var system, prompt, response string
	for i, m := range collated {
		switch m.Role {
		case "system":
			system = m.Content
		case "user":
			prompt = m.Content
		case "assistant":
			response = m.Content
		}

		if i != len(collated)-1 && prompt != "" && response != "" {
			if err := t.Template.Execute(&b, map[string]any{
				"System":   system,
				"Prompt":   prompt,
				"Response": response,
			}); err != nil {
				return err
			}

			system = ""
			prompt = ""
			response = ""
		}
	}

	var cut bool
	nodes := deleteNode(t.Template.Root.Copy(), func(n parse.Node) bool {
		switch t := n.(type) {
		case *parse.ActionNode:
		case *parse.FieldNode:
			if slices.Contains(t.Ident, "Response") {
				cut = true
			}
		}

		return cut
	})

	tree := parse.Tree{Root: nodes.(*parse.ListNode)}
	if err := template.Must(template.New("").AddParseTree("", &tree)).Execute(&b, map[string]any{
		"System": "",
		"Prompt": prompt,
	}); err != nil {
		return err
	}

	_, err := io.Copy(w, &b)
	return err
}

// collate messages based on role. consecutive messages of the same role are merged
// into a single message. collate also pulls out and merges messages with Role == "system"
// which are templated separately. As a side effect, it mangles message content adding image
// tags ([img-%d]) as needed
func collate(msgs []api.Message) (collated []*api.Message) {
	var n int
	for i := range msgs {
		msg := msgs[i]
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
	case *parse.TemplateNode:
		return parseNode(n.Pipe)
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
