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

func Parse(s string) (*Template, error) {
	tmpl := template.New("").Option("missingkey=zero").Funcs(template.FuncMap{
		"toJson": func(v any) string {
			b, err := json.Marshal(v)
			if err != nil {
				return ""
			}

			return string(b)
		},
		"isLastMessage": func(s []*api.Message, m *api.Message) bool {
			for i := len(s) - 1; i >= 0; i-- {
				if m.Role != s[i].Role {
					continue
				}

				return m == s[i]
			}

			return false
		},
	})

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
}

func (t *Template) Execute(w io.Writer, v Values) error {
	system, collated := collate(v.Messages)
	if slices.Contains(t.Vars(), "messages") {
		return t.Template.Execute(w, map[string]any{
			"System":   system,
			"Messages": collated,
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
		if slices.Contains(parseNode(n), "Response") {
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

func collate(msgs []api.Message) (system string, collated []*api.Message) {
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
