package template

import (
	"bytes"
	"embed"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"math"
	"slices"
	"strings"
	"sync"
	"text/template"
	"text/template/parse"
	"time"

	"github.com/agnivade/levenshtein"

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
	"json": func(v any) string {
		b, _ := json.Marshal(v)
		return string(b)
	},
	"currentDate": func(args ...string) string {
		// Currently ignoring the format argument, but accepting it for future use
		// Default format is YYYY-MM-DD
		return time.Now().Format("2006-01-02")
	},
	"yesterdayDate": func(args ...string) string {
		return time.Now().AddDate(0, 0, -1).Format("2006-01-02")
	},
	"toTypeScriptType": func(v any) string {
		if param, ok := v.(api.ToolProperty); ok {
			return param.ToTypeScriptType()
		}
		// Handle pointer case
		if param, ok := v.(*api.ToolProperty); ok && param != nil {
			return param.ToTypeScriptType()
		}
		return "any"
	},
}

func Parse(s string) (*Template, error) {
	tmpl := template.New("").Option("missingkey=zero").Funcs(funcs)

	tmpl, err := tmpl.Parse(s)
	if err != nil {
		return nil, err
	}

	t := Template{Template: tmpl, raw: s}
	vars, err := t.Vars()
	if err != nil {
		return nil, err
	}

	if !slices.Contains(vars, "messages") && !slices.Contains(vars, "response") {
		// touch up the template and append {{ .Response }}
		tmpl.Tree.Root.Nodes = append(tmpl.Tree.Root.Nodes, &response)
	}

	return &t, nil
}

func (t *Template) String() string {
	return t.raw
}

func (t *Template) Vars() ([]string, error) {
	var vars []string
	for _, tt := range t.Templates() {
		for _, n := range tt.Root.Nodes {
			v, err := Identifiers(n)
			if err != nil {
				return vars, err
			}
			vars = append(vars, v...)
		}
	}

	set := make(map[string]struct{})
	for _, n := range vars {
		set[strings.ToLower(n)] = struct{}{}
	}

	return slices.Sorted(maps.Keys(set)), nil
}

func (t *Template) Contains(s string) bool {
	return strings.Contains(t.raw, s)
}

type Values struct {
	Messages []api.Message
	api.Tools
	Prompt string
	Suffix string
	Think  bool
	// ThinkLevel contains the thinking level if Think is true and a string value was provided
	ThinkLevel string
	// whether or not the user explicitly set the thinking flag (vs. it being
	// implicitly false). Templates can't see whether `Think` is nil
	IsThinkSet bool

	// forceLegacy is a flag used to test compatibility with legacy templates
	forceLegacy bool
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
	system, messages := collate(v.Messages)
	vars, err := t.Vars()
	if err != nil {
		return err
	}
	if v.Prompt != "" && v.Suffix != "" {
		return t.Template.Execute(w, map[string]any{
			"Prompt":     v.Prompt,
			"Suffix":     v.Suffix,
			"Response":   "",
			"Think":      v.Think,
			"ThinkLevel": v.ThinkLevel,
			"IsThinkSet": v.IsThinkSet,
		})
	} else if !v.forceLegacy && slices.Contains(vars, "messages") {
		return t.Template.Execute(w, map[string]any{
			"System":     system,
			"Messages":   convertMessagesForTemplate(messages),
			"Tools":      convertToolsForTemplate(v.Tools),
			"Response":   "",
			"Think":      v.Think,
			"ThinkLevel": v.ThinkLevel,
			"IsThinkSet": v.IsThinkSet,
		})
	}

	system = ""
	var b bytes.Buffer
	var prompt, response string
	for _, m := range messages {
		execute := func() error {
			if err := t.Template.Execute(&b, map[string]any{
				"System":     system,
				"Prompt":     prompt,
				"Response":   response,
				"Think":      v.Think,
				"ThinkLevel": v.ThinkLevel,
				"IsThinkSet": v.IsThinkSet,
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
	nodes := deleteNode(t.Template.Root.Copy(), func(n parse.Node) bool {
		if field, ok := n.(*parse.FieldNode); ok && slices.Contains(field.Ident, "Response") {
			cut = true
			return false
		}

		return cut
	})

	tree := parse.Tree{Root: nodes.(*parse.ListNode)}
	if err := template.Must(template.New("").AddParseTree("", &tree)).Execute(&b, map[string]any{
		"System":     system,
		"Prompt":     prompt,
		"Response":   response,
		"Think":      v.Think,
		"ThinkLevel": v.ThinkLevel,
		"IsThinkSet": v.IsThinkSet,
	}); err != nil {
		return err
	}

	_, err = io.Copy(w, &b)
	return err
}

// collate messages based on role. consecutive messages of the same role are merged
// into a single message (except for tool messages which preserve individual metadata).
// collate also collects and returns all system messages.
// collate mutates message content adding image tags ([img-%d]) as needed
// todo(parthsareen): revisit for contextual image support
func collate(msgs []api.Message) (string, []*api.Message) {
	var system []string
	var collated []*api.Message
	for i := range msgs {
		if msgs[i].Role == "system" {
			system = append(system, msgs[i].Content)
		}

		// merges consecutive messages of the same role into a single message (except for tool messages)
		if len(collated) > 0 && collated[len(collated)-1].Role == msgs[i].Role && msgs[i].Role != "tool" {
			collated[len(collated)-1].Content += "\n\n" + msgs[i].Content
		} else {
			collated = append(collated, &msgs[i])
		}
	}

	return strings.Join(system, "\n\n"), collated
}

// templateTools is a slice of templateTool that marshals to JSON.
type templateTools []templateTool

func (t templateTools) String() string {
	bts, _ := json.Marshal(t)
	return string(bts)
}

// templateArgs is a map type with JSON string output for templates.
type templateArgs map[string]any

func (t templateArgs) String() string {
	if t == nil {
		return "{}"
	}
	bts, _ := json.Marshal(t)
	return string(bts)
}

// templateProperties is a map type with JSON string output for templates.
type templateProperties map[string]api.ToolProperty

func (t templateProperties) String() string {
	if t == nil {
		return "{}"
	}
	bts, _ := json.Marshal(t)
	return string(bts)
}

// templateTool is a template-compatible representation of api.Tool
// with Properties as a regular map for template ranging.
type templateTool struct {
	Type     string               `json:"type"`
	Items    any                  `json:"items,omitempty"`
	Function templateToolFunction `json:"function"`
}

type templateToolFunction struct {
	Name        string                         `json:"name"`
	Description string                         `json:"description"`
	Parameters  templateToolFunctionParameters `json:"parameters"`
}

type templateToolFunctionParameters struct {
	Type       string             `json:"type"`
	Defs       any                `json:"$defs,omitempty"`
	Items      any                `json:"items,omitempty"`
	Required   []string           `json:"required,omitempty"`
	Properties templateProperties `json:"properties"`
}

// templateToolCall is a template-compatible representation of api.ToolCall
// with Arguments as a regular map for template ranging.
type templateToolCall struct {
	ID       string
	Function templateToolCallFunction
}

type templateToolCallFunction struct {
	Index     int
	Name      string
	Arguments templateArgs
}

// templateMessage is a template-compatible representation of api.Message
// with ToolCalls converted for template use.
type templateMessage struct {
	Role       string
	Content    string
	Thinking   string
	Images     []api.ImageData
	ToolCalls  []templateToolCall
	ToolName   string
	ToolCallID string
}

// convertToolsForTemplate converts Tools to template-compatible format.
func convertToolsForTemplate(tools api.Tools) templateTools {
	if tools == nil {
		return nil
	}
	result := make(templateTools, len(tools))
	for i, tool := range tools {
		result[i] = templateTool{
			Type:  tool.Type,
			Items: tool.Items,
			Function: templateToolFunction{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters: templateToolFunctionParameters{
					Type:       tool.Function.Parameters.Type,
					Defs:       tool.Function.Parameters.Defs,
					Items:      tool.Function.Parameters.Items,
					Required:   tool.Function.Parameters.Required,
					Properties: templateProperties(tool.Function.Parameters.Properties.ToMap()),
				},
			},
		}
	}
	return result
}

// convertMessagesForTemplate converts Messages to template-compatible format.
func convertMessagesForTemplate(messages []*api.Message) []*templateMessage {
	if messages == nil {
		return nil
	}
	result := make([]*templateMessage, len(messages))
	for i, msg := range messages {
		var toolCalls []templateToolCall
		for _, tc := range msg.ToolCalls {
			toolCalls = append(toolCalls, templateToolCall{
				ID: tc.ID,
				Function: templateToolCallFunction{
					Index:     tc.Function.Index,
					Name:      tc.Function.Name,
					Arguments: templateArgs(tc.Function.Arguments.ToMap()),
				},
			})
		}
		result[i] = &templateMessage{
			Role:       msg.Role,
			Content:    msg.Content,
			Thinking:   msg.Thinking,
			Images:     msg.Images,
			ToolCalls:  toolCalls,
			ToolName:   msg.ToolName,
			ToolCallID: msg.ToolCallID,
		}
	}
	return result
}

// Identifiers walks the node tree returning any identifiers it finds along the way
func Identifiers(n parse.Node) ([]string, error) {
	switch n := n.(type) {
	case *parse.ListNode:
		var names []string
		for _, n := range n.Nodes {
			i, err := Identifiers(n)
			if err != nil {
				return names, err
			}
			names = append(names, i...)
		}

		return names, nil
	case *parse.TemplateNode:
		if n.Pipe == nil {
			return nil, errors.New("undefined template specified")
		}
		return Identifiers(n.Pipe)
	case *parse.ActionNode:
		if n.Pipe == nil {
			return nil, errors.New("undefined action in template")
		}
		return Identifiers(n.Pipe)
	case *parse.BranchNode:
		if n.Pipe == nil {
			return nil, errors.New("undefined branch")
		}
		names, err := Identifiers(n.Pipe)
		if err != nil {
			return names, err
		}
		for _, n := range []*parse.ListNode{n.List, n.ElseList} {
			if n != nil {
				i, err := Identifiers(n)
				if err != nil {
					return names, err
				}
				names = append(names, i...)
			}
		}
		return names, nil
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
				i, err := Identifiers(a)
				if err != nil {
					return names, err
				}
				names = append(names, i...)
			}
		}
		return names, nil
	case *parse.FieldNode:
		return n.Ident, nil
	case *parse.VariableNode:
		return n.Ident, nil
	}

	return nil, nil
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
