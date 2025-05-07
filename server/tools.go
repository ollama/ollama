package server

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"
	gotmpl "text/template"

	jsonv2 "github.com/go-json-experiment/json"

	"github.com/ollama/ollama/api"
)

type State int

const (
	SendTokens State = iota
	GreedyToolWithPrefix
	GreedyToolNoPrefix
	// ToolCall
	ForceTools
	ToolSuffix
	ContainsPartialPrefix
	Done
)

func (s State) String() string {
	switch s {
	case SendTokens:
		return "SendTokens"
	case GreedyToolWithPrefix:
		return "GreedyToolWithPrefix"
	case GreedyToolNoPrefix:
		return "GreedyToolNoPrefix"
	case ForceTools:
		return "ForceTools"
	case ToolSuffix:
		return "ToolSuffix"
	case Done:
		return "Done"
	case ContainsPartialPrefix:
		return "PartialPrefix"
	default:
		return fmt.Sprintf("Unknown State (%d)", s)
	}
}

type ToolParser struct {
	tmpl       *gotmpl.Template
	state      State
	sb         *strings.Builder
	toolPrefix string
}

// parseJSONToolCalls attempts to parse a JSON string into a slice of ToolCalls.
// Returns parsed tool calls and a boolean indicating if the JSON is incomplete
func (p *ToolParser) parseJSONToolCalls(s string) ([]api.ToolCall, bool, bool) {
	var b bytes.Buffer
	if err := p.tmpl.Execute(&b, map[string][]api.ToolCall{
		"ToolCalls": {
			{
				Function: api.ToolCallFunction{
					Name: "@@name@@",
					Arguments: api.ToolCallFunctionArguments{
						"@@argument@@": 1,
					},
				},
			},
		},
	}); err != nil {
		return nil, false, false
	}

	// ! this can be either a map or an array
	var temp any
	err := jsonv2.Unmarshal(b.Bytes(), &temp)
	if err != nil {
		return nil, false, false
	}

	var collect func(any) []map[string]any
	collect = func(obj any) (all []map[string]any) {
		switch o := obj.(type) {
		case map[string]any:
			all = append(all, o)
			for _, v := range o {
				all = append(all, collect(v)...)
			}
		case []any:
			for _, v := range o {
				all = append(all, collect(v)...)
			}
		}

		return all
	}

	var templateObjects []map[string]any
	switch t := temp.(type) {
	case map[string]any:
		templateObjects = []map[string]any{t}
	case []map[string]any:
		templateObjects = t
	// ! fallback?
	case []any:
		templateObjects = collect(t)
	}
	if len(templateObjects) == 0 {
		return nil, false, false
	}

	// find the keys that correspond to the name and arguments fields
	var name, arguments string
	for k, v := range templateObjects[0] {
		switch v.(type) {
		case string:
			name = k
		case map[string]any:
			arguments = k
		}
	}

	if name == "" || arguments == "" {
		return nil, false, false
	}
	var responseObjects any
	err = jsonv2.Unmarshal([]byte(s), &responseObjects)
	if err != nil {
		if errors.Is(err, io.ErrUnexpectedEOF) || err.Error() == "unexpected end of JSON input" {
			fmt.Println("Detected partial or incomplete JSON.")
			fmt.Println("state", p.state)
			return nil, true, false
		} else {
			fmt.Printf("Other error: %v\n", err)
			fmt.Println("exiting", p.state)
			return nil, false, false
		}
	}

	var objs []map[string]any
	objs = append(objs, collect(responseObjects)...)
	if len(objs) == 0 {
		return nil, false, false
	}

	slog.Debug("collected objects", "count", len(objs))

	var toolCalls []api.ToolCall
	for _, kv := range objs {
		n, nok := kv[name].(string)
		a, aok := kv[arguments].(map[string]any)
		if nok && aok {
			slog.Debug("found valid tool call", "name", n)
			toolCalls = append(toolCalls, api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      n,
					Arguments: a,
				},
			})
		}
	}

	slog.Debug("parsed tool calls", "count", len(toolCalls))
	return toolCalls, false, true
}

func (p *ToolParser) updateOutputState(ok bool, partial bool, tcs []api.ToolCall) {
	switch {
	case !ok && !partial && p.state == ForceTools:
		fmt.Println("Case: !ok && !partial && ForceTools - staying in force tools, resetting buffer")
		// force partial tool if we have a prefix
		// no op and stay in force tools
		p.sb.Reset()
	case !ok && !partial:
		fmt.Println("Case: !ok && !partial")
		fmt.Println("state", p.state)
		if p.state == GreedyToolNoPrefix {
			fmt.Println("  Subcase: GreedyToolNoPrefix - marking as done")
			p.state = Done
		}
		if p.state == GreedyToolWithPrefix {
			fmt.Println("  Subcase: GreedyToolWithPrefix - switching to SendTokens")
			p.state = SendTokens
		}
		p.sb.Reset()
	case !ok && partial:
		fmt.Println("Case: !ok && partial - accumulating partial content")
		// ! acucumulate

	case len(tcs) > 0:
		fmt.Println("Case: tool calls found")
		// do not parse again in the greedy JSON case as soon as we have a tool call
		if p.state == GreedyToolWithPrefix {
			p.state = SendTokens
		} else if p.state == GreedyToolNoPrefix {
			fmt.Println("  Subcase: Greedy modes - marking done and switching to SendTokens")
			p.state = Done
		}
		p.sb.Reset()
	}
}

func (p *ToolParser) updateInputState(s string, hasPrefix bool) (string, bool) {
	if p.toolPrefix == "" {
		return s, true
	}

	if hasPrefix {
		p.state = ForceTools
		// partial tool possibly
	} else if strings.HasPrefix(p.toolPrefix, s) {
		slog.Debug("tool prefix partially", "prefix", p.toolPrefix, "content", s)
		// TODO: could possibly err maybe this should be greedy instead?
		p.state = ForceTools
		return "", false
	} else if strings.Contains(s, p.toolPrefix) {
		idx := strings.Index(s, p.toolPrefix)
		if idx != -1 {
			// still keeps the prefix
			p.state = ContainsPartialPrefix
			p.sb.Reset()
			p.sb.WriteString(s[idx:])
			return s[:idx], false
		}
	}
	// Special token end case
	if strings.HasSuffix(s, p.toolPrefix[2:]) {
		// can be with string or just the token
		if hasPrefix {
			s = strings.TrimSpace(s[:len(s)-(len(p.toolPrefix)+1)])
		} else {
			p.state = ToolSuffix
			p.sb.Reset()
			return "", false
		}
		slog.Debug("setting to no tool", "content", s)
	}
	return s, true
}

// ParseToolCalls extracts tool calls from a string using a tool token prefix or direct JSON parsing.
// Returns tool calls, whether parsing is incomplete, and any errors.
func (p *ToolParser) ParseToolCalls(s string) ([]api.ToolCall, string, bool) {
	// append input
	p.sb.WriteString(s)
	s = p.sb.String()
	s = strings.TrimSpace(s)

	if len(s) == 0 {
		return nil, "", false
	}

	s, hasPrefix := strings.CutPrefix(s, p.toolPrefix)

	s, ok := p.updateInputState(s, hasPrefix)
	if !ok {
		if p.state == ContainsPartialPrefix {
			return nil, s, false
		}
		return nil, "", false
	}

	if p.state == SendTokens {
		return nil, "", false
	}

	var tcs []api.ToolCall
	var partial bool
	tcs, partial, ok = p.parseJSONToolCalls(s)
	p.updateOutputState(ok, partial, tcs)
	if !ok {
		return nil, "", false
	}

	return tcs, "", true
}

func NewToolParser(model *Model) *ToolParser {
	templateToolPrefix, _ := ToolPrefix(model.Template.Template)
	templateToolPrefix = strings.TrimSpace(templateToolPrefix)
	tmpl, ok := ToolTemplate(model)
	if !ok {
		return nil
	}

	var state State
	if templateToolPrefix == "" {
		state = GreedyToolNoPrefix
	} else {
		state = GreedyToolWithPrefix
	}
	fmt.Println("state", state)
	return &ToolParser{
		tmpl:       tmpl,
		sb:         &strings.Builder{},
		toolPrefix: templateToolPrefix,
		state:      state,
	}
}
