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
	jsontext "github.com/go-json-experiment/json/jsontext"

	"github.com/ollama/ollama/api"
)

type State int

// TODO: potentially coalesce states
const (
	SendTokens State = iota
	GreedyToolWithPrefix
	GreedyToolNoPrefix
	ForceTools
	ToolSuffix
	ContainsPartialPrefix
	Done
)

type ExternalState int

const (
	ToolCallFound ExternalState = iota
	ToolCallSendPartial
	ToolCallAccumulate
	ToolCallSendTokens
)

func (s ExternalState) String() string {
	switch s {
	case ToolCallFound:
		return "ToolCallFound"
	case ToolCallSendPartial:
		return "ToolCallSendPartial"
	case ToolCallAccumulate:
		return "ToolCallAccumulate"
	case ToolCallSendTokens:
		return "ToolCallSendTokens"
	default:
		return fmt.Sprintf("Unknown ExternalState (%d)", s)
	}
}

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

// TODO: simplify if possible
type ToolParser struct {
	tmpl        *gotmpl.Template
	state       State
	sb          *strings.Builder
	toolPrefix  string
	toolIndex   int
	ParserState ExternalState
	Done        bool
}

// ? move to a separate file
// parseJSONToolCalls attempts to parse a JSON string into a slice of ToolCalls.
// Returns parsed tool calls, a boolean indicating if the JSON is incomplete, and a boolean indicating if the tool calls were found
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

	// this can be either a map or an array
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
		default:
			// TODO: err or fallback
			return nil
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

	// TODO: there is probably some underlying repeat work here to avoid
	// This incrementally decodes the JSON string and returns the first parsedobject
	dec := jsontext.NewDecoder(strings.NewReader(s))
	if got, err := dec.ReadValue(); err == nil {
		s = got.String()
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
			fmt.Println("exiting from JSON parsing", p.state)
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
			// p.ParserState = DoneFR
			p.ParserState = ToolCallSendTokens
			p.Done = true
		}
		if p.state == GreedyToolWithPrefix {
			fmt.Println("  Subcase: GreedyToolWithPrefix - switching to SendTokens")
			p.state = SendTokens
			p.ParserState = ToolCallSendTokens
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
			p.ParserState = ToolCallFound
			p.state = Done
			p.Done = true
		} else if p.state == GreedyToolNoPrefix {
			fmt.Println("  Subcase: Greedy modes - marking done and switching to SendTokens")
			p.state = Done
			p.Done = true
		}
		p.sb.Reset()
	}
	p.updateExternalState(tcs)
}

func (p *ToolParser) updateExternalState(tcs []api.ToolCall) {
	if (p.state == GreedyToolWithPrefix || p.state == GreedyToolNoPrefix || p.state == ToolSuffix) || (p.state == ForceTools && len(tcs) == 0) {
		p.ParserState = ToolCallAccumulate
	} else if p.state == ContainsPartialPrefix {
		p.ParserState = ToolCallSendPartial
	} else if len(tcs) > 0 {
		p.ParserState = ToolCallFound
	} else if p.state == SendTokens {
		p.ParserState = ToolCallSendTokens
	}
}

// string, and if it has a prefix
func (p *ToolParser) checkPrefix(s string) (string, bool) {
	if p.toolPrefix == "" {
		return s, true
	}
	original := s
	// s = strings.TrimSpace(s)
	s, hasPrefix := strings.CutPrefix(s, p.toolPrefix)
	if hasPrefix {
		fmt.Println("has prefix", s)
		p.state = ForceTools
		// partial tool possibly
	} else if strings.HasPrefix(p.toolPrefix, s) {
		slog.Debug("tool prefix partially", "prefix", p.toolPrefix, "content", s)
		// TODO: could possibly err maybe this should be greedy instead?
		p.state = ForceTools
		// this would basically be a no op on rest of the input
		return "", false
		// the case where "token<tool_call>" - send "token" back
		// accounts for spaces in prefix or suffix to avoid breaking cache
	} else if strings.Contains(original, p.toolPrefix) {
		idx := strings.Index(original, p.toolPrefix)
		if idx != -1 {
			// still keeps the prefix
			p.state = ContainsPartialPrefix
			p.sb.Reset()
			// todo: see if there is a simpler way for this
			idx2 := strings.Index(s, p.toolPrefix)
			p.sb.WriteString(s[idx2:])
			return original[:idx], false
		}
	}

	return s, true
}

// TODO: simplify the flow of this function
// ParseToolCalls extracts tool calls from a string using a tool token prefix or direct JSON parsing.
// Returns tool calls, whether parsing is incomplete, and any errors.
func (p *ToolParser) ParseToolCalls(s string) ([]api.ToolCall, string) {
	fmt.Println("checking tool calls", s)
	fmt.Println("external state", p.ParserState)
	fmt.Println("internal state", p.state)
	p.sb.WriteString(s)
	s = p.sb.String()

	s = strings.TrimSpace(s)
	fmt.Println("sb", s)

	p.updateExternalState(nil)
	if len(s) == 0 {
		return nil, ""
	}

	s, cont := p.checkPrefix(s)
	if !cont {
		p.updateExternalState(nil)
		if p.state == ContainsPartialPrefix {
			return nil, s
		}
		return nil, ""
	}

	// stay in SendTokens unless we have a prefix
	if p.state == SendTokens {
		fmt.Println("SendTokens - resetting buffer")
		p.updateExternalState(nil)
		p.sb.Reset()
		return nil, ""
	}

	tcs, partial, ok := p.parseJSONToolCalls(s)
	p.updateOutputState(ok, partial, tcs)
	fmt.Println("output state", p.ParserState, p.state)
	if !ok {
		fmt.Println("returning empty tool calls")
		return nil, ""
	}
	for _, tc := range tcs {
		tc.Function.Index = p.toolIndex
		p.toolIndex++
	}
	return tcs, ""
}

func NewToolParser(model *Model) *ToolParser {
	// TODO: use new template parsing to get all tokens for the prefix
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
	fmt.Println("setup state", state)
	return &ToolParser{
		tmpl:        tmpl,
		sb:          &strings.Builder{},
		toolPrefix:  templateToolPrefix,
		state:       state,
		ParserState: ToolCallAccumulate,
	}
}
