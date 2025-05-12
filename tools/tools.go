package tools

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"strings"
	gotmpl "text/template"
	"text/template/parse"

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
	ContainsPrefix
	PartialPrefix
	NotPartialPrefix
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
	case PartialPrefix:
		return "PossiblePrefix"
	case Done:
		return "Done"
	case ContainsPrefix:
		return "PartialPrefix"
	default:
		return fmt.Sprintf("Unknown State (%d)", s)
	}
}

// TODO: simplify if possible
type Parser struct {
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
func (p *Parser) parseJSONToolCalls(s string) ([]api.ToolCall, bool, bool) {
	fmt.Printf("attempting to parse JSON tool calls: input=%s\n", s)

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
		fmt.Printf("failed to execute template: error=%v\n", err)
		return nil, false, false
	}

	// this can be either a map or an array
	var temp any
	err := jsonv2.Unmarshal(b.Bytes(), &temp)
	if err != nil {
		fmt.Printf("failed to unmarshal template: error=%v\n", err)
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
			fmt.Printf("collect encountered unknown type: type=%T\n", obj)
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
		fmt.Println("no template objects found")
		return nil, false, false
	}

	// find the keys that correspond to the name and arguments fields
	var name, arguments string
	for k, v := range templateObjects[0] {
		switch v.(type) {
		case string:
			name = k
			fmt.Printf("found name field: key=%s\n", k)
		case map[string]any:
			arguments = k
			fmt.Printf("found arguments field: key=%s\n", k)
		}
	}

	if name == "" || arguments == "" {
		fmt.Printf("missing required fields: name_found=%v arguments_found=%v\n", name != "", arguments != "")
		return nil, false, false
	}

	// TODO: there is probably some underlying repeat work here to avoid
	// This incrementally decodes the JSON string and returns the first parsedobject
	dec := jsontext.NewDecoder(strings.NewReader(s))
	if got, err := dec.ReadValue(); err == nil {
		s = got.String()
		fmt.Printf("decoded JSON value: value=%s\n", s)
	}

	var responseObjects any
	err = jsonv2.Unmarshal([]byte(s), &responseObjects)
	if err != nil {
		if errors.Is(err, io.ErrUnexpectedEOF) || err.Error() == "unexpected end of JSON input" {
			fmt.Println("incomplete JSON detected")
			return nil, true, false
		} else {
			fmt.Printf("failed to unmarshal response: error=%v\n", err)
			return nil, false, false
		}
	}

	var objs []map[string]any
	objs = append(objs, collect(responseObjects)...)
	if len(objs) == 0 {
		return nil, false, false
	}

	fmt.Printf("collected objects: count=%d\n", len(objs))

	var toolCalls []api.ToolCall
	for _, kv := range objs {
		n, nok := kv[name].(string)
		a, aok := kv[arguments].(map[string]any)
		if nok && aok {
			fmt.Printf("found valid tool call: name=%s\n", n)
			toolCalls = append(toolCalls, api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      n,
					Arguments: a,
				},
			})
		}
	}

	fmt.Printf("parsed tool calls: count=%d\n", len(toolCalls))
	return toolCalls, false, true
}

// TODO: clean up the boundary of internal and external state transitions
func (p *Parser) updateStateAfterJSONParse(ok bool, partial bool, tcs []api.ToolCall) {
	fmt.Printf("updating output state: ok=%v partial=%v tool_calls=%d current_state=%s\n", ok, partial, len(tcs), p.state)

	// state transition logic
	switch {
	case !ok && !partial && p.state == ForceTools:
		// force partial tool if we have a prefix
		// no op and stay in force tools
		p.sb.Reset()
	case !ok && !partial:
		if p.state == GreedyToolNoPrefix {
			p.state = Done
			// ? the output parser state is the same even though internal can we not leak the external state?
			p.Done = true
		}
		if p.state == GreedyToolWithPrefix {
			p.state = SendTokens
		}
		if p.state == PartialPrefix {
			p.state = NotPartialPrefix
		}
	case !ok && partial:
		// acucumulate

	case len(tcs) > 0:
		// do not parse again in the greedy JSON case as soon as we have a tool call
		p.sb.Reset()
	}
	p.updateExternalState(tcs)
	fmt.Printf("state updated: new_state=%s parser_state=%s\n", p.state, p.ParserState)
}

func (p *Parser) updateExternalState(tcs []api.ToolCall) {
	fmt.Printf("updating external state: current_state=%s tool_calls=%d\n", p.state, len(tcs))

	switch {
	case len(tcs) > 0:
		// do not parse again in the greedy JSON case as soon as we have a tool call
		if p.state == GreedyToolWithPrefix {
			p.state = SendTokens
		} else if p.state == GreedyToolNoPrefix {
			p.state = Done
			p.Done = true
		}
		p.ParserState = ToolCallFound
	case p.state == GreedyToolWithPrefix || p.state == GreedyToolNoPrefix ||
		p.state == ToolSuffix || p.state == PartialPrefix ||
		(p.state == ForceTools && len(tcs) == 0):
		p.ParserState = ToolCallAccumulate
	case p.state == ContainsPrefix:
		p.ParserState = ToolCallSendPartial
	case p.state == SendTokens || p.state == Done:
		p.ParserState = ToolCallSendTokens
	case p.state == NotPartialPrefix:
		p.ParserState = ToolCallSendPartial
	default:
		p.ParserState = ToolCallSendTokens
		p.sb.Reset()
		p.state = SendTokens
	}
}

// string, and if it has a prefix
func (p *Parser) checkPrefix(s string) (string, bool) {
	fmt.Printf("checking prefix: input=%s prefix=%s\n", s, p.toolPrefix)

	if p.toolPrefix == "" {
		return s, true
	}
	original := s
	s, hasPrefix := strings.CutPrefix(s, p.toolPrefix)
	if hasPrefix {
		p.state = ForceTools
		fmt.Printf("found exact prefix match: remaining=%s\n", s)
		// partial tool possibly - accumulate
	} else if suffixOverlap(s, p.toolPrefix) > 0 {
		p.state = PartialPrefix
		fmt.Printf("found partial prefix: remaining=%s\n", s)
		return "", false
		// the case where "token<tool_call>" - send "token" back
		// accounts for spaces in prefix or suffix to avoid breaking cache
	} else if strings.Contains(original, p.toolPrefix) {
		idx := strings.Index(original, p.toolPrefix)
		if idx != -1 {
			// still keeps the prefix
			p.state = ContainsPrefix
			p.sb.Reset()
			// todo: see if there is a simpler way for this
			idx2 := strings.Index(s, p.toolPrefix)
			// buffer now only has the prefix
			p.sb.WriteString(s[idx2:])
			fmt.Printf("found prefix in middle: prefix_start=%d content_before=%s\n", idx, original[:idx])
			return original[:idx], false
		}
	}

	return s, true
}

// TODO: simplify the flow of this function
// ParseToolCalls extracts tool calls from a string using a tool token prefix or direct JSON parsing.
// Returns tool calls, whether parsing is incomplete, and any errors.
func (p *Parser) ParseToolCalls(s string) ([]api.ToolCall, string) {
	fmt.Printf("parsing tool calls: input=%s current_state=%s\n", s, p.state)

	p.sb.WriteString(s)
	s = p.sb.String()

	s = strings.TrimSpace(s)

	if len(s) == 0 {
		p.updateExternalState(nil)
		return nil, ""
	}

	s, cont := p.checkPrefix(s)
	if !cont {
		p.updateExternalState(nil)
		if p.state == ContainsPrefix {
			fmt.Printf("returning partial prefix: remaining=%s\n", s)
			return nil, s
		}
		// * we'd be returning here for just accumulating with possible prefix
		// * ext state is accumulation
		return nil, ""
	}
	// * lets say the check fails here and now we're still in external state accumulation here

	// stay in SendTokens unless we have a prefix
	if p.state == SendTokens {
		p.updateExternalState(nil)
		p.sb.Reset()
		fmt.Printf("returning send tokens: remaining=%s\n", s)
		return nil, s
	}

	// * we'd parse here as json to see if it's a tool call
	tcs, partial, ok := p.parseJSONToolCalls(s)
	// * it would not be a tool call here
	p.updateStateAfterJSONParse(ok, partial, tcs)
	if !ok {
		// * and so we should send the data here
		// * we also need to move out of that internal state after sending the tokens
		if p.state == NotPartialPrefix {
			p.state = SendTokens
			// the string would have acc until here
			return nil, p.sb.String()
		}
		return nil, ""
	}
	for _, tc := range tcs {
		tc.Function.Index = p.toolIndex
		p.toolIndex++
	}
	fmt.Printf("finished parsing tool calls: tool_calls_found=%d\n", len(tcs))
	return tcs, ""
}

func suffixOverlap(s, delim string) int {
	max := min(len(delim), len(s))
	for i := max; i > 0; i-- {
		if strings.HasSuffix(s, delim[:i]) {
			return i
		}
	}
	return 0
}

// extractToolCallsTemplate finds the immediate following text after any IfNode containing ".ToolCalls"
func extractToolCallsTemplate(tmpl *gotmpl.Template) (string, bool) {
	if tmpl == nil || tmpl.Tree == nil {
		slog.Debug("TextAfterToolCalls: template or tree is nil")
		return "", false
	}

	var result string
	var found bool

	var walk func(nodes []parse.Node)
	walk = func(nodes []parse.Node) {
		for _, node := range nodes {
			if found {
				return
			}

			switch n := node.(type) {
			case *parse.IfNode:
				if nodeContainsToolCalls(n) {
					// Collect immediate TextNode(s) at start of IfNode's list
					var sb strings.Builder
					for _, innerNode := range n.List.Nodes {
						if tn, ok := innerNode.(*parse.TextNode); ok {
							sb.Write(tn.Text)
						} else {
							// Stop at first non-text node
							break
						}
					}
					result = sb.String()
					found = true
					return
				}
				// Recurse into child nodes
				walk(n.List.Nodes)
				if n.ElseList != nil {
					walk(n.ElseList.Nodes)
				}
			case *parse.ListNode:
				walk(n.Nodes)
			case *parse.RangeNode:
				walk(n.List.Nodes)
				if n.ElseList != nil {
					walk(n.ElseList.Nodes)
				}
			case *parse.WithNode:
				walk(n.List.Nodes)
				if n.ElseList != nil {
					walk(n.ElseList.Nodes)
				}
			default:
				// Continue to next node
				continue
			}

			if found {
				return
			}
		}
	}

	walk(tmpl.Tree.Root.Nodes)
	return result, found
}

// Helper to detect if a node's condition includes ".ToolCalls"
func nodeContainsToolCalls(n *parse.IfNode) bool {
	for _, cmd := range n.Pipe.Cmds {
		for _, arg := range cmd.Args {
			if field, ok := arg.(*parse.FieldNode); ok {
				if slices.Contains(field.Ident, "ToolCalls") {
					return true
				}
			}
		}
	}
	return false
}

func ToolPrefix(tmpl *gotmpl.Template) (string, bool) {
	tokenText, ok := extractToolCallsTemplate(tmpl)
	if !ok {
		return "", false
	}
	tokenText = strings.TrimSpace(tokenText)
	if tokenText == "" {
		return "", false
	}
	first := strings.Fields(tokenText)[0]

	start := -1
	end := -1
	for i, r := range tokenText {
		if r == '<' || r == '[' {
			start = i
		}
		if (r == '>' || r == ']') && start != -1 {
			end = i
			break
		}
	}
	if start != -1 && end != -1 {
		// return the token including the [ or < and the ] or >
		return tokenText[start : end+1], true
	} else if start != -1 {
		// get until the [ or < - in the case tag was not closed
		return tokenText[:start], true
	} else if end != -1 {
		// get after the ] or > - in the case tag was not opened
		return tokenText[end+1:], true
	}
	return first, true
}

func NewParser(tmpl *gotmpl.Template, toolTemplate *gotmpl.Template) *Parser {
	// TODO: use new template parsing to get all tokens for the prefix
	if tmpl == nil {
		return nil
	}
	if toolTemplate == nil {
		return nil
	}

	prefix, _ := ToolPrefix(tmpl)
	prefix = strings.TrimSpace(prefix)

	var state State
	if prefix == "" {
		state = GreedyToolNoPrefix
	} else {
		state = GreedyToolWithPrefix
	}
	fmt.Printf("creating new tool parser: prefix=%s initial_state=%s\n", prefix, state)
	return &Parser{
		tmpl:        toolTemplate,
		sb:          &strings.Builder{},
		toolPrefix:  prefix,
		state:       state,
		ParserState: ToolCallAccumulate,
	}
}
