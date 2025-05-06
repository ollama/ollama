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
	NoTool State = iota
	PartialTool
	ToolCall
)

type ToolParser struct {
	tmpl       *gotmpl.Template
	state      State
	sb         *strings.Builder
	toolPrefix string
	done       bool
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

	// slog.Debug("template", "template", b.String())

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
	// fmt.Println("template objects", templateObjects)

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
	return toolCalls, len(toolCalls) > 0, true
}

// ParseToolCalls extracts tool calls from a string using a tool token prefix or direct JSON parsing.
// Returns tool calls, whether parsing is incomplete, and any errors.
func (p *ToolParser) ParseToolCalls(s string) ([]api.ToolCall, bool) {
	p.sb.WriteString(s)
	s = p.sb.String()
	s = strings.TrimSpace(s)
	slog.Debug("parse tool calls", "content", s)

	if len(s) == 0 {
		return nil, false
	}
	hasPrefix := false
	if p.toolPrefix != "" {
		if strings.HasPrefix(s, p.toolPrefix) {
			s = strings.TrimSpace(s[len(p.toolPrefix):])
			slog.Debug("tool prefix", "prefix", p.toolPrefix, "content", s)
			p.state = PartialTool
			hasPrefix = true
			// Special token end case
		} else if strings.HasSuffix(s, p.toolPrefix[2:]) {
			p.state = PartialTool
			p.sb.Reset()
			slog.Debug("setting to no tool", "content", s)
			return nil, false
		}
	}
	tcs, partial, ok := p.parseJSONToolCalls(s)

	//  TODO: figure out how to return the remaining string if not partial anymore
	// update state
	switch {
	case !ok && !partial && hasPrefix:
		p.state = PartialTool
	case !ok && !partial:
		p.state = NoTool
	case !ok && partial:
		p.state = PartialTool
	case len(tcs) > 0:
		p.state = ToolCall
	}

	if p.state == NoTool || p.state == ToolCall {
		slog.Debug("resetting string builder", "state", p.state)
		p.sb.Reset()
	}

	if !ok {
		return nil, false
	}

	slog.Debug("returning tool calls", "tool calls", tcs)
	fmt.Println("end state", p.state)
	if p.toolPrefix == "" {
		p.done = true
	}

	fmt.Println("len tcs", len(tcs))
	return tcs, true
}

func NewToolParser(model *Model) *ToolParser {
	templateToolPrefix, _ := ToolPrefix(model.Template.Template)
	slog.Debug("tool prefix", "prefix", templateToolPrefix)
	tmpl, ok := ToolTemplate(model)
	if !ok {
		return nil
	}

	return &ToolParser{
		tmpl:       tmpl,
		sb:         &strings.Builder{},
		toolPrefix: templateToolPrefix,
		done:       false,
	}
}
