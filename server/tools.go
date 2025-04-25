package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	gotmpl "text/template"

	"github.com/ollama/ollama/api"
)

func parseObjects(s string) []map[string]any {
	var objs []map[string]any
	for offset := 0; offset < len(s); {
		var obj map[string]any
		decoder := json.NewDecoder(strings.NewReader(s[offset:]))
		err := decoder.Decode(&obj)
		switch {
		case errors.Is(err, io.EOF), errors.Is(err, io.ErrUnexpectedEOF):
			return objs
		case err != nil:
			var syntax *json.SyntaxError
			var unmarshalType *json.UnmarshalTypeError
			switch {
			case errors.As(err, &syntax):
				offset += int(syntax.Offset)
				continue
			case errors.As(err, &unmarshalType):
				offset += int(unmarshalType.Offset)
				continue
			default:
				return nil
			}
		}
		offset += int(decoder.InputOffset())
		objs = append(objs, obj)
	}
	return objs
}

// parseJSONToolCalls attempts to parse a JSON string into a slice of ToolCalls.
// Returns parsed tool calls and a boolean indicating if the JSON is incomplete
func parseJSONToolCalls(tmpl *gotmpl.Template, s string) ([]api.ToolCall, bool) {
	var b bytes.Buffer
	if err := tmpl.Execute(&b, map[string][]api.ToolCall{
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
		return nil, false
	}

	templateObjects := parseObjects(b.String())
	if len(templateObjects) == 0 {
		return nil, false
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
		return nil, false
	}

	responseObjects := parseObjects(s)
	if len(responseObjects) == 0 {
		return nil, false
	}

	// collect all nested objects
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

	var objs []map[string]any
	for _, p := range responseObjects {
		objs = append(objs, collect(p)...)
	}

	var toolCalls []api.ToolCall
	for _, kv := range objs {
		n, nok := kv[name].(string)
		a, aok := kv[arguments].(map[string]any)
		if nok && aok {
			toolCalls = append(toolCalls, api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      n,
					Arguments: a,
				},
			})
		}
	}

	return toolCalls, len(toolCalls) > 0
}

// routeToolParsing is a helper function that routes what kind of tool parsing to use
func routeToolParsing(s string, tmpl *gotmpl.Template) ([]api.ToolCall, bool, bool) {
	if strings.HasPrefix(s, "[{") || strings.HasPrefix(s, "```") || strings.HasPrefix(s, "{") {
		if toolCalls, ok := parseJSONToolCalls(tmpl, s); ok {
			return toolCalls, false, true
		}
		// in the case the JSON never finishes, the acuumulated content should be sent downstream
		return nil, true, true
	}
	// TODO(parthsareen): add python tool call support
	return nil, false, false
}

// ParseToolCalls extracts tool calls from a string using a tool token prefix or direct JSON parsing.
// Returns tool calls, whether parsing is incomplete, and any errors.
func ParseToolCalls(s string, toolToken string, tmpl *gotmpl.Template) ([]api.ToolCall, bool, error) {
	if tmpl == nil {
		return nil, false, fmt.Errorf("no template provided")
	}
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return nil, false, fmt.Errorf("empty input string")
	}
	if toolToken != "" {
		if strings.HasPrefix(s, toolToken) {
			s = strings.TrimSpace(s[len(toolToken):])
			tc, _, ok := routeToolParsing(s, tmpl)
			if len(tc) == 0 || !ok {
				return nil, true, nil
			}
			return tc, false, nil
			// Special token end case
		} else if strings.HasSuffix(s, toolToken[2:]) {
			tc := api.ToolCall{
				Function: api.ToolCallFunction{
					Name: toolToken,
				},
			}
			return []api.ToolCall{tc}, true, nil
		}
	}

	tc, partial, ok := routeToolParsing(s, tmpl)
	if !ok {
		return nil, false, fmt.Errorf("failed to parse tool calls for input: %q", s)
	}
	return tc, partial, nil
}
