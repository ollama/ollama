package tools

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
	"github.com/ollama/ollama/template"
)

// TODO: simplify if possible
type Parser struct {
	greedy        bool
	prefixFound   bool
	partialPrefix bool
	tmpl          *gotmpl.Template
	sb            *strings.Builder
	prefix        string
	index         int
	Done          bool
}

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

// prefix stripped string if any, prefix found, and if we should accumulate
func (p *Parser) checkPrefix(s string) (string, bool, bool) {

	if p.prefix == "" {
		return s, false, true
	}
	original := s
	s = strings.TrimSpace(s)
	s, hasPrefix := strings.CutPrefix(s, p.prefix)
	if hasPrefix {
		// partial tool possibly - accumulate
		return s, true, true
	} else if overlap := suffixOverlap(original, p.prefix); overlap > 0 {
		// p.state = PartialPrefix
		p.partialPrefix = true
		return original[0 : len(original)-overlap], false, false
	} else if idx := strings.Index(original, p.prefix); idx != -1 {
		// Found prefix in middle of string, keep only content before prefix
		// accounts for spaces in prefix or suffix to avoid breaking cache
		p.partialPrefix = true
		p.sb.Reset()

		p.sb.WriteString(strings.TrimSpace(original[idx:]))
		return original[:idx], false, false
	}

	p.partialPrefix = false
	return s, false, true
}

func (p *Parser) Add(s string) (tools []api.ToolCall, content string, err error) {
	slog.Debug("adding tool calls", "input", s)

	p.sb.WriteString(s)
	s = p.sb.String()

	if len(s) == 0 {
		return nil, "", nil
	}

	s, prefixFound, cont := p.checkPrefix(s)

	if !cont {
		if s != "" {
			// send only the content back, prefix exists
			return nil, s, nil
		}
		// accumulate case
		return nil, "", nil
	}

	// circuit breaker
	if prefixFound {
		p.prefixFound = true
	}

	// for cases with a prefix in template
	if p.prefix != "" && !p.greedy && !p.prefixFound {
		// send tokens down
		p.sb.Reset()
		return nil, "", errors.New("prefix not found")
	}
	// we have a prefix or are in json mode
	tcs, partial, ok := p.parseJSONToolCalls(s)
	if partial {
		// accumulate case
		return nil, "", nil
	}

	p.greedy = false
	if !ok {
		// will not be a partial at this point
		p.sb.Reset()
		// send tokens
		if p.prefix == "" {
			p.Done = true
		}
		if p.prefixFound {
			// drop tokens instead - sb is reset, no tokens sent to user
			return nil, "", nil
		}
		return nil, "", errors.New("failed to parse tool calls")
	}

	for _, tc := range tcs {
		tc.Function.Index = p.index
		p.index++
	}
	if p.prefix == "" {
		p.Done = true
	}
	p.sb.Reset()
	return tcs, "", nil
}

func NewParser(templateToProcess *gotmpl.Template) (*Parser, error) {
	parsedTemplate, err := template.Parse(templateToProcess.Root.String())
	if err != nil {
		return nil, err
	}
	if parsedTemplate == nil {
		return nil, errors.New("failed to parse template")
	}

	toolCallTemplate, hasToolCalls := toolTemplate(parsedTemplate)
	if !hasToolCalls {
		return nil, errors.New("failed to find tool template")
	}
	if toolCallTemplate == nil {
		return nil, errors.New("failed to find tool template")
	}

	toolPrefix, _ := ToolPrefix(templateToProcess)
	toolPrefix = strings.TrimSpace(toolPrefix)

	fmt.Printf("creating new tool parser: prefix=%s\n", toolPrefix)
	return &Parser{
		tmpl:   toolCallTemplate,
		sb:     &strings.Builder{},
		prefix: toolPrefix,
		greedy: true,
	}, nil
}
