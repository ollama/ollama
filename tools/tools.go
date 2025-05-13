package tools

import (
	"errors"
	"io"
	"log/slog"
	"strings"
	gotmpl "text/template"

	jsonv2 "github.com/go-json-experiment/json"
	jsontext "github.com/go-json-experiment/json/jsontext"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

type Parser struct {
	greedyParse   bool
	prefixFound   bool
	prefixPartial bool
	tmpl          *gotmpl.Template
	sb            *strings.Builder
	prefix        string
	index         int
	name          string
	arguments     string
	Done          bool
}

// parseJSONToolCalls attempts to parse a JSON string into a slice of ToolCalls.
// It first tries to incrementally decode the JSON to handle partial inputs.
// Returns:
//   - []api.ToolCall: The parsed tool calls if successful
//   - bool: True if JSON is incomplete and needs more input
func (p *Parser) parseJSONToolCalls(s string) ([]api.ToolCall, bool) {
	// First try incremental decoding to handle partial JSON
	dec := jsontext.NewDecoder(strings.NewReader(s))
	if got, err := dec.ReadValue(); err == nil {
		s = got.String()
	}

	// Attempt full unmarshal of the JSON
	var resp any
	err := jsonv2.Unmarshal([]byte(s), &resp)
	if err != nil {
		// Handle incomplete JSON cases
		if errors.Is(err, io.ErrUnexpectedEOF) || err.Error() == "unexpected end of JSON input" {
			slog.Debug("incomplete JSON detected", "input", s)
			return nil, true
		}
		slog.Debug("failed to unmarshal response", "error", err)
		return nil, false
	}

	// Collect all nested objects that could contain tool calls
	var objs []map[string]any
	objs = append(objs, collect(resp)...)
	if len(objs) == 0 {
		return nil, false
	}

	var toolCalls []api.ToolCall
	for _, kv := range objs {
		n, nok := kv[p.name].(string)
		a, aok := kv[p.arguments].(map[string]any)
		if nok && aok {
			toolCalls = append(toolCalls, api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      n,
					Arguments: a,
				},
			})
		}
	}

	// Valid JSON, no tool calls found
	if len(toolCalls) == 0 {
		return nil, false
	}

	return toolCalls, false
}

// checkPrefix processes a string to find and handle a prefix pattern.
//
// Returns:
//   - The processed string with prefix removed if found
//   - Whether the prefix was found at the start of the string
//   - Whether to continue parsing
func (p *Parser) checkPrefix(s string) (string, bool, bool) {
	// Keep original for overlap checks
	original := s
	s = strings.TrimSpace(s)
	if s == "" {
		return "", false, true
	}

	// If no prefix defined, just return trimmed string
	if p.prefix == "" {
		return s, false, true
	}

	// Check for prefix at start of string
	if processedStr, hasPrefix := strings.CutPrefix(s, p.prefix); hasPrefix {
		// Found prefix at start - accumulate for potential tool
		return processedStr, true, true
	}

	// Check if prefix overlaps end of string
	if overlap := suffixOverlap(original, p.prefix); overlap > 0 {
		p.prefixPartial = true
		// Return everything except overlapping portion
		p.sb.Reset()
		p.sb.WriteString(original[len(original)-overlap:])
		return original[0 : len(original)-overlap], false, false
	}

	// Check if prefix appears in middle of string
	if idx := strings.Index(original, p.prefix); idx != -1 {
		p.prefixPartial = true
		// Save remainder starting at prefix for next pass
		p.sb.Reset()
		p.sb.WriteString(strings.TrimSpace(original[idx:]))
		// Return everything before prefix
		return original[:idx], false, false
	}

	// No prefix found
	p.prefixPartial = false
	return s, false, true
}

// Add processes a string input to parse tool calls and content.
// It handles prefix detection and JSON parsing to extract tool calls.
//
// Returns:
//   - tools: Any parsed tool calls
//   - content: Non-tool call content
//   - err: Error if parsing failed
func (p *Parser) Add(s string) (tools []api.ToolCall, content string, err error) {
	if len(s) == 0 {
		return nil, "", nil
	}

	p.sb.WriteString(s)
	s = p.sb.String()

	// Check for prefix pattern in input
	s, prefixFound, shouldContinue := p.checkPrefix(s)
	if !shouldContinue {
		if s != "" {
			// Return content before prefix
			return nil, s, nil
		}
		// Need more input to complete prefix
		return nil, "", nil
	}

	// Update prefix found state
	if prefixFound {
		p.prefixFound = true
	}

	// Exit if prefix exists in template, greedy parsing is off, and prefix not found
	if !p.greedyParse && !p.prefixFound {
		p.sb.Reset()
		return nil, "", errors.New("prefix not found")
	}

	toolCalls, isPartial := p.parseJSONToolCalls(s)
	if isPartial {
		// Need more input to complete JSON
		return nil, "", nil
	}

	// Do not try greedy parsing if partial JSON not found
	p.greedyParse = false

	// Handle invalid tool call format
	if len(toolCalls) == 0 {
		p.sb.Reset()
		if p.prefix == "" {
			p.Done = true
		}
		if p.prefixFound {
			// Drop tokens since prefix was found
			return nil, "", nil
		}
		return nil, s, nil
	}

	for _, tc := range toolCalls {
		tc.Function.Index = p.index
		p.index++
	}

	// Mark as done if no prefix needed
	if p.prefix == "" {
		p.Done = true
	}

	p.sb.Reset()
	return toolCalls, "", nil
}

// NewParser creates a new tool call parser from a template. It extracts the tool call format,
// prefix, and field names from the template to use for parsing tool calls from model output.
//
// Returns an error if the template does not contain valid tool call formatting.
func NewParser(templateToProcess *gotmpl.Template) (*Parser, error) {
	parsed, err := template.Parse(templateToProcess.Root.String())
	if err != nil {
		return nil, err
	}
	if parsed == nil {
		return nil, errors.New("failed to parse template")
	}

	tt, tc := toolTemplate(parsed)
	if !tc {
		return nil, errors.New("failed to find tool calls in template")
	}
	if tt == nil {
		return nil, errors.New("failed to find tool template")
	}

	tp := toolPrefix(templateToProcess)
	tp = strings.TrimSpace(tp)

	name, arguments, err := extractToolArgs(tt)
	if err != nil {
		return nil, err
	}

	return &Parser{
		tmpl:        tt,
		sb:          &strings.Builder{},
		prefix:      tp,
		greedyParse: true,
		name:        name,
		arguments:   arguments,
	}, nil
}
