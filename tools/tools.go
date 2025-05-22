package tools

import (
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	gotmpl "text/template"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

var (
	errInvalidToolCall = errors.New("invalid tool call format")
	errAccumulateMore  = errors.New("need to accumulate more content")
)

type Parser struct {
	parseLeadingJSON bool
	prefix           string
	prefixFound      bool
	tmpl             gotmpl.Template
	sb               strings.Builder
	index            int
	name             string
	arguments        string
	done             bool
}

// parseJSONToolCalls attempts to parse a JSON string into a slice of ToolCalls.
//
// Parameters:
//   - s: The string to parse
//   - name: The field name from template that identifies the tool call name
//   - arguments: The field name from template that identifies the tool call arguments
//
// Returns:
//   - []api.ToolCall: The parsed tool calls if successful
//   - error: ErrAccumulateMore if braces unbalanced, ErrInvalidToolCall if invalid, or nil if successful
func parseJSONToolCalls(s string, name, arguments string, prefix string) ([]api.ToolCall, error) {
	// Check for balanced braces before attempting to parse
	braceCount := 0
	squareCount := 0
	startIndex := -1
	var rawToolCalls []string

	for i, c := range s {
		switch c {
		case '{':
			braceCount++
			if startIndex == -1 {
				startIndex = i
			}
		case '}':
			braceCount--
			if braceCount == 0 {
				rawToolCalls = append(rawToolCalls, s[startIndex:i+1])
				startIndex = -1
			}
		case '[':
			// Only track these if we don't have a prefix as it will be cut off from the prefix
			if prefix == "" || !strings.HasSuffix(prefix, "[") {
				squareCount++
			}
		case ']':
			// if prefix == "" {
			if prefix == "" || !strings.HasSuffix(prefix, "[") {
				squareCount--
			}
		}

		// Negative means we have an extra closing brace/bracket
		if braceCount < 0 || squareCount < 0 {
			return nil, errInvalidToolCall
		}
	}

	// If braces/brackets aren't balanced, need more input
	if braceCount > 0 || squareCount > 0 {
		return nil, errAccumulateMore
	}

	t := strings.TrimSpace(s)
	if len(t) == 0 {
		return nil, errAccumulateMore
	}
	// If the input is a single square bracket, it's not a valid tool call
	if t[0] == '[' && len(t) == 1 {
		return nil, errAccumulateMore
	}

	// Attempt full unmarshal of the JSON
	var toolCalls []api.ToolCall
	for _, rawToolCall := range rawToolCalls {
		var resp any
		if err := json.Unmarshal([]byte(rawToolCall), &resp); err != nil {
			continue
		}

		// Collect nested objects that could contain tool calls
		objs := collect(resp)
		if len(objs) == 0 {
			continue
		}

		// Extract tool calls from objects
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
			} else {
				slog.Debug("No valid tool call found in object.", "object", kv)
			}
		}
	}

	// Valid JSON, no tool calls found
	if len(toolCalls) == 0 {
		slog.Debug("No valid tool calls found in any raw tool calls.", "rawToolCalls", rawToolCalls)
		return nil, errInvalidToolCall
	}

	return toolCalls, nil
}

// checkPrefix processes a string to find and handle a prefix pattern.
//
// Returns:
//   - The processed string with prefix removed if found
//   - error: ErrAccumulateMore if prefix is incomplete, or nil if successful
func (p *Parser) checkPrefix(s string) (string, error) {
	original := s
	s = strings.ReplaceAll(s, "\n", " ")

	if s == "" || p.prefix == "" {
		return s, nil
	}

	// Check for prefix at start of string
	if cut, hasPrefix := strings.CutPrefix(s, p.prefix); hasPrefix {
		// Found prefix at start - accumulate for potential tool
		p.prefixFound = true
		return cut, nil
	}

	// Check if prefix overlaps end of string
	if idx := suffixOverlap(s, p.prefix); idx != -1 {
		// Return everything except overlapping portion
		p.sb.Reset()
		p.sb.WriteString(s[idx:])
		return original[:idx], errAccumulateMore
	}

	// Check if prefix appears in middle of string
	if idx := strings.Index(s, p.prefix); idx != -1 {
		// Save remainder starting at prefix for next pass
		p.sb.Reset()
		p.sb.WriteString(strings.TrimSpace(s[idx:]))
		// Return everything before prefix
		return original[:idx], errAccumulateMore
	}

	// No partial prefix found
	return s, nil
}

// Add processes a string input to parse tool calls and content.
// It handles prefix detection and JSON parsing to extract tool calls.
//
// Returns:
//   - tools: Any parsed tool calls
//   - content: Non-tool call content
func (p *Parser) Add(s string) (tools []api.ToolCall, content string) {
	if strings.TrimSpace(s) == "" {
		return nil, s
	}
	if p.done {
		if p.index == 0 {
			// Return original string if no tool calls found at start
			return nil, s
		}
		// Return empty if no tool calls found after start
		return nil, ""
	}
	p.sb.WriteString(s)
	s = p.sb.String()

	// Check for prefix pattern in input
	s, err := p.checkPrefix(s)
	if err != nil {
		// Need more input to complete prefix
		return nil, s
	}

	// Exit if prefix exists in template, greedy parsing is off, and prefix not found
	if !p.parseLeadingJSON && !p.prefixFound {
		p.sb.Reset()
		return nil, s
	}

	toolCalls, err := parseJSONToolCalls(s, p.name, p.arguments, p.prefix)
	if err != nil {
		if errors.Is(err, errAccumulateMore) {
			return nil, ""
		}
		p.sb.Reset()
		// Do not try parsing leading JSON if JSON not found
		p.parseLeadingJSON = false
		if p.prefix == "" {
			p.done = true
		}
		if p.index != 0 && p.prefix == "" {
			return nil, ""
		}
		if p.prefixFound {
			// Drop tokens since prefix was found
			return nil, ""
		}
		return nil, s
	}

	for _, tc := range toolCalls {
		tc.Function.Index = p.index
		p.index++
	}

	p.sb.Reset()
	return toolCalls, ""
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

	tt, err := toolTemplate(parsed)
	if err != nil {
		return nil, err
	}

	tp := toolPrefix(templateToProcess)

	name, arguments, err := extractToolArgs(tt)
	if err != nil {
		return nil, err
	}

	return &Parser{
		tmpl:             *tt,
		sb:               strings.Builder{},
		prefix:           tp,
		parseLeadingJSON: true,
		name:             name,
		arguments:        arguments,
	}, nil
}
