package tools

import (
	"errors"
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
	prefixFound      bool
	tmpl             gotmpl.Template
	sb               strings.Builder
	prefix           string
	index            int
	name             string
	arguments        string
	done             bool
}

// checkPrefix processes a string to find and handle a prefix pattern.
//
// Returns:
//   - The processed string with prefix removed if found
//   - error: ErrAccumulateMore if prefix is incomplete, or nil if successful
func (p *Parser) checkPrefix(s string) (string, error) {
	if s == "" {
		return "", nil
	}

	// If no prefix defined, just return trimmed string
	if p.prefix == "" {
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
		return s[:idx], errAccumulateMore
	}

	// Check if prefix appears in middle of string
	if idx := strings.Index(s, p.prefix); idx != -1 {
		// Save remainder starting at prefix for next pass
		p.sb.Reset()
		p.sb.WriteString(strings.TrimSpace(s[idx:]))
		// Return everything before prefix
		return s[:idx], errAccumulateMore
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
	if s == "" {
		return nil, ""
	}
	if p.done {
		return nil, s
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

	toolCalls, err := parseJSONToolCalls(s, p.name, p.arguments)
	if err != nil {
		if errors.Is(err, errAccumulateMore) {
			return nil, ""
		} else {
			p.sb.Reset()
			// Do not try parsing leading JSON if JSON not found
			p.parseLeadingJSON = false
			if p.prefix == "" {
				p.done = true
			}
			if p.prefixFound {
				// Drop tokens since prefix was found
				return nil, ""
			}
			return nil, s
		}
	}

	for _, tc := range toolCalls {
		tc.Function.Index = p.index
		p.index++
	}

	// Mark as done if no prefix in template
	if p.prefix == "" {
		p.done = true
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
	tp = strings.TrimSpace(tp)

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
