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

// Sentinel errors for parsing states
var (
	ErrPartialPrefix = errors.New("partial prefix detected")

	ErrPrefixNotFound = errors.New("prefix not found")

	ErrInvalidToolCall = errors.New("invalid tool call format")

	ErrAccumulateMore = errors.New("need to accumulate more content")
)

type Parser struct {
	greedyParse bool
	prefixFound bool
	tmpl        gotmpl.Template
	sb          strings.Builder
	prefix      string
	index       int
	name        string
	arguments   string
	Done        bool
}

// parseJSONToolCalls attempts to parse a JSON string into a slice ToolCalls.
// It first tries to incrementally decode the JSON to handle partial inputs.
// Returns:
//   - []api.ToolCall: The parsed tool calls if successful
//   - error: ErrPartialJSON if JSON is incomplete, ErrInvalidToolCall if invalid, or nil if successful
func (p *Parser) parseJSONToolCalls(s string) ([]api.ToolCall, error) {
	// First try incremental decoding to handle partial JSON
	dec := jsontext.NewDecoder(strings.NewReader(s))
	if got, err := dec.ReadValue(); err == nil {
		s = got.String()
	}

	// Attempt full unmarshal of the JSON
	var resp any
	if err := jsonv2.Unmarshal([]byte(s), &resp); errors.Is(err, io.ErrUnexpectedEOF) {
		slog.Debug("incomplete JSON detected", "input", s)
		return nil, ErrAccumulateMore
	} else if err != nil {
		slog.Debug("failed to unmarshal response", "error", err)
		return nil, ErrInvalidToolCall
	}

	// Collect all nested objects that could contain tool calls
	objs := collect(resp)
	if len(objs) == 0 {
		return nil, ErrInvalidToolCall
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
		return nil, ErrInvalidToolCall
	}

	return toolCalls, nil
}

// checkPrefix processes a string to find and handle a prefix pattern.
//
// Returns:
//   - The processed string with prefix removed if found
//   - error: ErrPartialPrefix if prefix is incomplete, ErrPrefixNotFound if not found, or nil if successful
func (p *Parser) checkPrefix(s string) (string, error) {
	// Keep original for overlap checks
	original := s
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}

	// If no prefix defined, just return trimmed string
	if p.prefix == "" {
		return s, nil
	}

	// Check for prefix at start of string
	if processedStr, hasPrefix := strings.CutPrefix(s, p.prefix); hasPrefix {
		// Found prefix at start - accumulate for potential tool
		p.prefixFound = true
		return processedStr, nil
	}

	// Check if prefix overlaps end of string
	if overlap := suffixOverlap(original, p.prefix); overlap > 0 {
		// Return everything except overlapping portion
		p.sb.Reset()
		p.sb.WriteString(original[len(original)-overlap:])
		return original[0 : len(original)-overlap], ErrAccumulateMore
	}

	// Check if prefix appears in middle of string
	if idx := strings.Index(original, p.prefix); idx != -1 {
		// Save remainder starting at prefix for next pass
		p.sb.Reset()
		p.sb.WriteString(strings.TrimSpace(original[idx:]))
		// Return everything before prefix
		return original[:idx], ErrAccumulateMore
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
//   - error: One of the sentinel errors or nil if successful
func (p *Parser) Add(s string) (tools []api.ToolCall, content string, err error) {
	p.sb.WriteString(s)
	s = p.sb.String()

	// Check for prefix pattern in input
	s, err = p.checkPrefix(s)
	if err != nil {
		if s != "" {
			// Return content before prefix
			return nil, s, nil
		}
		// Need more input to complete prefix
		return nil, "", ErrAccumulateMore
	}

	// Exit if prefix exists in template, greedy parsing is off, and prefix not found
	if !p.greedyParse && !p.prefixFound {
		p.sb.Reset()
		return nil, "", ErrPrefixNotFound
	}

	toolCalls, err := p.parseJSONToolCalls(s)
	if err != nil {
		if errors.Is(err, ErrAccumulateMore) {
			return nil, "", err
		} else {
			p.sb.Reset()
			// Do not try greedy parsing if JSON not found
			p.greedyParse = false
			if p.prefix == "" {
				p.Done = true
			}
			if p.prefixFound {
				// Drop tokens since prefix was found
				return nil, "", ErrAccumulateMore
			}
			return nil, s, nil
		}
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
func NewParser(templateToProcess *gotmpl.Template) (Parser, error) {
	parsed, err := template.Parse(templateToProcess.Root.String())
	if err != nil {
		return Parser{}, err
	}

	tt, err := toolTemplate(parsed)
	if err != nil {
		return Parser{}, err
	}

	tp := toolPrefix(templateToProcess)
	tp = strings.TrimSpace(tp)

	name, arguments, err := extractToolArgs(tt)
	if err != nil {
		return Parser{}, err
	}

	return Parser{
		tmpl:        *tt,
		sb:          strings.Builder{},
		prefix:      tp,
		greedyParse: true,
		name:        name,
		arguments:   arguments,
	}, nil
}
