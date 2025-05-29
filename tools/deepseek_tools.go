package tools

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	gotmpl "text/template"

	"github.com/ollama/ollama/api"
)

type DeepSeekToolParser struct {
	parser *Parser // Embed the base parser as a field
}

func (p *DeepSeekToolParser) Add(s string) (tools []api.ToolCall, content string) {
	fmt.Println("prefix", p.parser.prefix)
	fmt.Println("DeepSeekToolParser.Add: Starting with input:", s)
	p.parser.sb.WriteString(s)
	s = p.parser.sb.String()
	fmt.Println("DeepSeekToolParser.Add: After StringBuilder:", s)

	// Check for prefix pattern in input
	s, err := p.parser.checkPrefix(s)
	fmt.Println("DeepSeekToolParser.Add: After checkPrefix:", s, "error:", err)
	if err != nil {
		// Need more input to complete prefix
		return nil, s
	}

	// Exit if prefix exists in template, greedy parsing is off, and prefix not found
	if !p.parser.prefixFound {
		fmt.Println("DeepSeekToolParser.Add: Prefix not found, resetting")
		p.parser.sb.Reset()
		return nil, s
	}

	toolCalls, err := parseDeepSeekToolCalls(s)
	fmt.Println("DeepSeekToolParser.Add: After parseDeepSeekToolCalls:", toolCalls, "error:", err)
	if err != nil {
		if errors.Is(err, errAccumulateMore) {
			return nil, ""
		}
		p.parser.sb.Reset()
		// Only do greedy JSON parsing if there is no prefix from template
		if p.parser.prefix != "" {
			fmt.Println("DeepSeekToolParser.Add: Disabling greedy parsing")
			p.parser.greedyParseJSON = false
		}
		if p.parser.index != 0 && p.parser.prefix == "" {
			return nil, ""
		}
		if p.parser.prefixFound {
			fmt.Println("DeepSeekToolParser.Add: Prefix found but invalid tool call")
			// Drop tokens since prefix was found
			return nil, ""
		}
		return nil, s
	}

	fmt.Println("DeepSeekToolParser.Add: Processing tool calls")
	for _, tc := range toolCalls {
		tc.Function.Index = p.parser.index
		p.parser.index++
	}

	p.parser.sb.Reset()
	fmt.Println("DeepSeekToolParser.Add: Returning tool calls:", toolCalls)
	return toolCalls, ""
}

func (p *DeepSeekToolParser) NewParser(templateToProcess *gotmpl.Template) (ToolParser, error) {
	return NewDeepSeekToolParser(templateToProcess)
}

func NewDeepSeekToolParser(templateToProcess *gotmpl.Template) (ToolParser, error) {
	// Create base parser first
	baseParser, err := NewParser(templateToProcess)
	if err != nil {
		return nil, fmt.Errorf("failed to create base parser: %w", err)
	}

	return &DeepSeekToolParser{
		parser: baseParser,
	}, nil
}

func parseDeepSeekToolCalls(s string) ([]api.ToolCall, error) {
	fmt.Println("parseDeepSeekToolCalls: Starting with input:", s)
	fields := strings.Fields(s)
	fmt.Println("parseDeepSeekToolCalls: Split fields:", fields)

	sep := "<｜tool▁sep｜>"
	var functionNames []string
	for _, field := range fields {
		fmt.Println("parseDeepSeekToolCalls: Processing field:", field)
		// TODO: check if brittle
		if strings.Contains(field, "function") {
			idx := strings.Index(field, "function")
			if idx == -1 {
				fmt.Println("parseDeepSeekToolCalls: No 'function' prefix found")
				return nil, errAccumulateMore
			}
			functionName := field[idx+len("function"):]
			// functionName, cut := strings.CutPrefix(field, "function")
			// if !cut {
			// 	fmt.Println("parseDeepSeekToolCalls: Failed to cut 'function' prefix")
			// 	return nil, errAccumulateMore
			// }
			// pass through on this is fine as it doesn't always come down
			functionName, _ = strings.CutPrefix(functionName, sep)
			fmt.Println("parseDeepSeekToolCalls: Found function name:", functionName)
			functionNames = append(functionNames, functionName)
		}
	}

	if len(functionNames) == 0 {
		fmt.Println("parseDeepSeekToolCalls: No function names found")
		return nil, errAccumulateMore
	}
	fmt.Println("parseDeepSeekToolCalls: Found function names:", functionNames)

	braceCount := 0
	startIndex := -1

	var rawToolArgs []string
	for i, c := range s {
		switch c {
		case '{':
			braceCount++
			if startIndex == -1 {
				startIndex = i
				fmt.Printf("parseDeepSeekToolCalls: Found opening brace at index %d\n", i)
			}
		case '}':
			braceCount--
			if braceCount == 0 {
				rawToolArgs = append(rawToolArgs, s[startIndex:i+1])
				fmt.Printf("parseDeepSeekToolCalls: Found closing brace at index %d, captured: %s\n", i, s[startIndex:i+1])
				startIndex = -1
			}
		}
	}
	fmt.Println("parseDeepSeekToolCalls: Raw tool arguments:", rawToolArgs)

	var toolCalls []api.ToolCall
	// unmarshal args
	var args map[string]any
	for i, rawToolArg := range rawToolArgs {
		fmt.Printf("parseDeepSeekToolCalls: Unmarshaling tool arg %d: %s\n", i, rawToolArg)
		if err := json.Unmarshal([]byte(rawToolArg), &args); err != nil {
			fmt.Println("parseDeepSeekToolCalls: Failed to unmarshal JSON:", err)
			return nil, err
		}

		toolCalls = append(toolCalls, api.ToolCall{
			Function: api.ToolCallFunction{
				Name:      functionNames[i],
				Arguments: args,
			},
		})
		fmt.Printf("parseDeepSeekToolCalls: Created tool call %d with name %s and args %v\n", i, functionNames[i], args)
	}

	if len(toolCalls) == 0 {
		fmt.Println("parseDeepSeekToolCalls: No tool calls created")
		// todo: check err here
		return nil, errInvalidToolCall
	}

	fmt.Println("parseDeepSeekToolCalls: Returning tool calls:", toolCalls)
	return toolCalls, nil
}

// ! use as prefix
// {{"<｜tool▁call▁begin｜>
// ! send to tc parser
// * function<｜tool▁sep｜><function_name>\n```json\n<function_arguments_in_json_format>\n```<｜tool▁call▁end｜>"}}
