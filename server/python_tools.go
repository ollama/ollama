package server

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
)

var (
	pythonFuncRegex = regexp.MustCompile(`(\w+)\((.*?)\)`)
	braces          = map[rune]rune{
		'[':  ']',
		'{':  '}',
		'(':  ')',
		'"':  '"',
		'\'': '\'',
	}
)

// parsePythonValue converts a Python value string to its appropriate Go type
func parsePythonValue(value string) (any, error) {
	value = strings.TrimSpace(value)

	// string
	if (strings.HasPrefix(value, "\"") && strings.HasSuffix(value, "\"")) ||
		(strings.HasPrefix(value, "'") && strings.HasSuffix(value, "'")) {
		// Remove quotes
		result := value[1 : len(value)-1]
		return result, nil
	}

	// bool
	switch strings.ToLower(value) {
	case "true":
		return true, nil
	case "false":
		return false, nil
	case "none":
		return nil, nil
	}

	// int
	if i, err := strconv.Atoi(value); err == nil {
		return i, nil
	}

	// float
	if f, err := strconv.ParseFloat(value, 64); err == nil {
		return f, nil
	}

	// list
	if strings.HasPrefix(value, "[") && strings.HasSuffix(value, "]") {
		listStr := value[1 : len(value)-1]
		var list []any
		stack := []rune{}
		start := 0

		for i, char := range listStr {
			if len(stack) != 0 && char == braces[stack[len(stack)-1]] {
				stack = stack[:len(stack)-1]
			} else if _, ok := braces[char]; ok {
				stack = append(stack, char)
			}

			if len(stack) == 0 && (char == ',' || i == len(listStr)-1) {
				end := i
				if i == len(listStr)-1 {
					end = i + 1
				}
				item := strings.TrimSpace(listStr[start:end])
				if val, err := parsePythonValue(item); err == nil {
					list = append(list, val)
				} else {
					return nil, fmt.Errorf("invalid list item: %s", item)
				}
				start = i + 1
			}
		}
		return list, nil
	}

	// dictionary
	if strings.HasPrefix(value, "{") && strings.HasSuffix(value, "}") && strings.Contains(value, ":") {
		dictStr := value[1 : len(value)-1]
		dict := make(map[any]any)
		stack := []rune{}
		start := 0
		for i, char := range dictStr {
			if len(stack) != 0 && char == braces[stack[len(stack)-1]] {
				stack = stack[:len(stack)-1]
			} else if _, ok := braces[char]; ok {
				stack = append(stack, char)
			}
			if len(stack) == 0 && (char == ',' || i == len(dictStr)-1) {
				end := i
				if i == len(dictStr)-1 {
					end = i + 1
				}
				item := strings.TrimSpace(dictStr[start:end])
				kv := strings.SplitN(item, ":", 2)
				if len(kv) != 2 {
					return nil, fmt.Errorf("invalid dictionary key-value pair: %s", item)
				}

				key, err := parsePythonValue(strings.TrimSpace(kv[0]))
				if err != nil {
					return nil, fmt.Errorf("invalid dictionary key: %s", kv[0])
				}

				val, err := parsePythonValue(strings.TrimSpace(kv[1]))
				if err != nil {
					return nil, fmt.Errorf("invalid dictionary value: %s", kv[1])
				}

				dict[key] = val
				start = i + 1
			}
		}
		return dict, nil
	}

	// sets (stored as lists)
	if strings.HasPrefix(value, "{") && strings.HasSuffix(value, "}") {
		setStr := value[1 : len(value)-1]
		var list []any
		stack := []rune{}
		start := 0
		for i, char := range setStr {
			if len(stack) != 0 && char == braces[stack[len(stack)-1]] {
				stack = stack[:len(stack)-1]
			} else if _, ok := braces[char]; ok {
				stack = append(stack, char)
			}
			if len(stack) == 0 && (char == ',' || i == len(setStr)-1) {
				end := i
				if i == len(setStr)-1 {
					end = i + 1
				}
				item := strings.TrimSpace(setStr[start:end])
				if val, err := parsePythonValue(item); err == nil {
					list = append(list, val)
				} else {
					return nil, fmt.Errorf("invalid set item: %s", item)
				}
				start = i + 1
			}
		}
		return list, nil
	}

	return nil, fmt.Errorf("invalid Python value: %s", value)
}

// parsePythonFunctionCall parses Python function calls from a string
// it supports keyword arguments, as well as multiple functions in a single string
func parsePythonFunctionCall(s string) ([]api.ToolCall, error) {
	matches := pythonFuncRegex.FindAllStringSubmatchIndex(s, -1)
	if len(matches) == 0 {
		return nil, fmt.Errorf("no Python function calls found")
	}

	var toolCalls []api.ToolCall
	for _, match := range matches {
		name := s[match[2]:match[3]]
		args := s[match[4]:match[5]]
		arguments := make(api.ToolCallFunctionArguments)
		if len(args) == 0 {
			toolCalls = append(toolCalls, api.ToolCall{
				Function: api.ToolCallFunction{
					Name: name,
				},
			})
			continue
		}

		start := 0
		stack := []rune{}
		for i, char := range args {
			if len(stack) != 0 && char == braces[stack[len(stack)-1]] {
				stack = stack[:len(stack)-1]
			} else if _, ok := braces[char]; ok {
				stack = append(stack, char)
			}
			if len(stack) == 0 && (char == ',' || i == len(args)-1) {
				end := i
				if i == len(args)-1 {
					end = i + 1
				}
				kv := strings.SplitN(args[start:end], "=", 2)
				if len(kv) == 2 {
					key := strings.TrimSpace(kv[0])
					valueStr := strings.TrimSpace(kv[1])

					// Parse the value into appropriate type
					value, err := parsePythonValue(valueStr)
					if err != nil {
						return nil, fmt.Errorf("failed to parse value for key %q: %v", key, err)
					}

					arguments[key] = value
				} else {
					return nil, fmt.Errorf("invalid argument format: %q", args[start:end])
				}
				start = i + 1
			}
		}

		if len(arguments) > 0 {
			toolCalls = append(toolCalls, api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      name,
					Arguments: arguments,
				},
			})
		}
	}

	if len(toolCalls) > 0 {
		return toolCalls, nil
	}
	return nil, fmt.Errorf("failed to parse any valid tool calls")
}
