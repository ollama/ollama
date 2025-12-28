package parsers

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

// MiniMaxM2Parser handles MiniMax-M2's XML-style tool call format:
// <minimax:tool_call>
// <invoke name="function_name">
// <parameter name="key">value</parameter>
// </invoke>
// </minimax:tool_call>
//
// And thinking format:
// <think>thinking content</think>
type MiniMaxM2Parser struct {
	state  MiniMaxM2ParserState
	buffer strings.Builder
	tools  []api.Tool
	err    error // Store critical errors (like unknown tools)
}

type MiniMaxM2ParserState int

const (
	MiniMaxM2CollectingContent MiniMaxM2ParserState = iota
	MiniMaxM2CollectingThinking
	MiniMaxM2CollectingToolCalls
)

const (
	minimaxm2ThinkingOpenTag   = "<think>"
	minimaxm2ThinkingCloseTag  = "</think>"
	minimaxm2ToolCallOpenTag   = "<minimax:tool_call>"
	minimaxm2ToolCallCloseTag  = "</minimax:tool_call>"
	minimaxm2InvokeOpenPrefix  = "<invoke"
	minimaxm2InvokeCloseTag    = "</invoke>"
	minimaxm2ParameterOpenPrefix = "<parameter"
	minimaxm2ParameterCloseTag = "</parameter>"
)

func (p *MiniMaxM2Parser) HasToolSupport() bool {
	return true
}

func (p *MiniMaxM2Parser) HasThinkingSupport() bool {
	return true
}

func (p *MiniMaxM2Parser) setInitialState(lastMessage *api.Message, tools []api.Tool, thinkValue *api.ThinkValue) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	// Check both model capability AND request preference
	thinkingEnabled := thinkValue != nil && thinkValue.Bool()

	// If tools are present, we don't start in thinking mode
	if len(tools) > 0 {
		p.state = MiniMaxM2CollectingContent
		return
	}

	if !thinkingEnabled {
		p.state = MiniMaxM2CollectingContent
		return
	}

	if prefill && lastMessage.Content != "" {
		p.state = MiniMaxM2CollectingContent
		return
	}

	p.state = MiniMaxM2CollectingThinking
}

func (p *MiniMaxM2Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.err = nil
	p.setInitialState(lastMessage, tools, thinkValue)
	return tools
}

// Event types
type minimaxm2Event interface {
	isMiniMaxM2Event()
}

type minimaxm2EventContent struct {
	content string
}

type minimaxm2EventThinkingContent struct {
	content string
}

type minimaxm2EventToolCall struct {
	toolCall api.ToolCall
}

func (minimaxm2EventContent) isMiniMaxM2Event()         {}
func (minimaxm2EventThinkingContent) isMiniMaxM2Event() {}
func (minimaxm2EventToolCall) isMiniMaxM2Event()        {}

func (p *MiniMaxM2Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	// Check for critical errors
	if p.err != nil {
		return "", "", nil, p.err
	}

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case minimaxm2EventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		case minimaxm2EventThinkingContent:
			thinkingSb.WriteString(event.content)
		case minimaxm2EventContent:
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *MiniMaxM2Parser) parseEvents() []minimaxm2Event {
	var all []minimaxm2Event

	keepLooping := true
	for keepLooping && p.err == nil {
		var events []minimaxm2Event
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

func (p *MiniMaxM2Parser) eat() ([]minimaxm2Event, bool) {
	var events []minimaxm2Event
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case MiniMaxM2CollectingThinking:
		if strings.Contains(bufStr, minimaxm2ThinkingCloseTag) {
			// thinking[</think>] -> content
			split := strings.SplitN(bufStr, minimaxm2ThinkingCloseTag, 2)
			thinking := split[0]
			thinking = strings.TrimRightFunc(thinking, unicode.IsSpace)

			remaining := split[1]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = MiniMaxM2CollectingContent

			if len(thinking) > 0 {
				events = append(events, minimaxm2EventThinkingContent{content: thinking})
			}
			return events, true
		} else if overlapLen := overlap(bufStr, minimaxm2ThinkingCloseTag); overlapLen > 0 {
			// partial </think>
			beforePartialTag := bufStr[:len(bufStr)-overlapLen]
			trailingLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, minimaxm2EventThinkingContent{content: unambiguous})
			}
			return events, false
		} else {
			// otherwise it's thinking content
			whitespaceLen := trailingWhitespaceLen(bufStr)
			ambiguousStart := len(bufStr) - whitespaceLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, minimaxm2EventThinkingContent{content: unambiguous})
			}
			return events, false
		}

	case MiniMaxM2CollectingContent:
		// Check which tag appears first
		toolCallIdx := strings.Index(bufStr, minimaxm2ToolCallOpenTag)
		thinkIdx := strings.Index(bufStr, minimaxm2ThinkingOpenTag)

		// Determine which tag comes first
		var tagIdx int
		var tagName string
		var nextState MiniMaxM2ParserState

		if toolCallIdx >= 0 && (thinkIdx < 0 || toolCallIdx < thinkIdx) {
			tagIdx = toolCallIdx
			tagName = minimaxm2ToolCallOpenTag
			nextState = MiniMaxM2CollectingToolCalls
		} else if thinkIdx >= 0 {
			tagIdx = thinkIdx
			tagName = minimaxm2ThinkingOpenTag
			nextState = MiniMaxM2CollectingThinking
		} else {
			tagIdx = -1
		}

		if tagIdx >= 0 {
			// Found a tag - emit content before it
			before := bufStr[:tagIdx]
			if before != "" {
				events = append(events, minimaxm2EventContent{content: before})
			}

			// Move past the tag
			remaining := bufStr[tagIdx+len(tagName):]
			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = nextState

			logutil.Trace("minimaxm2: found tag", "tag", tagName, "before", before)
			return events, true
		}

		// No complete tag found - check for partial tags at end
		toolCallOverlap := overlap(bufStr, minimaxm2ToolCallOpenTag)
		thinkingOverlap := overlap(bufStr, minimaxm2ThinkingOpenTag)
		maxOverlap := max(toolCallOverlap, thinkingOverlap)

		if maxOverlap > 0 {
			// Hold back potential partial tag
			content := bufStr[:len(bufStr)-maxOverlap]
			if content != "" {
				events = append(events, minimaxm2EventContent{content: content})
			}
			// Keep the potential partial tag in buffer
			p.buffer.Reset()
			p.buffer.WriteString(bufStr[len(bufStr)-maxOverlap:])
			logutil.Trace("minimaxm2: holding potential partial tag", "overlap", maxOverlap)
			return events, false
		}

		// No partial tag - emit everything
		if bufStr != "" {
			events = append(events, minimaxm2EventContent{content: bufStr})
			p.buffer.Reset()
		}
		return events, false

	case MiniMaxM2CollectingToolCalls:
		if strings.Contains(bufStr, minimaxm2ToolCallCloseTag) {
			// Found closing tag - parse all tool calls in the block
			split := strings.SplitN(bufStr, minimaxm2ToolCallCloseTag, 2)
			toolCallBlock := split[0]
			remaining := split[1]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = MiniMaxM2CollectingContent

			// Parse all <invoke> blocks within this tool call block
			toolCalls, errs := p.parseToolCallBlock(toolCallBlock)
			// If there were critical errors (unknown tools), don't emit any events
			// This allows errors to propagate properly
			if len(errs) == 0 {
				for _, tc := range toolCalls {
					events = append(events, minimaxm2EventToolCall{toolCall: tc})
				}
			} else {
				// Log warnings for non-critical errors, but still emit successful tool calls
				for _, err := range errs {
					slog.Warn("minimaxm2 tool call parsing failed", "error", err)
				}
				// Only emit successfully parsed tool calls
				for _, tc := range toolCalls {
					events = append(events, minimaxm2EventToolCall{toolCall: tc})
				}
			}

			return events, true
		}

		// Check for partial closing tag
		if overlapLen := overlap(bufStr, minimaxm2ToolCallCloseTag); overlapLen > 0 {
			// Hold back potential partial closing tag
			return events, false
		}

		// Still collecting tool call content
		return events, false
	}

	return events, false
}

// parseToolCallBlock extracts all <invoke> blocks from a <minimax:tool_call> content
func (p *MiniMaxM2Parser) parseToolCallBlock(content string) ([]api.ToolCall, []error) {
	var toolCalls []api.ToolCall
	var errors []error

	remaining := content
	for {
		// Find next <invoke> tag
		invokeStartIdx := strings.Index(remaining, minimaxm2InvokeOpenPrefix)
		if invokeStartIdx == -1 {
			break
		}

		// Find the end of the opening <invoke> tag (the '>')
		tagEndIdx := strings.Index(remaining[invokeStartIdx:], ">")
		if tagEndIdx == -1 {
			logutil.Trace("minimaxm2: incomplete invoke opening tag")
			break
		}
		tagEndIdx += invokeStartIdx

		// Extract the opening tag to get the name attribute
		openingTag := remaining[invokeStartIdx : tagEndIdx+1]
		functionName := extractNameAttribute(openingTag)
		if functionName == "" {
			err := fmt.Errorf("invoke tag missing name attribute: %s", openingTag)
			slog.Warn("minimaxm2: invoke tag missing name attribute", "tag", openingTag)
			errors = append(errors, err)
			remaining = remaining[tagEndIdx+1:]
			continue
		}

		// Find the closing </invoke> tag
		invokeEndIdx := strings.Index(remaining[tagEndIdx+1:], minimaxm2InvokeCloseTag)
		if invokeEndIdx == -1 {
			// Missing closing tag - conservative recovery
			slog.Warn("minimaxm2: missing </invoke> tag, recovering", "function", functionName)
			// Try to parse what we have up to the end
			invokeContent := remaining[tagEndIdx+1:]
			toolCall, err := p.parseInvoke(functionName, invokeContent)
			if err != nil {
				errors = append(errors, err)
			} else {
				toolCalls = append(toolCalls, toolCall)
			}
			break
		}
		invokeEndIdx += tagEndIdx + 1

		// Extract the content between <invoke> and </invoke>
		invokeContent := remaining[tagEndIdx+1 : invokeEndIdx]

		// Parse this invoke
		toolCall, err := p.parseInvoke(functionName, invokeContent)
		if err != nil {
			slog.Warn("minimaxm2: failed to parse invoke", "function", functionName, "error", err)
			errors = append(errors, err)
		} else {
			toolCalls = append(toolCalls, toolCall)
			logutil.Trace("minimaxm2: parsed tool call", "name", functionName, "args", toolCall.Function.Arguments)
		}

		// Move past this invoke
		remaining = remaining[invokeEndIdx+len(minimaxm2InvokeCloseTag):]
	}

	return toolCalls, errors
}

// parseInvoke extracts the function name and parameters from an <invoke> block
func (p *MiniMaxM2Parser) parseInvoke(functionName string, content string) (api.ToolCall, error) {
	// Validate tool exists
	tool := p.findToolByName(functionName)
	if tool == nil {
		availableTools := make([]string, len(p.tools))
		for i, t := range p.tools {
			availableTools[i] = t.Function.Name
		}
		// Store critical error to halt processing
		p.err = fmt.Errorf("model called unknown tool %q - available tools: %v (ensure tools are provided in API request)", functionName, availableTools)
		slog.Error("MiniMaxM2 model attempted to call unregistered tool",
			"tool", functionName,
			"available_tools", availableTools,
			"recommendation", "ensure tools array includes this tool in API request")
		return api.ToolCall{}, p.err
	}

	// Extract all <parameter> tags
	params := make(map[string]any)
	remaining := content

	for {
		// Find next <parameter> tag
		paramStartIdx := strings.Index(remaining, minimaxm2ParameterOpenPrefix)
		if paramStartIdx == -1 {
			break
		}

		// Find the end of the opening <parameter> tag (the '>')
		tagEndIdx := strings.Index(remaining[paramStartIdx:], ">")
		if tagEndIdx == -1 {
			logutil.Trace("minimaxm2: incomplete parameter opening tag")
			break
		}
		tagEndIdx += paramStartIdx

		// Extract the opening tag to get the name attribute
		openingTag := remaining[paramStartIdx : tagEndIdx+1]
		paramName := extractNameAttribute(openingTag)
		if paramName == "" {
			slog.Warn("minimaxm2: parameter tag missing name attribute", "tag", openingTag)
			remaining = remaining[tagEndIdx+1:]
			continue
		}

		// Find the closing </parameter> tag
		paramEndIdx := strings.Index(remaining[tagEndIdx+1:], minimaxm2ParameterCloseTag)
		if paramEndIdx == -1 {
			// Missing closing tag - conservative recovery
			slog.Warn("minimaxm2: missing </parameter> tag for parameter, recovering", "parameter", paramName)
			// Take everything until the next parameter or end of content
			nextParamIdx := strings.Index(remaining[tagEndIdx+1:], minimaxm2ParameterOpenPrefix)
			if nextParamIdx == -1 {
				// No more parameters, take rest of content
				paramValue := strings.TrimSpace(remaining[tagEndIdx+1:])
				params[paramName] = parseParameterValue(paramValue)
				break
			} else {
				// Take up to next parameter
				paramValue := strings.TrimSpace(remaining[tagEndIdx+1 : tagEndIdx+1+nextParamIdx])
				params[paramName] = parseParameterValue(paramValue)
				remaining = remaining[tagEndIdx+1+nextParamIdx:]
				continue
			}
		}
		paramEndIdx += tagEndIdx + 1

		// Extract the parameter value
		paramValue := strings.TrimSpace(remaining[tagEndIdx+1 : paramEndIdx])
		params[paramName] = parseParameterValue(paramValue)

		logutil.Trace("minimaxm2: parsed parameter", "name", paramName, "value", params[paramName])

		// Move past this parameter
		remaining = remaining[paramEndIdx+len(minimaxm2ParameterCloseTag):]
	}

	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      tool.Function.Name,
			Arguments: params,
		},
	}, nil
}

// extractNameAttribute extracts the value of name="..." from an opening tag
// Handles: <invoke name="function_name"> or <parameter name="param_name">
func extractNameAttribute(tag string) string {
	// Look for name="..."
	nameAttrIdx := strings.Index(tag, `name="`)
	if nameAttrIdx == -1 {
		// Try with single quotes: name='...'
		nameAttrIdx = strings.Index(tag, `name='`)
		if nameAttrIdx == -1 {
			return ""
		}
		// Extract with single quotes
		start := nameAttrIdx + len(`name='`)
		end := strings.Index(tag[start:], `'`)
		if end == -1 {
			return ""
		}
		return tag[start : start+end]
	}

	// Extract with double quotes
	start := nameAttrIdx + len(`name="`)
	end := strings.Index(tag[start:], `"`)
	if end == -1 {
		return ""
	}
	return tag[start : start+end]
}

// parseParameterValue attempts to parse value as JSON, falls back to string
func parseParameterValue(value string) any {
	// Remove leading and trailing newlines
	value = strings.TrimPrefix(value, "\n")
	value = strings.TrimSuffix(value, "\n")

	// Check for null
	if strings.ToLower(value) == "null" {
		return nil
	}

	// Try to parse as JSON (handles objects, arrays, numbers, booleans)
	var jsonValue any
	if err := json.Unmarshal([]byte(value), &jsonValue); err == nil {
		return jsonValue
	}

	// Fallback to string
	return value
}

func (p *MiniMaxM2Parser) findToolByName(name string) *api.Tool {
	name = strings.TrimSpace(name)
	for i := range p.tools {
		if p.tools[i].Function.Name == name {
			return &p.tools[i]
		}
	}
	return nil
}
