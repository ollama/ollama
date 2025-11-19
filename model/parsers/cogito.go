package parsers

import (
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

type CogitoParserState int

const (
	CogitoCollectingThinking CogitoParserState = iota
	CogitoCollectingContent
	CogitoCollectingToolCalls
	CogitoCollectingToolOutput
	CogitoThinkingDoneEatingWhitespace
	CogitoContentTransition
)

const (
	cogitoThinkingOpenTag     = "<think>"
	cogitoThinkingCloseTag    = "</think>"
	cogitoToolCallsBeginTag   = "<｜tool▁calls▁begin｜>"
	cogitoToolCallsEndTag     = "<｜tool▁calls▁end｜>"
	cogitoToolCallBeginTag    = "<｜tool▁call▁begin｜>"
	cogitoToolCallEndTag      = "<｜tool▁call▁end｜>"
	cogitoToolSepTag          = "<｜tool▁sep｜>"
	cogitoToolOutputBeginTag  = "<｜tool▁output▁begin｜>"
	cogitoToolOutputEndTag    = "<｜tool▁output▁end｜>"
	cogitoToolOutputsBeginTag = "<｜tool▁outputs▁begin｜>"
	cogitoToolOutputsEndTag   = "<｜tool▁outputs▁end｜>"
	cogitoEndOfSentenceTag    = "<｜end▁of▁sentence｜>"
	cogitoAssistantTag        = "<｜Assistant｜>"
	cogitoUserTag             = "<｜User｜>"
	cogitoBeginOfSentenceTag  = "<｜begin▁of▁sentence｜>"
)

type CogitoParser struct {
	state  CogitoParserState
	buffer strings.Builder
	tools  []api.Tool
}

func (p *CogitoParser) HasToolSupport() bool {
	return true
}

func (p *CogitoParser) HasThinkingSupport() bool {
	return true
}

func (p *CogitoParser) setInitialState(lastMessage *api.Message) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"
	if prefill && lastMessage.Content != "" {
		p.state = CogitoCollectingContent
		return
	}
	p.state = CogitoCollectingThinking
}

func (p *CogitoParser) Init(tools []api.Tool, lastMessage *api.Message) []api.Tool {
	p.tools = tools
	p.setInitialState(lastMessage)
	return tools
}

func (p *CogitoParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)

	var contentSb strings.Builder
	var thinkingSb strings.Builder
	var toolCalls []api.ToolCall

	for {
		addedContent, addedThinking, addedCalls, keepGoing := p.processBuffer()

		contentSb.WriteString(addedContent)
		thinkingSb.WriteString(addedThinking)
		toolCalls = append(toolCalls, addedCalls...)

		if !keepGoing {
			break
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *CogitoParser) processBuffer() (content string, thinking string, calls []api.ToolCall, keepGoing bool) {
	bufStr := p.buffer.String()
	if bufStr == "" {
		return "", "", nil, false
	}

	switch p.state {
	case CogitoCollectingThinking:
		if strings.HasPrefix(bufStr, cogitoThinkingOpenTag) {
			if idx := strings.Index(bufStr, cogitoThinkingCloseTag); idx != -1 {
				thinkContent := bufStr[len(cogitoThinkingOpenTag):idx]
				thinkContent = strings.TrimRightFunc(thinkContent, unicode.IsSpace)

				remaining := bufStr[idx+len(cogitoThinkingCloseTag):]
				remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

				p.buffer.Reset()
				p.buffer.WriteString(remaining)
				p.state = CogitoCollectingContent

				return "", thinkContent, nil, true
			}
			return "", "", nil, false
		}

		p.state = CogitoCollectingContent
		return "", "", nil, true

	case CogitoCollectingContent:
		for _, tag := range []string{cogitoEndOfSentenceTag, cogitoAssistantTag, cogitoUserTag, cogitoBeginOfSentenceTag} {
			if idx := strings.Index(bufStr, tag); idx != -1 {
				contentBefore := bufStr[:idx]
				contentBefore = strings.TrimRightFunc(contentBefore, unicode.IsSpace)

				remaining := bufStr[idx+len(tag):]
				remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

				p.buffer.Reset()
				p.buffer.WriteString(remaining)

				return contentBefore, "", nil, true
			}
		}

		if idx := strings.Index(bufStr, cogitoToolCallsBeginTag); idx != -1 {
			contentBefore := bufStr[:idx]
			contentBefore = strings.TrimRightFunc(contentBefore, unicode.IsSpace)

			remaining := bufStr[idx+len(cogitoToolCallsBeginTag):]
			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = CogitoCollectingToolCalls

			return contentBefore, "", nil, true
		}

		if idx := strings.Index(bufStr, cogitoToolOutputsBeginTag); idx != -1 {
			contentBefore := bufStr[:idx]
			contentBefore = strings.TrimRightFunc(contentBefore, unicode.IsSpace)

			remaining := bufStr[idx+len(cogitoToolOutputsBeginTag):]
			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = CogitoCollectingToolOutput

			return contentBefore, "", nil, true
		}

		p.buffer.Reset()
		return bufStr, "", nil, false

	case CogitoCollectingToolCalls:
		if idx := strings.Index(bufStr, cogitoToolCallBeginTag); idx != -1 {
			startIdx := idx + len(cogitoToolCallBeginTag)
			if endIdx := strings.Index(bufStr[startIdx:], cogitoToolCallEndTag); endIdx != -1 {
				toolCallContent := bufStr[startIdx : startIdx+endIdx]

				if toolCall, err := p.parseToolCallContent(toolCallContent); err == nil {
					remaining := bufStr[startIdx+endIdx+len(cogitoToolCallEndTag):]
					remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

					p.buffer.Reset()
					p.buffer.WriteString(remaining)

					return "", "", []api.ToolCall{toolCall}, true
				} else {
					slog.Warn("cogito tool call parsing failed", "error", err)
				}
			}
		}

		if idx := strings.Index(bufStr, cogitoToolCallsEndTag); idx != -1 {
			remaining := bufStr[idx+len(cogitoToolCallsEndTag):]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = CogitoCollectingContent

			return "", "", nil, true
		}

		return "", "", nil, false

	case CogitoCollectingToolOutput:
		if idx := strings.Index(bufStr, cogitoToolOutputBeginTag); idx != -1 {
			startIdx := idx + len(cogitoToolOutputBeginTag)
			if endIdx := strings.Index(bufStr[startIdx:], cogitoToolOutputEndTag); endIdx != -1 {
				remaining := bufStr[startIdx+endIdx+len(cogitoToolOutputEndTag):]
				remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

				p.buffer.Reset()
				p.buffer.WriteString(remaining)

				return "", "", nil, true
			}
		}

		if idx := strings.Index(bufStr, cogitoToolOutputsEndTag); idx != -1 {
			remaining := bufStr[idx+len(cogitoToolOutputsEndTag):]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = CogitoCollectingContent

			return "", "", nil, true
		}

		return "", "", nil, false
	}

	return "", "", nil, false
}

func (p *CogitoParser) parseToolCallContent(content string) (api.ToolCall, error) {
	// Format: function<｜tool▁sep｜>tool_name\n```json\n{args}\n```
	parts := strings.SplitN(content, cogitoToolSepTag, 2)
	if len(parts) != 2 {
		return api.ToolCall{}, errors.New("invalid tool call format: missing separator")
	}

	// Verify function type
	functionType := strings.TrimSpace(parts[0])
	if functionType != "function" {
		return api.ToolCall{}, errors.New("invalid tool call format: expected 'function'")
	}

	nameAndArgs := parts[1]

	// Find JSON block
	jsonStart := strings.Index(nameAndArgs, "```json\n")
	if jsonStart == -1 {
		jsonStart = strings.Index(nameAndArgs, "```json")
		if jsonStart == -1 {
			return api.ToolCall{}, errors.New("invalid tool call format: missing JSON block start")
		}
		jsonStart += len("```json")
	} else {
		jsonStart += len("```json\n")
	}

	jsonEnd := strings.Index(nameAndArgs[jsonStart:], "\n```")
	if jsonEnd == -1 {
		jsonEnd = strings.Index(nameAndArgs[jsonStart:], "```")
		if jsonEnd == -1 {
			return api.ToolCall{}, errors.New("invalid tool call format: missing JSON block end")
		}
	}

	argsJSON := nameAndArgs[jsonStart : jsonStart+jsonEnd]
	toolName := strings.TrimSpace(nameAndArgs[:strings.Index(nameAndArgs, "```")])

	var args api.ToolCallFunctionArguments
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return api.ToolCall{}, err
	}

	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      toolName,
			Arguments: args,
		},
	}, nil
}
