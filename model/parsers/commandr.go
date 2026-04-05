package parsers

import (
	"encoding/json"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/api"
)

type commandRParserState int

const (
	commandRStateContent commandRParserState = iota
	commandRStateToolCalls
)

const (
	// Command-R outputs tool calls as: Action: ```json\n[...]\n```
	commandRActionTag     = "Action:"
	commandRJSONBlockOpen = "```json"
	commandRJSONBlockEnd  = "```"
)

type CommandRParser struct {
	state  commandRParserState
	buffer strings.Builder
}

func (p *CommandRParser) HasToolSupport() bool {
	return true
}

func (p *CommandRParser) HasThinkingSupport() bool {
	return false
}

func (p *CommandRParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.state = commandRStateContent
	return tools
}

func (p *CommandRParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)

	if done {
		bufStr := p.buffer.String()
		p.buffer.Reset()
		if p.state == commandRStateContent && len(bufStr) > 0 {
			return bufStr, "", nil, nil
		}
		// If we were collecting tool calls but never found the closing tag,
		// try to parse what we have
		if p.state == commandRStateToolCalls && len(bufStr) > 0 {
			if toolCalls, parseErr := parseCommandRToolCalls(bufStr); parseErr == nil && len(toolCalls) > 0 {
				return "", "", toolCalls, nil
			}
			// Failed to parse as tool calls, return as content
			return bufStr, "", nil, nil
		}
		return "", "", nil, nil
	}

	var allContent strings.Builder
	var allCalls []api.ToolCall

	for {
		events, cont := p.eat()
		for _, event := range events {
			switch e := event.(type) {
			case commandREventContent:
				allContent.WriteString(e.content)
			case commandREventToolCalls:
				allCalls = append(allCalls, e.calls...)
			}
		}
		if !cont {
			break
		}
	}

	return allContent.String(), "", allCalls, nil
}

type commandREvent interface {
	isCommandREvent()
}

type commandREventContent struct {
	content string
}

type commandREventToolCalls struct {
	calls []api.ToolCall
}

func (commandREventContent) isCommandREvent()   {}
func (commandREventToolCalls) isCommandREvent() {}

func (p *CommandRParser) eat() ([]commandREvent, bool) {
	var events []commandREvent
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case commandRStateContent:
		// Look for "Action:" which signals the start of tool calls
		if idx := strings.Index(bufStr, commandRActionTag); idx != -1 {
			content := strings.TrimRight(bufStr[:idx], " \t\n\r")
			remaining := bufStr[idx+len(commandRActionTag):]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = commandRStateToolCalls

			if len(content) > 0 {
				events = append(events, commandREventContent{content: content})
			}
			return events, true
		}

		// Check for partial overlap with "Action:"
		if overlapLen := overlap(bufStr, commandRActionTag); overlapLen > 0 {
			unambiguous := bufStr[:len(bufStr)-overlapLen]
			ambiguous := bufStr[len(bufStr)-overlapLen:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, commandREventContent{content: unambiguous})
			}
			return events, false
		}

		// Regular content
		p.buffer.Reset()
		if len(bufStr) > 0 {
			events = append(events, commandREventContent{content: bufStr})
		}
		return events, false

	case commandRStateToolCalls:
		// We're after "Action:" — look for the closing ``` after ```json\n[...]\n```
		// First, we need to find the opening ```json
		jsonBlockStart := strings.Index(bufStr, commandRJSONBlockOpen)
		if jsonBlockStart == -1 {
			// Haven't found ```json yet, keep buffering
			return events, false
		}

		// Find content after ```json
		afterOpen := bufStr[jsonBlockStart+len(commandRJSONBlockOpen):]

		// Find the closing ``` (but not the opening one)
		// The closing ``` must come after newline or content
		closeIdx := findClosingCodeFence(afterOpen)
		if closeIdx == -1 {
			// Haven't found closing ``` yet, keep buffering
			return events, false
		}

		// Extract the JSON content between ```json and closing ```
		jsonContent := strings.TrimSpace(afterOpen[:closeIdx])
		remaining := afterOpen[closeIdx+len(commandRJSONBlockEnd):]

		p.buffer.Reset()
		p.buffer.WriteString(remaining)
		p.state = commandRStateContent

		toolCalls, err := parseCommandRToolCalls(jsonContent)
		if err != nil {
			slog.Warn("command-r tool call parsing failed", "error", err, "content", jsonContent)
			// Return the raw content since we couldn't parse it
			raw := bufStr[:jsonBlockStart+len(commandRJSONBlockOpen)+closeIdx+len(commandRJSONBlockEnd)]
			events = append(events, commandREventContent{content: raw})
		} else if len(toolCalls) > 0 {
			events = append(events, commandREventToolCalls{calls: toolCalls})
		}

		return events, true
	}

	return events, false
}

// findClosingCodeFence finds the position of the closing ``` in content that
// comes after a ```json opening. It skips the first character to avoid matching
// immediately, and looks for ``` preceded by a newline or at the very end.
func findClosingCodeFence(s string) int {
	// The content after ```json typically starts with a newline.
	// We need to find ``` that is NOT part of the opening.
	// Search for \n``` or just ``` after some content.
	searchStart := 0
	for {
		idx := strings.Index(s[searchStart:], commandRJSONBlockEnd)
		if idx == -1 {
			return -1
		}
		absIdx := searchStart + idx
		// Make sure this isn't at position 0 (which would be empty content)
		if absIdx > 0 {
			return absIdx
		}
		searchStart = absIdx + 1
		if searchStart >= len(s) {
			return -1
		}
	}
}

// commandRToolCall represents the JSON structure command-r uses for tool calls
type commandRToolCall struct {
	ToolName   string         `json:"tool_name"`
	Parameters map[string]any `json:"parameters"`
}

// parseCommandRToolCalls parses command-r's JSON tool call format:
//
//	[{"tool_name": "func_name", "parameters": {"arg1": "val1"}}]
//
// Also handles the single-object case without array wrapper.
func parseCommandRToolCalls(jsonStr string) ([]api.ToolCall, error) {
	jsonStr = strings.TrimSpace(jsonStr)
	if jsonStr == "" {
		return nil, nil
	}

	// Try parsing as array first (most common)
	var calls []commandRToolCall
	if err := json.Unmarshal([]byte(jsonStr), &calls); err != nil {
		// Try parsing as single object
		var single commandRToolCall
		if err2 := json.Unmarshal([]byte(jsonStr), &single); err2 != nil {
			return nil, err
		}
		calls = []commandRToolCall{single}
	}

	var toolCalls []api.ToolCall
	for i, call := range calls {
		if call.ToolName == "" && call.ToolName != "directly-answer" {
			continue
		}
		// Skip the "directly-answer" pseudo-tool that command-r uses
		// when it wants to respond without calling any tools
		if call.ToolName == "directly-answer" {
			continue
		}

		args := api.NewToolCallFunctionArguments()
		for k, v := range call.Parameters {
			args.Set(k, v)
		}

		toolCalls = append(toolCalls, api.ToolCall{
			Function: api.ToolCallFunction{
				Name:      call.ToolName,
				Arguments: args,
				Index:     i,
			},
		})
	}

	return toolCalls, nil
}
