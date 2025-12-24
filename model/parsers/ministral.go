package parsers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

type ministralParserState int

const (
	ministralCollectingContent = iota
	ministralCollectingThinkingContent
	ministralCollectingToolName
	ministralCollectingToolArgs
)

type MinistralParser struct {
	state              ministralParserState
	buffer             strings.Builder
	tools              []api.Tool
	hasThinkingSupport bool
	currentTool        *api.Tool
}

func (p *MinistralParser) HasToolSupport() bool {
	return true
}

func (p *MinistralParser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *MinistralParser) setInitialState(lastMessage *api.Message) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"
	if !p.HasThinkingSupport() {
		p.state = ministralCollectingContent
		return
	}

	if prefill && lastMessage.Content != "" {
		p.state = ministralCollectingContent
		return
	}

	p.state = ministralCollectingThinkingContent
}

func (p *MinistralParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.setInitialState(lastMessage)
	return tools
}

func toolByName(tools []api.Tool, n string) (*api.Tool, error) {
	for i := range tools {
		if tools[i].Function.Name == n {
			return &tools[i], nil
		}
	}
	return nil, fmt.Errorf("tool '%s' not found", n)
}

func (p *MinistralParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)

	switch p.state {
	case ministralCollectingContent:
		if strings.Contains(p.buffer.String(), "[TOOL_CALLS]") {
			before, _ := splitAtTag(&p.buffer, "[TOOL_CALLS]", false)
			if before != "" {
				return before, "", calls, nil
			}
			p.state = ministralCollectingToolName
		} else if strings.Contains(p.buffer.String(), "[THINK]") {
			p.state = ministralCollectingThinkingContent
			return "", "", calls, nil
		} else {
			p.buffer.Reset()
			return s, "", calls, nil
		}
	case ministralCollectingThinkingContent:
		if strings.Contains(p.buffer.String(), "[/THINK]") {
			thinkingContent, after := splitAtTag(&p.buffer, "[/THINK]", true)
			p.state = ministralCollectingContent
			if after != "" {
				p.buffer.Reset()
				return after, thinkingContent, calls, nil
			}
			return "", thinkingContent, calls, nil
		} else {
			p.buffer.Reset()
			return "", s, calls, nil
		}
	case ministralCollectingToolName:
		if strings.Contains(p.buffer.String(), "[ARGS]") {
			name, _ := splitAtTag(&p.buffer, "[ARGS]", false)

			t, err := toolByName(p.tools, name)
			if err != nil {
				return "", "", calls, err
			}
			p.currentTool = t
			p.state = ministralCollectingToolArgs
			return "", "", calls, nil
		}
		return "", "", calls, nil
	case ministralCollectingToolArgs:
		if strings.Contains(p.buffer.String(), "}") {
			before, _ := splitAtTag(&p.buffer, "}", false)
			before += "}"

			var args api.ToolCallFunctionArguments
			if err := json.Unmarshal([]byte(before), &args); err != nil {
				// todo - throw a better error
				return "", "", calls, err
			}

			p.state = ministralCollectingContent

			call := api.ToolCall{
				Function: api.ToolCallFunction{
					Name:      p.currentTool.Function.Name,
					Arguments: args,
				},
			}
			calls = append(calls, call)
			return "", "", calls, nil
		}
		return "", "", calls, nil
	}

	return p.buffer.String(), thinking, calls, nil
}
