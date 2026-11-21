package parsers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

const (
	deepseek41ThinkClose = "</think>"
	deepseek41CallsOpen  = "<｜DSML｜ calls>"
	deepseek41CallsClose = "</｜DSML｜ calls>"
	deepseek41InvokeOpen = "<｜DSML｜ invoke name=\""
	deepseek41InvokeEnd  = "</｜DSML｜ invoke>"
	deepseek41ParamOpen  = "<｜DSML｜ parameter name=\""
	deepseek41ParamEnd   = "</｜DSML｜ parameter>"
)

type deepseek41ParserState uint8

const (
	deepseek41Reasoning deepseek41ParserState = iota
	deepseek41Content
	deepseek41Calls
)

type DeepSeekV41Parser struct {
	state     deepseek41ParserState
	buffer    string
	callIndex int
}

func (*DeepSeekV41Parser) HasToolSupport() bool     { return true }
func (*DeepSeekV41Parser) HasThinkingSupport() bool { return true }

func (p *DeepSeekV41Parser) PreservedTokens() []string {
	return []string{deepseek41ThinkClose, "｜DSML｜"}
}

func (p *DeepSeekV41Parser) ThinkingClose() []string {
	if p.state == deepseek41Reasoning {
		return []string{deepseek41ThinkClose}
	}
	return nil
}

func (p *DeepSeekV41Parser) Init(tools []api.Tool, lastMessage *api.Message, think *api.ThinkValue) []api.Tool {
	p.buffer, p.callIndex = "", 0
	p.state = deepseek41Reasoning
	if (think != nil && think.Value != nil && !think.Bool()) || (lastMessage != nil && lastMessage.Role == "assistant") {
		p.state = deepseek41Content
	}
	return tools
}

func (p *DeepSeekV41Parser) Add(chunk string, done bool) (content, thinking string, calls []api.ToolCall, err error) {
	p.buffer += chunk
	for {
		switch p.state {
		case deepseek41Reasoning:
			if at := strings.Index(p.buffer, deepseek41ThinkClose); at >= 0 {
				thinking += p.buffer[:at]
				p.buffer = p.buffer[at+len(deepseek41ThinkClose):]
				p.state = deepseek41Content
				continue
			}
			if done {
				thinking += p.buffer
				p.buffer = ""
			} else {
				keep := overlap(p.buffer, deepseek41ThinkClose)
				thinking += p.buffer[:len(p.buffer)-keep]
				p.buffer = p.buffer[len(p.buffer)-keep:]
			}
			return content, thinking, calls, nil
		case deepseek41Content:
			if at := strings.Index(p.buffer, deepseek41CallsOpen); at >= 0 {
				content += strings.TrimSuffix(p.buffer[:at], "\n\n")
				p.buffer = p.buffer[at+len(deepseek41CallsOpen):]
				p.state = deepseek41Calls
				continue
			}
			if done {
				content += p.buffer
				p.buffer = ""
			} else {
				keep := overlap(p.buffer, deepseek41CallsOpen)
				keep += trailingWhitespaceLen(p.buffer[:len(p.buffer)-keep])
				content += p.buffer[:len(p.buffer)-keep]
				p.buffer = p.buffer[len(p.buffer)-keep:]
			}
			return content, thinking, calls, nil
		case deepseek41Calls:
			at := strings.Index(p.buffer, deepseek41CallsClose)
			if at < 0 {
				if done {
					return content, thinking, calls, fmt.Errorf("incomplete DeepSeek V4.1 tool call block")
				}
				return content, thinking, calls, nil
			}
			parsed, parseErr := parseDeepSeek41Calls(p.buffer[:at], p.callIndex)
			if parseErr != nil {
				return content, thinking, calls, parseErr
			}
			calls = append(calls, parsed...)
			p.callIndex += len(parsed)
			p.buffer = p.buffer[at+len(deepseek41CallsClose):]
			p.state = deepseek41Content
			continue
		}
	}
}

func parseDeepSeek41Calls(block string, firstIndex int) ([]api.ToolCall, error) {
	var calls []api.ToolCall
	for remaining := strings.TrimSpace(block); remaining != ""; {
		if !strings.HasPrefix(remaining, deepseek41InvokeOpen) {
			return nil, fmt.Errorf("invalid DeepSeek V4.1 tool invocation")
		}
		remaining = remaining[len(deepseek41InvokeOpen):]
		nameEnd := strings.Index(remaining, "\">")
		if nameEnd < 0 {
			return nil, fmt.Errorf("incomplete DeepSeek V4.1 tool name")
		}
		name := remaining[:nameEnd]
		remaining = remaining[nameEnd+2:]
		invokeEnd := strings.Index(remaining, deepseek41InvokeEnd)
		if invokeEnd < 0 {
			return nil, fmt.Errorf("incomplete DeepSeek V4.1 tool invocation")
		}
		params := strings.TrimSpace(remaining[:invokeEnd])
		args := api.NewToolCallFunctionArguments()
		for params != "" {
			if !strings.HasPrefix(params, deepseek41ParamOpen) {
				return nil, fmt.Errorf("invalid DeepSeek V4.1 tool parameter")
			}
			params = params[len(deepseek41ParamOpen):]
			keyEnd := strings.Index(params, "\" string=\"")
			if keyEnd < 0 {
				return nil, fmt.Errorf("incomplete DeepSeek V4.1 parameter name")
			}
			key := params[:keyEnd]
			params = params[keyEnd+len("\" string=\""):]
			flagEnd := strings.Index(params, "\">")
			if flagEnd < 0 {
				return nil, fmt.Errorf("incomplete DeepSeek V4.1 parameter type")
			}
			stringValue := params[:flagEnd]
			params = params[flagEnd+2:]
			valueEnd := strings.Index(params, deepseek41ParamEnd)
			if valueEnd < 0 {
				return nil, fmt.Errorf("incomplete DeepSeek V4.1 parameter value")
			}
			var value any = params[:valueEnd]
			if stringValue == "false" {
				if err := json.Unmarshal([]byte(params[:valueEnd]), &value); err != nil {
					return nil, fmt.Errorf("DeepSeek V4.1 parameter %q: %w", key, err)
				}
			} else if stringValue != "true" {
				return nil, fmt.Errorf("invalid DeepSeek V4.1 parameter type %q", stringValue)
			}
			args.Set(key, value)
			params = strings.TrimSpace(params[valueEnd+len(deepseek41ParamEnd):])
		}
		calls = append(calls, api.ToolCall{Function: api.ToolCallFunction{Index: firstIndex + len(calls), Name: name, Arguments: args}})
		remaining = strings.TrimSpace(remaining[invokeEnd+len(deepseek41InvokeEnd):])
	}
	return calls, nil
}
