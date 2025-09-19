package parsers

import (
	"github.com/ollama/ollama/api"
)

type Parser interface {
	Add(s string, tools []api.Tool) (content string, thinking string, calls []api.ToolCall, err error)
	HasToolSupport() bool
	HasThinkingSupport() bool
}

func ParserForName(name string) Parser {
	switch name {
	case "qwen3-coder":
		parser := &Qwen3CoderParser{}
		return parser
	case "passthrough":
		return &PassthroughParser{}
	default:
		return nil
	}
}

type PassthroughParser struct{}

func (p *PassthroughParser) Add(s string, tools []api.Tool) (content string, thinking string, calls []api.ToolCall, err error) {
	return s, "", nil, nil
}

func (p *PassthroughParser) HasToolSupport() bool {
	return false
}

func (p *PassthroughParser) HasThinkingSupport() bool {
	return false
}
