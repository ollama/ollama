package parsers

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/harmony"
)

type Parser interface {
	// Init initializes the parser with tools and optional last message for chat prefill
	// Returns processed tools if the parser needs to modify them (e.g., harmony renames them)
	Init(tools []api.Tool, lastMessage *api.Message) []api.Tool
	// Add processes streamed content and returns parsed content, thinking, and tool calls
	// The done flag indicates if this is the last chunk (used for draining accumulators)
	Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error)
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
	case "harmony":
		return harmony.NewHarmonyMessageHandler()
	default:
		return nil
	}
}

type PassthroughParser struct{}

func (p *PassthroughParser) Init(tools []api.Tool, lastMessage *api.Message) []api.Tool {
	return tools // passthrough doesn't modify tools
}

func (p *PassthroughParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	return s, "", nil, nil
}

func (p *PassthroughParser) HasToolSupport() bool {
	return false
}

func (p *PassthroughParser) HasThinkingSupport() bool {
	return false
}
