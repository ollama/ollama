package parsers

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/thinking"
)

// Intellect3Parser combines thinking support using
// the built-in thinking parser, with tool call support
// via qwen3-coder's parser.
type Intellect3Parser struct {
	thinkingParser thinking.Parser
	toolParser     Qwen3CoderParser
}

func (p *Intellect3Parser) HasToolSupport() bool {
	return true
}

func (p *Intellect3Parser) HasThinkingSupport() bool {
	return true
}

func (p *Intellect3Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.thinkingParser = thinking.Parser{
		OpeningTag: "<think>",
		ClosingTag: "</think>",
	}
	p.toolParser = Qwen3CoderParser{}
	return p.toolParser.Init(tools, lastMessage, thinkValue)
}

func (p *Intellect3Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	// First extract thinking content
	thinkingContent, remainingContent := p.thinkingParser.AddContent(s)

	// Then process the remaining content for tool calls
	toolContent, _, toolCalls, err := p.toolParser.Add(remainingContent, done)
	if err != nil {
		return "", thinkingContent, nil, err
	}

	return toolContent, thinkingContent, toolCalls, nil
}
