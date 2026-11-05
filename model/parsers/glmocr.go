package parsers

import "github.com/ollama/ollama/api"

// GlmOcrParser is the GLM46 parser with thinking disabled.
type GlmOcrParser struct {
	GLM46Parser
}

func (p *GlmOcrParser) HasThinkingSupport() bool {
	return false
}

func (p *GlmOcrParser) Init(tools []api.Tool, _ *api.Message, _ *api.ThinkValue) []api.Tool {
	p.tools = tools
	return tools
}
