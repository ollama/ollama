package parsers

import "github.com/ollama/ollama/api"

// GLM47Parser extends GLM46Parser with thinking-aware initialization.
// GLM-4.7's prompt ends with <think> when thinking is enabled, so the parser
// must start in CollectingThinking state (the model outputs thinking content directly).
type GLM47Parser struct {
	GLM46Parser
}

func (p *GLM47Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	// When thinking is enabled (nil or true), the prompt ends with <think>,
	// so model output starts directly with thinking content (no opening tag).
	if thinkValue == nil || thinkValue.Bool() {
		p.state = glm46ParserState_CollectingThinking
	}
	return tools
}
