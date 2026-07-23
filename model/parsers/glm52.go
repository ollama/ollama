package parsers

import "github.com/ollama/ollama/api"

// GLM52Parser extends GLM47Parser for GLM-5.2 models.
// GLM-5.2 maintains the same parsing format as GLM-4.7 with enhanced
// response handling for complex prompts and longer context windows.
type GLM52Parser struct {
	GLM47Parser
}

func (p *GLM52Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.callIndex = 0
	// When thinking is enabled (nil or true), the prompt ends with <think>,
	// so model output starts directly with thinking content (no opening tag).
	if thinkValue == nil || thinkValue.Bool() {
		p.state = glm46ParserState_CollectingThinking
	}
	return tools
}
