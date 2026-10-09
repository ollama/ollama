package parsers

import "github.com/ollama/ollama/api"

// Kolibri uses the Qwen3 thinking and JSON tool grammar. Its renderer closes
// the thinking block before continuing an assistant message.
type Kolibri1Parser struct {
	Qwen3Parser
}

func (p *Kolibri1Parser) Init(tools []api.Tool, lastMessage *api.Message, think *api.ThinkValue) []api.Tool {
	if lastMessage != nil && lastMessage.Role == "assistant" && len(lastMessage.ToolCalls) == 0 {
		think = &api.ThinkValue{Value: false}
	}
	return p.Qwen3Parser.Init(tools, lastMessage, think)
}
