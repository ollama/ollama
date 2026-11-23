package parsers

import "github.com/ollama/ollama/api"

// FlintParser handles the publisher's bare assistant prompt and explicit
// thinking prefills. When thinking is disabled in the API, the model still
// decides whether to reason; the parser withholds that reasoning from output.
type FlintParser struct {
	GLM46Parser
	noThinking bool
}

func (p *FlintParser) Init(tools []api.Tool, lastMessage *api.Message, think *api.ThinkValue) []api.Tool {
	prepared := p.GLM46Parser.Init(tools, lastMessage, think)
	p.buffer.Reset()
	p.noThinking = think != nil && !think.Bool()
	p.state = glm46ParserState_LookingForThinkingOpen
	if think != nil && think.Bool() {
		p.state = glm46ParserState_CollectingThinking
	}
	return prepared
}

// The publisher's bare prompt marks reasoning with <think>. An earlier
// untagged-reasoning sample followed a forced </think> prompt, which Flint no
// longer uses for think:false. Without an opening tag, direct content and
// untagged reasoning cannot be distinguished while streaming.
func (p *FlintParser) Add(s string, done bool) (string, string, []api.ToolCall, error) {
	content, thinking, calls, err := p.GLM46Parser.Add(s, done)
	if p.noThinking {
		thinking = ""
	}
	return content, thinking, calls, err
}
