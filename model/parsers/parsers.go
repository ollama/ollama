package parsers

import (
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/harmony"
)

type Parser interface {
	// Init initializes the parser with tools, optional last message for chat prefill, and think value
	// Returns processed tools if the parser needs to modify them (e.g., harmony renames them)
	Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool
	// Add processes streamed content and returns parsed content, thinking, and tool calls
	// The done flag indicates if this is the last chunk (used for draining accumulators)
	Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error)
	HasToolSupport() bool
	HasThinkingSupport() bool
}

type ParserConstructor func() Parser

type ParserRegistry struct {
	constructors map[string]ParserConstructor
}

func (r *ParserRegistry) Register(name string, constructor ParserConstructor) {
	r.constructors[name] = constructor
}

var registry = ParserRegistry{
	constructors: make(map[string]ParserConstructor),
}

func Register(name string, constructor ParserConstructor) {
	registry.Register(name, constructor)
}

func ParserForName(name string) Parser {
	if parser, ok := registry.constructors[name]; ok {
		return parser()
	}
	var p Parser

	switch name {
	case "qwen3-coder":
		p = &Qwen3CoderParser{}
	case "qwen3-vl-instruct":
		p = &Qwen3VLParser{hasThinkingSupport: false}
	case "qwen3-vl-thinking":
		p = &Qwen3VLParser{hasThinkingSupport: true}
	case "ministral":
		p = &MinistralParser{hasThinkingSupport: false}
	case "passthrough":
		return &PassthroughParser{}
	case "harmony":
		return harmony.NewHarmonyMessageHandler()
	case "cogito":
		return &CogitoParser{}
	case "deepseek3":
		return &DeepSeek3Parser{hasThinkingSupport: true}
	case "olmo3":
		return &Olmo3Parser{}
	case "olmo3-think":
		return &Olmo3ThinkParser{}
	case "nemotron-3-nano":
		return &Nemotron3NanoParser{}
	case "functiongemma":
		return &FunctionGemmaParser{}
	default:
		return nil
	}
	return p
}

type PassthroughParser struct{}

func (p *PassthroughParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
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

func splitAtTag(sb *strings.Builder, tag string, trimAfter bool) (string, string) {
	split := strings.SplitN(sb.String(), tag, 2)
	if len(split) == 1 {
		sb.Reset()
		return split[0], ""
	}
	before := split[0]
	before = strings.TrimRightFunc(before, unicode.IsSpace)
	after := split[1]
	if trimAfter {
		after = strings.TrimLeftFunc(after, unicode.IsSpace)
	}
	sb.Reset()
	sb.WriteString(after)
	return before, after // return events
}
