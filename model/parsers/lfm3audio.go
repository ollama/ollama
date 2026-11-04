package parsers

import (
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

type lfm3AudioState int

const (
	lfm3AudioStart lfm3AudioState = iota
	lfm3AudioThinking
	lfm3AudioAfterThinking
	lfm3AudioContent
)

// LFM3AudioParser parses LFM3-Audio replies. Without a thinking prefill the
// model decides for itself whether to open with a <think> block, so the block
// is optional; tool calls use the Qwen3-Coder XML format.
type LFM3AudioParser struct {
	content Qwen3CoderParser
	state   lfm3AudioState
	buf     strings.Builder
	// filter removes reasoning the model still emits with thinking off.
	filter *lfm3ThinkFilter
}

func (p *LFM3AudioParser) HasToolSupport() bool     { return true }
func (p *LFM3AudioParser) HasThinkingSupport() bool { return false }
func (p *LFM3AudioParser) ThinkingClose() []string  { return nil }
func (p *LFM3AudioParser) PreservedTokens() []string {
	return []string{"<think>", "</think>", toolOpenTag, toolCloseTag}
}

func (p *LFM3AudioParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.buf.Reset()
	p.content = Qwen3CoderParser{}
	p.content.Init(tools, nil, nil)
	p.filter = nil
	switch {
	case lastMessage != nil && lastMessage.Role == "assistant" && lastMessage.Content != "":
		// The renderer continues a trailing assistant message.
		p.state = lfm3AudioContent
	case thinkValue != nil && thinkValue.Bool():
		p.state = lfm3AudioThinking
	case thinkValue != nil:
		// The prompt already closed an empty thinking block.
		p.state = lfm3AudioAfterThinking
		p.filter = &lfm3ThinkFilter{}
	default:
		p.state = lfm3AudioStart
	}
	return tools
}

func (p *LFM3AudioParser) Add(s string, done bool) (content, thinking string, calls []api.ToolCall, err error) {
	if p.filter != nil {
		s = p.filter.add(s, done)
	}
	p.buf.WriteString(s)
	var think strings.Builder
	for {
		text := p.buf.String()
		switch p.state {
		case lfm3AudioStart, lfm3AudioAfterThinking:
			trimmed := strings.TrimLeftFunc(text, unicode.IsSpace)
			if trimmed == "" && !done {
				p.buf.Reset()
				return p.finish(think.String(), "", false)
			}
			if p.state == lfm3AudioStart {
				if rest, ok := strings.CutPrefix(trimmed, "<think>"); ok {
					p.reset(rest)
					p.state = lfm3AudioThinking
					continue
				}
				if strings.HasPrefix("<think>", trimmed) && !done {
					return p.finish(think.String(), "", false)
				}
			}
			p.reset(trimmed)
			p.state = lfm3AudioContent
		case lfm3AudioThinking:
			if before, after, ok := strings.Cut(text, "</think>"); ok {
				think.WriteString(before)
				p.reset(after)
				p.state = lfm3AudioAfterThinking
				continue
			}
			keep := 0
			if !done {
				keep = overlap(text, "</think>")
			}
			think.WriteString(text[:len(text)-keep])
			p.reset(text[len(text)-keep:])
			return p.finish(think.String(), "", done)
		case lfm3AudioContent:
			p.buf.Reset()
			return p.finish(think.String(), text, done)
		}
	}
}

func (p *LFM3AudioParser) reset(s string) {
	p.buf.Reset()
	p.buf.WriteString(s)
}

func (p *LFM3AudioParser) finish(thinking, text string, done bool) (string, string, []api.ToolCall, error) {
	if text == "" && !done {
		return "", thinking, nil, nil
	}
	content, _, calls, err := p.content.Add(text, done)
	return content, thinking, calls, err
}
