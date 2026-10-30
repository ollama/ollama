package parsers

import (
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

// LFM3Parser is the Qwen3.5 parser plus the LFM3 text and VL checkpoints'
// output with thinking off: see lfm3ThinkFilter.
type LFM3Parser struct {
	Qwen35Parser
	suppress bool
	filter   lfm3ThinkFilter
}

func (p *LFM3Parser) Init(tools []api.Tool, lastMessage *api.Message, think *api.ThinkValue) []api.Tool {
	tools = p.Qwen35Parser.Init(tools, lastMessage, think)
	p.filter = lfm3ThinkFilter{}
	p.suppress = think != nil && !think.Bool() && p.state == qwen35ParserStateCollectingContent
	return tools
}

func (p *LFM3Parser) Add(s string, done bool) (string, string, []api.ToolCall, error) {
	if !p.suppress {
		return p.Qwen35Parser.Add(s, done)
	}
	content, _, calls, err := p.Qwen35Parser.Add(p.filter.add(s, done), done)
	return content, "", calls, err
}

// lfm3ThinkFilter handles LFM3 output with thinking off. After the
// <think></think> prefill the model may still reason, either in an explicit
// <think> block or untagged and closed by a second </think> (the publisher
// reference does both). Like the other thinking-off parsers, output streams:
// explicit reasoning blocks are dropped and stray </think> tags removed, but
// untagged reasoning cannot be told apart from the answer while streaming.
// A paragraph break replaces a stray tag between text so the two sides (often
// reasoning then answer, or a repeated answer) are not run together.
type lfm3ThinkFilter struct {
	inThinking  bool
	trimLeading bool
	wrote       bool
	sep         bool
	pending     strings.Builder
}

func (f *lfm3ThinkFilter) emit(out *strings.Builder, s string) {
	if s == "" {
		return
	}
	if f.sep {
		out.WriteString("\n\n")
		f.sep = false
	}
	out.WriteString(s)
	f.wrote = f.wrote || strings.TrimSpace(s) != ""
}

// add returns the output so far with thinking tags and explicit reasoning
// removed, keeping back a suffix that could still become a tag.
func (f *lfm3ThinkFilter) add(s string, done bool) string {
	f.pending.WriteString(s)
	acc := f.pending.String()
	f.pending.Reset()
	var out strings.Builder
	for {
		if f.inThinking {
			end := strings.Index(acc, qwen35ThinkingCloseTag)
			if end < 0 {
				if !done {
					f.pending.WriteString(acc[len(acc)-overlap(acc, qwen35ThinkingCloseTag):])
				}
				return out.String()
			}
			acc = acc[end+len(qwen35ThinkingCloseTag):]
			f.inThinking = false
			f.trimLeading = true
			continue
		}
		if f.trimLeading {
			acc = strings.TrimLeftFunc(acc, unicode.IsSpace)
			if acc == "" {
				return out.String()
			}
			f.trimLeading = false
		}
		open := strings.Index(acc, qwen35ThinkingOpenTag)
		closing := strings.Index(acc, qwen35ThinkingCloseTag)
		switch {
		case closing >= 0 && (open < 0 || closing < open):
			f.emit(&out, acc[:closing])
			acc = acc[closing+len(qwen35ThinkingCloseTag):]
			f.trimLeading = true
			f.sep = f.wrote
		case open >= 0:
			f.emit(&out, acc[:open])
			acc = acc[open+len(qwen35ThinkingOpenTag):]
			f.inThinking = true
		default:
			keep := 0
			if !done {
				keep = max(overlap(acc, qwen35ThinkingOpenTag), overlap(acc, qwen35ThinkingCloseTag))
			}
			f.emit(&out, acc[:len(acc)-keep])
			f.pending.WriteString(acc[len(acc)-keep:])
			return out.String()
		}
	}
}
