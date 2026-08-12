package parsers

import (
	"fmt"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

type Nemotron3NanoParserState int

const (
	Nemotron3NanoCollectingThinking Nemotron3NanoParserState = iota
	Nemotron3NanoSkipWhitespaceAfterThinking
	Nemotron3NanoCollectingContent
)

const (
	nemotronThinkOpen    = "<think>"
	nemotronThinkClose   = "</think>"
	nemotronToolCallOpen = "<tool_call>"
)

type Nemotron3NanoParser struct {
	state                  Nemotron3NanoParserState
	buffer                 strings.Builder
	toolParser             *Qwen3CoderParser
	maybeThinkingOpenAtBOL bool
	skipThinkingLeadingWS  bool
}

func (p *Nemotron3NanoParser) HasToolSupport() bool     { return true }
func (p *Nemotron3NanoParser) HasThinkingSupport() bool { return true }

func (p *Nemotron3NanoParser) PreservedTokens() []string {
	return []string{
		nemotronThinkOpen,
		nemotronThinkClose,
		toolOpenTag,
		toolCloseTag,
	}
}

func (p *Nemotron3NanoParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.toolParser = &Qwen3CoderParser{}
	p.toolParser.Init(tools, nil, nil)
	p.buffer.Reset()
	p.maybeThinkingOpenAtBOL = false
	p.skipThinkingLeadingWS = false

	thinkingEnabled := thinkValue == nil || thinkValue.Bool()
	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	if !thinkingEnabled || (prefill && lastMessage.Content != "") {
		p.state = Nemotron3NanoCollectingContent
	} else {
		p.state = Nemotron3NanoCollectingThinking
		p.maybeThinkingOpenAtBOL = true
	}

	return tools
}

func (p *Nemotron3NanoParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	if p.state == Nemotron3NanoCollectingContent {
		return p.addContent(s, done)
	}

	if p.state == Nemotron3NanoSkipWhitespaceAfterThinking {
		s = strings.TrimLeftFunc(s, unicode.IsSpace)
		if s == "" {
			return "", "", nil, nil
		}
		p.state = Nemotron3NanoCollectingContent
		return p.addContent(s, done)
	}

	// Nemotron3NanoCollectingThinking - buffer and look for end markers
	p.buffer.WriteString(s)
	if p.skipThinkingLeadingWS {
		trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
		p.buffer.Reset()
		p.buffer.WriteString(trimmed)
		if trimmed == "" {
			return "", "", nil, nil
		}
		p.skipThinkingLeadingWS = false
	}
	if p.stripOpeningThinkTag() {
		return p.Add("", done)
	}
	if p.maybeThinkingOpenAtBOL {
		bufStr := p.buffer.String()
		trimmed := strings.TrimLeftFunc(bufStr, unicode.IsSpace)
		if trimmed == "" || overlap(trimmed, nemotronThinkOpen) == len(trimmed) {
			if len(trimmed) != len(bufStr) {
				p.buffer.Reset()
				p.buffer.WriteString(trimmed)
			}
			if done {
				thinking = p.buffer.String()
				p.buffer.Reset()
				p.maybeThinkingOpenAtBOL = false
				return "", thinking, nil, nil
			}
			return "", "", nil, nil
		}
	}
	bufStr := p.buffer.String()

	// Look for end of thinking: </think> or <tool_call> (model may skip </think>)
	thinkIdx := strings.Index(bufStr, nemotronThinkClose)
	toolIdx := strings.Index(bufStr, nemotronToolCallOpen)

	var endIdx int = -1
	var remainder string

	if thinkIdx != -1 && (toolIdx == -1 || thinkIdx < toolIdx) {
		endIdx = thinkIdx
		remainder = strings.TrimLeftFunc(bufStr[thinkIdx+len(nemotronThinkClose):], unicode.IsSpace)
	} else if toolIdx != -1 {
		endIdx = toolIdx
		remainder = bufStr[toolIdx:] // Include <tool_call> tag
	}

	if endIdx != -1 {
		thinking = strings.TrimRightFunc(bufStr[:endIdx], unicode.IsSpace)
		p.buffer.Reset()

		if remainder == "" {
			p.state = Nemotron3NanoSkipWhitespaceAfterThinking
		} else {
			p.state = Nemotron3NanoCollectingContent
			content, _, calls, err = p.addContent(remainder, done)
		}
		return content, thinking, calls, err
	}

	// No end marker - emit unambiguous thinking
	thinking = p.emitThinking(bufStr)
	if done {
		thinking += p.buffer.String()
		p.buffer.Reset()
	}
	return "", thinking, nil, nil
}

// addContent keeps Qwen's shared parser behavior unchanged while making the
// Nemotron wrapper account for everything still buffered at end of stream.
func (p *Nemotron3NanoParser) addContent(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	content, thinking, calls, err = p.toolParser.Add(s, done)
	if err != nil || !done {
		return content, thinking, calls, err
	}

	switch p.toolParser.state {
	case qwenParserState_LookingForToolStart:
		content += p.toolParser.acc.String()
		p.toolParser.acc.Reset()
	case qwenParserState_CollectingToolContent:
		return content, thinking, calls, fmt.Errorf("unterminated %s", toolOpenTag)
	}
	return content, thinking, calls, nil
}

// emitThinking returns unambiguous thinking content, keeping potential partial tags in buffer
func (p *Nemotron3NanoParser) emitThinking(bufStr string) string {
	// Check for partial </think> or <tool_call> at end
	thinkOverlap := overlap(bufStr, nemotronThinkClose)
	toolOverlap := overlap(bufStr, nemotronToolCallOpen)
	maxOverlap := max(thinkOverlap, toolOverlap)

	if maxOverlap > 0 {
		beforePartialTag := bufStr[:len(bufStr)-maxOverlap]
		ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen(beforePartialTag)
		unambiguous := bufStr[:ambiguousStart]
		p.buffer.Reset()
		p.buffer.WriteString(bufStr[ambiguousStart:])
		return unambiguous
	}

	// No partial tags - emit all but trailing whitespace
	wsLen := trailingWhitespaceLen(bufStr)
	if wsLen > 0 {
		unambiguous := bufStr[:len(bufStr)-wsLen]
		p.buffer.Reset()
		p.buffer.WriteString(bufStr[len(bufStr)-wsLen:])
		return unambiguous
	}

	// Nothing to hold back
	p.buffer.Reset()
	return bufStr
}

func (p *Nemotron3NanoParser) stripOpeningThinkTag() bool {
	if !p.maybeThinkingOpenAtBOL {
		return false
	}

	bufStr := p.buffer.String()
	trimmed := strings.TrimLeftFunc(bufStr, unicode.IsSpace)
	if trimmed == "" {
		p.buffer.Reset()
		return false
	}

	if strings.HasPrefix(trimmed, nemotronThinkOpen) {
		p.buffer.Reset()
		p.buffer.WriteString(strings.TrimLeftFunc(trimmed[len(nemotronThinkOpen):], unicode.IsSpace))
		p.maybeThinkingOpenAtBOL = false
		p.skipThinkingLeadingWS = true
		return true
	}

	if overlap(trimmed, nemotronThinkOpen) == len(trimmed) {
		if len(trimmed) != len(bufStr) {
			p.buffer.Reset()
			p.buffer.WriteString(trimmed)
		}
		return false
	}

	p.maybeThinkingOpenAtBOL = false
	return false
}
