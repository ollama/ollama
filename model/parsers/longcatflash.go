package parsers

import (
	"encoding/json"
	"errors"
	"log/slog"
	"strconv"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
)

type LongcatParserState int

const (
	LongcatCollectingThinking LongcatParserState = iota
	LongcatCollectingContent
	LongcatCollectingToolCalls
)

const (
	lcThinkOpen     = "<longcat_think>"
	lcThinkClose    = "</longcat_think>"
	lcToolCallOpen  = "<longcat_tool_call>"
	lcToolCallClose = "</longcat_tool_call>"
	lcArgKeyOpen    = "<longcat_arg_key>"
	lcArgKeyClose   = "</longcat_arg_key>"
	lcArgValueOpen  = "<longcat_arg_value>"
	lcArgValueClose = "</longcat_arg_value>"
)

type LongcatParser struct {
	state                    LongcatParserState
	buffer                   strings.Builder
	hasThinkingSupport       bool
	needsThinkingLeadingTrim bool
	needsContentLeadingTrim  bool
}

func (p *LongcatParser) HasToolSupport() bool {
	return true
}

func (p *LongcatParser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *LongcatParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"
	thinkingEnabled := p.HasThinkingSupport() && (thinkValue != nil && thinkValue.Bool())

	if !thinkingEnabled {
		p.state = LongcatCollectingContent
		return tools
	}

	if prefill && lastMessage.Content != "" {
		p.state = LongcatCollectingContent
		return tools
	}

	p.state = LongcatCollectingThinking
	p.needsThinkingLeadingTrim = true
	return tools
}

type longcatEvent interface {
	isLongcatEvent()
}

type longcatEventThinkingContent struct {
	content string
}

type longcatEventContent struct {
	content string
}

type longcatEventToolCall struct {
	toolCall api.ToolCall
}

func (longcatEventThinkingContent) isLongcatEvent() {}
func (longcatEventContent) isLongcatEvent()         {}
func (longcatEventToolCall) isLongcatEvent()        {}

func (p *LongcatParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder

	for _, event := range events {
		switch event := event.(type) {
		case longcatEventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		case longcatEventThinkingContent:
			thinkingSb.WriteString(event.content)
		case longcatEventContent:
			contentSb.WriteString(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *LongcatParser) parseEvents() []longcatEvent {
	var all []longcatEvent
	keepLooping := true
	for keepLooping {
		var events []longcatEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}
	return all
}

func (p *LongcatParser) eat() ([]longcatEvent, bool) {
	var events []longcatEvent
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case LongcatCollectingThinking:
		if strings.HasPrefix(bufStr, lcThinkOpen) {
			bufStr = bufStr[len(lcThinkOpen):]
			p.needsThinkingLeadingTrim = true
			p.buffer.Reset()
			p.buffer.WriteString(bufStr)
		}

		if p.needsThinkingLeadingTrim {
			if trimmed := strings.TrimLeftFunc(bufStr, unicode.IsSpace); trimmed != bufStr {
				bufStr = trimmed
				p.buffer.Reset()
				p.buffer.WriteString(bufStr)
			}
			if len(bufStr) > 0 {
				p.needsThinkingLeadingTrim = false
			}
		}

		if idx := strings.Index(bufStr, lcThinkClose); idx != -1 {
			thinkingContent := bufStr[:idx]
			remaining := bufStr[idx+len(lcThinkClose):]

			thinkingContent = strings.TrimRightFunc(thinkingContent, unicode.IsSpace)
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			if len(thinkingContent) > 0 {
				events = append(events, longcatEventThinkingContent{content: thinkingContent})
			}

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = LongcatCollectingContent
			p.needsContentLeadingTrim = len(remaining) == 0
			return events, true
		}

		if overlapLen := overlap(bufStr, lcThinkClose); overlapLen > 0 {
			safeLen := len(bufStr) - overlapLen
			if safeLen > 0 {
				safeContent := bufStr[:safeLen]
				events = append(events, longcatEventThinkingContent{content: safeContent})
				p.buffer.Reset()
				p.buffer.WriteString(bufStr[safeLen:])
			}
			return events, false
		} else {
			p.buffer.Reset()
			if len(bufStr) > 0 {
				events = append(events, longcatEventThinkingContent{content: bufStr})
			}
			return events, false
		}

	case LongcatCollectingContent:
		if p.needsContentLeadingTrim {
			if trimmed := strings.TrimLeftFunc(bufStr, unicode.IsSpace); trimmed != bufStr {
				bufStr = trimmed
				p.buffer.Reset()
				p.buffer.WriteString(bufStr)
			}
			if len(bufStr) > 0 {
				p.needsContentLeadingTrim = false
			}
		}

		if idx := strings.Index(bufStr, lcToolCallOpen); idx != -1 {
			contentBefore := bufStr[:idx]
			remaining := bufStr[idx:]
			remaining = remaining[len(lcToolCallOpen):]

			if len(contentBefore) > 0 {
				events = append(events, longcatEventContent{content: contentBefore})
			}

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = LongcatCollectingToolCalls
			return events, true
		}

		if overlapLen := overlap(bufStr, lcToolCallOpen); overlapLen > 0 {
			safeLen := len(bufStr) - overlapLen
			if safeLen > 0 {
				events = append(events, longcatEventContent{content: bufStr[:safeLen]})
				p.buffer.Reset()
				p.buffer.WriteString(bufStr[safeLen:])
			}
			return events, false
		}

		p.buffer.Reset()
		if len(bufStr) > 0 {
			events = append(events, longcatEventContent{content: bufStr})
		}
		return events, false

	case LongcatCollectingToolCalls:
		// We are inside a block of tool calls. Longcat tools are wrapped individually:
		// <longcat_tool_call>... content ...</longcat_tool_call>
		// We entered this state after eating one Open tag.

		// Look for the closing tag
		if idx := strings.Index(bufStr, lcToolCallClose); idx != -1 {
			// We have a full tool call block
			rawToolCall := bufStr[:idx]
			remaining := bufStr[idx+len(lcToolCallClose):]

			toolCall, err := p.parseLongcatToolRaw(rawToolCall)
			if err == nil {
				events = append(events, longcatEventToolCall{toolCall: toolCall})
			} else {
				slog.Warn("longcat tool call parsing failed", "error", err, "content", rawToolCall)
			}

			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)
			if strings.HasPrefix(remaining, lcToolCallOpen) {
				remaining = remaining[len(lcToolCallOpen):]
				p.buffer.Reset()
				p.buffer.WriteString(remaining)
				return events, true
			} else {
				p.buffer.Reset()
				p.buffer.WriteString(remaining)
				p.state = LongcatCollectingContent
				return events, true
			}
		}

		// If we don't have a closing tag yet, simply wait for more data.
		return events, false
	}

	return events, false
}

func (p *LongcatParser) parseLongcatToolRaw(content string) (api.ToolCall, error) {
	content = strings.TrimSpace(content)

	nameEnd := strings.Index(content, "<")
	var funcName string
	var argsContent string

	if nameEnd == -1 {
		funcName = strings.TrimSpace(content)
		argsContent = ""
	} else {
		funcName = strings.TrimSpace(content[:nameEnd])
		argsContent = content[nameEnd:]
	}

	if funcName == "" {
		return api.ToolCall{}, errors.New("empty function name")
	}

	args := api.NewToolCallFunctionArguments()
	for len(argsContent) > 0 {
		kStart := strings.Index(argsContent, lcArgKeyOpen)
		if kStart == -1 {
			break
		}
		kContentStart := kStart + len(lcArgKeyOpen)
		kEnd := strings.Index(argsContent[kContentStart:], lcArgKeyClose)
		if kEnd == -1 {
			break
		}
		kEnd += kContentStart

		key := strings.TrimSpace(argsContent[kContentStart:kEnd])
		argsContent = argsContent[kEnd+len(lcArgKeyClose):]

		vStart := strings.Index(argsContent, lcArgValueOpen)
		if vStart == -1 {
			break
		}
		vContentStart := vStart + len(lcArgValueOpen)
		vEnd := strings.Index(argsContent[vContentStart:], lcArgValueClose)
		if vEnd == -1 {
			break
		}
		vEnd += vContentStart

		valStr := strings.TrimSpace(argsContent[vContentStart:vEnd])
		parseAndSetArg(args, key, valStr)
		argsContent = argsContent[vEnd+len(lcArgValueClose):]
	}

	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      funcName,
			Arguments: args,
		},
	}, nil
}

func parseAndSetArg(args api.ToolCallFunctionArguments, key string, valStr string) {
	var v interface{}
	if err := json.Unmarshal([]byte(valStr), &v); err == nil {
		args.Set(key, v)
		return
	}

	if i, err := strconv.ParseInt(valStr, 10, 64); err == nil {
		args.Set(key, i)
		return
	}
	if f, err := strconv.ParseFloat(valStr, 64); err == nil {
		args.Set(key, f)
		return
	}
	if valStr == "true" {
		args.Set(key, true)
		return
	}
	if valStr == "false" {
		args.Set(key, false)
		return
	}

	args.Set(key, valStr)
}
