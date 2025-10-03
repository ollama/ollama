package parsers

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"encoding/json"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

// type parserState int

const (
	CollectingContent         qwenParserState = iota
	CollectingThinkingContent                 // this is because qwen3vl starts with <thinking>
	// parserState_CompletedThinkingContent
	CollectingToolContent
	// parserState_CompletedToolContent
)

const (
	thinkingOpenTag  = "<thinking>"
	thinkingCloseTag = "</thinking>"
)

type Qwen3VLParser struct {
	state  qwenParserState
	buffer strings.Builder
	tools  []api.Tool
}

func (p *Qwen3VLParser) HasToolSupport() bool {
	return true
}

func (p *Qwen3VLParser) HasThinkingSupport() bool {
	return true
}

func (p *Qwen3VLParser) Init(tools []api.Tool, lastMessage *api.Message) []api.Tool {
	p.tools = tools
	return tools // Qwen doesn't modify tools
	// does qwenvl modify tools?
}

// Add processes a chunk of string output from the model, accumulating it in the parser's buffer,
// and then parses any complete events (such as tool calls or content) that can be extracted from the buffer.
// It returns the parsed content (as a string), an empty string for "thinking" (since this parser does not support it),
// a slice of parsed tool calls, and an error if any occurred during parsing.
//
// Specifically, it works as follows:
//   1. Appends the new string chunk 's' to the internal accumulator.
//   2. Calls parseEvents() to extract any complete events (tool calls or content) from the buffer.
//   3. Iterates over the events:
//        - For tool call events, attempts to parse them into api.ToolCall objects and collects them.
//        - For content events, appends their content to a string builder.
//   4. Returns the accumulated content, an empty string for thinking, the collected tool calls, and any error encountered.

type qwenEventThinkingContent struct {
	content string
}

func (qwenEventThinkingContent) isQwenEvent() {}

func (p *Qwen3VLParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	// is s the complete content (aka the for sure unambiguous content)
	p.buffer.WriteString(s)
	// why do we write the entire string?

	events := p.parseEvents()
	// parse events:
	// - parses the entire content
	// parses an entire tool call
	// parses an entire thinking content

	var toolCalls []api.ToolCall
	var sb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case qwenEventRawToolCall:
			toolCall, err := parseToolCall(event, p.tools)
			if err != nil {
				slog.Warn("qwen tool call parsing failed", "error", err)
				return "", "", nil, err
			}
			toolCalls = append(toolCalls, toolCall)
		case qwenEventThinkingContent: // maybe we only need one?
			print("unimplemented")
			// how exactly does thinking work?
		case qwenEventContent:
			// TODO(drifkin): if the same turn contains multiple interleaved content
			// events, we naively append them together here. See the note below about
			// `qwenEvent`s for more details
			sb.WriteString(event.content)
		}
	}

	return sb.String(), "", toolCalls, nil
}

func (p *Qwen3VLParser) parseEvents() []qwenEvent {
	var all []qwenEvent

	keepLooping := true
	for keepLooping {
		var events []qwenEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "qwen events parsed", "events", all, "state", p.state, "buffer", p.buffer.String())
	}

	return all
}

// type qwenEventRawToolCall struct {
// 	raw string
// }

// type qwenEventContent struct {
// 	content string
// }

// think if a better name
func emitContentBeforeTag(p *Qwen3VLParser, events []qwenEvent, tag string) []qwenEvent {
	split := strings.SplitN(p.buffer.String(), tag, 2) // what is his 2 for?
	before := split[0]                                 // before the tag
	// before = strings.TrimRightFunc(before, unicode.IsSpace) // trim all the space after the bfire
	if len(before) > 0 {
		events = append(events, qwenEventContent{content: before})
	}
	after := split[1]
	p.buffer.Reset()
	p.buffer.WriteString(after)
	return events
}

// overlap = ambiguous

// findFirstTag returns the tag that appears first in the buffer among the provided tags.
// If no tag is found, it returns an empty string.
func findFirstTag(p *Qwen3VLParser, tags []string) string {
	minIdx := -1
	var firstTag string
	for _, tag := range tags {
		idx := strings.Index(p.buffer.String(), tag)
		if idx != -1 && (minIdx == -1 || idx < minIdx) {
			minIdx = idx
			firstTag = tag
		}
	}
	if minIdx == -1 { // just content
		return ""
	}
	return firstTag // there is a possibility that there is no tag, can you return nil for that?
}

func (p *Qwen3VLParser) eat() ([]qwenEvent, bool) {
	var events []qwenEvent

	// certain events:
	// - thinking opening tag
	// - tool opening tag

	// since there is multiple tags, we need to think about which tag comes first
	// we also need to create a list for
	firstTag := findFirstTag(p, []string{thinkingOpenTag, toolOpenTag})

	switch p.state {
	case CollectingContent: // we  can only look for thinking content if we're collecting content

		// if strings.Contains(p.buffer.String(), thinkingOpenTag) { // found thinking
		if firstTag == thinkingOpenTag {
			// string contains the openThinkingTag, we move it to the CollectingThinkingContent state
			events = emitContentBeforeTag(p, events, thinkingOpenTag)
			p.state = CollectingThinkingContent // <found a thinking>
			return events, true
			// } else if strings.Contains(p.buffer.String(), toolOpenTag) { // found tool call
		} else if firstTag == toolOpenTag {
			events = emitContentBeforeTag(p, events, toolOpenTag)
			p.state = CollectingToolContent // found a <tool_call>
			return events, true
		} else if overlapLen := overlap(p.buffer.String(), thinkingOpenTag); overlapLen > 0 { // found a partial thinking tag
			// it is only possible that they find 1
			// found a partial think tag, emit the unambiguous before the partial tool call
			// hello </think -> hello, so ambiguous start includes all the whitespace before the tag
			beforePartialTag := p.buffer.String()[:len(p.buffer.String())-overlapLen]
			ambiguousStart := len(beforePartialTag)
			// HAVENT ADDED TRAILING WHITESPACE YET...
			unambiguous := p.buffer.String()[:ambiguousStart]
			ambiguous := p.buffer.String()[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			events = append(events, qwenEventContent{content: unambiguous})
			return events, false
		} else if overlapLen := overlap(p.buffer.String(), toolOpenTag); overlapLen > 0 { // found a partial tool call tag
			beforePartialTag := p.buffer.String()[:len(p.buffer.String())-overlapLen]
			ambiguousStart := len(beforePartialTag)

			unambiguous := p.buffer.String()[:ambiguousStart]
			ambiguous := p.buffer.String()[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			events = append(events, qwenEventContent{content: unambiguous})
			return events, false
		} else { // no partial or full thinking or tool call tag found
			// whitespaceLen := trailingWhitespaceLen(p.buffer.String()) <- all the trailing space we consider ambiguous
			ambiguousStart := len(p.buffer.String()) // - whitespaceLen
			unambiguous := p.buffer.String()[:ambiguousStart]
			ambiguous := p.buffer.String()[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwenEventContent{content: unambiguous})
			}
			return events, false
		}
	case CollectingToolContent: // we only move towards the CollectingContent state
		if strings.Contains(p.buffer.String(), toolCloseTag) {
			split := strings.SplitN(p.buffer.String(), toolCloseTag, 2) // this one splits by the first one
			before := split[0]
			if len(before) == 0 {
				slog.Warn("qwen tool call closing tag found but no content before it")
			}
			after := split[1]                                          // no whit space yet
			events = append(events, qwenEventRawToolCall{raw: before}) // do these need to be "seperated"?
			p.buffer.Reset()
			p.buffer.WriteString(after)
			p.state = CollectingContent
			return events, true
		} else {
			return events, false
		}
	case CollectingThinkingContent:
		if strings.Contains(p.buffer.String(), thinkingCloseTag) {
			split := strings.SplitN(p.buffer.String(), thinkingCloseTag, 2)
			// so it looks like before contains the open tag
			fmt.Println("split", split)
			before := split[0]
			if len(before) == 0 {
				slog.Warn("qwen tool call closing tag found but no content before it")
			}
			after := split[1] // no whit space yet
			events = append(events, qwenEventThinkingContent{content: before})
			p.buffer.Reset()
			p.buffer.WriteString(after)
			p.state = CollectingContent
			return events, true
		} else {
			return events, false
		}
	default:
		panic("unreachable")
	}
}

func parseJSONToolCall(raw qwenEventRawToolCall, tools []api.Tool) (api.ToolCall, error) {
	// Expected JSON shape: {"name": "...", "arguments": { ... }}
	// var in struct {
	// 	Name      string          `json:"name"`
	// 	Arguments json.RawMessage `json:"arguments"`
	// }
	fmt.Println(raw.raw)

	var toolCall api.ToolCall
	if err := json.Unmarshal([]byte(raw.raw), &toolCall); err != nil {
		return api.ToolCall{}, err
	}

	// args := make(api.ToolCallFunctionArguments)
	// 	if len(in.Arguments) > 0 && string(in.Arguments) != "null" {
	// 	var obj map[string]any
	// 	if err := json.Unmarshal(in.Arguments, &obj); err == nil {
	// 		for k, v := range obj {
	// 			args[k] = v
	// 		}
	// 	}
	// }
	fmt.Println(toolCall)
	return toolCall, nil
}

// do we need to parse values
