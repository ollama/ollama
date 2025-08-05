package server

import (
	"context"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

type harmonyParserState int

const (
	harmonyParserState_LookingForMessageStart harmonyParserState = iota
	harmonyParserState_ParsingHeader
	harmonyParserState_ParsingContent
)

func shouldUseHarmony(model Model) bool {
	if model.Config.ModelFamily == "gptoss" {
		// heuristic to check whether the template expects to be parsed via harmony:
		// search for harmony tags that are nearly always used
		if model.Template.Contains("<|start|>") && model.Template.Contains("<|end|>") {
			return true
		}
	}

	return false
}

func (s harmonyParserState) String() string {
	switch s {
	// we're looking for the message start tag
	case harmonyParserState_LookingForMessageStart:
		return "LookingForMessageStart"
	case harmonyParserState_ParsingHeader:
		return "ParsingHeader"
	case harmonyParserState_ParsingContent:
		return "ParsingContent"
	default:
		return "Unknown"
	}
}

type HarmonyParser struct {
	state           harmonyParserState
	MessageStartTag string
	MessageEndTag   string
	HeaderEndTag    string
	acc             strings.Builder
	lifetimeAcc     strings.Builder
}

type HarmonyEvent interface {
	isHarmonyEvent()
}

type HarmonyEventMessageStart struct{}

func (HarmonyEventMessageStart) isHarmonyEvent() {}

type HarmonyEventHeaderComplete struct {
	Header HarmonyHeader
}

func (HarmonyEventHeaderComplete) isHarmonyEvent() {}

type HarmonyEventContentEmitted struct {
	Content string
}

func (HarmonyEventContentEmitted) isHarmonyEvent() {}

type HarmonyEventMessageEnd struct{}

func (HarmonyEventMessageEnd) isHarmonyEvent() {}

type HarmonyHeader struct {
	Role      string
	Channel   string
	Recipient string
}

func (s *HarmonyParser) AddImplicitStart() {
	s.acc.WriteString("<|start|>assistant")
}

func (s *HarmonyParser) AddImplicitStartOrPrefill(lastMessage *api.Message) {
	if lastMessage != nil && lastMessage.Role == "assistant" {
		// handle prefilling conditions
		if lastMessage.Content != "" {
			s.acc.WriteString("<|start|>assistant<|channel|>final<|message|>")
			return
		} else if lastMessage.Thinking != "" {
			s.acc.WriteString("<|start|>assistant<|channel|>analysis<|message|>")
			return
		}
	}
	s.AddImplicitStart()
}

func (s *HarmonyParser) AddContent(content string) []HarmonyEvent {
	s.lifetimeAcc.WriteString(content)
	s.acc.WriteString(content)

	var events []HarmonyEvent

	keepLooping := true
	// we loop because we might pass through multiple parsing states in a single
	// call to addContent, and we want to make sure callers don't have to wait for
	// data that's already unambiguous
	for keepLooping {
		var newEvents []HarmonyEvent
		newEvents, keepLooping = eat(s)
		events = append(events, newEvents...)
	}

	return events
}

// the additional bool return is true iff we should continue eating
func eat(s *HarmonyParser) ([]HarmonyEvent, bool) {
	switch s.state {
	case harmonyParserState_LookingForMessageStart:
		// does the acc contain the message start tag?
		if strings.Contains(s.acc.String(), s.MessageStartTag) {
			// split the acc into the message start tag and the rest
			split := strings.SplitN(s.acc.String(), s.MessageStartTag, 2)
			before := split[0]
			if before != "" {
				slog.Warn("harmony parser: found message start tag in the middle of the content", "content", s.acc.String())
			}
			after := split[1]
			s.acc.Reset()
			s.acc.WriteString(after)
			s.state = harmonyParserState_ParsingHeader
			return []HarmonyEvent{HarmonyEventMessageStart{}}, true
		}

		// no match, so we keep accumulating
		return nil, false
	case harmonyParserState_ParsingHeader:
		if strings.Contains(s.acc.String(), s.HeaderEndTag) {
			split := strings.SplitN(s.acc.String(), s.HeaderEndTag, 2)
			header := split[0]
			after := split[1]
			s.acc.Reset()
			s.acc.WriteString(after)
			s.state = harmonyParserState_ParsingContent
			return []HarmonyEvent{HarmonyEventHeaderComplete{Header: s.parseHeader(header)}}, true
		}
		return nil, false
	case harmonyParserState_ParsingContent:
		if strings.Contains(s.acc.String(), s.MessageEndTag) {
			// if we already have the message end tag, we can emit the content up to it
			split := strings.SplitN(s.acc.String(), s.MessageEndTag, 2)
			content := split[0]
			after := split[1]
			s.acc.Reset()
			s.acc.WriteString(after)
			s.state = harmonyParserState_LookingForMessageStart
			events := []HarmonyEvent{}
			if content != "" {
				events = append(events, HarmonyEventContentEmitted{Content: content})
			}
			events = append(events, HarmonyEventMessageEnd{})
			return events, true
		} else if overlapLen := overlap(s.acc.String(), s.MessageEndTag); overlapLen > 0 {
			// if our suffix contains the start of the message end tag, we can emit
			// the content up to the start of the message end tag
			content := s.acc.String()[:len(s.acc.String())-overlapLen]
			remaining := s.acc.String()[len(s.acc.String())-overlapLen:]
			s.acc.Reset()
			s.acc.WriteString(remaining)
			// emit the content we know isn't part of the message end tag, and keep
			// accumulating to disambiguate the rest
			if content == "" {
				return nil, false
			}
			return []HarmonyEvent{HarmonyEventContentEmitted{Content: content}}, false
		} else {
			// no end tag, so it's still normal content that we can immediately emit
			content := s.acc.String()
			if content == "" {
				return nil, false
			}
			s.acc.Reset()
			return []HarmonyEvent{HarmonyEventContentEmitted{Content: content}}, false
		}
	}

	return nil, false
}

func (s *HarmonyParser) parseHeader(raw string) HarmonyHeader {
	harmonyHeader := HarmonyHeader{}

	// if `<|constrain|>` is present, ensure it has a space before it so it gets
	// parsed as a separate token, even if the model didn't include the space
	if strings.Contains(raw, "<|constrain|>") {
		raw = strings.Replace(raw, "<|constrain|>", " <|constrain|>", 1)
		raw = strings.TrimSpace(raw)
	}

	// look for the optional channel tag, which is `<|channel|>` followed by the
	// channel name, all without any whitespace
	channelIndex := strings.Index(raw, "<|channel|>")
	if channelIndex != -1 {
		before := raw[:channelIndex]
		after := raw[channelIndex+len("<|channel|>"):]
		// the channel name is `after` all the way up to the first (if any) whitespace character
		idx := strings.IndexFunc(after, func(r rune) bool {
			return unicode.IsSpace(r)
		})
		if idx == -1 {
			idx = len(after)
		}
		harmonyHeader.Channel = after[:idx]
		after = after[idx:]
		// now we remove the channel tag from the raw string to further process
		raw = before + after
		raw = strings.TrimSpace(raw)
	}

	// split the header into whitespace-separated tokens
	tokens := strings.Fields(raw)

	// the first token is treated as the role
	if len(tokens) == 0 {
		slog.Error("harmony parser: missing role in header", "header", raw)
		return harmonyHeader
	}
	role := tokens[0]
	tokens = tokens[1:]
	// special case: if role starts with to= then it's a tool call
	if strings.HasPrefix(role, "to=") {
		harmonyHeader.Recipient = role[3:]
		harmonyHeader.Role = "tool"
	} else {
		harmonyHeader.Role = role
	}

	// the recipient (if any) can be specified before or after the channel tag, so
	// we check it at the end once we've already parsed the channel and role
	if harmonyHeader.Recipient == "" && len(tokens) > 0 && strings.HasPrefix(tokens[0], "to=") {
		harmonyHeader.Recipient = tokens[0][3:]
	}

	return harmonyHeader
}

// longest overlap between suffix of s and prefix of delim
func overlap(s, delim string) int {
	max := min(len(delim), len(s))
	for i := max; i > 0; i-- {
		if strings.HasSuffix(s, delim[:i]) {
			return i
		}
	}
	return 0
}

// harmonyMessageState represents the current state of message processing
type harmonyMessageState int

const (
	harmonyMessageState_Normal harmonyMessageState = iota
	harmonyMessageState_Thinking
	harmonyMessageState_ToolCalling
)

// HarmonyMessageHandler processes harmony events and accumulates content appropriately.
// This is a higher level interface that maps harmony concepts into ollama concepts
type HarmonyMessageHandler struct {
	state         harmonyMessageState
	harmonyParser *HarmonyParser
}

// NewHarmonyMessageHandler creates a new message handler
func NewHarmonyMessageHandler() *HarmonyMessageHandler {
	return &HarmonyMessageHandler{
		state: harmonyMessageState_Normal,
		harmonyParser: &HarmonyParser{
			MessageStartTag: "<|start|>",
			MessageEndTag:   "<|end|>",
			HeaderEndTag:    "<|message|>",
		},
	}
}

// AddContent processes the content and returns the content, thinking, and tool content.
// content and thinking are already fully parsed, but tool content still needs to be passed to the tool parser
func (h *HarmonyMessageHandler) AddContent(content string, toolParser *HarmonyToolCallAccumulator) (string, string, string) {
	contentSb := strings.Builder{}
	thinkingSb := strings.Builder{}
	toolContentSb := strings.Builder{}

	events := h.harmonyParser.AddContent(content)
	for _, event := range events {
		switch event := event.(type) {
		case HarmonyEventHeaderComplete:
			slog.Log(context.TODO(), logutil.LevelTrace, "harmony event header complete", "header", event.Header)
			switch event.Header.Channel {
			case "analysis":
				if event.Header.Recipient != "" {
					h.state = harmonyMessageState_ToolCalling
					// event.Header.Recipient is the tool name, something like
					// "browser.search" for a built-in, or "functions.calc" for a
					// custom one
					toolParser.SetToolName(event.Header.Recipient)
				} else {
					h.state = harmonyMessageState_Thinking
				}
			case "commentary":
				if event.Header.Recipient != "" {
					h.state = harmonyMessageState_ToolCalling
					toolParser.SetToolName(event.Header.Recipient)
				} else {
					h.state = harmonyMessageState_Normal
				}
			case "final":
				h.state = harmonyMessageState_Normal
			}
		case HarmonyEventContentEmitted:
			slog.Log(context.TODO(), logutil.LevelTrace, "harmony event content", "content", event.Content, "state", h.state)
			if h.state == harmonyMessageState_Normal {
				contentSb.WriteString(event.Content)
			} else if h.state == harmonyMessageState_Thinking {
				thinkingSb.WriteString(event.Content)
			} else if h.state == harmonyMessageState_ToolCalling {
				toolContentSb.WriteString(event.Content)
			}
		case HarmonyEventMessageEnd:
			h.state = harmonyMessageState_Normal
		}
	}
	return contentSb.String(), thinkingSb.String(), toolContentSb.String()
}

func (h *HarmonyMessageHandler) CreateToolParser() *HarmonyToolCallAccumulator {
	return &HarmonyToolCallAccumulator{
		state:           harmonyToolCallState_Normal,
		currentToolName: nil,
	}
}

type harmonyToolCallState int

const (
	harmonyToolCallState_Normal harmonyToolCallState = iota
	harmonyToolCallState_ToolCalling
)

type HarmonyToolCallAccumulator struct {
	state           harmonyToolCallState
	acc             strings.Builder
	currentToolName *string
}

func (a *HarmonyToolCallAccumulator) SetToolName(toolName string) {
	a.currentToolName = &toolName
}

func (a *HarmonyToolCallAccumulator) Add(content string) {
	a.acc.WriteString(content)
}

func (a *HarmonyToolCallAccumulator) Drain() (*string, string) {
	str := a.acc.String()
	a.state = harmonyToolCallState_Normal
	a.acc.Reset()
	return a.currentToolName, str
}

func (a *HarmonyToolCallAccumulator) Content() string {
	return a.acc.String()
}
