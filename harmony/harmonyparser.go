package harmony

import (
	"encoding/json"
	"fmt"
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
	state           harmonyMessageState
	HarmonyParser   *HarmonyParser
	FunctionNameMap *FunctionNameMap
	toolAccumulator *HarmonyToolCallAccumulator
	convertedTools  map[string]struct{}
}

// NewHarmonyMessageHandler creates a new message handler
func NewHarmonyMessageHandler() *HarmonyMessageHandler {
	return &HarmonyMessageHandler{
		state: harmonyMessageState_Normal,
		HarmonyParser: &HarmonyParser{
			MessageStartTag: "<|start|>",
			MessageEndTag:   "<|end|>",
			HeaderEndTag:    "<|message|>",
		},
		FunctionNameMap: NewFunctionNameMap(),
		convertedTools:  make(map[string]struct{}),
	}
}

// AddContent processes the content and returns the content, thinking, and tool content.
// content and thinking are already fully parsed, but tool content still needs to be passed to the tool parser
func (h *HarmonyMessageHandler) AddContent(content string, toolParser *HarmonyToolCallAccumulator) (string, string, string) {
	contentSb := strings.Builder{}
	thinkingSb := strings.Builder{}
	toolContentSb := strings.Builder{}

	events := h.HarmonyParser.AddContent(content)
	for _, event := range events {
		switch event := event.(type) {
		case HarmonyEventHeaderComplete:
			logutil.Trace("harmony event header complete", "header", event.Header)
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
			logutil.Trace("harmony event content", "content", event.Content, "state", h.state)
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

// FunctionNameMap maps a user-specified function name to a valid function
// name for harmony (which look like TypeScript identifiers). This is needed to
// transform user-specified function names, which might contain characters that
// are not allowed in TypeScript identifiers
type FunctionNameMap struct {
	userToHarmony map[string]string
	harmonyToUser map[string]string
}

func NewFunctionNameMap() *FunctionNameMap {
	return &FunctionNameMap{
		userToHarmony: make(map[string]string),
		harmonyToUser: make(map[string]string),
	}
}

// Init initializes the handler with tools, optional last message, and think value
// Implements the Parser interface
func (h *HarmonyMessageHandler) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	// Initialize the harmony parser
	if h.HarmonyParser == nil {
		h.HarmonyParser = &HarmonyParser{
			MessageStartTag: "<|start|>",
			MessageEndTag:   "<|end|>",
			HeaderEndTag:    "<|message|>",
		}
	}

	// Handle prefill for chat mode
	if lastMessage != nil {
		h.HarmonyParser.AddImplicitStartOrPrefill(lastMessage)
	} else {
		h.HarmonyParser.AddImplicitStart()
	}

	// Initialize tool accumulator
	h.toolAccumulator = h.CreateToolParser()

	// Process tools and return renamed versions
	if len(tools) == 0 {
		return tools
	}

	processedTools := make([]api.Tool, len(tools))
	copy(processedTools, tools)
	for i, tool := range processedTools {
		if tool.Function.Name != "" {
			processedTools[i].Function.Name = h.FunctionNameMap.ConvertAndAdd(tool.Function.Name)
			h.convertedTools[tool.Function.Name] = struct{}{}
		}
	}
	return processedTools
}

// Add implements the Parser interface - processes streamed content and extracts content, thinking, and tool calls
func (h *HarmonyMessageHandler) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	content, thinking, toolContent := h.AddContent(s, h.toolAccumulator)
	if toolContent != "" {
		h.toolAccumulator.Add(toolContent)
	}

	// tool calls always happen one at a time, and always at the end of a message,
	// so for simplicity we defer parsing them until we know we're done
	if done {
		toolName, raw := h.toolAccumulator.Drain()
		if toolName != nil {
			name := strings.TrimPrefix(*toolName, "functions.")
			name = h.FunctionNameMap.OriginalFromConverted(name)
			var args api.ToolCallFunctionArguments
			if err := json.Unmarshal([]byte(raw), &args); err != nil {
				return "", "", nil, fmt.Errorf("error parsing tool call: raw='%s', err=%w", raw, err)
			}
			calls = append(calls, api.ToolCall{Function: api.ToolCallFunction{Name: name, Arguments: args}})
		}
	}

	return content, thinking, calls, nil
}

// HasToolSupport implements the Parser interface
func (h *HarmonyMessageHandler) HasToolSupport() bool {
	return true
}

// HasThinkingSupport implements the Parser interface
func (h *HarmonyMessageHandler) HasThinkingSupport() bool {
	return true
}

func (m *FunctionNameMap) ConvertAndAdd(userFunctionName string) string {
	harmonyFunctionName := m.deriveName(userFunctionName)
	// built-in functions should not be renamed
	if userFunctionName == "browser.open" || userFunctionName == "browser.search" || userFunctionName == "browser.find" || userFunctionName == "python" {
		harmonyFunctionName = userFunctionName
	}
	m.userToHarmony[userFunctionName] = harmonyFunctionName
	m.harmonyToUser[harmonyFunctionName] = userFunctionName
	return harmonyFunctionName
}

// OriginalFromConverted looks up the reverse-mapping of a previously-converted
// user->harmony function name. To unmap reliably, the mapping must exist, as
// the conversion process is not reversible without the appropriate state
func (m *FunctionNameMap) OriginalFromConverted(harmonyFunctionName string) string {
	if userFunctionName, ok := m.harmonyToUser[harmonyFunctionName]; ok {
		return userFunctionName
	}
	slog.Warn("harmony parser: no reverse mapping found for function name", "harmonyFunctionName", harmonyFunctionName)
	// fallback to the original function name if we can't find a mapping
	return harmonyFunctionName
}

// convertToValidChars converts a user-specified function name to a valid
// TypeScript identifier.
//
// Limitations:
//
//   - This doesn't restrict reserved TypeScript keywords.
//   - We don't perform a real ID_Start/ID_Continue check, and instead use the more
//     restrictive unicode.IsLetter/unicode.IsDigit check. Unclear what kind of
//     identifiers these models were trained on, so in the end we might want to
//     convert unicode-heavy identifiers to their closest ASCII equivalents.
func (m *FunctionNameMap) convertToValidChars(userFunctionName string) string {
	mapper := func(r rune) rune {
		// first, replace certain characters with underscores
		if r == ' ' || r == '-' || r == '.' {
			return '_'
		}

		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '$' {
			return r
		}

		// finally, remove any other characters
		return -1
	}
	candidate := strings.Map(mapper, userFunctionName)

	// set a default name if we end up with nothing left
	if candidate == "" {
		return "unnamed"
	}

	// if the candidate starts with a number, prepend an underscore to make it a
	// valid identifier
	if unicode.IsDigit(rune(candidate[0])) {
		candidate = "_" + candidate
	}

	return candidate
}

func (m *FunctionNameMap) deriveName(userFunctionName string) string {
	originalCandidate := m.convertToValidChars(userFunctionName)
	candidate := originalCandidate

	// Check for dupes, and if so, add a number to the end.
	// We start at 2 because if we have dupes and the first is never renamed, it
	// makes sense for them to be named, say, `f`, `f_2`, `f_3`
	count := 2
	for {
		if _, exists := m.harmonyToUser[candidate]; !exists {
			break
		}
		candidate = fmt.Sprintf("%s_%d", originalCandidate, count)
		count++
	}

	return candidate
}
