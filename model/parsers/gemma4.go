package parsers

import (
	"encoding/json"
	"errors"
	"log/slog"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
)

type Gemma4ParserState int

const (
	Gemma4CollectingContent Gemma4ParserState = iota
	Gemma4CollectingThinking
	Gemma4CollectingToolCall
	Gemma4IgnoringPostToolCallNoise
)

const (
	gemma4ThinkingOpenTag  = "<|channel>"
	gemma4ThinkingCloseTag = "<channel|>"
	gemma4ToolCallOpenTag  = "<|tool_call>"
	gemma4ToolCallCloseTag = "<tool_call|>"
	gemma4StringDelimiter  = `<|"|>`
)

var (
	gemma4QuotedStringRe = regexp.MustCompile(`(?s)<\|"\|>(.*?)<\|"\|>`)
	gemma4BareKeyRe      = regexp.MustCompile(`([,{])(\w+):`)
)

type Gemma4Parser struct {
	state                 Gemma4ParserState
	buffer                strings.Builder
	tools                 []api.Tool
	callIndex             int
	hasThinkingSupport    bool
	thinkingEnabled       bool // true when both model supports and user requested thinking
	needsChannelNameStrip bool // true when we just entered thinking and need to strip "thought\n"
}

func (p *Gemma4Parser) HasToolSupport() bool {
	return true
}

func (p *Gemma4Parser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *Gemma4Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.callIndex = 0

	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	p.thinkingEnabled = p.HasThinkingSupport() && (thinkValue != nil && thinkValue.Bool())

	if !p.thinkingEnabled {
		p.state = Gemma4CollectingContent
		return tools
	}

	if prefill && lastMessage.Content != "" {
		p.state = Gemma4CollectingContent
		return tools
	}

	// When thinking is enabled, start in content mode but we'll switch to
	// thinking when we see <|channel>. The model typically starts with
	// <|channel> immediately when thinking is enabled.
	p.state = Gemma4CollectingContent
	return tools
}

type gemma4Event interface {
	isGemma4Event()
}

type gemma4EventThinkingContent struct {
	content string
}

type gemma4EventContent struct {
	content string
}

type gemma4EventToolCall struct {
	toolCall api.ToolCall
}

func (gemma4EventThinkingContent) isGemma4Event() {}
func (gemma4EventContent) isGemma4Event()         {}
func (gemma4EventToolCall) isGemma4Event()        {}

func (p *Gemma4Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents(done)

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case gemma4EventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		case gemma4EventThinkingContent:
			if p.thinkingEnabled {
				thinkingSb.WriteString(event.content)
			}
			// When thinking is disabled, silently discard channel content
		case gemma4EventContent:
			contentSb.WriteString(event.content)
		}
	}

	for i := range toolCalls {
		toolCalls[i].Function.Index = p.callIndex
		p.callIndex++
	}

	return contentSb.String(), thinkingSb.String(), toolCalls, nil
}

func (p *Gemma4Parser) parseEvents(done bool) []gemma4Event {
	var all []gemma4Event

	keepLooping := true
	for keepLooping {
		var events []gemma4Event
		events, keepLooping = p.eat(done)
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

// longestOverlap returns the longest overlap between the suffix of bufStr and
// a prefix of any of the given tags.
func longestOverlap(bufStr string, tags ...string) int {
	maxOverlap := 0
	for _, tag := range tags {
		if o := overlap(bufStr, tag); o > maxOverlap {
			maxOverlap = o
		}
	}
	return maxOverlap
}

func (p *Gemma4Parser) eat(done bool) ([]gemma4Event, bool) {
	var events []gemma4Event
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case Gemma4CollectingContent:
		// Check for thinking open tag
		if idx := strings.Index(bufStr, gemma4ThinkingOpenTag); idx != -1 {
			contentBefore := bufStr[:idx]
			remaining := bufStr[idx+len(gemma4ThinkingOpenTag):]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Gemma4CollectingThinking
			p.needsChannelNameStrip = true

			if contentBefore = strings.TrimRightFunc(contentBefore, unicode.IsSpace); len(contentBefore) > 0 {
				events = append(events, gemma4EventContent{content: contentBefore})
			}
			return events, true
		}

		// Check for tool call open tag
		if idx := strings.Index(bufStr, gemma4ToolCallOpenTag); idx != -1 {
			contentBefore := bufStr[:idx]
			remaining := bufStr[idx+len(gemma4ToolCallOpenTag):]

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Gemma4CollectingToolCall

			if contentBefore = strings.TrimRightFunc(contentBefore, unicode.IsSpace); len(contentBefore) > 0 {
				events = append(events, gemma4EventContent{content: contentBefore})
			}
			return events, true
		}

		// Check for partial tag overlap
		if !done {
			if overlapLen := longestOverlap(bufStr, gemma4ThinkingOpenTag, gemma4ToolCallOpenTag); overlapLen > 0 {
				beforePartialTag := bufStr[:len(bufStr)-overlapLen]
				trailingLen := trailingWhitespaceLen(beforePartialTag)
				ambiguousStart := len(beforePartialTag) - trailingLen

				unambiguous := bufStr[:ambiguousStart]
				ambiguous := bufStr[ambiguousStart:]
				p.buffer.Reset()
				p.buffer.WriteString(ambiguous)
				if len(unambiguous) > 0 {
					events = append(events, gemma4EventContent{content: unambiguous})
				}
				return events, false
			}
		}

		// No tags found, emit all content
		p.buffer.Reset()
		if len(bufStr) > 0 {
			events = append(events, gemma4EventContent{content: bufStr})
		}
		return events, false

	case Gemma4CollectingThinking:
		// Strip channel name (e.g., "thought\n") after <|channel>.
		// Gemma 4 format: <|channel>thought\n...content...<channel|>
		// In streaming mode, "thought" and "\n" may arrive in separate chunks.
		if p.needsChannelNameStrip {
			if strings.HasPrefix(bufStr, "thought\n") {
				bufStr = bufStr[len("thought\n"):]
				p.buffer.Reset()
				p.buffer.WriteString(bufStr)
				p.needsChannelNameStrip = false
			} else if !done && (bufStr == "thought" || strings.HasPrefix("thought\n", bufStr)) {
				// Partial match — wait for more data.
				return events, false
			} else {
				// No match (different channel name or no newline) — don't strip.
				p.needsChannelNameStrip = false
			}
		}

		if strings.Contains(bufStr, gemma4ThinkingCloseTag) {
			split := strings.SplitN(bufStr, gemma4ThinkingCloseTag, 2)
			thinking := strings.TrimRightFunc(split[0], unicode.IsSpace)
			remaining := strings.TrimLeftFunc(split[1], unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Gemma4CollectingContent

			if len(thinking) > 0 {
				events = append(events, gemma4EventThinkingContent{content: thinking})
			}
			return events, true
		}

		// Check for partial close tag
		if !done {
			if overlapLen := overlap(bufStr, gemma4ThinkingCloseTag); overlapLen > 0 {
				beforePartialTag := bufStr[:len(bufStr)-overlapLen]
				trailingLen := trailingWhitespaceLen(beforePartialTag)
				ambiguousStart := len(beforePartialTag) - trailingLen

				unambiguous := bufStr[:ambiguousStart]
				ambiguous := bufStr[ambiguousStart:]
				p.buffer.Reset()
				p.buffer.WriteString(ambiguous)
				if len(unambiguous) > 0 {
					events = append(events, gemma4EventThinkingContent{content: unambiguous})
				}
				return events, false
			}
		}

		// No close tag, emit thinking content (hold back trailing whitespace)
		if !done {
			whitespaceLen := trailingWhitespaceLen(bufStr)
			ambiguousStart := len(bufStr) - whitespaceLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, gemma4EventThinkingContent{content: unambiguous})
			}
		} else {
			p.buffer.Reset()
			if len(bufStr) > 0 {
				events = append(events, gemma4EventThinkingContent{content: bufStr})
			}
		}
		return events, false

	case Gemma4CollectingToolCall:
		if idx := strings.Index(bufStr, gemma4ToolCallCloseTag); idx != -1 {
			toolCallContent := bufStr[:idx]
			remaining := bufStr[idx+len(gemma4ToolCallCloseTag):]
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.state = Gemma4IgnoringPostToolCallNoise

			if toolCall, err := parseGemma4ToolCall(toolCallContent, p.tools); err == nil {
				events = append(events, gemma4EventToolCall{toolCall: toolCall})
			} else {
				slog.Warn("gemma4 tool call parsing failed", "error", err, "content", toolCallContent)
			}
			return events, true
		}

		// If done, flush any accumulated tool call content even without closing tag.
		// The model may hit a stop token before emitting <tool_call|>.
		if done && len(bufStr) > 0 {
			p.buffer.Reset()
			p.state = Gemma4CollectingContent
			if toolCall, err := parseGemma4ToolCall(bufStr, p.tools); err == nil {
				events = append(events, gemma4EventToolCall{toolCall: toolCall})
			} else {
				slog.Warn("gemma4 tool call flush on done failed", "error", err, "content", bufStr)
			}
			return events, false
		}

		// Wait for closing tag
		return events, false

	case Gemma4IgnoringPostToolCallNoise:
		// We've observed Gemma 4 occasionally emitting extra <tool_call|> tags
		// after a valid tool call. We suppress leading close tags in this immediate
		// post-tool-call state so the extra close tags do not leak into assistant
		// content.  The tradeoff is that if the model intentionally begins its next
		// content span with the literal string "<tool_call|>", we will erroneously
		// treat it as noise and drop it.
		bufStr = strings.TrimLeftFunc(bufStr, unicode.IsSpace)
		p.buffer.Reset()
		p.buffer.WriteString(bufStr)

		for strings.HasPrefix(bufStr, gemma4ToolCallCloseTag) {
			bufStr = strings.TrimLeftFunc(bufStr[len(gemma4ToolCallCloseTag):], unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(bufStr)
		}

		if bufStr == "" {
			return events, false
		}

		if strings.HasPrefix(gemma4ToolCallCloseTag, bufStr) {
			if done {
				p.buffer.Reset()
				p.state = Gemma4CollectingContent
			}
			return events, false
		}

		p.state = Gemma4CollectingContent
		return events, true
	}

	return events, false
}

// parseGemma4ToolCall parses a tool call in Gemma 4 format:
// call:NAME{key:value,key:value}
func parseGemma4ToolCall(content string, tools []api.Tool) (api.ToolCall, error) {
	// Expected format: call:NAME{args}
	if !strings.HasPrefix(content, "call:") {
		return api.ToolCall{}, errors.New("expected 'call:' prefix")
	}
	content = content[len("call:"):]

	// Find the opening brace for args
	braceIdx := strings.Index(content, "{")
	if braceIdx == -1 {
		return api.ToolCall{}, errors.New("expected '{' in tool call")
	}

	toolName := strings.TrimSpace(content[:braceIdx])
	argsStr := content[braceIdx:]

	// Convert Gemma 4 argument format to JSON
	jsonStr := gemma4ArgsToJSON(argsStr)

	var args api.ToolCallFunctionArguments
	if err := json.Unmarshal([]byte(jsonStr), &args); err != nil {
		repairedArgs, repairErr := repairGemma4ToolCallArgs(argsStr, toolName, tools)
		if repairErr != nil {
			return api.ToolCall{}, errors.Join(err, repairErr)
		}
		args = repairedArgs
	}

	return api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      toolName,
			Arguments: args,
		},
	}, nil
}

// gemma4ArgsToJSON converts Gemma 4's custom argument format to valid JSON.
func gemma4ArgsToJSON(s string) string {
	var quotedStrings []string
	text := gemma4QuotedStringRe.ReplaceAllStringFunc(s, func(match string) string {
		submatches := gemma4QuotedStringRe.FindStringSubmatch(match)
		quotedStrings = append(quotedStrings, submatches[1])
		return "\x00" + string(rune(len(quotedStrings)-1)) + "\x00"
	})

	text = gemma4BareKeyRe.ReplaceAllString(text, `$1"$2":`)

	for i, value := range quotedStrings {
		escaped, _ := json.Marshal(value)
		text = strings.ReplaceAll(text, "\x00"+string(rune(i))+"\x00", string(escaped))
	}

	return text
}

// repairGemma4ToolCallArgs is a best-effort repair after strict parsing fails.
// For example, if the model emits an unclosed gemma string as the last value,
// we can repair it by closing it with the gemma string delimiter.
func repairGemma4ToolCallArgs(argsStr, toolName string, tools []api.Tool) (api.ToolCallFunctionArguments, error) {
	for _, candidate := range gemma4RepairCandidates(argsStr, toolName, tools) {
		jsonStr := gemma4ArgsToJSON(candidate)

		var args api.ToolCallFunctionArguments
		if err := json.Unmarshal([]byte(jsonStr), &args); err == nil {
			return args, nil
		}
	}

	return api.ToolCallFunctionArguments{}, errors.New("repair failed to produce valid JSON arguments")
}

func gemma4ToolProperties(toolName string, tools []api.Tool) *api.ToolPropertiesMap {
	for i := range tools {
		if tools[i].Function.Name == toolName {
			return tools[i].Function.Parameters.Properties
		}
	}
	return nil
}

// gemma4RepairCandidates returns the small set of repaired argument strings we
// are willing to try after strict parsing fails. Each candidate still has to
// pass the normal Gemma4-to-JSON conversion and JSON unmarshal before it is used.
func gemma4RepairCandidates(argsStr, toolName string, tools []api.Tool) []string {
	seen := map[string]bool{}
	var candidates []string
	addCandidate := func(candidate string, allowMissingObjectClose bool) {
		original := candidate
		candidate = repairGemma4SingleQuotedValues(candidate)
		candidate = repairGemma4MissingStringDelimiter(candidate)
		if allowMissingObjectClose || candidate != original {
			candidate = repairGemma4MissingObjectClose(candidate)
		}
		if !seen[candidate] {
			candidates = append(candidates, candidate)
			seen[candidate] = true
		}
	}

	addCandidate(argsStr, false)
	if raw, ok := repairGemma4RawTerminalStringValue(argsStr, toolName, tools); ok {
		addCandidate(raw, true)
	}

	return candidates
}

// repairGemma4MissingStringDelimiter closes an unbalanced Gemma string marker.
// When the value is immediately followed by a closing brace/bracket, the marker
// is inserted before that structural close rather than after it.
func repairGemma4MissingStringDelimiter(s string) string {
	if strings.Count(s, gemma4StringDelimiter)%2 == 0 {
		return s
	}

	insertAt := gemma4TrimRightSpaceIndex(s)
	if insertAt > 0 && (s[insertAt-1] == '}' || s[insertAt-1] == ']') {
		insertAt--
	}

	var sb strings.Builder
	sb.Grow(len(s) + len(gemma4StringDelimiter))
	sb.WriteString(s[:insertAt])
	sb.WriteString(gemma4StringDelimiter)
	sb.WriteString(s[insertAt:])
	return sb.String()
}

// repairGemma4MissingObjectClose adds a final object close after another repair
// has made a truncated object plausible. Callers decide when that guardrail is
// satisfied; this helper only performs the mechanical insertion.
func repairGemma4MissingObjectClose(s string) string {
	trimmedStart := strings.TrimLeftFunc(s, unicode.IsSpace)
	if !strings.HasPrefix(trimmedStart, "{") {
		return s
	}

	trimmedEnd := gemma4TrimRightSpaceIndex(s)
	if trimmedEnd > 0 && s[trimmedEnd-1] == '}' {
		return s
	}

	return s[:trimmedEnd] + "}" + s[trimmedEnd:]
}

// repairGemma4SingleQuotedValues converts single-quoted argument values into
// Gemma string-delimited values. It also drops a stray Gemma delimiter that
// sometimes appears immediately after the closing single quote.
func repairGemma4SingleQuotedValues(s string) string {
	var sb strings.Builder
	sb.Grow(len(s))

	for i := 0; i < len(s); {
		if strings.HasPrefix(s[i:], gemma4StringDelimiter) {
			end := strings.Index(s[i+len(gemma4StringDelimiter):], gemma4StringDelimiter)
			if end == -1 {
				sb.WriteString(s[i:])
				break
			}

			end = i + len(gemma4StringDelimiter) + end + len(gemma4StringDelimiter)
			sb.WriteString(s[i:end])
			i = end
			continue
		}

		if s[i] == '"' {
			end := gemma4JSONQuotedStringEnd(s, i)
			if end != -1 {
				sb.WriteString(s[i:end])
				i = end
				continue
			}
		}

		if s[i] != ':' {
			sb.WriteByte(s[i])
			i++
			continue
		}

		sb.WriteByte(s[i])
		i++

		spaceEnd := gemma4SkipSpace(s, i)
		sb.WriteString(s[i:spaceEnd])
		i = spaceEnd
		if i >= len(s) || s[i] != '\'' {
			continue
		}

		value, end, ok := gemma4SingleQuotedValue(s, i)
		if !ok {
			continue
		}

		sb.WriteString(gemma4StringDelimiter)
		sb.WriteString(value)
		sb.WriteString(gemma4StringDelimiter)
		i = end
		if strings.HasPrefix(s[i:], gemma4StringDelimiter) {
			i += len(gemma4StringDelimiter)
		}
	}

	return sb.String()
}

func gemma4SingleQuotedValue(s string, start int) (string, int, bool) {
	var sb strings.Builder
	escaped := false
	for i := start + 1; i < len(s); i++ {
		if s[i] == '\'' && !escaped {
			return sb.String(), i + 1, true
		}

		sb.WriteByte(s[i])
		escaped = s[i] == '\\' && !escaped
		if s[i] != '\\' {
			escaped = false
		}
	}

	return "", start, false
}

// repairGemma4RawTerminalStringValue wraps a raw value in Gemma string
// delimiters only when the tool schema says that argument is a string. This is
// deliberately schema-gated because raw text is otherwise too ambiguous.
func repairGemma4RawTerminalStringValue(argsStr, toolName string, tools []api.Tool) (string, bool) {
	props := gemma4ToolProperties(toolName, tools)
	if props == nil {
		return "", false
	}

	for key, prop := range props.All() {
		if !gemma4PropertyAcceptsString(prop) {
			continue
		}

		if repaired, ok := repairGemma4RawTerminalStringValueForKey(argsStr, key, props); ok {
			return repaired, true
		}
	}

	return "", false
}

func repairGemma4RawTerminalStringValueForKey(s, key string, props *api.ToolPropertiesMap) (string, bool) {
	for searchStart := 0; searchStart < len(s); {
		valueStart, ok := gemma4FindValueStartForKey(s, key, searchStart)
		if !ok {
			return "", false
		}

		valueCheck := gemma4SkipSpace(s, valueStart)
		if valueCheck < len(s) && gemma4ValueStartsStructured(s, valueCheck) {
			searchStart = valueStart
			continue
		}

		valueEnd := gemma4RawStringValueEnd(s, valueStart, props)
		return s[:valueStart] + gemma4StringDelimiter + s[valueStart:valueEnd] + gemma4StringDelimiter + s[valueEnd:], true
	}

	return "", false
}

func gemma4FindValueStartForKey(s, key string, searchStart int) (int, bool) {
	for i := searchStart; i < len(s); i++ {
		if strings.HasPrefix(s[i:], gemma4StringDelimiter) {
			end := strings.Index(s[i+len(gemma4StringDelimiter):], gemma4StringDelimiter)
			if end == -1 {
				return 0, false
			}
			i += len(gemma4StringDelimiter) + end + len(gemma4StringDelimiter) - 1
			continue
		}

		if s[i] == '"' {
			if end := gemma4JSONQuotedStringEnd(s, i); end != -1 {
				i = end - 1
				continue
			}
		}

		if s[i] != '{' && s[i] != ',' {
			continue
		}

		keyStart := gemma4SkipSpace(s, i+1)
		if !strings.HasPrefix(s[keyStart:], key) {
			continue
		}

		colon := gemma4SkipSpace(s, keyStart+len(key))
		if colon < len(s) && s[colon] == ':' {
			return colon + 1, true
		}
	}

	return 0, false
}

func gemma4RawStringValueEnd(s string, start int, props *api.ToolPropertiesMap) int {
	for i := start; i < len(s); i++ {
		if s[i] != ',' {
			continue
		}

		keyStart := gemma4SkipSpace(s, i+1)
		keyEnd := keyStart
		for keyEnd < len(s) {
			r, size := utf8.DecodeRuneInString(s[keyEnd:])
			if !(r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)) {
				break
			}
			keyEnd += size
		}
		if keyEnd == keyStart {
			continue
		}

		colon := gemma4SkipSpace(s, keyEnd)
		if colon < len(s) && s[colon] == ':' {
			if _, ok := props.Get(s[keyStart:keyEnd]); ok {
				return i
			}
		}
	}

	end := gemma4TrimRightSpaceIndex(s)
	if end > start && s[end-1] == '}' {
		return end - 1
	}
	return len(s)
}

func gemma4ValueStartsStructured(s string, pos int) bool {
	if pos >= len(s) {
		return false
	}
	if strings.HasPrefix(s[pos:], gemma4StringDelimiter) {
		return true
	}

	switch s[pos] {
	case '\'', '"', '{', '[':
		return true
	}

	return gemma4LooksLikeJSONLiteralStart(s[pos])
}

func gemma4JSONQuotedStringEnd(s string, start int) int {
	escaped := false
	for i := start + 1; i < len(s); i++ {
		if s[i] == '"' && !escaped {
			return i + 1
		}

		escaped = s[i] == '\\' && !escaped
		if s[i] != '\\' {
			escaped = false
		}
	}

	return -1
}

func gemma4SkipSpace(s string, i int) int {
	for i < len(s) {
		r, size := utf8.DecodeRuneInString(s[i:])
		if !unicode.IsSpace(r) {
			return i
		}
		i += size
	}
	return i
}

func gemma4TrimRightSpaceIndex(s string) int {
	i := len(s)
	for i > 0 {
		r, size := utf8.DecodeLastRuneInString(s[:i])
		if !unicode.IsSpace(r) {
			return i
		}
		i -= size
	}
	return i
}

func gemma4PropertyAcceptsString(prop api.ToolProperty) bool {
	for _, typ := range prop.Type {
		if strings.EqualFold(typ, "string") {
			return true
		}
	}

	for _, anyOf := range prop.AnyOf {
		if gemma4PropertyAcceptsString(anyOf) {
			return true
		}
	}

	return false
}

func gemma4LooksLikeJSONLiteralStart(ch byte) bool {
	return ch == '-' || ('0' <= ch && ch <= '9') || ch == 't' || ch == 'f' || ch == 'n'
}
