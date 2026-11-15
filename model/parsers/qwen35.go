package parsers

import (
	"context"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

type qwen35ParserState int

const (
	qwen35ParserStateCollectingThinking qwen35ParserState = iota
	qwen35ParserStateThinkingDoneEatingWhitespace
	qwen35ParserStateCollectingContent
)

const (
	qwen35ThinkingOpenTag  = "<think>"
	qwen35ThinkingCloseTag = "</think>"
	qwen35ToolCallOpenTag  = "<tool_call>"
)

// Qwen35Parser handles qwen3.5 reasoning extraction and delegates post-thinking
// content (including XML tool calls) to Qwen3CoderParser.
type Qwen35Parser struct {
	toolParser Qwen3CoderParser

	state  qwen35ParserState
	buffer strings.Builder
	// Some checkpoints may emit an explicit leading <think> even when the
	// prompt already opened thinking. Strip at most one such tag.
	allowLeadingThinkOpenTag bool
	trimLeadingThinkingSpace bool
	// inCandidate reports whether the buffer holds a <tool_call> opener
	// whose classification (real implicit tool call vs literal thinking
	// text) is still pending, starting at byte offset candidateStart.
	// While it is set, the opener and everything after it must not be
	// emitted as thinking or content until the decision is made, or the
	// result would depend on where chunks happen to end (issue #19694,
	// cases 3 and 4). The thinking before the opener stays buffered too,
	// so no separator whitespace is lost or trimmed before the decision.
	inCandidate    bool
	candidateStart int
}

func (p *Qwen35Parser) HasToolSupport() bool {
	return true
}

func (p *Qwen35Parser) HasThinkingSupport() bool {
	return true
}

func (p *Qwen35Parser) ThinkingClose() []string {
	if p.state == qwen35ParserStateCollectingThinking {
		return []string{qwen35ThinkingCloseTag}
	}
	return nil
}

func (p *Qwen35Parser) PreservedTokens() []string {
	return []string{
		qwen35ThinkingOpenTag,
		qwen35ThinkingCloseTag,
		toolOpenTag,
		toolCloseTag,
	}
}

func (p *Qwen35Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.buffer.Reset()
	p.inCandidate = false
	p.candidateStart = 0
	p.toolParser = Qwen3CoderParser{}
	p.toolParser.Init(tools, nil, nil)

	thinkingEnabled := thinkValue != nil && thinkValue.Bool()
	if thinkValue == nil {
		thinkingEnabled = true
	}

	assistantPrefill := lastMessage != nil && lastMessage.Role == "assistant" && lastMessage.Content != ""
	if thinkingEnabled && !assistantPrefill {
		p.state = qwen35ParserStateCollectingThinking
		p.allowLeadingThinkOpenTag = true
		p.trimLeadingThinkingSpace = false
	} else {
		p.state = qwen35ParserStateCollectingContent
		p.allowLeadingThinkOpenTag = false
		p.trimLeadingThinkingSpace = false
	}

	return tools
}

type qwen35Event interface {
	isQwen35Event()
}

type qwen35EventContent struct {
	content string
}

func (qwen35EventContent) isQwen35Event() {}

type qwen35EventThinkingContent struct {
	content string
}

func (qwen35EventThinkingContent) isQwen35Event() {}

func (p *Qwen35Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents(done)

	var thinkingSb strings.Builder
	var toolInput strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case qwen35EventContent:
			toolInput.WriteString(event.content)
		case qwen35EventThinkingContent:
			thinkingSb.WriteString(event.content)
		}
	}

	// Forward to the delegated tool parser exactly once, with the outer done
	// flag — even when this Add produced no content. An empty final chunk
	// (Add("", true) in content state) must still finalize the tool parser,
	// or a call it is still holding is silently lost (issue #19694, case
	// 5). Aggregating first also avoids finalizing the delegate partway
	// through one Add when several content events arrive together.
	// Skip the call only while still collecting thinking: the delegate
	// cannot be holding anything then, and nothing has been forwarded to it.
	if toolInput.Len() > 0 || (done && p.state != qwen35ParserStateCollectingThinking) {
		parsedContent, _, parsedCalls, err := p.toolParser.Add(toolInput.String(), done)
		if err != nil {
			slog.Warn("qwen3.5 tool call parsing failed", "error", err)
			return "", "", nil, err
		}
		calls = append(calls, parsedCalls...)
		return parsedContent, thinkingSb.String(), calls, nil
	}

	return "", thinkingSb.String(), nil, nil
}

func (p *Qwen35Parser) parseEvents(done bool) []qwen35Event {
	var all []qwen35Event

	keepLooping := true
	for keepLooping {
		var events []qwen35Event
		events, keepLooping = p.eat(done)
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "qwen3.5 events parsed", "events", all, "state", p.state, "buffer", p.buffer.String())
	}

	return all
}

func (p *Qwen35Parser) splitAtTag(tag string, trimAfter bool) (string, string) {
	return splitAtTag(&p.buffer, tag, trimAfter)
}

func (p *Qwen35Parser) eatLeadingWhitespaceAndTransitionTo(nextState qwen35ParserState) ([]qwen35Event, bool) {
	trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
	p.buffer.Reset()
	if trimmed == "" {
		return nil, false
	}
	p.state = nextState
	p.buffer.WriteString(trimmed)
	return nil, true
}

// maybeConsumeLeadingThinkOpenTag handles a single optional leading <think> tag.
// Returns (handled, shouldContinueParsingNow).
func (p *Qwen35Parser) maybeConsumeLeadingThinkOpenTag(acc string) (bool, bool) {
	if !p.allowLeadingThinkOpenTag {
		return false, false
	}

	trimmed := strings.TrimLeftFunc(acc, unicode.IsSpace)
	if strings.HasPrefix(trimmed, qwen35ThinkingOpenTag) {
		after := strings.TrimPrefix(trimmed, qwen35ThinkingOpenTag)
		after = strings.TrimLeftFunc(after, unicode.IsSpace)
		p.buffer.Reset()
		p.buffer.WriteString(after)
		p.allowLeadingThinkOpenTag = false
		p.trimLeadingThinkingSpace = after == ""
		if after == "" {
			return true, false
		}
		return true, true
	}

	if strings.HasPrefix(qwen35ThinkingOpenTag, trimmed) {
		return true, false
	}

	p.allowLeadingThinkOpenTag = false
	return false, false
}

func (p *Qwen35Parser) eat(done bool) ([]qwen35Event, bool) {
	var events []qwen35Event

	switch p.state {
	case qwen35ParserStateCollectingThinking:
		// A held <tool_call> candidate is resolved before anything else:
		// its classification decides whether the bytes after it are
		// thinking, a tool call, or the end of thinking.
		if p.inCandidate {
			return p.eatThinkingCandidate(done)
		}

		acc := p.buffer.String()

		if handled, continueNow := p.maybeConsumeLeadingThinkOpenTag(acc); handled {
			return events, continueNow
		}
		if p.trimLeadingThinkingSpace {
			acc = strings.TrimLeftFunc(acc, unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(acc)
			if acc == "" {
				return events, false
			}
			p.trimLeadingThinkingSpace = false
		}

		idxThink := strings.Index(acc, qwen35ThinkingCloseTag)
		idxCall := strings.Index(acc, qwen35ToolCallOpenTag)
		if idxThink >= 0 && (idxCall < 0 || idxThink < idxCall) {
			thinking, remaining := p.splitAtTag(qwen35ThinkingCloseTag, true)
			if len(thinking) > 0 {
				events = append(events, qwen35EventThinkingContent{content: thinking})
			}
			if remaining == "" {
				p.state = qwen35ParserStateThinkingDoneEatingWhitespace
			} else {
				p.state = qwen35ParserStateCollectingContent
			}
			return events, true
		} else if idxCall >= 0 {
			// A <tool_call> opener appears before any </think>. Do not
			// decide yet whether it is a real implicit call or literal
			// thinking text, and do not emit anything yet either: hold
			// the whole buffer for the candidate decision in
			// eatThinkingCandidate, so the result cannot depend on where
			// chunks happen to end, and no whitespace is lost or trimmed
			// before the decision.
			p.candidateStart = idxCall
			p.inCandidate = true
			return events, true
		} else if overlapLen := max(overlap(acc, qwen35ThinkingCloseTag), overlap(acc, qwen35ToolCallOpenTag)); overlapLen > 0 {
			// Only partial delimiters can reach here: a complete
			// </think> or <tool_call> is handled above. Hold the
			// possible partial tag at the end of the buffer.
			beforePartialTag := acc[:len(acc)-overlapLen]
			trailingWsLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWsLen

			unambiguous := acc[:ambiguousStart]
			ambiguous := acc[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwen35EventThinkingContent{content: unambiguous})
			}
			return events, false
		}

		whitespaceLen := trailingWhitespaceLen(acc)
		ambiguousStart := len(acc) - whitespaceLen
		unambiguous := acc[:ambiguousStart]
		ambiguous := acc[ambiguousStart:]
		p.buffer.Reset()
		p.buffer.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, qwen35EventThinkingContent{content: unambiguous})
		}
		return events, false

	case qwen35ParserStateThinkingDoneEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(qwen35ParserStateCollectingContent)

	case qwen35ParserStateCollectingContent:
		if p.buffer.Len() == 0 {
			return events, false
		}

		content := p.buffer.String()
		p.buffer.Reset()
		if len(content) > 0 {
			events = append(events, qwen35EventContent{content: content})
		}
		return events, false

	default:
		slog.Warn("qwen3.5 parser entered unknown state; resetting to content mode", "state", p.state)
		p.state = qwen35ParserStateCollectingContent
		return events, false
	}
}

type functionPrefixResult int

const (
	// prefixNoMatch: the bytes after <tool_call> differ from <function=,
	// so the opener is literal thinking text.
	prefixNoMatch functionPrefixResult = iota
	// prefixWait: the bytes so far are empty or a strict prefix of
	// <function=; more input is needed to decide.
	prefixWait
	// prefixMatch: the bytes after <tool_call> start with <function=.
	prefixMatch
)

// classifyFunctionPrefix compares the bytes following a <tool_call> opener
// with the <function= prefix a real call must have.
func classifyFunctionPrefix(after string) functionPrefixResult {
	const fnPrefix = "<function="
	switch {
	case strings.HasPrefix(after, fnPrefix):
		return prefixMatch
	case after == "" || strings.HasPrefix(fnPrefix, after):
		return prefixWait
	default:
		return prefixNoMatch
	}
}

// eatThinkingCandidate resolves a <tool_call> opener held in the buffer
// (starting at byte offset p.candidateStart) while collecting thinking.
// The opener is a real implicit tool call only if all three hold: it is
// followed by <function=, its </tool_call> arrives before any </think>, and
// after the call the stream ends or another call starts. Otherwise it is
// literal thinking text. An undecided candidate stays buffered; the
// decision is the same for any chunking.
func (p *Qwen35Parser) eatThinkingCandidate(done bool) ([]qwen35Event, bool) {
	var events []qwen35Event
	acc := p.buffer.String()
	openerEnd := p.candidateStart + len(qwen35ToolCallOpenTag)
	afterOpener := acc[openerEnd:]

	// Stage 1: the opener must be followed by <function=.
	switch classifyFunctionPrefix(afterOpener) {
	case prefixWait:
		if done {
			// Incomplete opener at end of stream: literal thinking.
			return p.releaseCandidateAsThinking(len(acc))
		}
		return events, false
	case prefixNoMatch:
		// Not a call: release the thinking before the opener plus just
		// the opener as thinking, and re-scan what follows it (it may
		// contain </think> or another candidate).
		return p.releaseCandidateAsThinking(openerEnd)
	case prefixMatch:
		// Continue to stage 2.
	}

	// Stage 2: the call's </tool_call> must arrive before any </think>.
	// The terminator is the structural one (the text before it ends with
	// </function> and parses as a tool call), so a literal </tool_call>
	// inside a parameter value does not end the call early. Likewise, a
	// </think> inside a parameter value is only known to be inside the
	// call once the structural terminator is found, so it is ignored here;
	// it only ends thinking if it arrives before any structural terminator
	// is known.
	idxClose := findCallTerminator(acc[p.candidateStart:])
	if idxClose < 0 {
		if idxThink := strings.Index(acc, qwen35ThinkingCloseTag); idxThink >= 0 {
			// </think> arrives before the call closes: the candidate is
			// literal thinking. Release everything up to it as thinking
			// and let the next iteration close thinking normally. The
			// released text ends thinking, so trim its trailing
			// whitespace exactly as the </think> split does.
			thinking := strings.TrimRightFunc(acc[:idxThink], unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(acc[idxThink:])
			p.inCandidate = false
			p.candidateStart = 0
			if len(thinking) > 0 {
				events = append(events, qwen35EventThinkingContent{content: thinking})
			}
			return events, true
		}
		if done {
			// Incomplete call at end of stream: literal thinking.
			return p.releaseCandidateAsThinking(len(acc))
		}
		return events, false
	}

	// Stage 3: after the call, the stream must end or another call must
	// start. Anything else (ordinary text, a </think>, an undecided
	// second opener at end of stream) means the candidate was literal
	// thinking text.
	callEnd := p.candidateStart + idxClose + len(toolCloseTag)
	afterCall := acc[callEnd:]
	trimmedAfter := strings.TrimLeftFunc(afterCall, unicode.IsSpace)
	switch {
	case trimmedAfter == "":
		if !done {
			return events, false
		}
		return p.confirmCandidate()
	case strings.HasPrefix(trimmedAfter, qwen35ToolCallOpenTag):
		// Another call may be starting; it confirms this one only once
		// its own <function= prefix is confirmed.
		switch classifyFunctionPrefix(trimmedAfter[len(qwen35ToolCallOpenTag):]) {
		case prefixMatch:
			return p.confirmCandidate()
		case prefixWait:
			if done {
				return p.releaseCandidateAsThinking(len(acc))
			}
			return events, false
		case prefixNoMatch:
			return p.releaseCandidateAsThinking(callEnd)
		}
	case strings.HasPrefix(trimmedAfter, qwen35ThinkingCloseTag):
		return p.releaseCandidateAsThinking(callEnd)
	default:
		// Ordinary text after the call: not an implicit call. A partial
		// tag would need more input to decide, so wait unless the stream
		// has ended.
		if !done && (strings.HasPrefix(qwen35ToolCallOpenTag, trimmedAfter) || strings.HasPrefix(qwen35ThinkingCloseTag, trimmedAfter)) {
			return events, false
		}
		return p.releaseCandidateAsThinking(callEnd)
	}
	// Unreachable: every case above returns.
	return events, false
}

// findCallTerminator returns the index of the </tool_call> that closes the
// candidate call starting at the start of acc, using the same structural
// check as the tool-call collector (see selectToolCallTerminator): the text
// before it must end with </function> and parse as a tool call. A literal
// </tool_call> inside a parameter value fails, so it is skipped. Returns -1
// if no such terminator is present yet.
func findCallTerminator(acc string) int {
	for start := len(qwen35ToolCallOpenTag); ; {
		idx := strings.Index(acc[start:], toolCloseTag)
		if idx < 0 {
			return -1
		}
		end := start + idx
		raw := acc[len(qwen35ToolCallOpenTag):end]
		if endsWithFunctionClose(raw) && isValidToolCall(raw) {
			return end
		}
		start = end + len(toolCloseTag)
	}
}

// releaseCandidateAsThinking emits the first n bytes of the held candidate
// buffer as thinking, leaves the rest buffered for re-scanning, and clears
// candidate mode. It returns keepLooping=true so the rest is processed.
func (p *Qwen35Parser) releaseCandidateAsThinking(n int) ([]qwen35Event, bool) {
	var events []qwen35Event
	acc := p.buffer.String()
	if n > 0 {
		events = append(events, qwen35EventThinkingContent{content: acc[:n]})
	}
	p.buffer.Reset()
	p.buffer.WriteString(acc[n:])
	p.inCandidate = false
	p.candidateStart = 0
	return events, true
}

// confirmCandidate applies the qwen3.5:9b workaround to a confirmed implicit
// tool call: it injects a synthetic </think> after the thinking that
// precedes the opener, so the ordinary thinking-close transition hands the
// call to the delegated tool parser. The thinking before the opener is
// trimmed exactly as the original workaround did.
func (p *Qwen35Parser) confirmCandidate() ([]qwen35Event, bool) {
	acc := p.buffer.String()
	thinking := strings.TrimRightFunc(acc[:p.candidateStart], unicode.IsSpace)
	p.buffer.Reset()
	p.buffer.WriteString(thinking + qwen35ThinkingCloseTag + acc[p.candidateStart:])
	p.inCandidate = false
	p.candidateStart = 0
	return nil, true
}
