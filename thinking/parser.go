package thinking

import (
	"strings"
	"sync"
	"unicode"
)

// thinkingState tracks where the parser is in the stream.
type thinkingState int

const (
	thinkingState_LookingForOpening thinkingState = iota
	thinkingState_ThinkingStartedEatingWhitespace
	thinkingState_Thinking
	thinkingState_ThinkingDoneEatingWhitespace
	thinkingState_ThinkingDone
)

func (s thinkingState) String() string {
	switch s {
	case thinkingState_LookingForOpening:
		return "LookingForOpening"
	case thinkingState_ThinkingStartedEatingWhitespace:
		return "ThinkingStartedEatingWhitespace"
	case thinkingState_Thinking:
		return "Thinking"
	case thinkingState_ThinkingDoneEatingWhitespace:
		return "ThinkingDoneEatingWhitespace"
	case thinkingState_ThinkingDone:
		return "ThinkingDone"
	default:
		return "Unknown"
	}
}

// Parser splits a streaming text into thinking content and non-thinking content.
// It is NOT safe for concurrent use; use SafeParser for that.
type Parser struct {
	state      thinkingState
	OpeningTag string
	ClosingTag string
	acc        strings.Builder
}

// NewParser creates a Parser with the given opening and closing tags.
func NewParser(openingTag, closingTag string) *Parser {
	return &Parser{
		OpeningTag: openingTag,
		ClosingTag: closingTag,
	}
}

// AddContent returns (thinkingContent, remainingContent) parsed from the new chunk.
// It buffers internally when the boundary is ambiguous.
func (s *Parser) AddContent(content string) (string, string) {
	s.acc.WriteString(content)

	var thinkingSb, remainingSb strings.Builder

	keepLooping := true
	for keepLooping {
		var thinking, remaining string
		thinking, remaining, keepLooping = eat(s)
		thinkingSb.WriteString(thinking)
		remainingSb.WriteString(remaining)
	}

	return thinkingSb.String(), remainingSb.String()
}

// Flush drains any content buffered while waiting for tag disambiguation.
// Call this when the stream ends to recover content that would otherwise be held.
// Returns (thinkingContent, remainingContent) for whatever is buffered.
//
// After Flush the parser is in a terminal state; call Reset to reuse it.
func (s *Parser) Flush() (string, string) {
	if s.acc.Len() == 0 {
		return "", ""
	}
	buffered := s.acc.String()
	s.acc.Reset()

	switch s.state {
	case thinkingState_LookingForOpening:
		// The buffer held a partial or full opening tag that never completed.
		// Treat it as plain content (no thinking block).
		s.state = thinkingState_ThinkingDone
		return "", buffered

	case thinkingState_ThinkingStartedEatingWhitespace:
		// Inside a thinking block, buffering leading whitespace that never ended.
		s.state = thinkingState_ThinkingDone
		return buffered, ""

	case thinkingState_Thinking:
		// Inside a thinking block; the partial closing tag in the buffer is NOT a
		// real close (stream ended mid-tag), so treat it as thinking content.
		s.state = thinkingState_ThinkingDone
		return buffered, ""

	case thinkingState_ThinkingDoneEatingWhitespace:
		// Eating whitespace after close tag — whitespace belongs to remaining.
		s.state = thinkingState_ThinkingDone
		return "", buffered

	case thinkingState_ThinkingDone:
		return "", buffered

	default:
		return "", buffered
	}
}

// Reset returns the parser to its initial state so it can be reused.
func (s *Parser) Reset() {
	s.state = thinkingState_LookingForOpening
	s.acc.Reset()
}

// Done reports whether the parser has finished processing the thinking block
// and all subsequent content is unambiguously non-thinking.
func (s *Parser) Done() bool {
	return s.state == thinkingState_ThinkingDone
}

// HasThinkingBlock reports whether a thinking block was found in the stream so far.
func (s *Parser) HasThinkingBlock() bool {
	return s.state == thinkingState_Thinking ||
		s.state == thinkingState_ThinkingStartedEatingWhitespace ||
		s.state == thinkingState_ThinkingDoneEatingWhitespace ||
		s.state == thinkingState_ThinkingDone
}

// State returns the current parser state (useful for debugging/testing).
func (s *Parser) State() thinkingState {
	return s.state
}

// eat advances the state machine by one step.
// Returns (thinkingContent, remainingContent, shouldContinue).
func eat(s *Parser) (string, string, bool) {
	switch s.state {

	case thinkingState_LookingForOpening:
		trimmed := strings.TrimLeftFunc(s.acc.String(), unicode.IsSpace)
		if strings.HasPrefix(trimmed, s.OpeningTag) {
			after := strings.Join(strings.Split(trimmed, s.OpeningTag)[1:], s.OpeningTag)
			after = strings.TrimLeftFunc(after, unicode.IsSpace)
			s.acc.Reset()
			s.acc.WriteString(after)
			if after == "" {
				s.state = thinkingState_ThinkingStartedEatingWhitespace
			} else {
				s.state = thinkingState_Thinking
			}
			return "", "", true
		} else if strings.HasPrefix(s.OpeningTag, trimmed) {
			return "", "", false
		} else if trimmed == "" {
			return "", "", false
		} else {
			s.state = thinkingState_ThinkingDone
			untrimmed := s.acc.String()
			s.acc.Reset()
			return "", untrimmed, false
		}

	case thinkingState_ThinkingStartedEatingWhitespace:
		trimmed := strings.TrimLeftFunc(s.acc.String(), unicode.IsSpace)
		s.acc.Reset()
		if trimmed == "" {
			return "", "", false
		}
		s.state = thinkingState_Thinking
		s.acc.WriteString(trimmed)
		return "", "", true

	case thinkingState_Thinking:
		acc := s.acc.String()
		if strings.Contains(acc, s.ClosingTag) {
			split := strings.Split(acc, s.ClosingTag)
			thinking := split[0]
			remaining := strings.Join(split[1:], s.ClosingTag)
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)
			s.acc.Reset()
			if remaining == "" {
				s.state = thinkingState_ThinkingDoneEatingWhitespace
			} else {
				s.state = thinkingState_ThinkingDone
			}
			return thinking, remaining, false
		} else if overlapLen := overlap(acc, s.ClosingTag); overlapLen > 0 {
			thinking := acc[:len(acc)-overlapLen]
			candidate := acc[len(acc)-overlapLen:]
			s.acc.Reset()
			s.acc.WriteString(candidate)
			return thinking, "", false
		}
		s.acc.Reset()
		return acc, "", false

	case thinkingState_ThinkingDoneEatingWhitespace:
		trimmed := strings.TrimLeftFunc(s.acc.String(), unicode.IsSpace)
		s.acc.Reset()
		if trimmed != "" {
			s.state = thinkingState_ThinkingDone
		}
		return "", trimmed, false

	case thinkingState_ThinkingDone:
		acc := s.acc.String()
		s.acc.Reset()
		return "", acc, false

	default:
		panic("thinking.Parser: unknown state " + s.state.String())
	}
}

// overlap returns the length of the longest suffix of s that is also a prefix
// of delim. Used to detect partial closing tags at the end of the buffer.
func overlap(s, delim string) int {
	max := min(len(delim), len(s))
	for i := max; i > 0; i-- {
		if strings.HasSuffix(s, delim[:i]) {
			return i
		}
	}
	return 0
}

// ---------------------------------------------------------------------------
// SafeParser — a thread-safe wrapper around Parser.
// ---------------------------------------------------------------------------

// SafeParser wraps Parser with a mutex so multiple goroutines can call
// AddContent concurrently (e.g. a writer goroutine + a progress monitor).
type SafeParser struct {
	mu sync.Mutex
	p  Parser
}

// NewSafeParser creates a SafeParser with the given tags.
func NewSafeParser(openingTag, closingTag string) *SafeParser {
	return &SafeParser{
		p: Parser{OpeningTag: openingTag, ClosingTag: closingTag},
	}
}

func (sp *SafeParser) AddContent(content string) (string, string) {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	return sp.p.AddContent(content)
}

func (sp *SafeParser) Flush() (string, string) {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	return sp.p.Flush()
}

func (sp *SafeParser) Reset() {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	sp.p.Reset()
}

func (sp *SafeParser) Done() bool {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	return sp.p.Done()
}

func (sp *SafeParser) HasThinkingBlock() bool {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	return sp.p.HasThinkingBlock()
}

func (sp *SafeParser) State() thinkingState {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	return sp.p.State()
}
