package thinking

import (
	"strings"
	"unicode"
)

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

type Parser struct {
	state      thinkingState
	OpeningTag string
	ClosingTag string
	acc        strings.Builder
}

func (s *Parser) AddContent(content string) (string, string) {
	s.acc.WriteString(content)

	var thinkingSb, remainingSb strings.Builder
	var thinking, remaining string
	keepLooping := true

	for keepLooping {
		thinking, remaining, keepLooping = eat(s)
		thinkingSb.WriteString(thinking)
		remainingSb.WriteString(remaining)
	}

	return thinkingSb.String(), remainingSb.String()
}

func eat(s *Parser) (string, string, bool) {
	trimmed := strings.TrimLeftFunc(s.acc.String(), unicode.IsSpace)

	switch s.state {
	case thinkingState_LookingForOpening:
		if strings.HasPrefix(trimmed, s.OpeningTag) {
			after := trimmed[len(s.OpeningTag):]
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
		if trimmed == "" {
			return "", "", false
		} else {
			s.state = thinkingState_Thinking
			s.acc.Reset()
			s.acc.WriteString(trimmed)
			return "", "", true
		}

	case thinkingState_Thinking:
		acc := s.acc.String()
		if strings.Contains(acc, s.ClosingTag) {
			idx := strings.Index(acc, s.ClosingTag)
			thinking := acc[:idx]
			remaining := acc[idx+len(s.ClosingTag):]
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
			remaining := acc[len(acc)-overlapLen:]
			s.acc.Reset()
			s.acc.WriteString(remaining)
			return thinking, "", false
		} else {
			s.acc.Reset()
			return acc, "", false
		}

	case thinkingState_ThinkingDoneEatingWhitespace:
		if trimmed != "" {
			s.state = thinkingState_ThinkingDone
		}
		s.acc.Reset()
		return "", trimmed, false

	case thinkingState_ThinkingDone:
		acc := s.acc.String()
		s.acc.Reset()
		return "", acc, false

	default:
		panic("unknown state")
	}
}

func overlap(s, delim string) int {
	max := min(len(delim), len(s))
	for i := max; i > 0; i-- {
		if strings.HasSuffix(s, delim[:i]) {
			return i
		}
	}
	return 0
}
