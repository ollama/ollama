package server

import (
	"strings"
	"unicode"
)

type thinkingParseState int

const (
	thinkingParseState_LookingForOpening thinkingParseState = iota
	thinkingParseState_Thinking
	thinkingParseState_ThinkingDone
)

func (s thinkingParseState) String() string {
	switch s {
	case thinkingParseState_LookingForOpening:
		return "LookingForOpening"
	case thinkingParseState_Thinking:
		return "Thinking"
	case thinkingParseState_ThinkingDone:
		return "ThinkingDone"
	default:
		return "Unknown"
	}
}

type thinkingParser struct {
	state      thinkingParseState
	openingTag string
	closingTag string
	acc        strings.Builder
}

// returns the thinking content and the normal content that should be
// immediately sent to the user. It will internally buffer if it needs to see
// more content to disambiguate
func (s *thinkingParser) addContent(content string) (string, string) {
	s.acc.WriteString(content)

	var thinkingAcc, remainingAcc strings.Builder

	var thinking, remaining string
	keepLooping := true
	// we loop because we might pass through multiple parsing states in a single
	// call to addContent, and we want to make sure callers don't have to wait for
	// data that's already unambiguous
	for keepLooping {
		thinking, remaining, keepLooping = eat(s)
		thinkingAcc.WriteString(thinking)
		remainingAcc.WriteString(remaining)
	}

	return thinkingAcc.String(), remainingAcc.String()
}

// the additional bool return is true iff we should continue eating
func eat(s *thinkingParser) (string, string, bool) {
	switch s.state {
	case thinkingParseState_LookingForOpening:
		trimmed := strings.TrimLeftFunc(s.acc.String(), unicode.IsSpace)
		if strings.HasPrefix(trimmed, s.openingTag) {
			after := strings.Join(strings.Split(trimmed, s.openingTag)[1:], s.openingTag)
			after = strings.TrimLeftFunc(after, unicode.IsSpace)
			// after might contain more than just thinking tokens, so we continue
			// parsing instead of returning it as thinking tokens here
			s.acc.Reset()
			s.acc.WriteString(after)
			s.state = thinkingParseState_Thinking
			return "", "", true
		} else if strings.HasPrefix(s.openingTag, trimmed) {
			// partial opening seen, so let's keep accumulating
			return "", "", false
		} else if trimmed == "" {
			// saw whitespace only, so let's keep accumulating
			return "", "", false
		} else {
			// didn't see an opening tag, but we have content, so thinking was skipped
			s.state = thinkingParseState_ThinkingDone
			// note that we use the original content, not the trimmed one because we
			// don't want to eat any whitespace in the real content if there were no
			// thinking tags
			return "", s.acc.String(), false
		}
	case thinkingParseState_Thinking:
		acc := s.acc.String()
		if strings.Contains(acc, s.closingTag) {
			split := strings.Split(acc, s.closingTag)
			thinking := split[0]
			remaining := strings.Join(split[1:], s.closingTag)
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)
			s.acc.Reset()
			s.state = thinkingParseState_ThinkingDone
			return thinking, remaining, false
		} else if overlapLen := overlap(acc, s.closingTag); overlapLen > 0 {
			thinking := acc[:len(acc)-overlapLen]
			remaining := acc[len(acc)-overlapLen:]
			s.acc.Reset()
			// keep track of the candidate closing tag. We have to buffer it until it
			// becomes disambiguated
			s.acc.WriteString(remaining)
			return thinking, "", false
		} else {
			// purely just thinking tokens, so we can return them
			s.acc.Reset()
			return acc, "", false
		}
	case thinkingParseState_ThinkingDone:
		acc := s.acc.String()
		s.acc.Reset()
		return "", acc, false
	default:
		panic("unknown state")
	}
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
