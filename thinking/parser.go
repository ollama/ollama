package thinking

import (
	"strings"
	"unicode"
)

type thinkingState int

const (
	// We're looking for the opening tag, but we haven't seen any non-whitespace
	// characters yet
	thinkingState_LookingForOpening thinkingState = iota
	// We've seen the opening tag, but we haven't seen any non-whitespace
	// characters yet (we want to eat any whitespace between the opening tag and
	// the thinking content)
	thinkingState_ThinkingStartedEatingWhitespace
	// We've seen non-whitespace characters after the opening tag, but we haven't
	// seen the closing tag yet
	thinkingState_Thinking
	// We've seen the closing tag, but we haven't seen any non-whitespace
	// characters after the closing tag yet (we want to eat any whitespace between
	// the closing tag and the content)
	thinkingState_ThinkingDoneEatingWhitespace
	// We've seen the closing tag and seen at least one non-whitespace character
	// after it
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

// AddContent returns the thinking content and the non-thinking content that
// should be immediately sent to the user. It will internally buffer if it needs
// to see more raw content to disambiguate
func (s *Parser) AddContent(content string) (string, string) {
	s.acc.WriteString(content)

	var thinkingSb, remainingSb strings.Builder

	var thinking, remaining string
	keepLooping := true
	// we loop because we might pass through multiple parsing states in a single
	// call to addContent, and we want to make sure callers don't have to wait for
	// data that's already unambiguous
	for keepLooping {
		thinking, remaining, keepLooping = eat(s)
		thinkingSb.WriteString(thinking)
		remainingSb.WriteString(remaining)
	}

	return thinkingSb.String(), remainingSb.String()
}

// the additional bool return is true iff we should continue eating
func eat(s *Parser) (string, string, bool) {
	switch s.state {
	case thinkingState_LookingForOpening:
		trimmed := strings.TrimLeftFunc(s.acc.String(), unicode.IsSpace)
		if strings.HasPrefix(trimmed, s.OpeningTag) {
			after := strings.Join(strings.Split(trimmed, s.OpeningTag)[1:], s.OpeningTag)
			after = strings.TrimLeftFunc(after, unicode.IsSpace)
			// after might contain more than just thinking tokens, so we continue
			// parsing instead of returning it as thinking tokens here
			s.acc.Reset()
			s.acc.WriteString(after)
			if after == "" {
				s.state = thinkingState_ThinkingStartedEatingWhitespace
			} else {
				s.state = thinkingState_Thinking
			}
			return "", "", true
		} else if strings.HasPrefix(s.OpeningTag, trimmed) {
			// partial opening seen, so let's keep accumulating
			return "", "", false
		} else if trimmed == "" {
			// saw whitespace only, so let's keep accumulating
			return "", "", false
		} else {
			// didn't see an opening tag, but we have content, so thinking was skipped
			s.state = thinkingState_ThinkingDone
			// note that we use the original content, not the trimmed one because we
			// don't want to eat any whitespace in the real content if there were no
			// thinking tags
			untrimmed := s.acc.String()
			s.acc.Reset()
			return "", untrimmed, false
		}
	case thinkingState_ThinkingStartedEatingWhitespace:
		trimmed := strings.TrimLeftFunc(s.acc.String(), unicode.IsSpace)
		s.acc.Reset()
		if trimmed == "" {
			return "", "", false
		} else {
			s.state = thinkingState_Thinking
			s.acc.WriteString(trimmed)
			return "", "", true
		}
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
	case thinkingState_ThinkingDoneEatingWhitespace:
		trimmed := strings.TrimLeftFunc(s.acc.String(), unicode.IsSpace)
		s.acc.Reset()
		// if we see non-whitespace, we're done eating the leading whitespace of the content
		if trimmed != "" {
			s.state = thinkingState_ThinkingDone
		}
		return "", trimmed, false
	case thinkingState_ThinkingDone:
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
