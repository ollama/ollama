package mlxrunner

import "strings"

// stopMatcher applies a request's stop sequences to the streamed output. Text
// that could be the start of a stop sequence is held back, so a sequence split
// across tokens is still caught and none of it reaches the client. Responses
// are held whole and only cut where a stop sequence matches, so a response's
// text and logprobs go out together.
type stopMatcher struct {
	stops   []string
	pending []CompletionResponse
}

func newStopMatcher(stops []string) *stopMatcher {
	s := &stopMatcher{}
	for _, stop := range stops {
		if stop != "" {
			s.stops = append(s.stops, stop)
		}
	}
	return s
}

// push adds a streamed response and returns the responses that are safe to
// send. When matched is true the returned responses end just before the
// earliest stop sequence and generation should end.
func (s *stopMatcher) push(resp CompletionResponse) (ready []CompletionResponse, matched bool) {
	if len(s.stops) == 0 {
		return []CompletionResponse{resp}, false
	}
	s.pending = append(s.pending, resp)

	var text strings.Builder
	for _, p := range s.pending {
		text.WriteString(p.Content)
	}
	held := text.String()

	cut := -1
	for _, stop := range s.stops {
		if i := strings.Index(held, stop); i >= 0 && (cut < 0 || i < cut) {
			cut = i
		}
	}
	if cut >= 0 {
		ready = s.truncate(cut)
		s.pending = nil
		return ready, true
	}

	keep := 0
	for _, stop := range s.stops {
		keep = max(keep, partialStopSuffix(held, stop))
	}
	return s.take(len(held) - keep), false
}

// flush returns the held responses once generation ends without a match.
func (s *stopMatcher) flush() []CompletionResponse {
	ready := s.pending
	s.pending = nil
	return ready
}

// take removes the leading responses that lie entirely within the first n
// bytes of held text and returns them. A response reaching past n stays held
// whole.
func (s *stopMatcher) take(n int) []CompletionResponse {
	var ready []CompletionResponse
	for len(s.pending) > 0 && len(s.pending[0].Content) <= n {
		n -= len(s.pending[0].Content)
		ready = append(ready, s.pending[0])
		s.pending = s.pending[1:]
	}
	return ready
}

// truncate returns the held responses cut to the first n bytes of held text.
// A response cut at n keeps its logprobs with the text before the stop.
func (s *stopMatcher) truncate(n int) []CompletionResponse {
	var ready []CompletionResponse
	for _, p := range s.pending {
		if n == 0 {
			break
		}
		if len(p.Content) > n {
			p.Content = p.Content[:n]
		}
		ready = append(ready, p)
		n -= len(p.Content)
	}
	return ready
}

// partialStopSuffix returns the length of the longest suffix of text that is
// a proper prefix of stop. Since stop is valid UTF-8, the suffix starts on a
// rune boundary.
func partialStopSuffix(text, stop string) int {
	for n := min(len(text), len(stop)-1); n > 0; n-- {
		if strings.HasSuffix(text, stop[:n]) {
			return n
		}
	}
	return 0
}
