package mlxrunner

import "strings"

// stopMatcher applies a request's stop sequences to the streamed output. Text
// that could be the start of a stop sequence is held back, so a sequence split
// across tokens is still caught and none of it reaches the client.
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
		ready = s.take(cut)
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

// take removes the first n bytes of held text from pending and returns them
// as responses. A response split at n keeps its logprobs with the part sent.
func (s *stopMatcher) take(n int) []CompletionResponse {
	var ready []CompletionResponse
	for n > 0 && len(s.pending) > 0 {
		p := s.pending[0]
		if len(p.Content) <= n {
			ready = append(ready, p)
			n -= len(p.Content)
			s.pending = s.pending[1:]
			continue
		}
		sent := p
		sent.Content = p.Content[:n]
		ready = append(ready, sent)
		s.pending[0] = CompletionResponse{Content: p.Content[n:]}
		n = 0
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
