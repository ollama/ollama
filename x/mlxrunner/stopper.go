package mlxrunner

import "strings"

// stopper truncates a streamed completion at the first occurrence of any
// stop sequence, holding back a trailing fragment that could still grow
// into a match. Stop text is never emitted, matching llama-server's stop
// semantics.
type stopper struct {
	stops   []string
	pending string
	maxHold int
}

func newStopper(stops []string) *stopper {
	s := &stopper{stops: stops}
	for _, st := range stops {
		if len(st)-1 > s.maxHold {
			s.maxHold = len(st) - 1
		}
	}
	return s
}

// feed appends chunk and returns the text that is safe to emit. When a
// stop sequence completes, emit ends just before it and stopped is true;
// the stopper must not be fed afterwards.
func (s *stopper) feed(chunk string) (emit string, stopped bool) {
	s.pending += chunk

	earliest := -1
	for _, st := range s.stops {
		if i := strings.Index(s.pending, st); i >= 0 && (earliest < 0 || i < earliest) {
			earliest = i
		}
	}
	if earliest >= 0 {
		emit = s.pending[:earliest]
		s.pending = ""
		return emit, true
	}

	// Hold back the longest tail that is a prefix of some stop sequence.
	hold := 0
	for h := min(s.maxHold, len(s.pending)); h > 0; h-- {
		tail := s.pending[len(s.pending)-h:]
		for _, st := range s.stops {
			if strings.HasPrefix(st, tail) {
				hold = h
				break
			}
		}
		if hold > 0 {
			break
		}
	}
	emit = s.pending[:len(s.pending)-hold]
	s.pending = s.pending[len(s.pending)-hold:]
	return emit, false
}

// flush returns the held-back tail once generation ends without a match.
func (s *stopper) flush() string {
	out := s.pending
	s.pending = ""
	return out
}
