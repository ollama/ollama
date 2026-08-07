package mlxrunner

import "testing"

func stopFeedAll(s *stopper, chunks ...string) (out string, matched bool) {
	for _, c := range chunks {
		emit, m := s.feed(c)
		out += emit
		if m {
			return out, true
		}
	}
	return out, false
}

func TestStopperTruncatesAtMatch(t *testing.T) {
	s := newStopper([]string{"</think>"})
	out, matched := stopFeedAll(s, "hello </think> world")
	if !matched {
		t.Fatal("stop not matched")
	}
	if out != "hello " {
		t.Fatalf("emitted %q, want %q", out, "hello ")
	}
}

func TestStopperMatchAcrossChunks(t *testing.T) {
	s := newStopper([]string{"</think>"})
	out, matched := stopFeedAll(s, "reasoning</th", "ink>after")
	if !matched {
		t.Fatal("split stop not matched")
	}
	if out != "reasoning" {
		t.Fatalf("emitted %q, want %q", out, "reasoning")
	}
}

func TestStopperHoldbackReleasedWhenDisproven(t *testing.T) {
	s := newStopper([]string{"</think>"})
	emit1, m1 := s.feed("a</thx")
	if m1 {
		t.Fatal("unexpected match")
	}
	emit2, m2 := s.feed("b")
	if m2 {
		t.Fatal("unexpected match")
	}
	if emit1+emit2+s.flush() != "a</thxb" {
		t.Fatalf("total = %q, want %q", emit1+emit2+s.flush(), "a</thxb")
	}
}

func TestStopperHoldsPossiblePrefixAtEnd(t *testing.T) {
	s := newStopper([]string{"STOP"}) //nolint:misspell
	emit, matched := s.feed("text ST")
	if matched {
		t.Fatal("unexpected match")
	}
	if emit != "text " {
		t.Fatalf("emitted %q, want %q (holding possible prefix)", emit, "text ")
	}
	if s.flush() != "ST" {
		t.Fatalf("flush = %q, want %q", s.flush(), "ST")
	}
}

func TestStopperEarliestOfMultipleStops(t *testing.T) {
	s := newStopper([]string{"XX", "Y"})
	out, matched := stopFeedAll(s, "abYcdXX")
	if !matched {
		t.Fatal("no match")
	}
	if out != "ab" {
		t.Fatalf("emitted %q, want %q", out, "ab")
	}
}

func TestStopperMatchAtStart(t *testing.T) {
	s := newStopper([]string{"</think>"})
	out, matched := stopFeedAll(s, "</think>rest")
	if !matched {
		t.Fatal("no match")
	}
	if out != "" {
		t.Fatalf("emitted %q, want empty", out)
	}
}

func TestStopperRepeatedPrefixes(t *testing.T) {
	s := newStopper([]string{"aab"})
	out, matched := stopFeedAll(s, "a", "a", "a", "b")
	if !matched {
		t.Fatal("no match")
	}
	if out != "a" {
		t.Fatalf("emitted %q, want %q", out, "a")
	}
}
