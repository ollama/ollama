package llm

import "slices"

// A model that has come apart and a payload that is legitimately repetitive
// look the same for a while: both emit the same token over and over. Base64 of
// an audio file, a hex dump, a run of indentation, a rule of dashes in a
// markdown table — all of them repeat inside a bounded region and then carry on
// with something else. A model that is stuck repeats until something stops it.
//
// So the guard measures two things the old one did not. It measures the run in
// characters rather than in stream events, with a budget no realistic payload
// reaches, and it looks for a repeating unit of several tokens rather than a
// single one, which is the shape a stuck model actually produces ("Wait, let me
// re-read the question." over and over) and which a single-token counter never
// saw at all.
const (
	// longest repeating unit recognised, in tokens
	repeatGuardMaxPeriod = 32
	// how many trailing tokens to keep: three turns of the longest unit, the
	// minimum needed to call something a repetition rather than a coincidence
	repeatGuardWindow = repeatGuardMaxPeriod * 3
	// characters of uninterrupted repetition tolerated before giving up
	repeatGuardBudgetBytes = 16 * 1024
)

// repeatGuard reports when a generation has spent too long repeating one short
// unit. The zero value is ready to use.
type repeatGuard struct {
	tokens []string
	bytes  int
}

// observe records the next piece of generated text and reports whether the
// stream has now repeated a single unit for longer than the budget allows.
func (g *repeatGuard) observe(s string) bool {
	if s == "" {
		return false
	}

	if len(g.tokens) == repeatGuardWindow {
		copy(g.tokens, g.tokens[1:])
		g.tokens[repeatGuardWindow-1] = s
	} else {
		g.tokens = append(g.tokens, s)
	}

	if g.period() == 0 {
		g.bytes = 0
		return false
	}

	g.bytes += len(s)
	return g.bytes > repeatGuardBudgetBytes
}

// period returns the length of the shortest unit that the tail of the window
// repeats three times over, or zero when the tail is not repeating. Three turns
// rather than two so that an ordinary "the ... the" pair does not read as a
// cycle.
func (g *repeatGuard) period() int {
	for k := 1; k <= repeatGuardMaxPeriod; k++ {
		if len(g.tokens) < 3*k {
			break
		}
		tail := g.tokens[len(g.tokens)-3*k:]
		if slices.Equal(tail[:k], tail[k:2*k]) && slices.Equal(tail[:k], tail[2*k:]) {
			return k
		}
	}
	return 0
}
