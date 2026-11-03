package llm

import (
	"fmt"
	"slices"
	"strings"
	"unicode"
)

// thinkingGrammar returns a grammar that leaves the text before any of the
// closings unconstrained, then constrains what follows the first complete
// closing to the root of the format grammar. The response may end before any
// closing. A closing that is a prefix of another ends the thinking as soon
// as it is complete.
func thinkingGrammar(closings []string, format string) string {
	// The rules added here take a prefix no rule of the format grammar starts
	// with.
	prefix := "ollama-"
	for grammarHasRulePrefix(format, prefix) {
		prefix += "ollama-"
	}
	thinking, formatted := prefix+"thinking-", prefix+"format"

	a := newClosingAutomaton(closings)
	var b strings.Builder
	fmt.Fprintf(&b, "root ::= %s0\n", thinking)
	for state := range a.states {
		fmt.Fprintf(&b, "%s%d ::=", thinking, state)
		var matching, terminal []rune
		next := map[int][]rune{}
		for _, r := range a.alphabet {
			to, done := a.next(state, r)
			if done {
				terminal = append(terminal, r)
			} else if to != 0 {
				next[to] = append(next[to], r)
			}
			if done || to != 0 {
				matching = append(matching, r)
			}
		}
		// Any other character restarts the search from the empty prefix.
		fmt.Fprintf(&b, " | %s %s0", gbnfCharClass(matching, true), thinking)
		for to := range a.states {
			if rs := next[to]; len(rs) > 0 {
				fmt.Fprintf(&b, " | %s %s%d", gbnfCharClass(rs, false), thinking, to)
			}
		}
		if len(terminal) > 0 {
			fmt.Fprintf(&b, " | %s %s", gbnfCharClass(terminal, false), formatted)
		}
		b.WriteByte('\n')
	}
	b.WriteString(renameGrammarRule(format, "root", formatted))
	return b.String()
}

// closingAutomaton is the Aho-Corasick automaton over the closings: each
// state is a proper prefix of a closing, state 0 the empty one.
type closingAutomaton struct {
	closings []string
	states   []string
	index    map[string]int
	alphabet []rune
}

func newClosingAutomaton(closings []string) *closingAutomaton {
	a := &closingAutomaton{closings: closings, index: map[string]int{}}
	add := func(prefix string) {
		if _, ok := a.index[prefix]; !ok {
			a.index[prefix] = len(a.states)
			a.states = append(a.states, prefix)
		}
	}
	add("")
	for _, closing := range closings {
		for i := range closing {
			add(closing[:i])
		}
		for _, r := range closing {
			if !slices.Contains(a.alphabet, r) {
				a.alphabet = append(a.alphabet, r)
			}
		}
	}
	slices.Sort(a.alphabet)
	return a
}

// next returns the state after reading r in state, or done when the text
// read so far ends with a closing.
func (a *closingAutomaton) next(state int, r rune) (to int, done bool) {
	s := a.states[state] + string(r)
	for _, closing := range a.closings {
		if strings.HasSuffix(s, closing) {
			return 0, true
		}
	}
	for i := range s {
		if to, ok := a.index[s[i:]]; ok {
			return to, false
		}
	}
	return 0, false
}

func gbnfCharClass(rs []rune, negate bool) string {
	var b strings.Builder
	b.WriteByte('[')
	if negate {
		b.WriteByte('^')
	}
	for _, r := range rs {
		switch {
		case r > unicode.MaxASCII || !unicode.IsPrint(r) || strings.ContainsRune(`]\^-`, r):
			fmt.Fprintf(&b, `\U%08X`, r)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte(']')
	return b.String()
}

func renameGrammarRule(grammar, from, to string) string {
	var b strings.Builder
	scanGrammar(grammar, func(text string, name bool) {
		if name && text == from {
			text = to
		}
		b.WriteString(text)
	})
	return b.String()
}

func grammarHasRulePrefix(grammar, prefix string) bool {
	found := false
	scanGrammar(grammar, func(text string, name bool) {
		found = found || name && strings.HasPrefix(text, prefix)
	})
	return found
}

// scanGrammar splits a grammar into rule names and everything else. Comments,
// quoted strings and character classes are passed whole, so a word inside
// them is never taken for a name.
func scanGrammar(grammar string, fn func(text string, name bool)) {
	isName := func(c byte) bool {
		return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_'
	}
	for i := 0; i < len(grammar); {
		start := i
		switch c := grammar[i]; {
		case c == '#':
			for i < len(grammar) && grammar[i] != '\n' {
				i++
			}
		case c == '"' || c == '[':
			end := byte('"')
			if c == '[' {
				end = ']'
			}
			for i++; i < len(grammar) && grammar[i] != end; i++ {
				if grammar[i] == '\\' {
					i++
				}
			}
			i = min(i+1, len(grammar))
		case isName(c):
			for i < len(grammar) && isName(grammar[i]) {
				i++
			}
			fn(grammar[start:i], true)
			continue
		default:
			i++
		}
		fn(grammar[start:i], false)
	}
}
