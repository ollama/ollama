package llm

import (
	"fmt"
	"strconv"
	"strings"
)

// toolCallGrammar permits delimited tool calls while retaining the requested
// format for assistant text. Tool bodies end at the first complete closing
// delimiter; neither an incomplete call nor trailing prose can satisfy it.
func toolCallGrammar(open, close, format string) string {
	prefix := "ollama-"
	for grammarHasRulePrefix(format, prefix) {
		prefix += "ollama-"
	}
	formatted, tool, body, ws := prefix+"format", prefix+"tool", prefix+"body-", prefix+"space"
	var b strings.Builder
	fmt.Fprintf(&b, "root ::= %s (%s | %s (%s %s)* (%s %s)?) %s\n", ws, formatted, tool, ws, tool, ws, formatted, ws)
	fmt.Fprintf(&b, "%s ::= [ \\t\\r\\n]*\n", ws)
	fmt.Fprintf(&b, "%s ::= %s %s0\n", tool, strconv.Quote(open), body)

	a := newClosingAutomaton([]string{close})
	for state := range a.states {
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
		fmt.Fprintf(&b, "%s%d ::= %s %s0", body, state, gbnfCharClass(matching, true), body)
		for to := range a.states {
			if rs := next[to]; len(rs) > 0 {
				fmt.Fprintf(&b, " | %s %s%d", gbnfCharClass(rs, false), body, to)
			}
		}
		if len(terminal) > 0 {
			fmt.Fprintf(&b, " | %s", gbnfCharClass(terminal, false))
		}
		b.WriteByte('\n')
	}
	b.WriteString(renameGrammarRule(format, "root", formatted))
	return b.String()
}
