package llm

import (
	"strings"
	"testing"
)

// TestRenameGrammarRule checks that only rule names change, not the same word
// in a comment, a string, a character class or a longer name.
func TestRenameGrammarRule(t *testing.T) {
	grammar := "# root stays in a comment\n" +
		"root ::= \"root\" | [root] | \"[\" root \"]\" | root-item\n" +
		"root-item ::= \"\\\"root\\\"\" root\n"
	want := "# root stays in a comment\n" +
		"out ::= \"root\" | [root] | \"[\" out \"]\" | root-item\n" +
		"root-item ::= \"\\\"root\\\"\" out\n"
	if got := renameGrammarRule(grammar, "root", "out"); got != want {
		t.Errorf("renamed grammar:\n%s\nwant:\n%s", got, want)
	}
}

// TestClosingAutomatonNext checks that the state after a character is the
// longest suffix of the text so far that is a proper prefix of a closing, and
// that the automaton is done exactly when the text ends with a closing. The
// closings share a prefix and overlap, so a reset to the empty prefix on a
// mismatch would lose matches.
func TestClosingAutomatonNext(t *testing.T) {
	a := newClosingAutomaton([]string{"<|x|>", "<|y|>", "y|>|"})
	state := 0
	for _, step := range []struct {
		r    rune
		want string
		done bool
	}{
		{r: '<', want: "<"},
		{r: '|', want: "<|"},
		{r: '<', want: "<"},
		{r: '|', want: "<|"},
		{r: 'y', want: "<|y"},
		{r: '|', want: "<|y|"},
		{r: '>', done: true},
		{r: '|', want: ""},
		{r: 'y', want: "y"},
		{r: '|', want: "y|"},
		{r: '>', want: "y|>"},
		{r: '<', want: "<"},
		{r: 'z', want: ""},
	} {
		to, done := a.next(state, step.r)
		if done != step.done || (!done && a.states[to] != step.want) {
			t.Fatalf("after %q: state %q done %v, want state %q done %v", step.r, a.states[to], done, step.want, step.done)
		}
		state = to
	}
}

// TestThinkingGrammarRules checks the emitted rules: every prefix state may
// end the response, completing the closing leads to the renamed format root,
// the format grammar's other rules keep their names, and the added rules
// avoid a prefix the format grammar already uses.
func TestThinkingGrammarRules(t *testing.T) {
	format := "root ::= \"{\" space \"}\"\nspace ::= | \" \"\n"
	grammar := thinkingGrammar([]string{"</think>"}, format)
	for _, want := range []string{
		"root ::= ollama-thinking-0\n",
		"ollama-thinking-0 ::= | [^<] ollama-thinking-0 | [<] ollama-thinking-1\n",
		"ollama-thinking-7 ::= | [^<>] ollama-thinking-0 | [<] ollama-thinking-1 | [>] ollama-format\n",
		"ollama-format ::= \"{\" space \"}\"\nspace ::= | \" \"\n",
	} {
		if !strings.Contains(grammar, want) {
			t.Errorf("grammar lacks %q:\n%s", want, grammar)
		}
	}

	grammar = thinkingGrammar([]string{"</think>"}, "root ::= ollama-x\nollama-x ::= \"{}\"\n")
	if !strings.HasPrefix(grammar, "root ::= ollama-ollama-thinking-0\n") || !strings.Contains(grammar, "ollama-ollama-format ::= ollama-x\n") {
		t.Errorf("grammar does not avoid the format's rule prefix:\n%s", grammar)
	}
}
