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

// TestThinkingGrammarRules checks the emitted rules: prefix states cannot end
// before a closing, completing the closing leads to the renamed format root,
// the format grammar's other rules keep their names, and the added rules avoid
// a prefix the format grammar already uses.
func TestThinkingGrammarRules(t *testing.T) {
	format := "root ::= \"{\" space \"}\"\nspace ::= | \" \"\n"
	grammar := thinkingGrammar(nil, []string{"</think>"}, format)
	want := "root ::= ollama-thinking-0\n" +
		"ollama-thinking-0 ::= [^<] ollama-thinking-0 | [<] ollama-thinking-1\n" +
		"ollama-thinking-1 ::= [^/<] ollama-thinking-0 | [<] ollama-thinking-1 | [/] ollama-thinking-2\n" +
		"ollama-thinking-2 ::= [^<t] ollama-thinking-0 | [<] ollama-thinking-1 | [t] ollama-thinking-3\n" +
		"ollama-thinking-3 ::= [^<h] ollama-thinking-0 | [<] ollama-thinking-1 | [h] ollama-thinking-4\n" +
		"ollama-thinking-4 ::= [^<i] ollama-thinking-0 | [<] ollama-thinking-1 | [i] ollama-thinking-5\n" +
		"ollama-thinking-5 ::= [^<n] ollama-thinking-0 | [<] ollama-thinking-1 | [n] ollama-thinking-6\n" +
		"ollama-thinking-6 ::= [^<k] ollama-thinking-0 | [<] ollama-thinking-1 | [k] ollama-thinking-7\n" +
		"ollama-thinking-7 ::= [^<>] ollama-thinking-0 | [<] ollama-thinking-1 | [>] ollama-format\n" +
		"ollama-format ::= \"{\" space \"}\"\nspace ::= | \" \"\n"
	if grammar != want {
		t.Errorf("thinking grammar =\n%s\nwant:\n%s", grammar, want)
	}

	grammar = thinkingGrammar(nil, []string{"</think>"}, "root ::= ollama-x\nollama-x ::= \"{}\"\n")
	if !strings.HasPrefix(grammar, "root ::= ollama-ollama-thinking-0\n") || !strings.Contains(grammar, "ollama-ollama-format ::= ollama-x\n") {
		t.Errorf("grammar does not avoid the format's rule prefix:\n%s", grammar)
	}
}

func TestThinkingGrammarAllowsDirectFormatOrOpenedThinking(t *testing.T) {
	grammar := thinkingGrammar([]string{"<|channel>"}, []string{"<channel|>"}, "root ::= object\nobject ::= \"{}\"\n")
	for _, want := range []string{
		"root ::= ollama-format | ollama-thinking-open\n",
		"ollama-thinking-open ::= \"<|channel>\" ollama-thinking-0\n",
		"ollama-format ::= object\n",
	} {
		if !strings.Contains(grammar, want) {
			t.Errorf("thinking grammar lacks %q:\n%s", want, grammar)
		}
	}
	for _, line := range strings.Split(grammar, "\n") {
		if strings.HasPrefix(line, "ollama-thinking-") && strings.Contains(line, " ::= |") {
			t.Errorf("opened-thinking grammar permits an empty alternative: %s", line)
		}
	}
}
