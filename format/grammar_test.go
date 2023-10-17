package format

import (
	"bufio"
	"regexp"
	"strings"
	"testing"
)

const objectSchama = `{
	"type": "object",
	"properties": {
		"foo": {
			"type": "number"
		},
		"bar": {
			"type": "string"
		}
	}
}`

const objectGrammar = `
space ::= " "?
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space
string ::=  "\"" (
		[^"\\] |
		"\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
		)* "\"" space
root ::= "{" space "\"foo\"" space ":" space number "," space "\"bar\"" space ":" space string "}" space
`

var GRAMMAR_BEGINNING_RE = regexp.MustCompile(`^[\w\d\s]+::=`)

func readGrammarRules(grammar string) ([]string, error) {
	scanner := bufio.NewScanner(strings.NewReader(grammar))
	var rules []string
	var rule strings.Builder
	for scanner.Scan() {
		line := scanner.Text()
		if GRAMMAR_BEGINNING_RE.MatchString(line) {
			if rule.Len() > 0 {
				rules = append(rules, rule.String())
				rule.Reset()
			}
		}
		rule.WriteString(line)
	}
	if rule.Len() > 0 {
		rules = append(rules, rule.String())
	}
	return rules, scanner.Err()
}

func assertGrammaticallyEqual(t *testing.T, actual, expected string) {
	actualRules, err := readGrammarRules(strings.TrimSpace(actual))
	if err != nil {
		t.Errorf("Error reading actual grammar: %s", err)
		return
	}
	expectedRules, err := readGrammarRules(strings.TrimSpace(expected))
	if err != nil {
		t.Errorf("Error reading expected grammar: %s", err)
		return
	}

	if len(actualRules) != len(expectedRules) {
		t.Errorf("Expected %d rules, got %d", len(expectedRules), len(actualRules))
		return
	}

	expectedSet := make(map[string]bool, len(expectedRules))
	for _, expectedRule := range expectedRules {
		expectedSet[expectedRule] = true
	}
	for _, actualRule := range actualRules {
		if !expectedSet[actualRule] {
			t.Errorf("Unexpected rule found: %s", actualRule)
			return
		}
	}
}

func TestGrammar(t *testing.T) {
	t.Run("object schema to grammar", func(t *testing.T) {
		assertGrammaticallyEqual(t, SchemaToGrammar(objectSchama, []string{"foo", "bar"}), objectGrammar)
	})
}
