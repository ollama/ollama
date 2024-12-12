package grammar

import (
	"bufio"
	"iter"
	"strings"
	"testing"
)

const tests = `
# Each group of lines below (seperated by a blank line) is a test case.
#
# This is a comment.
#
# The first line is a root property schema, the second line is the expected grammar.
{}
root ::= value;

{"properties": {"a": {"type": "array", "items": {"type": "string"}}}}
root_tuple_0 ::= string;
root_a       ::= "[" ( "," root_tuple_0 )* "]";
root         ::= "{" "a" ":" root_a "}";

{"properties": {"o": {"type": "object", "properties": {"s": {"type": "string"}}}}}
root_o_s ::= string;
root_o   ::= "{" "s" ":" root_o_s "}";
root     ::= "{" "o" ":" root_o "}";

{"properties": {"b": {"type": "boolean"}}}
root_b ::= boolean;
root   ::= "{" "b" ":" root_b "}";

{"properties": {"e": {}}}
root_e ::= value;
root   ::= "{" "e" ":" root_e "}";

# TODO: Implement support for number constraints.
{"properties": {"n": {"type": "number", "minimum": 123, "maximum": 4567}}}
root_n ::= number;
root   ::= "{" "n" ":" root_n "}";
`

func TestGrammar(t *testing.T) {
	for tt := range testCases() {
		t.Run("", func(t *testing.T) {
			t.Logf("schema:\n%s", tt.schema)
			g, err := GrammarFromSchema(nil, []byte(tt.schema))
			if err != nil {
				t.Fatalf("GrammarFromSchema: %v", err)
			}
			got := string(g)
			got = strings.TrimPrefix(got, jsonTerms)
			if got != tt.want {
				t.Errorf("got:\n%q", got)
				t.Errorf("want:\n%q", tt.want)
			}
		})
	}
}

type testCase struct {
	schema string
	want   string
}

func testCases() iter.Seq[testCase] {
	return func(yield func(testCase) bool) {
		sc := bufio.NewScanner(strings.NewReader(tests))
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" || line[0] == '#' {
				continue
			}

			s := sc.Text()
			g := ""
			for sc.Scan() {
				line := strings.TrimSpace(sc.Text())
				if line == "" || line[0] == '#' {
					break
				}
				g += sc.Text() + "\n"
			}
			if !yield(testCase{s, g}) {
				return
			}
		}
		if err := sc.Err(); err != nil {
			panic(err)
		}
	}
}
