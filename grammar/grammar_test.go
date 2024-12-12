package grammar

import (
	"bufio"
	"cmp"
	"iter"
	"strings"
	"testing"

	_ "embed"

	"github.com/ollama/ollama/grammar/internal/diff"
)

func TestFromSchema(t *testing.T) {
	for tt := range testCases(t) {
		t.Run(tt.name, func(t *testing.T) {
			g, err := FromSchema(nil, []byte(tt.schema))
			if err != nil {
				t.Fatalf("FromSchema: %v", err)
			}
			got := string(g)
			got = strings.TrimPrefix(got, jsonTerms)
			if got != tt.want {
				t.Logf("schema:\n%s", tt.schema)
				t.Fatal(string(diff.Diff("got", []byte(got), "want", []byte(tt.want))))
			}
		})
	}
}

type testCase struct {
	name   string
	schema string
	want   string
}

//go:embed testdata/schemas.txt
var tests string

func testCases(t testing.TB) iter.Seq[testCase] {
	t.Helper()
	return func(yield func(testCase) bool) {
		t.Helper()
		sc := bufio.NewScanner(strings.NewReader(tests))
		name := ""
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" {
				name = ""
				continue
			}
			if line[0] == '#' {
				name = cmp.Or(name, strings.TrimSpace(line[1:]))
				continue
			}
			s := sc.Text()
			g := ""
			for sc.Scan() {
				line = strings.TrimSpace(sc.Text())
				if line == "" || line[0] == '#' {
					break
				}
				g += sc.Text() + "\n"
			}
			if !yield(testCase{name, s, g}) {
				return
			}
			name = strings.TrimSpace(strings.TrimPrefix(line, "#"))
		}
		if err := sc.Err(); err != nil {
			t.Fatalf("error reading tests: %v", err)
		}
	}
}
