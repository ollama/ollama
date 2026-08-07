package structured

import (
	"testing"
)

// advanceString feeds s byte-by-byte and reports how many bytes were
// accepted before the first rejection.
func advanceString(m *Matcher, s string) int {
	for i := 0; i < len(s); i++ {
		if !m.AdvanceByte(s[i]) {
			return i
		}
	}
	return len(s)
}

func mustCompileJSON(t *testing.T) *Grammar {
	t.Helper()
	g, err := Compile([]byte(`"json"`))
	if err != nil {
		t.Fatalf(`Compile("json") error: %v`, err)
	}
	return g
}

func TestJSONAcceptsObjects(t *testing.T) {
	g := mustCompileJSON(t)
	cases := []string{
		`{}`,
		`{"a":1}`,
		`{"a":-1.5e+10}`,
		`{"a":"b"}`,
		`{"a":true,"b":false,"c":null}`,
		`{"a":[1,2,3]}`,
		`{"a":{"b":{"c":[]}}}`,
		`{"a":"he said \"hi\" é\n"}`,
		`{"a":"café → naïve"}`, // raw multi-byte UTF-8 in strings
		`{ "a" :1}`,            // ws inside object before key... see below
	}
	// grammarJSON has no ws between a key string and ":", so the last case
	// is actually invalid under it; keep it out of the accept table.
	cases = cases[:len(cases)-1]

	for _, c := range cases {
		m := g.NewMatcher()
		if n := advanceString(m, c); n != len(c) {
			t.Errorf("%q: rejected at byte %d (%q)", c, n, c[n])
			continue
		}
		if !m.CanComplete() {
			t.Errorf("%q: CanComplete() = false after full input", c)
		}
	}
}

func TestJSONWhitespaceParity(t *testing.T) {
	g := mustCompileJSON(t)

	// grammarJSON (llm/llama_server.go) puts ws after "{", after ",",
	// after ":", and before "}" — but not between a key string and ":".
	accepts := []string{
		"{ \"a\":1}",
		"{\"a\": 1}",
		"{\"a\":1 }",
		"{\"a\":1, \"b\":2}",
		"{\"a\":\t[ 1,\n2 ]\n}",
		"{\"a\":true }",
	}
	for _, c := range accepts {
		m := g.NewMatcher()
		if n := advanceString(m, c); n != len(c) || !m.CanComplete() {
			t.Errorf("%q: rejected at byte %d or incomplete", c, n)
		}
	}

	// No ws allowed between key and colon: parity with grammarJSON.
	m := g.NewMatcher()
	in := "{\"a\" :1}"
	if n := advanceString(m, in); n != 4 {
		t.Errorf("%q: expected rejection at byte 4 (space before colon), advanced %d", in, n)
	}
}

func TestJSONRootRequiresObject(t *testing.T) {
	g := mustCompileJSON(t)
	for _, c := range []string{`[1]`, `"s"`, `42`, `true`, `null`} {
		m := g.NewMatcher()
		if m.AdvanceByte(c[0]) {
			t.Errorf("%q: first byte %q accepted at root; grammarJSON root is an object", c, c[0])
		}
	}
}

func TestJSONRejectsMalformed(t *testing.T) {
	g := mustCompileJSON(t)
	cases := []struct {
		in     string
		reject int // index of the byte that must be rejected
	}{
		{`{a:1}`, 1},         // unquoted key
		{`{"a":1,}`, 7},      // trailing comma then close
		{`{"a" 1}`, 4},       // missing colon (ws before colon invalid too)
		{`{"a":01}`, 6},      // leading zero
		{`{"a":1}}`, 7},      // trailing garbage after complete root
		{`{"a":'b'}`, 5},     // single quotes
		{`{"a":.5}`, 5},      // bare fraction
		{"{\"a\x01\":1}", 3}, // raw control byte in string
	}
	for _, c := range cases {
		m := g.NewMatcher()
		if n := advanceString(m, c.in); n != c.reject {
			t.Errorf("%q: rejected at byte %d, want %d", c.in, n, c.reject)
		}
	}
}

func TestJSONIncompleteCannotComplete(t *testing.T) {
	g := mustCompileJSON(t)
	for _, c := range []string{`{`, `{"a"`, `{"a":`, `{"a":1`, `{"a":[1`, `{"a":"unterminated`} {
		m := g.NewMatcher()
		if n := advanceString(m, c); n != len(c) {
			t.Fatalf("%q: prefix rejected at %d", c, n)
		}
		if m.CanComplete() {
			t.Errorf("%q: CanComplete() = true for incomplete JSON", c)
		}
	}
	// {"a":1 is incomplete, but 1 could also extend (e.g. 12): both must hold.
	m := g.NewMatcher()
	advanceString(m, `{"a":1`)
	if !m.AdvanceByte('2') {
		t.Error(`{"a":1 then '2': digit extension rejected`)
	}
}

func TestAdvanceIsAtomic(t *testing.T) {
	g := mustCompileJSON(t)
	m := g.NewMatcher()
	if m.Advance([]byte(`{"a"!`)) {
		t.Fatal(`Advance({"a"!) = true, want false`)
	}
	// State must be unchanged: the same valid continuation must still work.
	if !m.Advance([]byte(`{"a":1}`)) {
		t.Error("state was mutated by a failed Advance")
	}
	if !m.CanComplete() {
		t.Error("CanComplete() = false after valid object")
	}
}

func TestStateKeyDistinguishesStates(t *testing.T) {
	g := mustCompileJSON(t)
	a, b := g.NewMatcher(), g.NewMatcher()
	if a.StateKey() != b.StateKey() {
		t.Error("fresh matchers disagree on StateKey")
	}
	a.AdvanceByte('{')
	if a.StateKey() == b.StateKey() {
		t.Error("StateKey unchanged after consuming a byte")
	}
	b.AdvanceByte('{')
	if a.StateKey() != b.StateKey() {
		t.Error("equal states disagree on StateKey")
	}
}

func TestCompileRejectsInvalidFormat(t *testing.T) {
	for _, c := range []string{`"yaml"`, `42`, `[1]`, `"JSON"`, `hello`} {
		if _, err := Compile([]byte(c)); err == nil {
			t.Errorf("Compile(%s): expected error", c)
		}
	}
	// Empty / null formats are the caller's job to skip; Compile treats
	// them as errors rather than guessing.
	for _, c := range []string{``, `null`} {
		if _, err := Compile([]byte(c)); err == nil {
			t.Errorf("Compile(%q): expected error", c)
		}
	}
}
