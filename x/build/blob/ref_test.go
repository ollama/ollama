package blob

import (
	"strings"
	"testing"
)

// test refs
const (
	refTooLong = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)

var testRefs = map[string]Ref{
	"mistral:latest":      {name: "mistral", tag: "latest"},
	"mistral":             {name: "mistral"},
	"mistral:30B":         {name: "mistral", tag: "30B"},
	"mistral:7b":          {name: "mistral", tag: "7b"},
	"mistral:7b+Q4_0":     {name: "mistral", tag: "7b", build: "Q4_0"},
	"mistral+KQED":        {name: "mistral", build: "KQED"},
	"mistral.x-3:7b+Q4_0": {name: "mistral.x-3", tag: "7b", build: "Q4_0"},
	"mistral:7b+q4_0":     {name: "mistral", tag: "7b", build: "Q4_0"},
	"llama2":              {name: "llama2"},

	// invalid
	"mistral:7b+Q4_0:latest": {},
	"mi tral":                {},

	// From fuzzing
	"/0": {},
}

func TestRefParts(t *testing.T) {
	const wantNumParts = 5
	var ref Ref
	if len(ref.Parts()) != wantNumParts {
		t.Errorf("Parts() = %d; want %d", len(ref.Parts()), wantNumParts)
	}
}

func TestParseRef(t *testing.T) {
	for s, want := range testRefs {
		t.Run(s, func(t *testing.T) {
			got := ParseRef(s)
			if got != want {
				t.Errorf("ParseRef(%q) = %q; want %q", s, got, want)
			}

			// test round-trip
			if ParseRef(got.String()) != got {
				t.Errorf("String() = %q; want %q", got.String(), s)
			}
		})
	}
}

func TestRefFull(t *testing.T) {
	const empty = "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/!(MISSING NAME):!(MISSING TAG)+!(MISSING BUILD)"

	cases := []struct {
		in       string
		wantFull string
	}{
		{"", empty},
		{"example.com/mistral:7b+x", "!(MISSING DOMAIN)/example.com/mistral:7b+X"},
		{"example.com/mistral:7b+Q4_0", "!(MISSING DOMAIN)/example.com/mistral:7b+Q4_0"},
		{"example.com/x/mistral:latest", "example.com/x/mistral:latest+!(MISSING BUILD)"},
		{"example.com/x/mistral:latest+Q4_0", "example.com/x/mistral:latest+Q4_0"},

		{"mistral:7b+x", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:7b+X"},
		{"mistral:7b+q4_0", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:7b+Q4_0"},
		{"mistral:7b+Q4_0", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:7b+Q4_0"},
		{"mistral:latest", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:latest+!(MISSING BUILD)"},
		{"mistral", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:!(MISSING TAG)+!(MISSING BUILD)"},
		{"mistral:30b", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:30b+!(MISSING BUILD)"},
	}

	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			ref := ParseRef(tt.in)
			t.Logf("ParseRef(%q) = %#v", tt.in, ref)
			if g := ref.Full(); g != tt.wantFull {
				t.Errorf("Full(%q) = %q; want %q", tt.in, g, tt.wantFull)
			}
		})
	}
}

func TestParseRefAllocs(t *testing.T) {
	// test allocations
	allocs := testing.AllocsPerRun(1000, func() {
		ParseRef("example.com/mistral:7b+Q4_0")
	})
	if allocs > 0 {
		t.Errorf("ParseRef allocs = %v; want 0", allocs)
	}
}

func BenchmarkParseRef(b *testing.B) {
	b.ReportAllocs()

	var r Ref
	for i := 0; i < b.N; i++ {
		r = ParseRef("example.com/mistral:7b+Q4_0")
	}
	_ = r
}

func FuzzParseRef(f *testing.F) {
	f.Add("example.com/mistral:7b+Q4_0")
	f.Add("example.com/mistral:7b+q4_0")
	f.Add("example.com/mistral:7b+x")
	f.Add("x/y/z:8n+I")
	f.Fuzz(func(t *testing.T, s string) {
		r0 := ParseRef(s)
		if !r0.Valid() {
			if r0 != (Ref{}) {
				t.Errorf("expected invalid ref to be zero value; got %#v", r0)
			}
			t.Skipf("invalid ref: %q", s)
		}

		if !strings.EqualFold(r0.String(), s) {
			t.Errorf("String() did not round-trip with case insensitivity: %q\ngot  = %q\nwant = %q", s, r0.String(), s)
		}

		r1 := ParseRef(r0.String())
		if r0 != r1 {
			t.Errorf("round-trip mismatch: %q != %q", r0, r1)
		}
	})
}
