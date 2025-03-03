package names

import (
	"strings"
	"testing"
)

func TestParseName(t *testing.T) {
	cases := []struct {
		in   string
		want Name
	}{
		{"", Name{}},
		{"m:t", Name{m: "m", t: "t"}},
		{"m", Name{m: "m"}},
		{"/m", Name{m: "m"}},
		{"/n/m:t", Name{n: "n", m: "m", t: "t"}},
		{"n/m", Name{n: "n", m: "m"}},
		{"n/m:t", Name{n: "n", m: "m", t: "t"}},
		{"n/m", Name{n: "n", m: "m"}},
		{"n/m", Name{n: "n", m: "m"}},
		{strings.Repeat("m", MaxNameLength+1), Name{}},
		{"h/n/m:t", Name{h: "h", n: "n", m: "m", t: "t"}},
		{"ollama.com/library/_:latest", Name{h: "ollama.com", n: "library", m: "_", t: "latest"}},

		// Invalids
		// TODO: {"n:t/m:t", Name{}},
		// TODO: {"/h/n/m:t", Name{}},
	}

	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			got := Parse(tt.in)
			if got.Compare(tt.want) != 0 {
				t.Errorf("parseName(%q) = %#v, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestString(t *testing.T) {
	cases := []string{
		"",
		"m:t",
		"m:t",
		"m",
		"n/m",
		"n/m:t",
		"n/m",
		"n/m",
		"h/n/m:t",
		"ollama.com/library/_:latest",

		// Special cased to "round trip" without the leading slash.
		"/m",
		"/n/m:t",
	}
	for _, s := range cases {
		t.Run(s, func(t *testing.T) {
			s = strings.TrimPrefix(s, "/")
			if g := Parse(s).String(); g != s {
				t.Errorf("parse(%q).String() = %q", s, g)
			}
		})
	}
}

func TestParseExtended(t *testing.T) {
	cases := []struct {
		in string

		wantScheme string
		wantName   Name
		wantDigest string
	}{
		{"", "", Name{}, ""},
		{"m", "", Name{m: "m"}, ""},
		{"http://m", "http", Name{m: "m"}, ""},
		{"http+insecure://m", "http+insecure", Name{m: "m"}, ""},
		{"http://m@sha256:deadbeef", "http", Name{m: "m"}, "sha256:deadbeef"},
	}
	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			scheme, name, digest := Split(tt.in)
			n := Parse(name)
			if scheme != tt.wantScheme || n.Compare(tt.wantName) != 0 || digest != tt.wantDigest {
				t.Errorf("ParseExtended(%q) = %q, %#v, %q, want %q, %#v, %q", tt.in, scheme, name, digest, tt.wantScheme, tt.wantName, tt.wantDigest)
			}
		})
	}
}

func TestMerge(t *testing.T) {
	cases := []struct {
		a, b string
		want string
	}{
		{"", "", ""},
		{"m", "", "m"},
		{"", "m", ""},
		{"x", "y", "x"},
		{"o.com/n/m:t", "o.com/n/m:t", "o.com/n/m:t"},
		{"o.com/n/m:t", "o.com/n/_:t", "o.com/n/m:t"},

		{"bmizerany/smol", "ollama.com/library/_:latest", "ollama.com/bmizerany/smol:latest"},
		{"localhost:8080/bmizerany/smol", "ollama.com/library/_:latest", "localhost:8080/bmizerany/smol:latest"},
	}
	for _, tt := range cases {
		t.Run("", func(t *testing.T) {
			a, b := Parse(tt.a), Parse(tt.b)
			got := Merge(a, b)
			if got.Compare(Parse(tt.want)) != 0 {
				t.Errorf("merge(%q, %q) = %#v, want %q", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func TestParseStringRoundTrip(t *testing.T) {
	cases := []string{
		"",
		"m",
		"m:t",
		"n/m",
		"n/m:t",
		"n/m:t",
		"n/m",
		"n/m",
		"h/n/m:t",
		"ollama.com/library/_:latest",
	}
	for _, s := range cases {
		t.Run(s, func(t *testing.T) {
			if got := Parse(s).String(); got != s {
				t.Errorf("parse(%q).String() = %q", s, got)
			}
		})
	}
}

var junkName Name

func BenchmarkParseName(b *testing.B) {
	b.ReportAllocs()
	for range b.N {
		junkName = Parse("h/n/m:t")
	}
}

const (
	part80  = "88888888888888888888888888888888888888888888888888888888888888888888888888888888"
	part350 = "33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333"
)

var testCases = map[string]bool{ // name -> valid
	"": false,

	"_why/_the/_lucky:_stiff": true,

	// minimal
	"h/n/m:t": true,

	"host/namespace/model:tag": true,
	"host/namespace/model":     true,
	"namespace/model":          true,
	"model":                    true,

	// long (but valid)
	part80 + "/" + part80 + "/" + part80 + ":" + part80:  true,
	part350 + "/" + part80 + "/" + part80 + ":" + part80: true,

	// too long
	part80 + "/" + part80 + "/" + part80 + ":" + part350:       false,
	"x" + part350 + "/" + part80 + "/" + part80 + ":" + part80: false,

	"h/nn/mm:t": true, // bare minimum part sizes

	// unqualified
	"m":     true,
	"n/m:":  true,
	"h/n/m": true,
	"@t":    false,
	"m@d":   false,

	// invalids
	"^":      false,
	"mm:":    true,
	"/nn/mm": true,
	"//":     false, // empty model
	"//mm":   true,
	"hh//":   false, // empty model
	"//mm:@": false,
	"00@":    false,
	"@":      false,

	// not starting with alphanum
	"-hh/nn/mm:tt": false,
	"hh/-nn/mm:tt": false,
	"hh/nn/-mm:tt": false,
	"hh/nn/mm:-tt": false,

	// smells like a flag
	"-h": false,

	// hosts
	"host:https/namespace/model:tag": true,

	// colon in non-host part before tag
	"host/name:space/model:tag": false,
}

func TestParseNameValidation(t *testing.T) {
	for s, valid := range testCases {
		got := Parse(s)
		if got.IsValid() != valid {
			t.Logf("got: %v", got)
			t.Errorf("Parse(%q).IsValid() = %v; want !%[2]v", s, got.IsValid())
		}
	}
}
