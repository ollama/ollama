package model

import (
	"fmt"
	"strings"
	"testing"
)

var testNames = map[string]Name{
	"mistral:latest":                 {model: "mistral", tag: "latest"},
	"mistral":                        {model: "mistral"},
	"mistral:30B":                    {model: "mistral", tag: "30B"},
	"mistral:7b":                     {model: "mistral", tag: "7b"},
	"mistral:7b+Q4_0":                {model: "mistral", tag: "7b", build: "Q4_0"},
	"mistral+KQED":                   {model: "mistral", build: "KQED"},
	"mistral.x-3:7b+Q4_0":            {model: "mistral.x-3", tag: "7b", build: "Q4_0"},
	"mistral:7b+q4_0":                {model: "mistral", tag: "7b", build: "Q4_0"},
	"llama2":                         {model: "llama2"},
	"user/model":                     {namespace: "user", model: "model"},
	"example.com/ns/mistral:7b+Q4_0": {host: "example.com", namespace: "ns", model: "mistral", tag: "7b", build: "Q4_0"},
	"example.com/ns/mistral:7b+x":    {host: "example.com", namespace: "ns", model: "mistral", tag: "7b", build: "X"},

	// invalid (includes fuzzing trophies)
	" / / : + ": {},
	" / : + ":   {},
	" : + ":     {},
	" + ":       {},
	" : ":       {},
	" / ":       {},
	" /":        {},
	"/ ":        {},
	"/":         {},
	":":         {},
	"+":         {},

	// (".") in namepsace is not allowed
	"invalid.com/7b+x": {},

	"invalid:7b+Q4_0:latest": {},
	"in valid":               {},
	"invalid/y/z/foo":        {},
	"/0":                     {},
	"0 /0":                   {},
	"0 /":                    {},
	"0/":                     {},
	":/0":                    {},
	"+0/00000":               {},
	"0+.\xf2\x80\xf6\x9d00000\xe5\x99\xe6\xd900\xd90\xa60\x91\xdc0\xff\xbf\x99\xe800\xb9\xdc\xd6\xc300\x970\xfb\xfd0\xe0\x8a\xe1\xad\xd40\x9700\xa80\x980\xdd0000\xb00\x91000\xfe0\x89\x9b\x90\x93\x9f0\xe60\xf7\x84\xb0\x87\xa5\xff0\xa000\x9a\x85\xf6\x85\xfe\xa9\xf9\xe9\xde00\xf4\xe0\x8f\x81\xad\xde00\xd700\xaa\xe000000\xb1\xee0\x91": {},
	"0//0":                        {},
	"m+^^^":                       {},
	"file:///etc/passwd":          {},
	"file:///etc/passwd:latest":   {},
	"file:///etc/passwd:latest+u": {},

	strings.Repeat("a", MaxNamePartLen):   {model: strings.Repeat("a", MaxNamePartLen)},
	strings.Repeat("a", MaxNamePartLen+1): {},
}

func TestNameParts(t *testing.T) {
	const wantNumParts = 5
	var p Name
	if len(p.Parts()) != wantNumParts {
		t.Errorf("Parts() = %d; want %d", len(p.Parts()), wantNumParts)
	}
}

func TestPartTooLong(t *testing.T) {
	for i := Host; i <= Build; i++ {
		t.Run(i.String(), func(t *testing.T) {
			var p Name
			switch i {
			case Host:
				p.host = strings.Repeat("a", MaxNamePartLen+1)
			case Namespace:
				p.namespace = strings.Repeat("a", MaxNamePartLen+1)
			case Model:
				p.model = strings.Repeat("a", MaxNamePartLen+1)
			case Tag:
				p.tag = strings.Repeat("a", MaxNamePartLen+1)
			case Build:
				p.build = strings.Repeat("a", MaxNamePartLen+1)
			}
			s := strings.Trim(p.String(), "+:/")
			if len(s) != MaxNamePartLen+1 {
				t.Errorf("len(String()) = %d; want %d", len(s), MaxNamePartLen+1)
				t.Logf("String() = %q", s)
			}
			if ParseName(p.String()).Valid() {
				t.Errorf("Valid(%q) = true; want false", p)
			}
		})
	}
}

func TestParseName(t *testing.T) {
	for baseName, want := range testNames {
		for _, prefix := range []string{"", "https://", "http://"} {
			// We should get the same results with or without the
			// http(s) prefixes
			s := prefix + baseName

			t.Run(s, func(t *testing.T) {
				got := ParseName(s)
				if !got.EqualFold(want) {
					t.Errorf("ParseName(%q) = %q; want %q", s, got, want)
				}

				// test round-trip
				if !ParseName(got.String()).EqualFold(got) {
					t.Errorf("String() = %s; want %s", got.String(), baseName)
				}

				if got.Valid() && got.Model() == "" {
					t.Errorf("Valid() = true; Model() = %q; want non-empty name", got.Model())
				} else if !got.Valid() && got.Model() != "" {
					t.Errorf("Valid() = false; Model() = %q; want empty name", got.Model())
				}
			})
		}
	}
}

func TestComplete(t *testing.T) {
	cases := []struct {
		in                   string
		complete             bool
		completeWithoutBuild bool
	}{
		{"", false, false},
		{"incomplete/mistral:7b+x", false, false},
		{"incomplete/mistral:7b+Q4_0", false, false},
		{"incomplete:7b+x", false, false},
		{"complete.com/x/mistral:latest+Q4_0", true, true},
		{"complete.com/x/mistral:latest", false, true},
	}

	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			p := ParseName(tt.in)
			t.Logf("ParseName(%q) = %#v", tt.in, p)
			if g := p.Complete(); g != tt.complete {
				t.Errorf("Complete(%q) = %v; want %v", tt.in, g, tt.complete)
			}
		})
	}
}

func TestNameStringVariants(t *testing.T) {
	cases := []struct {
		in              string
		nameAndTag      string
		nameTagAndBuild string
	}{
		{"x/y/z:8n+I", "z:8n", "z:8n+I"},
		{"x/y/z:8n", "z:8n", "z:8n"},
	}

	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			p := ParseName(tt.in)
			t.Logf("ParseName(%q) = %#v", tt.in, p)
			if g := p.DisplayCompact(); g != tt.nameAndTag {
				t.Errorf("ModelAndTag(%q) = %q; want %q", tt.in, g, tt.nameAndTag)
			}
			if g := p.DisplayShort(); g != tt.nameTagAndBuild {
				t.Errorf("ModelTagAndBuild(%q) = %q; want %q", tt.in, g, tt.nameTagAndBuild)
			}
		})
	}
}

func TestNameFull(t *testing.T) {
	const empty = "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/!(MISSING NAME):!(MISSING TAG)+!(MISSING BUILD)"

	cases := []struct {
		in       string
		wantFull string
	}{
		{"", empty},
		{"ns/mistral:7b+x", "!(MISSING DOMAIN)/ns/mistral:7b+X"},
		{"ns/mistral:7b+Q4_0", "!(MISSING DOMAIN)/ns/mistral:7b+Q4_0"},
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
			p := ParseName(tt.in)
			t.Logf("ParseName(%q) = %#v", tt.in, p)
			if g := p.DisplayFull(); g != tt.wantFull {
				t.Errorf("DisplayFull(%q) = %q; want %q", tt.in, g, tt.wantFull)
			}
		})
	}
}

func TestParseNameAllocs(t *testing.T) {
	// test allocations
	var r Name
	allocs := testing.AllocsPerRun(1000, func() {
		r = ParseName("example.com/mistral:7b+Q4_0")
	})
	_ = r
	if allocs > 0 {
		t.Errorf("ParseName allocs = %v; want 0", allocs)
	}
}

func BenchmarkParseName(b *testing.B) {
	b.ReportAllocs()

	var r Name
	for i := 0; i < b.N; i++ {
		r = ParseName("example.com/mistral:7b+Q4_0")
	}
	_ = r
}

func FuzzParseName(f *testing.F) {
	f.Add("example.com/mistral:7b+Q4_0")
	f.Add("example.com/mistral:7b+q4_0")
	f.Add("example.com/mistral:7b+x")
	f.Add("x/y/z:8n+I")
	f.Fuzz(func(t *testing.T, s string) {
		r0 := ParseName(s)
		if !r0.Valid() {
			if !r0.EqualFold(Name{}) {
				t.Errorf("expected invalid path to be zero value; got %#v", r0)
			}
			t.Skipf("invalid path: %q", s)
		}

		for _, p := range r0.Parts() {
			if len(p) > MaxNamePartLen {
				t.Errorf("part too long: %q", p)
			}
		}

		if !strings.EqualFold(r0.String(), s) {
			t.Errorf("String() did not round-trip with case insensitivity: %q\ngot  = %q\nwant = %q", s, r0.String(), s)
		}

		r1 := ParseName(r0.String())
		if !r0.EqualFold(r1) {
			t.Errorf("round-trip mismatch: %+v != %+v", r0, r1)
		}
	})
}

func TestFill(t *testing.T) {
	cases := []struct {
		dst  string
		src  string
		want string
	}{
		{"mistral", "o.com/library/PLACEHOLDER:latest+Q4_0", "o.com/library/mistral:latest+Q4_0"},
		{"o.com/library/mistral", "PLACEHOLDER:latest+Q4_0", "o.com/library/mistral:latest+Q4_0"},
		{"", "o.com/library/mistral:latest+Q4_0", "o.com/library/mistral:latest+Q4_0"},
	}

	for _, tt := range cases {
		t.Run(tt.dst, func(t *testing.T) {
			r := Fill(ParseName(tt.dst), ParseName(tt.src))
			if r.String() != tt.want {
				t.Errorf("Fill(%q, %q) = %q; want %q", tt.dst, tt.src, r, tt.want)
			}
		})
	}
}

func ExampleFill() {
	r := Fill(
		ParseName("mistral"),
		ParseName("registry.ollama.com/library/PLACEHOLDER:latest+Q4_0"),
	)
	fmt.Println(r)

	// Output:
	// registry.ollama.com/library/mistral:latest+Q4_0
}

func ExampleName_MapHash() {
	m := map[uint64]bool{}

	// key 1
	m[ParseName("mistral:latest+q4").MapHash()] = true
	m[ParseName("miSTRal:latest+Q4").MapHash()] = true
	m[ParseName("mistral:LATest+Q4").MapHash()] = true

	// key 2
	m[ParseName("mistral:LATest").MapHash()] = true

	fmt.Println(len(m))
	// Output:
	// 2
}
