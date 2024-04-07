package model

import (
	"bytes"
	"cmp"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"testing"
)

type fields struct {
	host, namespace, model, tag, build string
	digest                             string
}

func fieldsFromName(p Name) fields {
	return fields{
		host:      p.parts[Host],
		namespace: p.parts[Namespace],
		model:     p.parts[Model],
		tag:       p.parts[Tag],
		build:     p.parts[Build],
		digest:    p.parts[Digest],
	}
}

func mustParse(s string) Name {
	p := ParseName(s)
	if !p.Valid() {
		panic(fmt.Sprintf("invalid name: %q", s))
	}
	return p
}

var testNames = map[string]fields{
	"mistral:latest":                 {model: "mistral", tag: "latest"},
	"mistral":                        {model: "mistral"},
	"mistral:30B":                    {model: "mistral", tag: "30B"},
	"mistral:7b":                     {model: "mistral", tag: "7b"},
	"mistral:7b+Q4_0":                {model: "mistral", tag: "7b", build: "Q4_0"},
	"mistral+KQED":                   {model: "mistral", build: "KQED"},
	"mistral.x-3:7b+Q4_0":            {model: "mistral.x-3", tag: "7b", build: "Q4_0"},
	"mistral:7b+q4_0":                {model: "mistral", tag: "7b", build: "q4_0"},
	"llama2":                         {model: "llama2"},
	"user/model":                     {namespace: "user", model: "model"},
	"example.com/ns/mistral:7b+Q4_0": {host: "example.com", namespace: "ns", model: "mistral", tag: "7b", build: "Q4_0"},
	"example.com/ns/mistral:7b+X":    {host: "example.com", namespace: "ns", model: "mistral", tag: "7b", build: "X"},

	// invalid digest
	"mistral:latest@invalid256-": {},
	"mistral:latest@-123":        {},
	"mistral:latest@!-123":       {},
	"mistral:latest@1-!":         {},
	"mistral:latest@":            {},

	// resolved
	"x@sha123-1": {model: "x", digest: "sha123-1"},
	"@sha456-2":  {digest: "sha456-2"},

	// preserves case for build
	"x+b": {model: "x", build: "b"},

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
	var p Name
	if len(p.Parts()) != int(NumParts) {
		t.Errorf("Parts() = %d; want %d", len(p.Parts()), NumParts)
	}
}

func TestNamePartString(t *testing.T) {
	if g := NamePart(-2).String(); g != "Unknown" {
		t.Errorf("Unknown part = %q; want %q", g, "Unknown")
	}
	for kind, name := range kindNames {
		if g := kind.String(); g != name {
			t.Errorf("%s = %q; want %q", kind, g, name)
		}
	}
}

func TestIsValidDigestType(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"sha256", true},
		{"blake2", true},

		{"", false},
		{"-sha256", false},
		{"sha256-", false},
		{"Sha256", false},
		{"sha256(", false},
		{" sha256", false},
	}
	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			if g := isValidDigestType(tt.in); g != tt.want {
				t.Errorf("isValidDigestType(%q) = %v; want %v", tt.in, g, tt.want)
			}
		})
	}
}

func TestIsValidDigest(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"", false},
		{"sha256-123", true},
		{"sha256-1234567890abcdef", true},
		{"sha256-1234567890abcdef1234567890abcdeffffffffffffffffffffffffffffffffffffffffff", true},
		{"!sha256-123", false},
		{"sha256-123!", false},
		{"sha256-", false},
		{"-123", false},
	}
	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			if g := isValidDigest(tt.in); g != tt.want {
				t.Errorf("isValidDigest(%q) = %v; want %v", tt.in, g, tt.want)
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
				for kind, part := range Parts(s) {
					t.Logf("Part: %s: %q", kind, part)
				}

				name := ParseName(s)
				got := fieldsFromName(name)
				if got != want {
					t.Errorf("ParseName(%q) = %q; want %q", s, got, want)
				}

				// test round-trip
				if !ParseName(name.String()).EqualFold(name) {
					t.Errorf("String() = %s; want %s", name.String(), baseName)
				}

				if name.Valid() && name.DisplayModel() == "" {
					t.Errorf("Valid() = true; Model() = %q; want non-empty name", got.model)
				} else if !name.Valid() && name.DisplayModel() != "" {
					t.Errorf("Valid() = false; Model() = %q; want empty name", got.model)
				}

				if name.Resolved() && name.Digest() == "" {
					t.Errorf("Resolved() = true; Digest() = %q; want non-empty digest", got.digest)
				} else if !name.Resolved() && name.Digest() != "" {
					t.Errorf("Resolved() = false; Digest() = %q; want empty digest", got.digest)
				}
			})
		}
	}
}

func TestCompleteWithAndWithoutBuild(t *testing.T) {
	cases := []struct {
		in              string
		complete        bool
		completeNoBuild bool
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
			if g := p.CompleteNoBuild(); g != tt.completeNoBuild {
				t.Errorf("CompleteNoBuild(%q) = %v; want %v", tt.in, g, tt.completeNoBuild)
			}
		})
	}

	// Complete uses Parts which returns a slice, but it should be
	// inlined when used in Complete, preventing any allocations or
	// escaping to the heap.
	allocs := testing.AllocsPerRun(1000, func() {
		keep(ParseName("complete.com/x/mistral:latest+Q4_0").Complete())
	})
	if allocs > 0 {
		t.Errorf("Complete allocs = %v; want 0", allocs)
	}
}

func TestNameLogValue(t *testing.T) {
	cases := []string{
		"example.com/library/mistral:latest+Q4_0",
		"mistral:latest",
		"mistral:7b+Q4_0",
	}
	for _, s := range cases {
		t.Run(s, func(t *testing.T) {
			var b bytes.Buffer
			log := slog.New(slog.NewTextHandler(&b, nil))
			name := ParseName(s)
			log.Info("", "name", name)
			want := fmt.Sprintf("name=%s", name.GoString())
			got := b.String()
			if !strings.Contains(got, want) {
				t.Errorf("expected log output to contain %q; got %q", want, got)
			}
		})
	}
}

func TestNameDisplay(t *testing.T) {
	cases := []struct {
		name         string
		in           string
		wantShort    string
		wantLong     string
		wantComplete string
		wantString   string
		wantModel    string
		wantGoString string // default is tt.in
	}{
		{
			name:         "Complete Name",
			in:           "example.com/library/mistral:latest+Q4_0",
			wantShort:    "mistral:latest",
			wantLong:     "library/mistral:latest",
			wantComplete: "example.com/library/mistral:latest",
			wantModel:    "mistral",
			wantGoString: "example.com/library/mistral:latest+Q4_0@?",
		},
		{
			name:         "Short Name",
			in:           "mistral:latest",
			wantShort:    "mistral:latest",
			wantLong:     "mistral:latest",
			wantComplete: "mistral:latest",
			wantModel:    "mistral",
			wantGoString: "?/?/mistral:latest+?@?",
		},
		{
			name:         "Long Name",
			in:           "library/mistral:latest",
			wantShort:    "mistral:latest",
			wantLong:     "library/mistral:latest",
			wantComplete: "library/mistral:latest",
			wantModel:    "mistral",
			wantGoString: "?/library/mistral:latest+?@?",
		},
		{
			name:         "Case Preserved",
			in:           "Library/Mistral:Latest",
			wantShort:    "Mistral:Latest",
			wantLong:     "Library/Mistral:Latest",
			wantComplete: "Library/Mistral:Latest",
			wantModel:    "Mistral",
			wantGoString: "?/Library/Mistral:Latest+?@?",
		},
		{
			name:         "With digest",
			in:           "Library/Mistral:Latest@sha256-123456",
			wantShort:    "Mistral:Latest",
			wantLong:     "Library/Mistral:Latest",
			wantComplete: "Library/Mistral:Latest",
			wantModel:    "Mistral",
			wantGoString: "?/Library/Mistral:Latest+?@sha256-123456",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			p := ParseName(tt.in)
			if g := p.DisplayShort(); g != tt.wantShort {
				t.Errorf("DisplayShort = %q; want %q", g, tt.wantShort)
			}
			if g := p.DisplayLong(); g != tt.wantLong {
				t.Errorf("DisplayLong = %q; want %q", g, tt.wantLong)
			}
			if g := p.DisplayFullest(); g != tt.wantComplete {
				t.Errorf("DisplayFullest = %q; want %q", g, tt.wantComplete)
			}
			if g := p.String(); g != tt.in {
				t.Errorf("String(%q) = %q; want %q", tt.in, g, tt.in)
			}
			if g := p.DisplayModel(); g != tt.wantModel {
				t.Errorf("Model = %q; want %q", g, tt.wantModel)
			}

			tt.wantGoString = cmp.Or(tt.wantGoString, tt.in)
			if g := fmt.Sprintf("%#v", p); g != tt.wantGoString {
				t.Errorf("GoString() = %q; want %q", g, tt.wantGoString)
			}
		})
	}
}

func TestParseNameAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		keep(ParseName("example.com/mistral:7b+Q4_0"))
	})
	if allocs > 0 {
		t.Errorf("ParseName allocs = %v; want 0", allocs)
	}
}

func BenchmarkParseName(b *testing.B) {
	b.ReportAllocs()

	for range b.N {
		keep(ParseName("example.com/mistral:7b+Q4_0"))
	}
}

func BenchmarkNameDisplay(b *testing.B) {
	b.ReportAllocs()

	r := ParseName("example.com/mistral:7b+Q4_0")
	b.Run("Short", func(b *testing.B) {
		for range b.N {
			keep(r.DisplayShort())
		}
	})
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

func TestNameTextMarshal(t *testing.T) {
	cases := []struct {
		in      string
		want    string
		wantErr error
	}{
		{"example.com/mistral:latest+Q4_0", "", nil},
		{"mistral:latest+Q4_0", "mistral:latest+Q4_0", nil},
		{"mistral:latest", "mistral:latest", nil},
		{"mistral", "mistral", nil},
		{"mistral:7b", "mistral:7b", nil},
		{"example.com/library/mistral:latest+Q4_0", "example.com/library/mistral:latest+Q4_0", nil},
	}

	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			p := ParseName(tt.in)
			got, err := p.MarshalText()
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("MarshalText() error = %v; want %v", err, tt.wantErr)
			}
			if string(got) != tt.want {
				t.Errorf("MarshalText() = %q; want %q", got, tt.want)
			}

			var r Name
			if err := r.UnmarshalText(got); err != nil {
				t.Fatalf("UnmarshalText() error = %v; want nil", err)
			}
			if !r.EqualFold(p) {
				t.Errorf("UnmarshalText() = %q; want %q", r, p)
			}
		})
	}

	t.Run("UnmarshalText into valid Name", func(t *testing.T) {
		// UnmarshalText should not be called on a valid Name.
		p := mustParse("x")
		if err := p.UnmarshalText([]byte("mistral:latest+Q4_0")); err == nil {
			t.Error("UnmarshalText() = nil; want error")
		}
	})

	t.Run("TextMarshal allocs", func(t *testing.T) {
		var data []byte
		name := ParseName("example.com/ns/mistral:latest+Q4_0")
		if !name.Complete() {
			// sanity check
			panic("sanity check failed")
		}

		allocs := testing.AllocsPerRun(1000, func() {
			var err error
			data, err = name.MarshalText()
			if err != nil {
				t.Fatal(err)
			}
			if len(data) == 0 {
				t.Fatal("MarshalText() = 0; want non-zero")
			}
		})
		if allocs > 0 {
			// TODO: Update when/if this lands:
			// https://github.com/golang/go/issues/62384
			//
			// Currently, the best we can do is 1 alloc.
			t.Errorf("MarshalText allocs = %v; want <= 1", allocs)
		}
	})
}

func TestSQL(t *testing.T) {
	t.Run("Scan for already valid Name", func(t *testing.T) {
		p := mustParse("x")
		if err := p.Scan("mistral:latest+Q4_0"); err == nil {
			t.Error("Scan() = nil; want error")
		}
	})
	t.Run("Scan for invalid Name", func(t *testing.T) {
		p := Name{}
		if err := p.Scan("mistral:latest+Q4_0"); err != nil {
			t.Errorf("Scan() = %v; want nil", err)
		}
		if p.String() != "mistral:latest+Q4_0" {
			t.Errorf("String() = %q; want %q", p, "mistral:latest+Q4_0")
		}
	})
	t.Run("Value", func(t *testing.T) {
		p := mustParse("x")
		if g, err := p.Value(); err != nil {
			t.Errorf("Value() error = %v; want nil", err)
		} else if g != "x" {
			t.Errorf("Value() = %q; want %q", g, "x")
		}
	})
}

func TestNameStringAllocs(t *testing.T) {
	name := ParseName("example.com/ns/mistral:latest+Q4_0")
	allocs := testing.AllocsPerRun(1000, func() {
		keep(name.String())
	})
	if allocs > 1 {
		t.Errorf("String allocs = %v; want 0", allocs)
	}
}

func ExampleFill() {
	defaults := ParseName("registry.ollama.com/library/PLACEHOLDER:latest+Q4_0")
	r := Fill(ParseName("mistral"), defaults)
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

func ExampleName_CompareFold_sort() {
	names := []Name{
		ParseName("mistral:latest"),
		ParseName("mistRal:7b+q4"),
		ParseName("MIstral:7b"),
	}

	slices.SortFunc(names, Name.CompareFold)

	for _, n := range names {
		fmt.Println(n)
	}

	// Output:
	// MIstral:7b
	// mistRal:7b+q4
	// mistral:latest
}

func ExampleName_DisplayFullest() {
	for _, s := range []string{
		"example.com/jmorganca/mistral:latest+Q4_0",
		"mistral:latest+Q4_0",
		"mistral:latest",
	} {
		fmt.Println(ParseName(s).DisplayFullest())
	}

	// Output:
	// example.com/jmorganca/mistral:latest
	// mistral:latest
	// mistral:latest
}

func keep[T any](v T) T { return v }
