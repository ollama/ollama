package model

import (
	"reflect"
	"testing"
)

const (
	part80  = "88888888888888888888888888888888888888888888888888888888888888888888888888888888"
	part350 = "33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333"
)

func TestParseNameParts(t *testing.T) {
	cases := []struct {
		in              string
		want            Name
		wantValidDigest bool
	}{
		{
			in: "host/namespace/model:tag",
			want: Name{
				Host:      "host",
				Namespace: "namespace",
				Model:     "model",
				Tag:       "tag",
			},
		},
		{
			in: "host/namespace/model",
			want: Name{
				Host:      "host",
				Namespace: "namespace",
				Model:     "model",
			},
		},
		{
			in: "namespace/model",
			want: Name{
				Namespace: "namespace",
				Model:     "model",
			},
		},
		{
			in: "model",
			want: Name{
				Model: "model",
			},
		},
		{
			in: "h/nn/mm:t",
			want: Name{
				Host:      "h",
				Namespace: "nn",
				Model:     "mm",
				Tag:       "t",
			},
		},
		{
			in: part80 + "/" + part80 + "/" + part80 + ":" + part80,
			want: Name{
				Host:      part80,
				Namespace: part80,
				Model:     part80,
				Tag:       part80,
			},
		},
		{
			in: part350 + "/" + part80 + "/" + part80 + ":" + part80,
			want: Name{
				Host:      part350,
				Namespace: part80,
				Model:     part80,
				Tag:       part80,
			},
		},
		{
			in: "@digest",
			want: Name{
				RawDigest: "digest",
			},
			wantValidDigest: false,
		},
		{
			in: "model@sha256:123",
			want: Name{
				Model:     "model",
				RawDigest: "sha256:123",
			},
			wantValidDigest: true,
		},
	}

	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			got := ParseNameBare(tt.in)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseName(%q) = %v; want %v", tt.in, got, tt.want)
			}
		})
	}
}

var testCases = map[string]bool{ // name -> valid
	"": false,

	"_why/_the/_lucky:_stiff": true,

	// minimal
	"h/n/m:t@d": true,

	"host/namespace/model:tag": true,
	"host/namespace/model":     false,
	"namespace/model":          false,
	"model":                    false,
	"@sha256-1000000000000000000000000000000000000000000000000000000000000000":      false,
	"model@sha256-1000000000000000000000000000000000000000000000000000000000000000": false,
	"model@sha256:1000000000000000000000000000000000000000000000000000000000000000": false,

	// long (but valid)
	part80 + "/" + part80 + "/" + part80 + ":" + part80:  true,
	part350 + "/" + part80 + "/" + part80 + ":" + part80: true,

	"h/nn/mm:t@sha256-1000000000000000000000000000000000000000000000000000000000000000": true, // bare minimum part sizes
	"h/nn/mm:t@sha256:1000000000000000000000000000000000000000000000000000000000000000": true, // bare minimum part sizes

	// unqualified
	"m":     false,
	"n/m:":  false,
	"h/n/m": false,
	"@t":    false,
	"m@d":   false,

	// invalids
	"^":      false,
	"mm:":    false,
	"/nn/mm": false,
	"//":     false,
	"//mm":   false,
	"hh//":   false,
	"//mm:@": false,
	"00@":    false,
	"@":      false,

	// not starting with alphanum
	"-hh/nn/mm:tt@dd": false,
	"hh/-nn/mm:tt@dd": false,
	"hh/nn/-mm:tt@dd": false,
	"hh/nn/mm:-tt@dd": false,
	"hh/nn/mm:tt@-dd": false,

	// hosts
	"host:https/namespace/model:tag": true,

	// colon in non-host part before tag
	"host/name:space/model:tag": false,
}

func TestNameparseNameDefault(t *testing.T) {
	const name = "xx"
	n := ParseName(name)
	got := n.String()
	want := "registry.ollama.ai/library/xx:latest"
	if got != want {
		t.Errorf("parseName(%q).String() = %q; want %q", name, got, want)
	}
}

func TestNameIsValid(t *testing.T) {
	var numStringTests int
	for s, want := range testCases {
		n := ParseNameBare(s)
		got := n.IsValid()
		if got != want {
			t.Errorf("parseName(%q).IsValid() = %v; want %v", s, got, want)
		}

		// Test roundtrip with String
		if got {
			got := ParseNameBare(s).String()
			if got != s {
				t.Errorf("parseName(%q).String() = %q; want %q", s, got, s)
			}
			numStringTests++
		}
	}

	if numStringTests == 0 {
		t.Errorf("no tests for Name.String")
	}
}

func TestNameIsValidPart(t *testing.T) {
	cases := []struct {
		kind partKind
		s    string
		want bool
	}{
		{kind: kindHost, s: "", want: false},
		{kind: kindHost, s: "a", want: true},
		{kind: kindHost, s: "a.", want: true},
		{kind: kindHost, s: "a.b", want: true},
		{kind: kindHost, s: "a:123", want: true},
		{kind: kindHost, s: "a:123/aa/bb", want: false},
		{kind: kindNamespace, s: "bb", want: true},
		{kind: kindNamespace, s: "a.", want: false},
		{kind: kindModel, s: "-h", want: false},
		{kind: kindDigest, s: "sha256-1000000000000000000000000000000000000000000000000000000000000000", want: true},
	}
	for _, tt := range cases {
		t.Run(tt.s, func(t *testing.T) {
			got := isValidPart(tt.kind, tt.s)
			if got != tt.want {
				t.Errorf("isValidPart(%s, %q) = %v; want %v", tt.kind, tt.s, got, tt.want)
			}
		})
	}

}

func FuzzName(f *testing.F) {
	for s := range testCases {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		n := ParseNameBare(s)
		if n.IsValid() {
			parts := [...]string{n.Host, n.Namespace, n.Model, n.Tag, n.RawDigest}
			for _, part := range parts {
				if part == ".." {
					t.Errorf("unexpected .. as valid part")
				}
				if len(part) > 350 {
					t.Errorf("part too long: %q", part)
				}
			}
			if n.String() != s {
				t.Errorf("String() = %q; want %q", n.String(), s)
			}
		}

	})
}
