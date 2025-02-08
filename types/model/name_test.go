package model

import (
	"path/filepath"
	"reflect"
	"runtime"
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
		wantFilepath    string
		wantValidDigest bool
	}{
		{
			in: "registry.ollama.ai/library/dolphin-mistral:7b-v2.6-dpo-laser-q6_K",
			want: Name{
				Host:      "registry.ollama.ai",
				Namespace: "library",
				Model:     "dolphin-mistral",
				Tag:       "7b-v2.6-dpo-laser-q6_K",
			},
			wantFilepath: filepath.Join("registry.ollama.ai", "library", "dolphin-mistral", "7b-v2.6-dpo-laser-q6_K"),
		},
		{
			in: "scheme://host:port/namespace/model:tag",
			want: Name{
				Host:      "host:port",
				Namespace: "namespace",
				Model:     "model",
				Tag:       "tag",
			},
			wantFilepath: filepath.Join("host:port", "namespace", "model", "tag"),
		},
		{
			in: "host/namespace/model:tag",
			want: Name{
				Host:      "host",
				Namespace: "namespace",
				Model:     "model",
				Tag:       "tag",
			},
			wantFilepath: filepath.Join("host", "namespace", "model", "tag"),
		},
		{
			in: "host:port/namespace/model:tag",
			want: Name{
				Host:      "host:port",
				Namespace: "namespace",
				Model:     "model",
				Tag:       "tag",
			},
			wantFilepath: filepath.Join("host:port", "namespace", "model", "tag"),
		},
		{
			in: "host/namespace/model",
			want: Name{
				Host:      "host",
				Namespace: "namespace",
				Model:     "model",
			},
			wantFilepath: filepath.Join("host", "namespace", "model", "latest"),
		},
		{
			in: "host:port/namespace/model",
			want: Name{
				Host:      "host:port",
				Namespace: "namespace",
				Model:     "model",
			},
			wantFilepath: filepath.Join("host:port", "namespace", "model", "latest"),
		},
		{
			in: "namespace/model",
			want: Name{
				Namespace: "namespace",
				Model:     "model",
			},
			wantFilepath: filepath.Join("registry.ollama.ai", "namespace", "model", "latest"),
		},
		{
			in: "model",
			want: Name{
				Model: "model",
			},
			wantFilepath: filepath.Join("registry.ollama.ai", "library", "model", "latest"),
		},
		{
			in: "h/nn/mm:t",
			want: Name{
				Host:      "h",
				Namespace: "nn",
				Model:     "mm",
				Tag:       "t",
			},
			wantFilepath: filepath.Join("h", "nn", "mm", "t"),
		},
		{
			in: part80 + "/" + part80 + "/" + part80 + ":" + part80,
			want: Name{
				Host:      part80,
				Namespace: part80,
				Model:     part80,
				Tag:       part80,
			},
			wantFilepath: filepath.Join(part80, part80, part80, part80),
		},
		{
			in: part350 + "/" + part80 + "/" + part80 + ":" + part80,
			want: Name{
				Host:      part350,
				Namespace: part80,
				Model:     part80,
				Tag:       part80,
			},
			wantFilepath: filepath.Join(part350, part80, part80, part80),
		},
	}

	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			got := ParseNameBare(tt.in)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseName(%q) = %v; want %v", tt.in, got, tt.want)
			}

			got = ParseName(tt.in)
			if tt.wantFilepath != "" && got.Filepath() != tt.wantFilepath {
				t.Errorf("parseName(%q).Filepath() = %q; want %q", tt.in, got.Filepath(), tt.wantFilepath)
			}
		})
	}
}

var testCases = map[string]bool{ // name -> valid
	"": false,

	"_why/_the/_lucky:_stiff": true,

	// minimal
	"h/n/m:t": true,

	"host/namespace/model:tag": true,
	"host/namespace/model":     false,
	"namespace/model":          false,
	"model":                    false,

	// long (but valid)
	part80 + "/" + part80 + "/" + part80 + ":" + part80:  true,
	part350 + "/" + part80 + "/" + part80 + ":" + part80: true,

	"h/nn/mm:t": true, // bare minimum part sizes

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
	"-hh/nn/mm:tt": false,
	"hh/-nn/mm:tt": false,
	"hh/nn/-mm:tt": false,
	"hh/nn/mm:-tt": false,

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

func TestFilepathAllocs(t *testing.T) {
	n := ParseNameBare("HOST/NAMESPACE/MODEL:TAG")
	allocs := testing.AllocsPerRun(1000, func() {
		n.Filepath()
	})
	var allowedAllocs float64 = 1
	if runtime.GOOS == "windows" {
		allowedAllocs = 3
	}
	if allocs > allowedAllocs {
		t.Errorf("allocs = %v; allowed %v", allocs, allowedAllocs)
	}
}

func TestParseNameFromFilepath(t *testing.T) {
	cases := map[string]Name{
		filepath.Join("host", "namespace", "model", "tag"):      {Host: "host", Namespace: "namespace", Model: "model", Tag: "tag"},
		filepath.Join("host:port", "namespace", "model", "tag"): {Host: "host:port", Namespace: "namespace", Model: "model", Tag: "tag"},
		filepath.Join("namespace", "model", "tag"):              {},
		filepath.Join("model", "tag"):                           {},
		"model":                                                 {},
		filepath.Join("..", "..", "model", "tag"):               {},
		filepath.Join("", "namespace", ".", "tag"):              {},
		filepath.Join(".", ".", ".", "."):                       {},
		filepath.Join("/", "path", "to", "random", "file"):      {},
	}

	for in, want := range cases {
		t.Run(in, func(t *testing.T) {
			got := ParseNameFromFilepath(in)

			if !reflect.DeepEqual(got, want) {
				t.Errorf("parseNameFromFilepath(%q) = %v; want %v", in, got, want)
			}
		})
	}
}

func TestDisplayShortest(t *testing.T) {
	cases := map[string]string{
		"registry.ollama.ai/library/model:latest": "model:latest",
		"registry.ollama.ai/library/model:tag":    "model:tag",
		"registry.ollama.ai/namespace/model:tag":  "namespace/model:tag",
		"host/namespace/model:tag":                "host/namespace/model:tag",
		"host/library/model:tag":                  "host/library/model:tag",
	}

	for in, want := range cases {
		t.Run(in, func(t *testing.T) {
			got := ParseNameBare(in).DisplayShortest()
			if got != want {
				t.Errorf("parseName(%q).DisplayShortest() = %q; want %q", in, got, want)
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
			parts := [...]string{n.Host, n.Namespace, n.Model, n.Tag}
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

func TestIsValidNamespace(t *testing.T) {
	cases := []struct {
		username string
		expected bool
	}{
		{"", false},
		{"a", true},
		{"a:b", false},
		{"a/b", false},
		{"a:b/c", false},
		{"a/b:c", false},
		{"a/b:c", false},
		{"a/b:c/d", false},
		{"a/b:c/d@e", false},
		{"a/b:c/d@sha256-100", false},
		{"himynameisjoe", true},
		{"himynameisreallyreallyreallyreallylongbutitshouldstillbevalid", true},
	}
	for _, tt := range cases {
		t.Run(tt.username, func(t *testing.T) {
			if got := IsValidNamespace(tt.username); got != tt.expected {
				t.Errorf("IsValidName(%q) = %v; want %v", tt.username, got, tt.expected)
			}
		})
	}
}
