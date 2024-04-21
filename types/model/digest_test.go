package model

import "testing"

var testDigests = map[string]Digest{
	"":                 {},
	"sha256-1234":      {s: "sha256-1234"},
	"sha256-5678":      {s: "sha256-5678"},
	"blake2-9abc":      {s: "blake2-9abc"},
	"-1234":            {},
	"sha256-":          {},
	"sha256-1234-5678": {},
	"sha256-P":         {}, //         invalid  hex
	"sha256-1234P":     {},
	"---":              {},
}

func TestDigestParse(t *testing.T) {
	// Test cases.
	for s, want := range testDigests {
		got := ParseDigest(s)
		t.Logf("ParseDigest(%q) = %#v", s, got)
		if got != want {
			t.Errorf("ParseDigest(%q) = %q; want %q", s, got, want)
		}
	}
}

func TestDigestString(t *testing.T) {
	// Test cases.
	for s, d := range testDigests {
		want := s
		if !d.IsValid() {
			want = ""
		}
		got := d.String()
		if got != want {
			t.Errorf("ParseDigest(%q).String() = %q; want %q", s, got, want)
		}

		got = ParseDigest(s).String()
		if got != want {
			t.Errorf("roundtrip ParseDigest(%q).String() = %q; want %q", s, got, want)
		}
	}
}
