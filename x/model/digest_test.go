package model

import "testing"

// - test scan
// - test marshal text
// - test unmarshal text
// - test log value
// - test string
// - test type
// - test digest
// - test valid
// - test driver valuer
// - test sql scanner
// - test parse digest

var testDigests = map[string]Digest{
	"":                 {},
	"sha256-1234":      {typ: "sha256", digest: "1234"},
	"sha256-5678":      {typ: "sha256", digest: "5678"},
	"blake2-9abc":      {typ: "blake2", digest: "9abc"},
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
		if !d.Valid() {
			want = ""
		}
		got := d.String()
		if got != want {
			t.Errorf("ParseDigest(%q).String() = %q; want %q", s, got, want)
		}
	}
}
