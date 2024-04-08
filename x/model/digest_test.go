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

func TestDigestUnmarshalText(t *testing.T) {
	const testDigest = "sha256-1234"
	t.Run("UnmarshalText (into Valid)", func(t *testing.T) {
		d := ParseDigest(testDigest)
		if !d.IsValid() {
			panic("invalid test")
		}
		if err := d.UnmarshalText(nil); err == nil {
			t.Errorf("UnmarshalText on valid Digest did not return error")
		}
		if d.String() != testDigest {
			t.Errorf("UnmarshalText on valid Digest changed Digest: %q", d.String())
		}
	})
	t.Run("UnmarshalText make safe copy", func(t *testing.T) {
		data := []byte(testDigest)
		var d Digest
		d.UnmarshalText(data)
		data[0] = 'x'
		if d.String() != testDigest {
			t.Errorf("UnmarshalText did not make a safe copy")
		}
	})
}
