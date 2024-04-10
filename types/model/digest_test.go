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
		if err := d.UnmarshalText(data); err != nil {
			t.Errorf("UnmarshalText on valid Digest returned error: %v", err)
		}
		data[0] = 'x'
		if d.String() != testDigest {
			t.Errorf("UnmarshalText did not make a safe copy")
		}
	})
}

func TestDigestScan(t *testing.T) {
	const testDigest = "sha256-1234"
	t.Run("Scan", func(t *testing.T) {
		var d Digest
		if err := d.Scan(testDigest); err != nil {
			t.Errorf("Scan on valid Digest returned error: %v", err)
		}
		if d.String() != testDigest {
			t.Errorf("Scan on valid Digest changed Digest: %q", d.String())
		}
		s, err := d.Value()
		if err != nil {
			t.Errorf("Value on valid Digest returned error: %v", err)
		}
		if s != testDigest {
			t.Errorf("Value on valid Digest changed Digest: %q", s)
		}
	})
	t.Run("Scan (into Valid)", func(t *testing.T) {
		d := ParseDigest(testDigest)
		if !d.IsValid() {
			panic("invalid test")
		}
		if err := d.Scan(nil); err == nil {
			t.Errorf("Scan on valid Digest did not return error")
		}
		if d.String() != testDigest {
			t.Errorf("Scan on valid Digest changed Digest: %q", d.String())
		}
	})
	t.Run("Scan make safe copy", func(t *testing.T) {
		data := []byte(testDigest)
		var d Digest
		if err := d.Scan(data); err != nil {
			t.Errorf("Scan on valid Digest returned error: %v", err)
		}
		data[0] = 'x'
		if d.String() != testDigest {
			t.Errorf("Scan did not make a safe copy")
		}
	})
}
