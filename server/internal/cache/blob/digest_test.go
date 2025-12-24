package blob

import (
	"encoding/json"
	"testing"
)

func FuzzParseDigest(f *testing.F) {
	// Add seed corpus
	seeds := []string{
		"sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		"sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		"sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde",
		"sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde",
		"sha255-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		"invalid",
		"",
		"sha256:",
		"sha256-",
		":",
		"-",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, input string) {
		d, err := ParseDigest(input)
		if err != nil {
			return // Invalid input is expected
		}

		// If valid, test round-trip consistency
		output := d.String()
		d2, err := ParseDigest(output)
		if err != nil {
			t.Errorf("Round-trip failed: ParseDigest(d.String()) returned error: %v\nOriginal input: %q\nString output: %q", err, input, output)
			return
		}

		if d.String() != d2.String() {
			t.Errorf("Round-trip changed digest: %q -> %q", d.String(), d2.String())
		}

		// Verify the digest is valid
		if !d.IsValid() {
			t.Errorf("ParseDigest returned a digest that IsValid() returns false for input: %q", input)
		}
	})
}

func TestParseDigest(t *testing.T) {
	cases := []struct {
		in    string
		valid bool
	}{
		{"sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", true},
		{"sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", true},

		// too short
		{"sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde", false},
		{"sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde", false},

		// too long
		{"sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0", false},
		{"sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0", false},

		// invalid prefix
		{"sha255-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", false},
		{"sha255:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", false},
		{"sha256!0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", false},

		// invalid hex
		{"sha256-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", false},
		{"sha256:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", false},
	}

	for _, tt := range cases {
		got, err := ParseDigest(tt.in)
		if tt.valid && err != nil {
			t.Errorf("ParseDigest(%q) = %v, %v; want valid", tt.in, got, err)
		}
		want := "sha256:" + tt.in[7:]
		if tt.valid && got.String() != want {
			t.Errorf("ParseDigest(%q).String() = %q, want %q", tt.in, got.String(), want)
		}
	}
}

func TestDigestMarshalText(t *testing.T) {
	const s = `"sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"`
	var d Digest
	if err := json.Unmarshal([]byte(s), &d); err != nil {
		t.Errorf("json.Unmarshal: %v", err)
	}
	out, err := json.Marshal(d)
	if err != nil {
		t.Errorf("json.Marshal: %v", err)
	}
	want := `"sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"`
	if string(out) != want {
		t.Errorf("json.Marshal: got %s, want %s", out, want)
	}
	if err := json.Unmarshal([]byte(`"invalid"`), &Digest{}); err == nil {
		t.Errorf("json.Unmarshal: expected error")
	}
}
