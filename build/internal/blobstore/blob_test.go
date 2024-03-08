package blobstore

import (
	"strings"
	"testing"
)

func TestParseID(t *testing.T) {
	const valid = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	var invalid = strings.Repeat("\x00", HashSize*2)

	cases := []struct {
		in   string
		want string
	}{
		{"", invalid},
		{"sha256-", invalid},
		{"sha256-" + valid, valid},

		{"" + valid, invalid},              // no prefix
		{"sha123-" + valid, invalid},       // invalid prefix
		{"sha256-" + valid[1:], invalid},   // too short
		{"sha256-" + valid + "a", invalid}, // too long
		{"sha256-!" + valid[1:], invalid},  // invalid hex
	}

	for _, tt := range cases {
		t.Run("", func(t *testing.T) {
			// sanity check
			if len(tt.want) > HashSize*2 {
				panic("invalid test")
			}

			got := ParseID(tt.in)

			wantValid := tt.want != invalid
			if wantValid {
				if !got.Valid() {
					t.Errorf("ParseID(%q).Valid() = false; want true", tt.in)
				}
				if got.String() != "sha256-"+tt.want {
					t.Errorf("ParseID(%q).String() = %q; want %q", tt.in, got.String(), "sha256-"+tt.want)
				}
			} else {
				if got.Valid() {
					t.Errorf("ParseID(%q).Valid() = true; want false", tt.in)
				}
				if got.String() != "" {
					t.Errorf("ParseID(%q).String() = %q; want %q", tt.in, got.String(), "")
				}
			}
		})
	}
}
