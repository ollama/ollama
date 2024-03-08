package blob

import "testing"

// test refs
const (
	refTooLong = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)

func TestParseRef(t *testing.T) {
	cases := []struct {
		in   string
		want Ref
	}{
		{"mistral:latest", Ref{"mistral", "latest", ""}},
		{"mistral", Ref{"mistral", "latest", ""}},
		{"mistral:30B", Ref{"mistral", "30B", ""}},
		{"mistral:7b", Ref{"mistral", "7b", ""}},
		{"mistral:7b+Q4_0", Ref{"mistral", "7b", "Q4_0"}},
		{"mistral+KQED", Ref{"mistral", "latest", "KQED"}},
		{"mistral.x-3:7b+Q4_0", Ref{"mistral.x-3", "7b", "Q4_0"}},

		// lowecase build
		{"mistral:7b+q4_0", Ref{"mistral", "7b", "Q4_0"}},

		// Invalid
		{"mistral:7b+Q4_0:latest", Ref{"", "", ""}},
		{"mi tral", Ref{"", "", ""}},
		{"llama2:+", Ref{"", "", ""}},

		// too long
		{refTooLong, Ref{"", "", ""}},
	}
	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			got := ParseRef(tt.in)
			if got != tt.want {
				t.Errorf("ParseRef(%q) = %q; want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestRefFull(t *testing.T) {
	cases := []struct {
		in        string
		wantShort string
		wantFull  string
	}{
		{"", "", ""},
		{"mistral:7b+x", "mistral:7b", "mistral:7b+X"},
		{"mistral:7b+Q4_0", "mistral:7b", "mistral:7b+Q4_0"},
		{"mistral:latest", "mistral:latest", "mistral:latest+!(MISSING BUILD)"},
		{"mistral", "mistral:latest", "mistral:latest+!(MISSING BUILD)"},
		{"mistral:30b", "mistral:30b", "mistral:30b+!(MISSING BUILD)"},
	}

	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			ref := ParseRef(tt.in)
			if g := ref.Short(); g != tt.wantShort {
				t.Errorf("Short(%q) = %q; want %q", tt.in, g, tt.wantShort)
			}
			if g := ref.Full(); g != tt.wantFull {
				t.Errorf("Full(%q) = %q; want %q", tt.in, g, tt.wantFull)
			}
		})
	}
}
