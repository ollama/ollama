package blob

import "testing"

// test refs
const (
	refTooLong = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)

func TestRefParts(t *testing.T) {
	const wantNumParts = 5
	var ref Ref
	if len(ref.Parts()) != wantNumParts {
		t.Errorf("Parts() = %d; want %d", len(ref.Parts()), wantNumParts)
	}
}

func TestParseRef(t *testing.T) {
	cases := []struct {
		in   string
		want Ref
	}{
		{"mistral:latest", Ref{
			name: "mistral",
			tag:  "latest",
		}},
		{"mistral", Ref{
			name: "mistral",
		}},
		{"mistral:30B", Ref{
			name: "mistral",
			tag:  "30B",
		}},
		{"mistral:7b", Ref{
			name: "mistral",
			tag:  "7b",
		}},
		{"mistral:7b+Q4_0", Ref{
			name:  "mistral",
			tag:   "7b",
			build: "Q4_0",
		}},
		{"mistral+KQED", Ref{
			name:  "mistral",
			build: "KQED",
		}},
		{"mistral.x-3:7b+Q4_0", Ref{
			name:  "mistral.x-3",
			tag:   "7b",
			build: "Q4_0",
		}},

		// lowecase build
		{"mistral:7b+q4_0", Ref{
			name:  "mistral",
			tag:   "7b",
			build: "Q4_0",
		}},
		{"llama2:+", Ref{name: "llama2"}},

		// Invalid
		{"mistral:7b+Q4_0:latest", Ref{}},
		{"mi tral", Ref{}},

		// too long
		{refTooLong, Ref{}},
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
	const empty = "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/!(MISSING NAME):!(MISSING TAG)+!(MISSING BUILD)"

	cases := []struct {
		in       string
		wantFull string
	}{
		{"", empty},
		{"example.com/mistral:7b+x", "!(MISSING DOMAIN)/example.com/mistral:7b+X"},
		{"example.com/mistral:7b+Q4_0", "!(MISSING DOMAIN)/example.com/mistral:7b+Q4_0"},
		{"example.com/x/mistral:latest", "example.com/x/mistral:latest+!(MISSING BUILD)"},
		{"example.com/x/mistral:latest+Q4_0", "example.com/x/mistral:latest+Q4_0"},

		{"mistral:7b+x", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:7b+X"},
		{"mistral:7b+Q4_0", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:7b+Q4_0"},
		{"mistral:latest", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:latest+!(MISSING BUILD)"},
		{"mistral", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:!(MISSING TAG)+!(MISSING BUILD)"},
		{"mistral:30b", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:30b+!(MISSING BUILD)"},
	}

	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			ref := ParseRef(tt.in)
			t.Logf("ParseRef(%q) = %#v", tt.in, ref)
			if g := ref.Full(); g != tt.wantFull {
				t.Errorf("Full(%q) = %q; want %q", tt.in, g, tt.wantFull)
			}
		})
	}
}
