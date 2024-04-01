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
		{"mistral:latest", Ref{"registry.ollama.ai", "mistral", "latest", ""}},
		{"mistral", Ref{"registry.ollama.ai", "mistral", "latest", ""}},
		{"mistral:30B", Ref{"registry.ollama.ai", "mistral", "30B", ""}},
		{"mistral:7b", Ref{"registry.ollama.ai", "mistral", "7b", ""}},
		{"mistral:7b+Q4_0", Ref{"registry.ollama.ai", "mistral", "7b", "Q4_0"}},
		{"mistral+KQED", Ref{"registry.ollama.ai", "mistral", "latest", "KQED"}},
		{"mistral.x-3:7b+Q4_0", Ref{"registry.ollama.ai", "mistral.x-3", "7b", "Q4_0"}},

		// lowecase build
		{"mistral:7b+q4_0", Ref{"registry.ollama.ai", "mistral", "7b", "Q4_0"}},

		// Invalid
		{"mistral:7b+Q4_0:latest", Ref{"", "", "", ""}},
		{"mi tral", Ref{"", "", "", ""}},
		{"llama2:+", Ref{"", "", "", ""}},

		// too long
		{refTooLong, Ref{"", "", "", ""}},
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
		{"mistral:7b+x", "registry.ollama.ai/mistral:7b", "registry.ollama.ai/mistral:7b+X"},
		{"mistral:7b+Q4_0", "registry.ollama.ai/mistral:7b", "registry.ollama.ai/mistral:7b+Q4_0"},
		{"mistral:latest", "registry.ollama.ai/mistral:latest", "registry.ollama.ai/mistral:latest+!(MISSING BUILD)"},
		{"mistral", "registry.ollama.ai/mistral:latest", "registry.ollama.ai/mistral:latest+!(MISSING BUILD)"},
		{"mistral:30b", "registry.ollama.ai/mistral:30b", "registry.ollama.ai/mistral:30b+!(MISSING BUILD)"},
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
