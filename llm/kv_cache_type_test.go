package llm

import "testing"

func TestResolveKVCacheType(t *testing.T) {
	cases := []struct {
		name, env, opt, want string
	}{
		{"both empty means llama-server default f16", "", "", ""},
		{"env only", "q8_0", "", "q8_0"},
		{"env case folded", "Q8_0", "", "q8_0"},
		{"option overrides env", "q8_0", "f16", "f16"},
		{"option sets when env empty", "", "q8_0", "q8_0"},
		{"option case and whitespace folded", "q8_0", " F16 ", "f16"},
		{"invalid option falls back to env", "q8_0", "q9_9", "q8_0"},
		{"invalid option with empty env", "", "banana", ""},
		{"bf16 allowed", "", "bf16", "bf16"},
		{"iq4_nl allowed", "f16", "iq4_nl", "iq4_nl"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := resolveKVCacheType(c.env, c.opt); got != c.want {
				t.Fatalf("resolveKVCacheType(%q, %q) = %q, want %q", c.env, c.opt, got, c.want)
			}
		})
	}
}
