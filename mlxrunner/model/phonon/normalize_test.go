package phonon

import "testing"

func TestNormalizeText(t *testing.T) {
	// Expected strings come from xn-ptts Normalize::for_lang(Lang::En).
	for _, tc := range []struct{ input, want string }{
		{"You’re sweet! Is there anything you’d like to talk about?", "You're sweet! Is there anything you'd like to talk about?"},
		{"Youʼre sweet!", "You're sweet!"},
		{"Of course—I would love to help! What’s on your mind?", "Of course I would love to help! What's on your mind?"},
		{"Well… I am not sure… maybe we could try again.", "Well. I am not sure. maybe we could try again."},
		{"🎉 Congratulations! You did an amazing job! 😊", "Congratulations! You did an amazing job!"},
		{"Hello👋there! It is nice to meet you.", "Hello there! It is nice to meet you."},
		{"I can help (if you would like); just let me know.", "I can help, if you would like, just let me know."},
		{"Here is the plan: • Take a walk → Get some rest.", "Here is the plan, Take a walk Get some rest."},
		{"That is [really] **wonderful** news!", "That is really wonderful news!"},
		{"She said, “It is time to go.”", "She said, \"It is time to go."},
		{"Hello\x00 there. How are you?", "Hello there. How are you?"},
		{"sun ☀️ dev 👩‍💻 done", "sun dev done"},
		{"time: 10:30", "time, 10:30"},
		{"foo ; bar  :  baz", "foo, bar, baz"},
		{"a;b:c", "a, b:c"},
		{"fish &. 12%. #.", "fish &. 12%. #."},
		{"hello... goodbye", "hello. goodbye"},
		{"'Hello!'", "'Hello!"},
		{"a ‐ b ‑ c ‒ d ― e – f — g", "a - b - c - d - e f g"},
		{"👋 \x00", ""},
	} {
		got := normalizeText(tc.input)
		if got != tc.want {
			t.Errorf("normalize(%q) = %q; want %q", tc.input, got, tc.want)
		}
		if again := normalizeText(got); again != got {
			t.Errorf("normalize is not idempotent for %q: %q", got, again)
		}
	}
}

func TestNormalizeTextPreservesLanguageSpecificMeaning(t *testing.T) {
	// Until a language-specific pass exists, do not silently erase these.
	for _, text := range []string{"I ❤ you", "Rated ★★★☆☆", "*5", "2 + 2 = 4", "21°C", "user@host", "It costs $5.", "12.50", "café", "日本語"} {
		if got := normalizeText(text); got != text {
			t.Errorf("normalize(%q) = %q", text, got)
		}
	}
}

func TestUnigramDecodeSpecialTokens(t *testing.T) {
	u, err := newUnigram([]byte(`{
		"added_tokens":[{"id":1,"content":"\u0000<s>","special":true}],
		"model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",-10],["\u0000<s>",0],["▁hello",-1],["▁world",-1]]}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := u.decode([]int32{1, 2, 1, 3, 1}); got != "hello world" {
		t.Fatalf("decode with skipped special tokens = %q", got)
	}
}
