package parsers

import "testing"

func TestOlmo3UnescapeUnicodeEscapes(t *testing.T) {
	cases := []struct{ in, want string }{
		{`\u003c`, "<"},
		{`\u00e9`, "é"},
		{`\u65e5\u672c`, "日本"},
		{`\ud83d\ude00`, "😀"},
		{`\u12`, `\u12`},
		{`\uZZZZ`, `\uZZZZ`},
		{`\\u003c`, `\u003c`},
		{`a\u000ab`, "a\nb"},
		// paolo-perrone's review case on #19878: a decoded \u escape must
		// not be able to inject the old placeholder byte or re-open an
		// escape after the original pass restored it.
		{`a\u0000b`, "a\x00b"},
		{`\u005cu003c`, `\u003c`},
		{`a\u005cb`, "a\\b"},
		{`\\`, `\`},
		{`a\u0041`, "aA"},
	}
	for _, c := range cases {
		if got := unescapeString(c.in); got != c.want {
			t.Errorf("unescapeString(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestOlmo3ParserDecodesUnicodeEscapesInArguments(t *testing.T) {
	p := &Olmo3Parser{}
	p.Init(nil, nil, nil)

	_, _, calls, err := p.Add(`<function_calls>write_file(content="\u003c")</function_calls>`, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	got, ok := calls[0].Function.Arguments.Get("content")
	if !ok {
		t.Fatal("content argument missing")
	}
	if got != "<" {
		t.Errorf("content = %q, want %q", got, "<")
	}
}
