package phonon

import "testing"

func TestNormalizeText(t *testing.T) {
	// Expected strings are xn-ptts 0.4.1 normalize_text(Lang::En, Rules::DEFAULT),
	// with the readings rewrite.go corrects.
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
		{"He said, \"go home!\"", "He said, \"go home!"},
		{"It's the boys'", "It's the boys'"},
		{"café résumé 日本語", "café résumé 日本語"},
		{"bǎohù zìjǐ in İstanbul, città", "baohù ziji in Istanbul, città"},

		// Symbols that are read aloud.
		{"I ❤ you", "I love you"},
		{"I ❤️ NY, dial *67, it's 4.5%.", "I love NY, dial star 67, it's 4.5%."},
		{"Rated ★★★★☆, or ★★★½.", "Rated 4 out of 5 stars, or 3 point 5 stars."},
		{"Wait ~10 minutes, ±2.", "Wait about 10 minutes, plus or minus 2."},
		{"2 + 2 = 4", "2 plus 2 equals 4"},
		{"x² + y³ = 10⁻⁶", "x squared plus y cubed equals 10 to the power of minus 6"},
		{"C₈H₁₈ and ¹³C", "C8H18 and 13C"},
		{"ΔH, α and ω", "delta H, alpha and omega"},
		{"√2, ∞, 5‰ and ∑", "square root of 2, infinity, 5 per mille and sigma"},
		{"1920×1080, 6÷2, x≤3, y≥4, a≠b, π≈3.14", "1920 times 1 thousand 80, 6 divided by 2, x less than or equal to 3, y greater than or equal to 4, a is not equal to b, pi approximately 3 point 14"},
		{"Water boils at 100°C, or 212 °F.", "Water boils at 100 degrees Celsius, or 212 degrees Fahrenheit."},
		{"It was -3.5°C, then 1°C.", "It was minus 3 point 5 degrees Celsius, then 1 degree Celsius."},
		{"A 45°Ceiling", "A 45 degrees Ceiling"},
		{"Turn 90°, then 10n°.", "Turn 90 degrees, then 10n degrees."},
		{"Call n.° 5 now.", "Call number 5 now."},
		{"Fish &, chips &.", "Fish &, chips &."},

		// The word rules.
		{"user@host @home", "user at host at home"},
		{"Write to laurent.mazare@gmail.com!", "Write to laurent dot mazare at gmail dot com!"},
		{"Mail (laurent+tag@gmail.com)", "Mail, laurent plus tag at gmail dot com,"},
		{"a@.", "a at."},
		{"C++.", "C plus plus."},
		{"1500+20", "1 thousand 500 plus 20"},
		{"It costs $1500 today.", "It costs 1 thousand 500 dollars today."},
		{"It costs 5$.", "It costs 5 dollars."},
		{"It costs -$5.", "It costs minus 5 dollars."},
		{"In $.", "In."},
		{"The fee is 2.9% plus 30¢.", "The fee is 2.9% plus 30 cents."},
		{"It sold for £3.2m.", "It sold for 3 point 2 million pounds."},
		{"We raised $2 million, then €3bn.", "We raised 2 million dollars, then 3 billion euros."},
		{"From 10 $20 bills.", "From 10 20 dollars bills."},
		{"Call 555-123-4567 before 2024-05-12, it costs 12345678.", "Call 555 123 4 5 6 7 before 2024-05-12, it costs 12 million 345 thousand 678."},
		{"Ring +44 20 7946 0958.", "Ring plus four four, two zero, seven nine four six, zero nine five eight."},
		{"Text 07700 900123 now.", "Text zero seven seven zero zero, nine zero zero one two three now."},
		{"In 2019 2020 2021", "In 2019 2020 2021"},
		{"It fell to -12.", "It fell to minus 12."},
		{"Your order number is 48213.", "Your order number is 48 thousand 213."},
		{"Visit www.kyutai.fr today.", "Visit W-W-W dot kyutai dot F-R today."},
		{"and/or, to/from, reality/the deity.", "and or, to from, reality the deity."},
		{"40 km/h, 5 mg/kg, c/o, 3/4, 05/04/2025", "40 km/h, 5 mg/kg, c/o, 3/4, 05/04/2025"},
		// Times and dates are opt-in upstream, and stay as written.
		{"It's 1:06pm on 20/12/2015.", "It's 1:06pm on 20/12/2015."},
	} {
		if got := normalizeText(tc.input); got != tc.want {
			t.Errorf("normalize(%q) = %q; want %q", tc.input, got, tc.want)
		}
	}
}

func TestRewriteWord(t *testing.T) {
	// Expected strings are the xn-ptts 0.4.1 rule tests, with the readings
	// rewrite.go corrects.
	for _, tc := range []struct{ word, want string }{
		{"1,234.56...", "1 thousand 234 point 56..."},
		{"2,500,000,", "2 million 500 thousand,"},
		{"1002003004", "1 billion 2 million 3 thousand 4"},
		{"-12.00", "minus 12 point 00"},
		{"2024", "2024"},
		{"1,999", "1 thousand 999"},
		{"007", "zero zero seven"},
		{"7,000-ton", "7 thousand-ton"},
		{"20-year-old", "20-year-old"},
		{"20-25", "20 to 25"},
		{"2024-25", "2024 to 25"},
		{"$1,234.56", "1 thousand 234 dollars and 56 cents"},
		{"$0.01", "1 cent"},
		{"$1.00", "1 dollar"},
		{"$1", "1 dollar"},
		{"$2.01", "2 dollars and 1 cent"},
		{"£1.50", "1 pound and 50 pence"},
		{"£0.01", "1 penny"},
		{"€0.5", "0 point 5 euros"},
		{"1¢", "1 cent"},
		{"1.50¢", "1 point 50 cents"},
		{"1234$", "1 thousand 234 dollars"},
		{"-$500.", "minus 500 dollars."},
		{"978-3-16-148410-0", "978 3 16 1 4 8 4 1 0 0"},
		{"555_123_4567", "555 underscore 123 underscore 4 5 6 7"},
		{"555-123-4567?!", "555 123 4 5 6 7?!"},
		{"first_last@x.co.uk.", "first underscore last at x dot co dot U-K."},
		{"a..b@x.io", "a dot dot b at x dot I-O"},
		{"https://www.example.com/path/to-page/", "H-T-T-P-S colon slash slash W-W-W dot example dot com slash path slash T-O dash page slash"},
		{"abc123.io/v2", "abc 1 2 3 dot I-O slash v 2"},
		{"https://kyutai.Fr", "H-T-T-P-S colon slash slash kyutai dot F-R"},
		{"EXAMPLE.COM", "EXAMPLE dot COM"},
		{"README.md", "README dot M-D"},
		{"Node.js", "Node dot J-S"},
		{"example.html", "example dot html"},
		{"Dec.,", "December,"},
		{"w/o", "without"},
	} {
		if got, ok := rewriteWord(tc.word); !ok || got != tc.want {
			t.Errorf("rewrite(%q) = %q, %v; want %q", tc.word, got, ok, tc.want)
		}
	}
	for _, word := range []string{
		"10-5", "3-5", "12:30", "3.5%", "1234567890123", "$5€", "$007", "2024-05-12",
		"12-05-24", "+1-555-123-4567", "user@host", "fé@bar.com", "it.Now",
		"e.g.", "v1.2", "Dr.", "dec.", "Zzz", "WWW",
	} {
		if got, ok := rewriteWord(word); ok {
			t.Errorf("rewrite(%q) = %q; want no rule to claim it", word, got)
		}
	}
}
