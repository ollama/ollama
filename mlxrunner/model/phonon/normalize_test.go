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
		{"fish &. 12%. #.", "fish &. 12%."},
		{"hello... goodbye", "hello. goodbye"},
		{"'Hello!'", "'Hello!"},
		{"a ‐ b ‑ c ‒ d ― e – f — g", "a - b - c - d - e f g"},
		{"👋 \x00", ""},
		{"He said, \"go home!\"", "He said, \"go home!"},
		{"It's the boys'", "It's the boys'"},
		{"café résumé 日本語", "café résumé 日本語"},
		// Letters are folded by the tokenizer, which knows its pieces.
		{"bǎohù in İstanbul", "bǎohù in İstanbul"},

		// Symbols that are read aloud.
		{"I ❤ you", "I love you"},
		{"I ❤️ NY, dial *67, it's 4.5%.", "I love NY, dial star 67, it's 4.5%."},
		{"Rated ★★★★☆, or ★★★½.", "Rated 4 out of 5 stars, or 3 point 5 stars."},
		{"★ New ★ Rated ★★★½☆, or ★½.", "New Rated 3 point 5 out of 5 stars, or 1 point 5 stars."},
		{"From 2020–2021, pages 10–20, $10–$20; a – b.", "From 2020 to 2021, pages 10 to 20, 10 dollars to 20 dollars, a b."},
		{"It was −5°C, then −12.", "It was minus 5 degrees Celsius, then minus 12."},
		{"Thanks for reading! ❤️ Made with ❤️ in Paris ❤❤", "Thanks for reading! Made with love in Paris"},
		{"Made with ❤️. Thanks! ❤️", "Made with love. Thanks!"},
		{"x < 5, y >= 2, a -> b, <br>done", "x < 5, y greater than or equal to 2, a b, done"},
		{"> Quoted line\n> more", "Quoted line more"},
		{"2*3 = 6, 2^10 and x^-1, a|b, ~/git \\o/", "2 times 3 equals 6, 2 to the power of 10 and x to the power of minus 1, a b, git o"},
		{"Studies show¹ it works.² E = mc², 5 km² and ¹³C.", "Studies show it works. E equals mc squared, 5 km squared and 13C."},
		{"Pay $20/month or $15/hr, $10/user/month.", "Pay 20 dollars per month or 15 dollars per hour, 10 dollars per user per month."},
		{"From 5,000-10,000 steps, $50,000-$70,000, $5-10 or 5-10€.", "From 5 thousand to 10 thousand steps, 50 thousand dollars to 70 thousand dollars, 5 dollars to 10 dollars or 5 euros to 10 euros."},
		{"A $5 Billion deal.", "A 5 billion dollars deal."},
		{"We met in Dec. Then we left on Mon. Oct. 5, in Dec. 2024.", "We met in December. Then we left on Monday October 5, in December 2024."},
		{"Mt. Everest, Sen. Smith vs. Jones, vol. II and the dept. They", "Mount Everest, Senator Smith versus Jones, volume II and the department. They"},
		{"Wait ~10 minutes, ±2.", "Wait about 10 minutes, plus or minus 2."},
		{"2 + 2 = 4", "2 plus 2 equals 4"},
		{"x² + y³ = 10⁻⁶", "x squared plus y cubed equals 10 to the power of minus 6"},
		{"C₈H₁₈ and ¹³C", "C8H18 and 13C"},
		{"ΔH, α and ω", "delta H, alpha and omega"},
		{"√2, ∞, 5‰ and ∑", "square root of 2, infinity, 5 per mille and sigma"},
		{"1920×1080, 6÷2, x≤3, y≥4, a≠b, π≈3.14", "1920 times 10 80, 6 divided by 2, x less than or equal to 3, y greater than or equal to 4, a is not equal to b, pi approximately 3 point 14"},
		{"Water boils at 100°C, or 212 °F.", "Water boils at 100 degrees Celsius, or 212 degrees Fahrenheit."},
		{"It was -3.5°C, then 1°C.", "It was minus 3 point 5 degrees Celsius, then 1 degree Celsius."},
		{"A 45°Ceiling", "A 45 degrees Ceiling"},
		{"Turn 90°, then 10n°.", "Turn 90 degrees, then 10n degrees."},
		{"Call n.° 5 now.", "Call number 5 now."},
		{"Fish &, chips &.", "Fish &, chips &."},
		{"Order #12345 in C#, #tag, ## Notes", "Order number 12 thousand 345 in C sharp, tag, Notes"},
		{"Use snake_case, not _emphasis_ or __init__.", "Use snake underscore case, not emphasis or init."},

		// The word rules.
		{"user@host @home", "user at host at home"},
		{"Write to laurent.mazare@gmail.com!", "Write to laurent dot mazare at gmail dot com!"},
		{"Mail (laurent+tag@gmail.com)", "Mail, laurent plus tag at gmail dot com,"},
		{"a@.", "a at."},
		{"C++.", "C plus plus."},
		{"1500+20", "15 hundred plus 20"},
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
		{"The scores were 21 17 14, call 555 123 4567.", "The scores were 21 17 14, call five five five, one two three, four five six seven."},
		{"Dial +33 1 23 45 67 89.", "Dial plus three three, one, two three, four five, six seven, eight nine."},
		{"It fell to -12.", "It fell to minus 12."},
		{"Your order number is 48213.", "Your order number is 48 thousand 213."},
		{"In 1492 and 1805, the RTX 4090 had 4096 tokens.", "In 14 92 and 18 oh 5, the RTX 4 thousand 90 had 4 thousand 96 tokens."},
		{"Wait 5-10 minutes, 0-60 in 3-4 s, a 10-5 win.", "Wait 5 to 10 minutes, 0 to 60 in 3 to 4 s, a 10-5 win."},
		{"Visit www.kyutai.fr today.", "Visit W-W-W dot kyutai dot F-R today."},
		{"and/or, to/from, reality/the deity.", "and or, to from, reality the deity."},
		{"40 km/h, 5 mg/kg, c/o, 3/4, 05/04/2025", "40 km per hour, 5 mg per kg, care of, 3 quarters, 05/04/2025"},
		{"Drive 60 km/h at 9.8 m/s², N/A.", "Drive 60 km per hour at 9 point 8 m per second squared, N A."},
		{"Add 1 1/2 cups, 3/4 tsp and 2/3 cup, 24/7.", "Add 1 and 1 half cups, 3 quarters tsp and 2 thirds cup, 24/7."},
		{"Use Ctrl+C to copy, Cmd+V to paste.", "Use Control plus C to copy, Command plus V to paste."},
		// The model reads times and dates as written; upstream's opt-in "10-30"
		// reads as a range.
		{"It's 1:06pm on 20/12/2015, 16:9.", "It's 1:06pm on 20/12/2015, 16:9."},
		{"Sooo good, Nooooo!", "So good, No!"},
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
		{"3-5", "3 to 5"},
		{"1492", "14 92"},
		{"1805", "18 oh 5"},
		{"1500.", "15 hundred."},
		{"4096", "4 thousand 96"},
		{"3/4", "3 quarters"},
		{"1/2,", "1 half,"},
		{"sooo", "so"},
		{"Nooooo,", "No,"},
		{"Hiii", "Hi"},
		{"Whyyy", "Why"},
		{"Ewww,", "Ew,"},
		{"Ahhh!", "Ah!"},
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
		{"$20/month.", "20 dollars per month."},
		{"$1.50/lb", "1 dollar and 50 cents per lb"},
		{"5,000-10,000,", "5 thousand to 10 thousand,"},
		{"$50k-$70k", "50 thousand dollars to 70 thousand dollars"},
		{"$50-70k", "50 to 70 thousand dollars"},
		{"978-3-16-148410-0", "978 3 16 1 4 8 4 1 0 0"},
		{"555_123_4567", "555 underscore 123 underscore 4 5 6 7"},
		{"555-123-4567?!", "555 123 4 5 6 7?!"},
		{"first_last@x.co.uk.", "first underscore last at x dot co dot U-K."},
		{"a..b@x.io", "a dot dot b at x dot I-O"},
		{"https://www.example.com/path/to-page/", "H-T-T-P-S colon slash slash W-W-W dot example dot com slash path slash to dash page slash"},
		{"my-site.co.uk/how-to-go", "my dash site dot C-O dot U-K slash how dash to dash go"},
		{"https://example.com/docs/index.html.", "H-T-T-P-S colon slash slash example dot com slash docs slash index dot html."},
		{"abc123.io/v2", "abc 1 2 3 dot I-O slash v 2"},
		{"https://kyutai.Fr", "H-T-T-P-S colon slash slash kyutai dot F-R"},
		{"EXAMPLE.COM", "EXAMPLE dot COM"},
		{"README.md", "README dot M-D"},
		{"Node.js", "Node dot J-S"},
		{"example.html", "example dot html"},
		{"Dec.,", "December,"},
		{"w/o", "without"},
		{"c/o", "care of"},
	} {
		if got, ok := rewriteWord(tc.word); !ok || got != tc.want {
			t.Errorf("rewrite(%q) = %q, %v; want %q", tc.word, got, ok, tc.want)
		}
	}
	for _, word := range []string{
		"10-5", "12:30", "3:45PM", "10/23/2025", "50/50", "3.5%", "1234567890123", "$5€", "$007", "2024-05-12",
		"12-05-24", "+1-555-123-4567", "user@host", "fé@bar.com", "it.Now",
		"e.g.", "v1.2", "Dr.", "dec.", "Zzz", "WWW", "Sun.", "Jan.", "good", "Brrr", "Mmm",
	} {
		if got, ok := rewriteWord(word); ok {
			t.Errorf("rewrite(%q) = %q; want no rule to claim it", word, got)
		}
	}
}
