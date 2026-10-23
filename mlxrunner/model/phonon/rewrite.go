package phonon

import (
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// The word rules of xn-ptts preprocess::rewrite for English: everything but
// phone numbers, times and dates, with the readings fixed where upstream gets
// them wrong.

// rewriteText runs the rules over the words of text. A few readings claim
// several words; otherwise the first rule that claims a word wins, and the
// symbols of a word no rule claims are spelled out or dropped.
func rewriteText(text string) string {
	words := strings.Split(text, " ")
	out := make([]string, 0, len(words))
	for i := 0; i < len(words); {
		if s, n := rewriteSpan(words[i:]); n > 0 {
			out = append(out, s)
			i += n
			continue
		}
		s := words[i]
		if r, ok := rewriteWord(s); ok {
			s = r
		} else {
			s = spellSymbols(s)
		}
		if s != "" {
			out = append(out, s)
		}
		i++
	}
	return strings.Join(out, " ")
}

// rewriteSpan reads the readings that need their neighbours: "$2 million", "1
// 1/2", an abbreviation that ends a sentence, and runs of digit groups ("+44 20
// 7946 0958"). It returns how many words it used, or 0.
func rewriteSpan(words []string) (string, int) {
	if s, ok := scaledAmount(words); ok {
		return s, 2
	}
	if s, ok := mixedNumber(words); ok {
		return s, 2
	}
	if s, ok := sentenceEnd(words); ok {
		return s, 1
	}
	if s, n := digitRun(words); n > 0 {
		return s, n
	}
	return "", 0
}

var rules = []func(string) (string, bool){numbers, currency, ranges, fraction, dashedDigits, emails, urls, abbreviations, elongations}

func rewriteWord(word string) (string, bool) {
	for _, rule := range rules {
		if s, ok := rule(word); ok {
			return s, true
		}
	}
	return "", false
}

// The sentence punctuation a rule carries through at the end of a word.
const punctuation = "!?.:,;…"

func splitSuffix(word string) (string, string) {
	body := strings.TrimRight(word, punctuation)
	return body, word[len(body):]
}

// isSuffix reports whether rest is only trailing punctuation, which a rule
// carries through; anything else means the rule did not read the whole word.
func isSuffix(rest string) bool {
	return strings.Trim(rest, punctuation) == ""
}

func allDigits(s string) bool {
	return s != "" && strings.Trim(s, "0123456789") == ""
}

var digitNames = strings.Fields("zero one two three four five six seven eight nine")

// spell reads s digit by digit, dropping dashes: "007" is "zero zero seven".
func spell(s string) string {
	var words []string
	for _, c := range s {
		switch {
		case c == '-':
		case isASCIIDigit(c):
			words = append(words, digitNames[c-'0'])
		default:
			words = append(words, string(c))
		}
	}
	return strings.Join(words, " ")
}

// spellShort spells out a two-letter label of an address as an abbreviation
// reads, "fr" as "F-R", and "www" as "W-W-W". Anything else is read as a word.
func spellShort(word string) string {
	switch {
	case word == "www" || word == "WWW":
		return "W-W-W"
	case len(word) == 2 && word[0] < utf8.RuneSelf && word[1] < utf8.RuneSelf:
		return strings.ToUpper(word[:1]) + "-" + strings.ToUpper(word[1:])
	}
	return word
}

// numbers reads large numbers with their scale words: "1,234.56" is "1
// thousand 234 point 56". Values under a thousand stay digits, as do the years
// 1900 to 2099, and earlier years are read as years are: "1492" is "14 92". A
// leading zero marks a code read digit by digit, and "20-25" is a range.
func numbers(word string) (string, bool) {
	body, end := splitSuffix(word)
	if len(body) >= 2 && body[0] == '0' && allDigits(body) {
		return spell(body) + end, true
	}
	if len(body) == 4 && allDigits(body) && body < "1900" {
		return yearStyle(body) + end, true
	}
	if from, to, ok := strings.Cut(body, "-"); ok && isRange(from, to) {
		return from + " to " + to + end, true
	}
	if s, suffix, ok := numberWords(word); ok {
		return s + suffix, true
	}
	// "7,000-ton", "20-year-old".
	i := strings.IndexByte(word, '-')
	if i < 0 {
		return "", false
	}
	head, tail := word[:i], word[i:]
	if c, _ := utf8.DecodeRuneInString(tail[1:]); !isAlphabetic(c) {
		return "", false
	}
	if s, suffix, ok := numberWords(head); ok && suffix == "" {
		return s + tail, true
	}
	return "", false
}

// yearStyle reads a four-digit number as a year is read: "1492" is "14 92",
// "1805" "18 oh 5", "1500" "15 hundred" and "1000" "1 thousand".
func yearStyle(s string) string {
	switch {
	case s[1:] == "000":
		return s[:1] + " thousand"
	case s[2:] == "00":
		return s[:2] + " hundred"
	case s[2] == '0':
		return s[:2] + " oh " + s[3:]
	}
	return s[:2] + " " + s[2:]
}

// isRange reports whether from-to is a range: numbers of up to three digits
// going up, or years. Going down, "10-5" is more likely a score.
func isRange(from, to string) bool {
	small := func(s string) bool { return len(s) <= 3 && allDigits(s) && (s[0] != '0' || s == "0") }
	year := func(s string) bool { return len(s) == 4 && allDigits(s) && (s[0] == '1' || strings.HasPrefix(s, "20")) }
	switch {
	case small(from) && small(to):
		f, _ := strconv.Atoi(from)
		t, _ := strconv.Atoi(to)
		return f < t
	case len(from) == 4 && len(to) == 4:
		return year(from) && year(to) && from < to
	case len(from) == 4 && len(to) == 2:
		return year(from) && allDigits(to) && from[2:] < to
	}
	return false
}

// numberWords reads word as a number, with the trailing punctuation it
// carries through kept apart so that an amount can put its currency between.
func numberWords(word string) (string, string, bool) {
	if len(word) >= 2 && word[0] == '0' && isASCIIDigit(rune(word[1])) {
		return "", "", false
	}
	rest, negative := strings.CutPrefix(word, "-")
	end := strings.IndexFunc(rest, func(c rune) bool { return !isASCIIDigit(c) && c != ',' })
	if end < 0 {
		end = len(rest)
	}
	whole := strings.TrimRight(rest[:end], ",")
	if whole == "" {
		return "", "", false
	}
	rest = rest[len(whole):]
	decimal := ""
	if after, ok := strings.CutPrefix(rest, "."); ok {
		if d, after, ok := runOf(after, isASCIIDigit); ok {
			decimal, rest = d, after
		}
	}
	if !isSuffix(rest) {
		return "", "", false
	}
	value, err := strconv.ParseInt(strings.ReplaceAll(whole, ",", ""), 10, 64)
	if err != nil || value >= 1_000_000_000_000 {
		return "", "", false
	}
	s := strconv.FormatInt(value, 10)
	// A year is never written with thousands commas.
	if value >= 1000 && (value < 1900 || value >= 2100 || strings.Contains(whole, ",")) {
		var parts []string
		for _, scale := range []struct {
			n    int64
			name string
		}{{1_000_000_000, "billion"}, {1_000_000, "million"}, {1000, "thousand"}} {
			if group := value / scale.n % 1000; group > 0 {
				parts = append(parts, strconv.FormatInt(group, 10)+" "+scale.name)
			}
		}
		if value%1000 != 0 {
			parts = append(parts, strconv.FormatInt(value%1000, 10))
		}
		s = strings.Join(parts, " ")
	}
	if decimal != "" {
		s += " point " + decimal
	}
	if negative {
		s = "minus " + s
	}
	return s, rest, true
}

// digitRun reads groups of digits, each a word of its own, digit by digit, as
// phone and card numbers are read: "+44 20 7946 0958". A run needs a leading
// "+" or zero, or three or more groups, not all years, with one of four digits
// or more, so that "21 17 14" stays a list of numbers.
func digitRun(words []string) (string, int) {
	group := func(s string) bool {
		return len(s) >= 1 && len(s) <= 8 && strings.Trim(s, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") == "" && strings.ContainsAny(s, "0123456789")
	}
	plus := strings.HasPrefix(words[0], "+")
	var bodies []string
	end := ""
	for _, word := range words {
		body, suffix := splitSuffix(word)
		if len(bodies) == 0 && plus {
			body = body[1:]
		}
		if !group(body) {
			break
		}
		bodies = append(bodies, body)
		if suffix != "" {
			end = suffix
			break
		}
	}
	years := !slices.ContainsFunc(bodies, func(b string) bool { return len(b) != 4 || b[:2] != "19" && b[:2] != "20" })
	zero := slices.ContainsFunc(bodies, func(b string) bool { return b[0] == '0' })
	long := slices.ContainsFunc(bodies, func(b string) bool { return len(b) >= 4 })
	if !(len(bodies) >= 2 && (plus || zero)) && !(len(bodies) >= 3 && !years && long) {
		return "", 0
	}
	spelled := make([]string, len(bodies))
	for i, b := range bodies {
		spelled[i] = spell(b)
	}
	s := strings.Join(spelled, ", ")
	if plus {
		s = "plus " + s
	}
	return s + end, len(bodies)
}

type money struct{ one, many, subOne, subMany string }

var currencies = map[rune]money{
	'$': {"dollar", "dollars", "cent", "cents"},
	'€': {"euro", "euros", "cent", "cents"},
	'£': {"pound", "pounds", "penny", "pence"},
	'¥': {"yen", "yen", "", ""},
	'¢': {"cent", "cents", "", ""},
}

// perUnits are the short units after a slash that read as words: "$15/hr".
var perUnits = map[string]string{"h": "hour", "hr": "hour", "hrs": "hour", "s": "second", "sec": "second", "min": "minute", "wk": "week", "mo": "month", "yr": "year"}

// currency reads an amount with its currency symbol before or after it:
// "$1,234.56" is "1 thousand 234 dollars and 56 cents", "30¢" is "30 cents",
// one unit is singular: "$1" is "1 dollar", and "$20/month" is "20 dollars per
// month".
func currency(word string) (string, bool) {
	body, suffix := splitSuffix(word)
	if price, per, ok := strings.Cut(body, "/"); ok {
		units := strings.Split(per, "/")
		for i, unit := range units {
			if unit == "" || strings.IndexFunc(unit, func(c rune) bool { return !isAlphabetic(c) }) >= 0 {
				return "", false
			}
			if w := perUnits[unit]; w != "" {
				units[i] = w
			}
		}
		s, ok := currency(price)
		return s + " per " + strings.Join(units, " per ") + suffix, ok
	}
	// "-$500" reads as "$-500" does.
	if rest, ok := strings.CutPrefix(body, "-"); ok {
		if c, n := utf8.DecodeRuneInString(rest); currencies[c].many != "" {
			return currency(string(c) + "-" + rest[n:] + suffix)
		}
	}
	symbol, amount, ok := splitSymbol(body)
	if !ok {
		return "", false
	}
	if number, scale, ok := splitScale(amount); ok {
		s, ok := withScale(number, scale, symbol)
		return s + suffix, ok
	}
	if s, ok := withCents(amount, symbol); ok {
		return s + suffix, true
	}
	s, _, ok := numberWords(amount)
	if !ok {
		return "", false
	}
	name := currencies[symbol].many
	if strings.TrimPrefix(s, "minus ") == "1" {
		name = currencies[symbol].one
	}
	return s + " " + name + suffix, true
}

func splitSymbol(word string) (rune, string, bool) {
	if c, n := utf8.DecodeRuneInString(word); currencies[c].many != "" {
		return c, word[n:], true
	}
	if c, n := utf8.DecodeLastRuneInString(word); currencies[c].many != "" {
		return c, word[:len(word)-n], true
	}
	return 0, "", false
}

// scaleSuffixes maps a scale word or abbreviation to the scale it reads as.
var scaleSuffixes = []struct{ suffix, scale string }{
	{"thousand", "thousand"},
	{"million", "million"},
	{"billion", "billion"},
	{"trillion", "trillion"},
	{"bn", "billion"},
	{"BN", "billion"},
	{"mn", "million"},
	{"tn", "trillion"},
	{"k", "thousand"},
	{"K", "thousand"},
	{"m", "million"},
	{"M", "million"},
	{"b", "billion"},
	{"B", "billion"},
}

// splitScale splits a scale off an amount: "3.2m" is 3.2 million.
func splitScale(amount string) (string, string, bool) {
	for _, s := range scaleSuffixes {
		if number, ok := strings.CutSuffix(amount, s.suffix); ok && number != "" && isASCIIDigit(rune(number[len(number)-1])) {
			return number, s.scale, true
		}
	}
	return "", "", false
}

func withScale(amount, scale string, symbol rune) (string, bool) {
	s, rest, ok := numberWords(amount)
	if !ok || rest != "" {
		return "", false
	}
	return s + " " + scale + " " + currencies[symbol].many, true
}

// scaledAmount reads an amount followed by a scale word: "$2 million".
func scaledAmount(words []string) (string, bool) {
	if len(words) < 2 {
		return "", false
	}
	scale, suffix := splitSuffix(words[1])
	scale = strings.ToLower(scale)
	symbol, amount, ok := splitSymbol(words[0])
	if !ok || !slices.Contains([]string{"thousand", "million", "billion", "trillion"}, scale) {
		return "", false
	}
	s, ok := withScale(amount, scale, symbol)
	return s + suffix, ok
}

// ranges reads a range of amounts, or of numbers with thousands commas: "$5-$10",
// "$5-10", "$50-70k" and "5,000-10,000". Plain numbers are left to numbers.
func ranges(word string) (string, bool) {
	body, suffix := splitSuffix(word)
	from, to, ok := strings.Cut(body, "-")
	if !ok || from == "" || to == "" {
		return "", false
	}
	plain := func(s string) bool {
		_, rest, ok := numberWords(s)
		return ok && rest == ""
	}
	fromSymbol, _, fromAmount := splitSymbol(from)
	toSymbol, _, toAmount := splitSymbol(to)
	switch {
	case !fromAmount && !toAmount:
		if !strings.Contains(body, ",") || !plain(from) || !plain(to) {
			return "", false
		}
		a, _, _ := numberWords(from)
		b, _, _ := numberWords(to)
		return a + " to " + b + suffix, true
	case !toAmount && plain(to):
		// "$5-10" reads as "$5-$10".
		to = string(fromSymbol) + to
	case !fromAmount && plain(from):
		from = string(toSymbol) + from
	case !toAmount:
		number, scale, ok := splitScale(to)
		_, amount, _ := splitSymbol(from)
		if !ok || !plain(number) {
			return "", false
		}
		if plain(amount) {
			// "$50-70k" is "50 to 70 thousand dollars".
			a, _, _ := numberWords(amount)
			b, _, _ := numberWords(number)
			return a + " to " + b + " " + scale + " " + currencies[fromSymbol].many + suffix, true
		}
		to = string(fromSymbol) + to
	case !fromAmount:
		return "", false
	}
	a, ok := currency(from)
	if !ok {
		return "", false
	}
	b, ok := currency(to)
	return a + " to " + b + suffix, ok
}

// withCents reads an amount with exactly two decimals as units and cents:
// "$4.59" is "4 dollars and 59 cents", "£1.50" is "1 pound and 50 pence". A
// currency without a subunit keeps the plain reading.
func withCents(amount string, symbol rune) (string, bool) {
	m := currencies[symbol]
	if m.subMany == "" {
		return "", false
	}
	i := strings.LastIndexByte(amount, '.')
	if i < 0 {
		return "", false
	}
	whole, cents := amount[:i], amount[i+1:]
	grouped := true
	for j, group := range strings.Split(whole, ",") {
		grouped = grouped && allDigits(group) && (j == 0 && len(group) <= 3 || j > 0 && len(group) == 3)
	}
	if !allDigits(whole) && !grouped || len(cents) != 2 || !allDigits(cents) {
		return "", false
	}
	whole, _, ok := numberWords(whole)
	if !ok {
		return "", false
	}
	units := whole + " " + m.many
	if whole == "1" {
		units = whole + " " + m.one
	}
	if cents == "00" {
		return units, true
	}
	// "05" is read "5".
	cents = strings.TrimPrefix(cents, "0")
	sub := m.subMany
	if cents == "1" {
		sub = m.subOne
	}
	if whole == "0" {
		return cents + " " + sub, true
	}
	return units + " and " + cents + " " + sub, true
}

// dashedDigits reads digits with two or more '-' or '_' separators, such as
// phone, card or ISBN numbers: groups of up to three digits as numbers,
// longer ones digit by digit, and underscores spoken. Dates are left alone.
func dashedDigits(word string) (string, bool) {
	body, suffix := splitSuffix(word)
	if strings.Trim(body, "0123456789-_") != "" || strings.Count(body, "-")+strings.Count(body, "_") < 2 {
		return "", false
	}
	type group struct {
		digits     string
		underscore bool
	}
	var groups []group
	start := 0
	for i := range len(body) {
		if body[i] == '-' || body[i] == '_' {
			if i == start {
				return "", false
			}
			groups = append(groups, group{body[start:i], body[i] == '_'})
			start = i + 1
		}
	}
	if start == len(body) {
		return "", false
	}
	groups = append(groups, group{digits: body[start:]})
	if !strings.Contains(body, "_") && len(groups) == 3 {
		a, b, c := groups[0].digits, groups[1].digits, groups[2].digits
		if len(a) == 4 && dayMonth(b, c) || (len(c) == 4 || len(c) == 2) && dayMonth(a, b) {
			return "", false
		}
	}
	var words []string
	for _, g := range groups {
		if len(g.digits) >= 4 {
			for i := range len(g.digits) {
				words = append(words, g.digits[i:i+1])
			}
		} else {
			words = append(words, g.digits)
		}
		if g.underscore {
			words = append(words, "underscore")
		}
	}
	return strings.Join(words, " ") + suffix, true
}

var fractionNames = strings.Fields("half third quarter fifth sixth seventh eighth ninth tenth")

// fraction reads a fraction up to tenths in words: "3/4" is "3 quarters".
// Other numbers split by slashes, as in dates and "50/50", the model reads
// itself, while "10/23" written as "10-23" would read as a range.
func fraction(word string) (string, bool) {
	body, suffix := splitSuffix(word)
	a, b, ok := strings.Cut(body, "/")
	if !ok || !allDigits(a) || !allDigits(b) {
		return "", false
	}
	n, _ := strconv.Atoi(a)
	d, _ := strconv.Atoi(b)
	if n < 1 || n >= d || d > 10 {
		return "", false
	}
	name := fractionNames[d-2]
	switch {
	case n == 1:
	case name == "half":
		name = "halves"
	default:
		name += "s"
	}
	return a + " " + name + suffix, true
}

// mixedNumber reads a whole number and a fraction: "1 1/2" is "1 and 1 half".
func mixedNumber(words []string) (string, bool) {
	if len(words) < 2 || !allDigits(words[0]) {
		return "", false
	}
	s, ok := fraction(words[1])
	return words[0] + " and " + s, ok
}

// dayMonth reports whether x and y are a day and month, in either order.
func dayMonth(x, y string) bool {
	if len(x) > 2 || len(y) > 2 {
		return false
	}
	a, _ := strconv.Atoi(x)
	b, _ := strconv.Atoi(y)
	return a >= 1 && a <= 31 && b >= 1 && b <= 12 || a >= 1 && a <= 12 && b >= 1 && b <= 31
}

// emails reads an email address with its separators spoken and a two-letter
// top-level domain spelled out: "a.b@gra-dium.fr" is "a dot b at gra dash
// dium dot F-R".
func emails(word string) (string, bool) {
	user, rest, ok := strings.Cut(word, "@")
	if !ok || user == "" || strings.IndexFunc(user, func(c rune) bool { return !isASCIIAlnum(c) && !strings.ContainsRune("-_.+", c) }) >= 0 {
		return "", false
	}
	domains, rest := labels(rest, func(c rune) bool { return isASCIIAlnum(c) || c == '-' })
	if len(domains) < 2 || !isSuffix(rest) {
		return "", false
	}
	words := spokenParts(user, nil, map[rune]string{'-': "dash", '.': "dot", '+': "plus", '_': "underscore"})
	words = append(words, "at")
	for i, label := range domains[:len(domains)-1] {
		if i > 0 {
			words = append(words, "dot")
		}
		words = spokenParts(label, words, map[rune]string{'-': "dash"})
	}
	return strings.Join(words, " ") + " dot " + spellShort(domains[len(domains)-1]) + rest, true
}

// labels splits s into the '.'-separated runs of runes keep accepts, and
// returns what follows them.
func labels(s string, keep func(rune) bool) ([]string, string) {
	var out []string
	for {
		label, rest, ok := runOf(s, keep)
		if !ok {
			return out, s
		}
		out = append(out, label)
		next, ok := strings.CutPrefix(rest, ".")
		if !ok {
			return out, rest
		}
		if _, _, ok := runOf(next, keep); !ok {
			return out, rest
		}
		s = next
	}
}

func spokenParts(part string, words []string, spoken map[rune]string) []string {
	start := 0
	for i, c := range part {
		word, ok := spoken[c]
		if !ok {
			continue
		}
		if start < i {
			words = append(words, part[start:i])
		}
		words = append(words, word)
		start = i + utf8.RuneLen(c)
	}
	if start < len(part) {
		words = append(words, part[start:])
	}
	return words
}

// runOf splits off the run of runes s starts with that keep accepts, if any.
func runOf(s string, keep func(rune) bool) (string, string, bool) {
	n := strings.IndexFunc(s, func(c rune) bool { return !keep(c) })
	if n < 0 {
		n = len(s)
	}
	return s[:n], s[n:], n > 0
}

// urls reads a web address, with an optional scheme and path, with its
// separators spoken: "https://www.kyutai.fr" is "H-T-T-P-S colon slash slash
// W-W-W dot kyutai dot F-R". Its last label must be two or more ASCII letters,
// and one in mixed case ("it.Now") is a sentence that lost its space unless a
// scheme or "www." says otherwise.
func urls(word string) (string, bool) {
	var words []string
	rest := word
	if after, ok := strings.CutPrefix(word, "https://"); ok {
		words, rest = []string{"H-T-T-P-S", "colon", "slash", "slash"}, after
	} else if after, ok := strings.CutPrefix(word, "http://"); ok {
		words, rest = []string{"H-T-T-P", "colon", "slash", "slash"}, after
	}
	isLabel := func(c rune) bool { return isAlphanumeric(c) || c == '-' }
	domains, rest := labels(rest, isLabel)
	path := ""
	if strings.HasPrefix(rest, "/") {
		path, rest, _ = runOf(rest, func(c rune) bool { return isLabel(c) || strings.ContainsRune("/._", c) })
		// A period that ends the sentence is not part of the path.
		trimmed := strings.TrimRight(path, ".")
		path, rest = trimmed, path[len(trimmed):]+rest
	}
	if len(domains) < 2 || !isSuffix(rest) || !asciiWord(domains[len(domains)-1]) {
		return "", false
	}
	tld := domains[len(domains)-1]
	if tld != strings.ToLower(tld) && tld != strings.ToUpper(tld) && len(words) == 0 && !strings.EqualFold(domains[0], "www") {
		return "", false
	}
	for i, label := range domains {
		if i > 0 {
			words = append(words, "dot")
		}
		words = addressWords(label, words)
	}
	if path != "" {
		words = addressWords(path, words)
	}
	return strings.Join(words, " ") + rest, true
}

// addressWords reads part with its separators and digits spoken. Only a part
// with none of them, such as "fr" or "www", can be spelled out.
func addressWords(part string, words []string) []string {
	start := 0
	for i := range len(part) {
		var word string
		switch c := part[i]; {
		case c == '-':
			word = "dash"
		case c == '/':
			word = "slash"
		case c == '.':
			word = "dot"
		case c == '_':
			word = "underscore"
		case c >= '0' && c <= '9':
			word = part[i : i+1]
		default:
			continue
		}
		if start < i {
			words = append(words, part[start:i])
		}
		words = append(words, word)
		start = i + 1
	}
	if start == 0 {
		return append(words, spellShort(part))
	}
	if start < len(part) {
		words = append(words, part[start:])
	}
	return words
}

// asciiWord reports whether s is two or more ASCII letters, as in "fr" and "md".
func asciiWord(s string) bool {
	return len(s) >= 2 && strings.Trim(s, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") == ""
}

// abbreviationWords come before a name or a number, so their period never ends
// a sentence.
var abbreviationWords = map[string]string{
	"Lt": "Lieutenant", "Lt.": "Lieutenant", "Col.": "Colonel", "Sgt.": "Sergeant", "Sgt": "Sergeant", "Capt.": "Captain",
	"Cpl.": "Corporal", "Gen.": "General", "Adm.": "Admiral", "Sen.": "Senator", "Sens.": "Senators", "Rep.": "Representative",
	"Reps.": "Representatives", "Gov.": "Governor", "Pres.": "President", "Det.": "Detective", "Insp.": "Inspector",
	"Supt.": "Superintendent", "Ft.": "Fort", "Mt.": "Mount", "vol.": "volume", "Vol.": "Volume", "vs.": "versus",
	"w/": "with", "W/": "With", "w/o": "without", "W/o": "Without", "c/o": "care of", "Ctrl": "Control", "Cmd": "Command",
}

// endingWords can also end a sentence. "Jan." and "Sun." are left out: they are
// more often a name or the Sun.
var endingWords = map[string]string{
	"Feb.": "February", "Mar.": "March", "Apr.": "April", "Jun.": "June", "Jul.": "July", "Aug.": "August",
	"Sep.": "September", "Sept.": "September", "Oct.": "October", "Nov.": "November", "Dec.": "December",
	"Mon.": "Monday", "Tue.": "Tuesday", "Tues.": "Tuesday", "Wed.": "Wednesday", "Thu.": "Thursday",
	"Thur.": "Thursday", "Thurs.": "Thursday", "Fri.": "Friday", "Sat.": "Saturday",
	"ed.": "edition", "approx.": "approximately", "Approx.": "Approximately", "dept.": "department", "Dept.": "Department",
}

// calendarNames start the names of months and days.
var calendarNames = strings.Fields("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec Mon Tue Wed Thu Fri Sat Sun")

func abbreviations(word string) (string, bool) {
	body, suffix := splitSuffix(word)
	abbreviation := body
	if rest, ok := strings.CutPrefix(suffix, "."); ok {
		abbreviation, suffix = word[:len(body)+1], rest
	}
	s, ok := abbreviationWords[abbreviation]
	if !ok {
		s, ok = endingWords[abbreviation]
	}
	return s + suffix, ok
}

// sentenceEnd keeps the period of an abbreviation that ends a sentence: "in Dec.
// Then" is "in December. Then", while "Dec. 5" and "Mon. Oct. 5" are dates.
func sentenceEnd(words []string) (string, bool) {
	s, ok := endingWords[words[0]]
	if !ok || len(words) < 2 {
		return "", false
	}
	next := words[1]
	if c, _ := utf8.DecodeRuneInString(next); !unicode.IsUpper(c) || len(next) >= 3 && slices.Contains(calendarNames, next[:3]) {
		return "", false
	}
	return s + ".", true
}

// elongations shortens a stretched word as upstream does: "sooo" is "so" and
// "Nooooo," "No,". A run of three or more of a vowel, 'h', 'w' or 'y' becomes
// one letter; "Zzz", "Brrr" and words in capitals are left alone.
func elongations(word string) (string, bool) {
	body, suffix := splitSuffix(word)
	if !asciiWord(body) || body == strings.ToUpper(body) {
		return "", false
	}
	var b strings.Builder
	stretched := false
	for i := 0; i < len(body); {
		j := i + 1
		for j < len(body) && body[j] == body[i] {
			j++
		}
		if j-i >= 3 && strings.IndexByte("aeiouyhwAEIOUYHW", body[i]) >= 0 {
			b.WriteByte(body[i])
			stretched = true
		} else {
			b.WriteString(body[i:j])
		}
		i = j
	}
	return b.String() + suffix, stretched
}
