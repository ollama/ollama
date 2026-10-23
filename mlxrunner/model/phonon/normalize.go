package phonon

import (
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// normalizeText is xn-ptts normalize_text for English with the default rules
// (ptts 0.4.1): typographic marks the tokenizer lacks are folded to ASCII or
// dropped, symbols that are read aloud are spelled out, then rewrite.go reads
// each word.
func normalizeText(input string) string {
	var res appender
	runes := []rune(input)
	at := func(i int) rune {
		if i < len(runes) {
			return runes[i]
		}
		return -1
	}
	prev := rune(-1)
	for i := 0; i < len(runes); i++ {
		c := runes[i]
		switch {
		case strings.ContainsRune("»«‹›“”„‟″‶⹂❝❞❠〝〞〟＂", c):
			res.push('"')
		case strings.ContainsRune("‘’‚‛′‵❛❜❟＇ʼʻʹ´`", c):
			res.push('\'')
		case strings.ContainsRune("‐‑‒―−", c):
			res.push('-')
		case c == '★' || c == '☆':
			// A rating such as "★★★½☆"; a lone star is decoration.
			full, empty, half := 0, 0, false
			for ; at(i) == '★' || at(i) == '☆' || at(i) == '½' && !half; i++ {
				switch at(i) {
				case '★':
					full++
				case '☆':
					empty++
				default:
					half = true
				}
			}
			i--
			n, of := strconv.Itoa(full), full+empty
			if half {
				n, of = n+".5", of+1
			}
			switch {
			case of < 2:
				res.pushSpace()
			case empty > 0:
				res.pushSpoken(n + " out of " + strconv.Itoa(of) + " stars")
			default:
				res.pushSpoken(n + " stars")
			}
		case (c == '°' || c == 'º') && res.endsWithN():
			// "n°5" is the number sign, not "n degrees".
			b := res.buf
			if b[len(b)-1] == '.' {
				b = b[:len(b)-1]
			}
			word := "number"
			if b[len(b)-1] == 'N' {
				word = "Number"
			}
			res.buf = b[:len(b)-1]
			res.pushSpoken(word)
		case c == '°':
			if res.endsWithOne() {
				res.pushSpoken("degree")
			} else {
				res.pushSpoken("degrees")
			}
			if scale := degreeScales[at(i+1)]; scale != "" && !isAlphanumeric(at(i+2)) {
				res.pushString(scale)
				i++
			}
		case subscripts[c] != 0:
			res.push(subscripts[c])
		case superscripts[c] != 0:
			n := string(superscripts[c])
			for superscripts[at(i+1)] != 0 {
				i++
				n += string(superscripts[runes[i]])
			}
			if isAlphanumeric(prev) || prev == ')' {
				res.pushSpoken(power(n))
			} else {
				res.pushString(n)
			}
		case greek(c) != "":
			res.pushSpoken(greek(c))
		case symbols[c] != "":
			res.pushSpoken(symbols[c])
		case c == '±':
			res.pushSpoken("plus or minus")
		case c == '❤' || c == '♥':
			// "I ❤ NY" reads "love"; a heart that is not between words is decoration.
			if at(i+1) == '️' {
				i++
			}
			j := i + 1
			for unicode.IsSpace(at(j)) {
				j++
			}
			if b := trimSpace(res.buf); len(b) > 0 && isAlphanumeric(b[len(b)-1]) && isAlphanumeric(at(j)) {
				res.pushSpoken("love")
			} else {
				res.pushSpace()
			}
		case c == '~' && isASCIIDigit(at(i+1)):
			res.pushSpoken("about")
		case c == '*' && (prev < 0 || unicode.IsSpace(prev)) && isASCIIDigit(at(i+1)):
			res.pushSpoken("star")
		case c == '#':
			// "#1" is "number 1" and "C#" "C sharp"; other hashes are markup.
			switch {
			case isASCIIDigit(at(i + 1)):
				res.pushSpoken("number")
			case isAlphanumeric(prev) && !isAlphanumeric(at(i+1)):
				res.pushSpoken("sharp")
			default:
				res.pushSpace()
			}
		case c == '_' && !(isAlphanumeric(prev) && isAlphanumeric(at(i+1))):
			// "_word_" is emphasis; the '_' of "snake_case" is read out later.
			res.pushSpace()
		case c == '–' && isASCIIDigit(prev) && (isASCIIDigit(at(i+1)) || currencies[at(i+1)].many != ""):
			// An en dash between amounts is a range: "2020–2021", "$10–$20".
			res.pushSpoken("to")
		case strings.ContainsRune("–*—[]{}•‣◦·→←↑↓➡➜", c):
			res.pushSpace()
		case currencies[c].many != "" && res.endsWithSpacedDigit() && !isASCIIDigit(at(i+1)):
			// "250 €" reads as "250€", unless the symbol leads the next amount.
			res.popSpace()
			res.push(c)
		case c == '…':
			res.push('.')
		case c == '=':
			res.pushSpoken("equals")
		case c == ':' && prev >= 0 && !unicode.IsSpace(prev) && at(i+1) >= 0 && !unicode.IsSpace(at(i+1)):
			res.push(c)
		case strings.ContainsRune(";:()", c):
			res.popSpace()
			res.push(',')
			res.pushSpace()
		case isEmoji(c) || unicode.IsControl(c) || unicode.IsSpace(c):
			res.pushSpace()
		default:
			res.push(c)
		}
		prev = c
	}
	return stripClosingQuote(rewriteText(res.String()))
}

// appender is the reference StringAppender: whitespace collapses to one space,
// and a '.' or ',' swallows the whitespace and punctuation before it.
type appender struct {
	buf []rune
}

func (a *appender) push(c rune) {
	if c == '.' || c == ',' {
		for len(a.buf) > 0 {
			last := a.buf[len(a.buf)-1]
			if !unicode.IsSpace(last) && (!isASCIIPunct(last) || a.keeps(last)) {
				break
			}
			a.buf = a.buf[:len(a.buf)-1]
		}
	}
	a.buf = append(a.buf, c)
}

func (a *appender) pushString(s string) {
	for _, c := range s {
		a.push(c)
	}
}

// keeps reports whether last stays before a '.' or ',': quotes, so that they
// stay balanced, '@' and '+', which are spelled out after the rules read them,
// the '$' of an amount ("5$."), and '%' and '&', which are read aloud.
func (a *appender) keeps(last rune) bool {
	if last == '$' {
		return len(a.buf) >= 2 && isASCIIDigit(a.buf[len(a.buf)-2])
	}
	return strings.ContainsRune("\"'@+%&", last)
}

// pushSpoken adds a symbol's spoken form as a word of its own.
func (a *appender) pushSpoken(word string) {
	a.pushSpace()
	a.pushString(word)
	a.pushSpace()
}

func (a *appender) pushSpace() {
	if len(a.buf) > 0 && !unicode.IsSpace(a.buf[len(a.buf)-1]) {
		a.push(' ')
	}
}

func (a *appender) popSpace() { a.buf = trimSpace(a.buf) }

// endsWithOne reports whether the text ends with the number 1, which takes
// "degree" rather than "degrees".
func (a *appender) endsWithOne() bool {
	b := trimSpace(a.buf)
	n := len(b)
	return n > 0 && b[n-1] == '1' && (n < 2 || !isASCIIDigit(b[n-2]) && b[n-2] != '.' && b[n-2] != ',')
}

// endsWithN reports whether the text ends with an "n" or "N" that is a word of
// its own, with or without a period.
func (a *appender) endsWithN() bool {
	b := a.buf
	if n := len(b); n > 0 && b[n-1] == '.' {
		b = b[:n-1]
	}
	n := len(b)
	return n > 0 && (b[n-1] == 'n' || b[n-1] == 'N') && (n < 2 || !isAlphanumeric(b[n-2]))
}

// endsWithSpacedDigit reports whether the text ends with a digit and then
// whitespace, as before the "€" of "250 €".
func (a *appender) endsWithSpacedDigit() bool {
	b := trimSpace(a.buf)
	return len(b) < len(a.buf) && len(b) > 0 && isASCIIDigit(b[len(b)-1])
}

func (a *appender) String() string {
	a.popSpace()
	return string(a.buf)
}

func trimSpace(b []rune) []rune {
	for len(b) > 0 && unicode.IsSpace(b[len(b)-1]) {
		b = b[:len(b)-1]
	}
	return b
}

func stripClosingQuote(text string) string {
	text = strings.TrimRight(text, "\"")
	rest, ok := strings.CutSuffix(text, "'")
	if last, _ := utf8.DecodeLastRuneInString(rest); ok && strings.ContainsRune(".!?,;", last) {
		return rest
	}
	return text
}

// spellSymbols spells out the '@', '+' and '_' of a word no rule claimed, and
// gives the pieces they leave their own turn at the rules: "1500+20" reads "1
// thousand 500 plus 20". A slash between words ("and/or") is read as a pause.
func spellSymbols(word string) string {
	if text, ok := splitSlash(word); ok {
		return rewriteEach(text)
	}
	if !strings.ContainsAny(word, "@+_") {
		return word
	}
	var res appender
	for _, c := range word {
		switch c {
		case '@':
			res.pushSpoken("at")
		case '+':
			res.pushSpoken("plus")
		case '_':
			res.pushSpoken("underscore")
		default:
			res.push(c)
		}
	}
	return rewriteEach(res.String())
}

func rewriteEach(text string) string {
	words := strings.Split(text, " ")
	for i, word := range words {
		if s, ok := rewriteWord(word); ok {
			words[i] = s
		}
	}
	return strings.Join(words, " ")
}

// Units written with a slash between letters ("mg/kg"), which keep it.
var slashUnits = strings.Fields("km kg mg ml mL dL mol kJ kW kWh Hz min hr hrs sec ft lb lbs oz mi mph kph yd cm mm yr wk mo kcal rpm")

// splitSlash replaces the slashes of word with spaces when every part they
// separate is a word of two letters or more and not a unit, which leaves
// "c/o", "km/h", "mg/kg", dates and fractions alone.
func splitSlash(word string) (string, bool) {
	body := strings.TrimRightFunc(word, func(c rune) bool { return isASCIIPunct(c) && c != '/' })
	parts := strings.Split(body, "/")
	if len(parts) < 2 {
		return "", false
	}
	for _, part := range parts {
		if utf8.RuneCountInString(part) < 2 || strings.IndexFunc(part, func(c rune) bool { return !isAlphabetic(c) }) >= 0 || slices.Contains(slashUnits, part) {
			return "", false
		}
	}
	return strings.Join(parts, " ") + word[len(body):], true
}

func power(n string) string {
	switch n {
	case "2":
		return "squared"
	case "3":
		return "cubed"
	}
	return "to the power of " + n
}

var symbols = map[rune]string{
	'×': "times",
	'÷': "divided by",
	'≤': "less than or equal to",
	'≥': "greater than or equal to",
	'≠': "is not equal to",
	'≈': "approximately",
	'√': "square root of",
	'∞': "infinity",
	'∑': "sigma",
	'‰': "per mille",
}

// degreeScales are the temperature scales that follow "°".
var degreeScales = map[rune]string{'C': "Celsius", 'F': "Fahrenheit"}

var greekNames = strings.Fields("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega")

func greek(c rune) string {
	switch {
	case c >= 'Α' && c <= 'Ρ':
		return greekNames[c-'Α']
	case c >= 'Σ' && c <= 'Ω':
		return greekNames[c-'Α'-1]
	case c >= 'α' && c <= 'ρ':
		return greekNames[c-'α']
	case c == 'ς':
		return greekNames[17]
	case c >= 'σ' && c <= 'ω':
		return greekNames[c-'α'-1]
	}
	return ""
}

// isEmoji covers the pictograph blocks, and the variation selector and joiner
// that emoji sequences leave.
func isEmoji(c rune) bool {
	return c >= 0x1F000 && c <= 0x1FAFF || c >= 0x2600 && c <= 0x27BF || c >= 0x2B00 && c <= 0x2BFF || c == 0xFE0F || c == 0x200D
}

// pairs maps each rune at an even position of s to the rune after it.
func pairs(s string) map[rune]rune {
	r := []rune(s)
	m := make(map[rune]rune, len(r)/2)
	for i := 0; i+1 < len(r); i += 2 {
		m[r[i]] = r[i+1]
	}
	return m
}

var (
	subscripts   = pairs("₀0₁1₂2₃3₄4₅5₆6₇7₈8₉9₊+₋-ₐaₑeₒoₓxₕhₖkₗlₘmₙnₚpₛsₜtᵢiⱼj")
	superscripts = pairs("⁰0¹1²2³3⁴4⁵5⁶6⁷7⁸8⁹9⁻-⁺+ⁿn")
)

// Rust's char classes, which unicode.IsLetter and friends only approximate.

func isAlphabetic(c rune) bool {
	return c >= 0 && (unicode.IsLetter(c) || unicode.In(c, unicode.Nl, unicode.Other_Alphabetic))
}

func isAlphanumeric(c rune) bool { return isAlphabetic(c) || c >= 0 && unicode.IsNumber(c) }

func isASCIIDigit(c rune) bool { return c >= '0' && c <= '9' }

func isASCIIAlnum(c rune) bool {
	return isASCIIDigit(c) || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}

func isASCIIPunct(c rune) bool {
	return c >= '!' && c <= '/' || c >= ':' && c <= '@' || c >= '[' && c <= '`' || c >= '{' && c <= '~'
}
