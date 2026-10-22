package phonon

import (
	"strings"
	"unicode"
)

// normalizeText applies the reference's language-independent punctuation rules.
// Number, currency, address and other language-specific rewrites belong in a
// separate pass once the model's language is known.
func normalizeText(text string) string {
	var out []rune
	push := func(c rune) {
		if c == '.' || c == ',' {
			for len(out) > 0 {
				last := out[len(out)-1]
				keep := strings.ContainsRune("\"'@+%#&", last) ||
					last == '$' && len(out) > 1 && out[len(out)-2] >= '0' && out[len(out)-2] <= '9'
				punctuation := last < 128 && (unicode.IsPunct(last) || unicode.IsSymbol(last))
				if !unicode.IsSpace(last) && (!punctuation || keep) {
					break
				}
				out = out[:len(out)-1]
			}
		}
		out = append(out, c)
	}
	space := func() {
		if len(out) > 0 && out[len(out)-1] != ' ' {
			out = append(out, ' ')
		}
	}
	runes := []rune(text)
	for i, c := range runes {
		switch {
		case strings.ContainsRune("»«‹›“”„‟″‶⹂❝❞❠〝〞〟＂", c):
			push('"')
		case strings.ContainsRune("‘’‚‛′‵❛❜❟＇ʼʻʹ´`", c):
			push('\'')
		case strings.ContainsRune("‐‑‒―", c):
			push('-')
		case strings.ContainsRune("–—[]{}•‣◦·→←↑↓➡➜", c):
			space()
		case c == '*':
			// A leading * before a number is a spoken star in the reference.
			// Preserve it for language-specific expansion; remove markup.
			if (i == 0 || unicode.IsSpace(runes[i-1])) && i+1 < len(runes) && runes[i+1] >= '0' && runes[i+1] <= '9' {
				push(c)
			} else {
				space()
			}
		case c == '…':
			push('.')
		case c == ':' && i > 0 && i+1 < len(runes) && !unicode.IsSpace(runes[i-1]) && !unicode.IsSpace(runes[i+1]):
			push(c)
		case strings.ContainsRune(";:()", c):
			push(',')
			space()
		case strings.ContainsRune("★☆❤♥", c):
			// These carry meaning ("stars", "love"), unlike decorative emoji.
			push(c)
		case c >= 0x1F000 && c <= 0x1FAFF || c >= 0x2600 && c <= 0x27BF ||
			c >= 0x2B00 && c <= 0x2BFF || c == 0xFE0F || c == 0x200D ||
			unicode.IsControl(c) || unicode.IsSpace(c):
			space()
		default:
			push(c)
		}
	}
	text = strings.TrimRight(strings.TrimSpace(string(out)), "\"")
	if len(text) >= 2 && text[len(text)-1] == '\'' && strings.ContainsRune(".!?,;", rune(text[len(text)-2])) {
		text = text[:len(text)-1]
	}
	return text
}
