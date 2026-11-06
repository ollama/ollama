package testutil

import (
	"regexp"
	"strings"
)

// wikitextDetokenizer is a faithful Go port of EleutherAI lm-evaluation-
// harness' preprocess_wikitext.wikitext_detokenizer. It restores the natural
// English form of a wikitext-2-raw-v1 document by reversing the dataset's
// tokenization marks (e.g. " @-@ " → "-", " : " → ": ", "\\s*\\(\\s*x\\s*\\)" → "(x)").
//
// This transform is applied to the input the model sees, but NOT to the
// text used as the denominator for word_perplexity / byte_perplexity --
// matching lm-eval-harness exactly. Without this transform, our PPL
// numbers will be ~2× higher than the canonical wikitext PPL because the
// model is being asked to predict alignment artifacts.
//
// Source for the original Python:
//
//	https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/preprocess_wikitext.py
func WikitextDetokenize(s string) string {
	// contractions
	s = strings.ReplaceAll(s, "s '", "s'")
	s = wikitextContractionRe.ReplaceAllStringFunc(s, func(m string) string {
		// `' [0-9]` -> `'[0-9]`: drop the space.
		return string([]byte{m[0], m[2]})
	})

	// number separators
	s = strings.ReplaceAll(s, " @-@ ", "-")
	s = strings.ReplaceAll(s, " @,@ ", ",")
	s = strings.ReplaceAll(s, " @.@ ", ".")

	// punctuation
	s = strings.ReplaceAll(s, " : ", ": ")
	s = strings.ReplaceAll(s, " ; ", "; ")
	s = strings.ReplaceAll(s, " . ", ". ")
	s = strings.ReplaceAll(s, " ! ", "! ")
	s = strings.ReplaceAll(s, " ? ", "? ")
	s = strings.ReplaceAll(s, " , ", ", ")

	// double brackets — strip whitespace immediately inside ( ) [ ] { } " '
	s = wikitextParensRe.ReplaceAllString(s, "($1)")
	s = wikitextSquareRe.ReplaceAllString(s, "[$1]")
	s = wikitextBraceRe.ReplaceAllString(s, "{$1}")
	s = wikitextDQuoteRe.ReplaceAllString(s, `"$1"`)
	s = wikitextSQuoteRe.ReplaceAllString(s, "'$1'")

	// miscellaneous
	s = strings.ReplaceAll(s, "= = = =", "====")
	s = strings.ReplaceAll(s, "= = =", "===")
	s = strings.ReplaceAll(s, "= =", "==")
	s = strings.ReplaceAll(s, " "+"\u00b0"+" ", "\u00b0") // " ° " → "°"
	s = strings.ReplaceAll(s, " \n", "\n")
	s = strings.ReplaceAll(s, "\n ", "\n")
	s = strings.ReplaceAll(s, " N ", " 1 ")
	s = strings.ReplaceAll(s, " 's", "'s")

	return s
}

var (
	// `' [0-9]` → `'[0-9]`. Original Python: `re.sub(r"' [0-9]", r"'[0-9]", s)`
	// (the original code uses `r"/' [0-9]/"` which is a slash-bracketed regex
	// literal in another language; the Python source we're porting writes the
	// pattern as `r"' [0-9]"`).
	wikitextContractionRe = regexp.MustCompile(`' [0-9]`)

	// Each of these strips inner whitespace from a balanced bracket pair.
	// Python: re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", s) etc.
	wikitextParensRe = regexp.MustCompile(`\(\s*([^)]*?)\s*\)`)
	wikitextSquareRe = regexp.MustCompile(`\[\s*([^\]]*?)\s*\]`)
	wikitextBraceRe  = regexp.MustCompile(`\{\s*([^}]*?)\s*\}`)
	wikitextDQuoteRe = regexp.MustCompile(`"\s*([^"]*?)\s*"`)
	wikitextSQuoteRe = regexp.MustCompile(`'\s*([^']*?)\s*'`)
)
