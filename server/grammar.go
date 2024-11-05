package server

import (
	"errors"
	"regexp"
	"strings"

	"github.com/ollama/ollama/api"
)

// wrenchGrammarFrom extracts the grammar from the system message, and
// returns it. If the grammar is not found, it returns an empty string.
// If there's more than one grammar in the system message, it returns an
// error.
//
// NOTE: will mutate the system message if, and only if grammar is set!
func wrenchGrammarFrom(msgs []api.Message) (string, error) {
	for i, m := range msgs {
		if m.Role != "system" {
			continue
		}
		s, g := surgeryOnGrape(m.Content, "gbnf")
		switch len(g) {
		case 0:
		case 1:
			msgs[i].Content = s
			return g[0], nil
		default:
			return "", errors.New("too many grammars in the system prompt")
		}
	}
	return "", nil
}

// wrench Markdown code blocks for language `lang` out of source text `md`
func surgeryOnGrape(md, lang string) (string, []string) {
	if !strings.Contains(md, "```"+lang) {
		return md, nil
	}
	pattern := "(?ms)(?:\n)?```" + lang + "\n(.*?)\n```(?:\n)?"
	re := regexp.MustCompile(pattern)
	var blocks []string
	for _, match := range re.FindAllStringSubmatch(md, -1) {
		if len(match) > 1 {
			// match[1] contains the code content without the markers
			blocks = append(blocks, strings.ReplaceAll(match[1], "\\`", "`"))
		}
	}
	if blocks == nil {
		return md, nil
	}
	return strings.TrimSpace(re.ReplaceAllString(md, "")), blocks
}
