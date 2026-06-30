package chat

import "strings"

func renderMarkdownForView(markdown string, width int) string {
	return strings.Join(wrapChatText(markdown, width), "\n")
}

func splitRenderedBody(body string) []string {
	body = strings.TrimRight(body, "\n")
	if body == "" {
		return []string{""}
	}
	return strings.Split(body, "\n")
}
