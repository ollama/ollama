package renderers

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// renderContentWithImageTags preserves the legacy server-side placeholder
// semantics for explicit [img] tokens: replace placeholders in order, and
// only prepend tags for any remaining images without placeholders.
func renderContentWithImageTags(content string, imageCount int, imageOffset int) (string, int) {
	content, prefix := PrepareMediaTags(content, imageCount, imageOffset)
	if prefix != "" && content != "" {
		if r, _ := utf8.DecodeRuneInString(content); r != utf8.RuneError && !unicode.IsSpace(r) {
			prefix += " "
		}
	}
	return prefix + content, imageOffset + imageCount
}

// PrepareMediaTags preserves explicit slot positions and returns markers for
// unplaced payloads. The caller owns whether those markers precede or follow text.
func PrepareMediaTags(content string, mediaCount, mediaOffset int) (string, string) {
	var unplaced strings.Builder
	for i := range mediaCount {
		tag := fmt.Sprintf("[img-%d]", mediaOffset+i)
		if strings.Contains(content, tag) {
			continue
		}
		if strings.Contains(content, "[img]") {
			content = strings.Replace(content, "[img]", tag, 1)
		} else {
			unplaced.WriteString(tag)
		}
	}
	return content, unplaced.String()
}
