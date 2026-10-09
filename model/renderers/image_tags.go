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
	if imageCount == 0 {
		return content, imageOffset
	}

	if strings.Contains(content, "[img-") {
		return content, imageOffset + imageCount
	}

	var prefix strings.Builder
	for i := range imageCount {
		imgTag := fmt.Sprintf("[img-%d]", imageOffset+i)
		if strings.Contains(content, "[img]") {
			content = strings.Replace(content, "[img]", imgTag, 1)
		} else {
			prefix.WriteString(imgTag)
		}
	}

	if prefix.Len() > 0 && content != "" {
		if r, _ := utf8.DecodeRuneInString(content); r != utf8.RuneError && !unicode.IsSpace(r) {
			prefix.WriteByte(' ')
		}
	}

	return prefix.String() + content, imageOffset + imageCount
}
