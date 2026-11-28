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

	if indexedImageTagPresent(content, imageCount, imageOffset) {
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

// indexedImageTagPresent reports whether content already has a numbered
// placeholder for every image in [imageOffset, imageOffset+imageCount).
// One match is not enough: arr[img-1] is not the tag for image 0, and a
// two-image message still needs [img-0].
func indexedImageTagPresent(content string, imageCount, imageOffset int) bool {
	for i := range imageCount {
		if !strings.Contains(content, fmt.Sprintf("[img-%d]", imageOffset+i)) {
			return false
		}
	}
	return true
}
