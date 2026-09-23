package mlxrunner

import (
	"bytes"
	"unicode/utf8"
)

// flushValidUTF8Prefix returns and consumes the longest valid UTF-8 prefix
// currently buffered, leaving any incomplete trailing bytes in place.
func flushValidUTF8Prefix(b *bytes.Buffer) string {
	data := b.Bytes()
	if len(data) == 0 {
		return ""
	}

	prefix := validUTF8PrefixLen(data)
	if prefix == 0 {
		return ""
	}

	text := string(data[:prefix])
	b.Next(prefix)
	return text
}

func validUTF8PrefixLen(data []byte) int {
	i := 0
	prefix := 0
	for i < len(data) {
		r, size := utf8.DecodeRune(data[i:])
		if r == utf8.RuneError && size == 1 {
			if !utf8.FullRune(data[i:]) {
				break
			}

			// Invalid UTF-8 byte; consume one byte to guarantee forward progress.
			i++
			prefix = i
			continue
		}

		i += size
		prefix = i
	}

	return prefix
}
