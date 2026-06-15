package imagegen

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExtractFileNames(t *testing.T) {
	input := `./directory.png/image.png /tmp/photo.png. (/tmp/wrapped.jpg) /tmp/a.png,/tmp/b.png "` + `/tmp/quoted.webp"` + `" '` + `/tmp/single.jpeg` + `' ` + "`/tmp/backtick.png`"
	res := extractFileNames(input)
	assert.Equal(t, []string{"./directory.png/image.png", "/tmp/photo.png", "/tmp/wrapped.jpg", "/tmp/a.png", "/tmp/b.png", "/tmp/quoted.webp", "/tmp/single.jpeg", "/tmp/backtick.png"}, res)

	input = `C:\images\dir.png\nested.jpg`
	res = extractFileNames(input)
	assert.Equal(t, []string{`C:\images\dir.png\nested.jpg`}, res)
}
