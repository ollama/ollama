package imagegen

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExtractFileNames(t *testing.T) {
	input := `./directory.png/image.png /tmp/photo.png. (/tmp/wrapped.jpg) /tmp/a.png,/tmp/b.png "` + `/tmp/quoted.webp"` + `" '` + `/tmp/single.jpeg` + `' ` + "`/tmp/backtick.png`"
	res := extractFileNames(input)
	assert.Equal(t, []string{"./directory.png/image.png", "/tmp/photo.png", "/tmp/wrapped.jpg", "/tmp/a.png", "/tmp/b.png", "/tmp/quoted.webp", "/tmp/single.jpeg", "/tmp/backtick.png"}, res)

	input = `compare /tmp/a.png&/tmp/b.png | /tmp/c.webp</tmp/d.jpg`
	res = extractFileNames(input)
	assert.Equal(t, []string{"/tmp/a.png", "/tmp/b.png", "/tmp/c.webp", "/tmp/d.jpg"}, res)

	input = `compare /tmp/a.png+/tmp/b.png`
	res = extractFileNames(input)
	assert.Equal(t, []string{"/tmp/a.png", "/tmp/b.png"}, res)

	input = `describe /tmp/cat.png请描述 and /tmp/dog.jpg。`
	res = extractFileNames(input)
	assert.Equal(t, []string{"/tmp/cat.png", "/tmp/dog.jpg"}, res)

	input = `/tmp/a.png./b.png`
	res = extractFileNames(input)
	assert.Equal(t, []string{"/tmp/a.png", "./b.png"}, res)

	input = `compare C:\tmp\a.png&D:\tmp\b.jpg`
	res = extractFileNames(input)
	assert.Equal(t, []string{`C:\tmp\a.png`, `D:\tmp\b.jpg`}, res)

	input = `attach {/tmp/photo.png} and {C:\tmp\a.png}`
	res = extractFileNames(input)
	assert.Equal(t, []string{"/tmp/photo.png", `C:\tmp\a.png`}, res)

	input = `C:\images\dir.png\nested.jpg`
	res = extractFileNames(input)
	assert.Equal(t, []string{`C:\images\dir.png\nested.jpg`}, res)
}
