package cmd

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExtractFilenames(t *testing.T) {
	// Unix style paths
	input := ` some preamble 
 ./relative\ path/one.png inbetween1 ./not a valid two.jpg inbetween2 ./1.svg
/unescaped space /three.jpeg inbetween3 /valid\ path/dir/four.png "./quoted with spaces/five.JPG
/unescaped space /six.webp inbetween6 /valid\ path/dir/seven.WEBP`
	res := extractFileNames(input)
	assert.Len(t, res, 7)
	assert.Contains(t, res[0], "one.png")
	assert.Contains(t, res[1], "two.jpg")
	assert.Contains(t, res[2], "three.jpeg")
	assert.Contains(t, res[3], "four.png")
	assert.Contains(t, res[4], "five.JPG")
	assert.Contains(t, res[5], "six.webp")
	assert.Contains(t, res[6], "seven.WEBP")
	assert.NotContains(t, res[4], '"')
	assert.NotContains(t, res, "inbetween1")
	assert.NotContains(t, res, "./1.svg")

	// Windows style paths
	input = ` some preamble
 c:/users/jdoe/one.png inbetween1 c:/program files/someplace/two.jpg inbetween2 
 /absolute/nospace/three.jpeg inbetween3 /absolute/with space/four.png inbetween4
./relative\ path/five.JPG inbetween5 "./relative with/spaces/six.png inbetween6
d:\path with\spaces\seven.JPEG inbetween7 c:\users\jdoe\eight.png inbetween8 
 d:\program files\someplace\nine.png inbetween9 "E:\program files\someplace\ten.PNG
c:/users/jdoe/eleven.webp inbetween11 c:/program files/someplace/twelve.WebP inbetween12
d:\path with\spaces\thirteen.WEBP some ending
`
	res = extractFileNames(input)
	assert.Len(t, res, 13)
	assert.NotContains(t, res, "inbetween2")
	assert.Contains(t, res[0], "one.png")
	assert.Contains(t, res[0], "c:")
	assert.Contains(t, res[1], "two.jpg")
	assert.Contains(t, res[1], "c:")
	assert.Contains(t, res[2], "three.jpeg")
	assert.Contains(t, res[3], "four.png")
	assert.Contains(t, res[4], "five.JPG")
	assert.Contains(t, res[5], "six.png")
	assert.Contains(t, res[6], "seven.JPEG")
	assert.Contains(t, res[6], "d:")
	assert.Contains(t, res[7], "eight.png")
	assert.Contains(t, res[7], "c:")
	assert.Contains(t, res[8], "nine.png")
	assert.Contains(t, res[8], "d:")
	assert.Contains(t, res[9], "ten.PNG")
	assert.Contains(t, res[9], "E:")
	assert.Contains(t, res[10], "eleven.webp")
	assert.Contains(t, res[10], "c:")
	assert.Contains(t, res[11], "twelve.WebP")
	assert.Contains(t, res[11], "c:")
	assert.Contains(t, res[12], "thirteen.WEBP")
	assert.Contains(t, res[12], "d:")
}

func TestExtractFilenamesWithMacOSPaths(t *testing.T) {
	// macOS drag-drop paths with escaped spaces (backslash-space)
	input := `describe /Users/jdoe/My\ Project/image.png and also /Users/jdoe/file.webp`
	res := extractFileNames(input)
	assert.Contains(t, res[0], "image.png", "should find macOS path with escaped space")
	assert.Contains(t, res[1], "file.webp", "should find normal macOS path")

	// macOS drag-drop with file:// URL prefix
	input = `what's in file:///Users/jdoe/photo.jpeg?`
	res = extractFileNames(input)
	assert.Contains(t, res[0], "photo.jpeg", "should find file:// path")

	// macOS drag-drop combining file:// with escaped spaces
	input = `analyze file:///Users/jdoe/My\ Project/screenshot.png`
	res = extractFileNames(input)
	assert.NotEmpty(t, res, "should find file:// path with escaped space")
	assert.Contains(t, res[0], "screenshot.png")

	// Quoted paths (single and double quotes)
	input = `look at '/path/to/my file.jpg' and "/another/file with spaces.png"`
	res = extractFileNames(input)
	assert.Contains(t, res[0], "my file.jpg", "should find single-quoted path")
	assert.Contains(t, res[1], "file with spaces.png", "should find double-quoted path")

	// Mixed content with macOS drag-drop path
	input = `describe the image file:///Users/jdoe/Desktop/sample.JPEG and /tmp/normal.webp`
	res = extractFileNames(input)
	assert.Len(t, res, 2, "should find both macOS drag-drop and normal paths")
}

func TestNormalizeFilePath(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"file:///Users/jdoe/image.png", "/Users/jdoe/image.png"},
		{"/Users/jdoe/My\\ Project/image.png", "/Users/jdoe/My Project/image.png"},
		{"file:///Users/jdoe/My\\ Project/image.png", "/Users/jdoe/My Project/image.png"},
		{"/path/with (parens)/file.png", "/path/with (parens)/file.png"},
		{"/path/with/file?.png", "/path/with/file?.png"},
	}
	for _, tt := range tests {
		got := normalizeFilePath(tt.input)
		if got != tt.expected {
			t.Errorf("normalizeFilePath(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

// Ensure that file paths wrapped in single quotes are removed with the quotes.
func TestExtractFileDataRemovesQuotedFilepath(t *testing.T) {
	dir := t.TempDir()
	fp := filepath.Join(dir, "img.jpg")
	data := make([]byte, 600)
	copy(data, []byte{
		0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 'J', 'F', 'I', 'F',
		0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0xff, 0xd9,
	})
	if err := os.WriteFile(fp, data, 0o600); err != nil {
		t.Fatalf("failed to write test image: %v", err)
	}

	input := "before '" + fp + "' after"
	cleaned, imgs, err := extractFileData(input)
	assert.NoError(t, err)
	assert.Len(t, imgs, 1)
	assert.Equal(t, cleaned, "before  after")
}

func TestExtractFileDataWAV(t *testing.T) {
	dir := t.TempDir()
	fp := filepath.Join(dir, "sample.wav")
	data := make([]byte, 600)
	copy(data[:44], []byte{
		'R', 'I', 'F', 'F',
		0x58, 0x02, 0x00, 0x00, // file size - 8
		'W', 'A', 'V', 'E',
		'f', 'm', 't', ' ',
		0x10, 0x00, 0x00, 0x00, // fmt chunk size
		0x01, 0x00, // PCM
		0x01, 0x00, // mono
		0x80, 0x3e, 0x00, 0x00, // 16000 Hz
		0x00, 0x7d, 0x00, 0x00, // byte rate
		0x02, 0x00, // block align
		0x10, 0x00, // 16-bit
		'd', 'a', 't', 'a',
		0x34, 0x02, 0x00, 0x00, // data size
	})
	if err := os.WriteFile(fp, data, 0o600); err != nil {
		t.Fatalf("failed to write test audio: %v", err)
	}

	input := "before " + fp + " after"
	cleaned, imgs, err := extractFileData(input)
	assert.NoError(t, err)
	assert.Len(t, imgs, 1)
	assert.Equal(t, "before  after", cleaned)
}
