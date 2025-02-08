package cmd

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// Test generated using Keploy
func TestNormalizeFilePath_EscapedCharacters(t *testing.T) {
	input := `C:\\path\\to\\file\\with\\escaped\\characters\\file\\name\\space\\test\\file.txt`
	expected := `C:\path\to\file\with\escaped\characters\file\name\space\test\file.txt`
	result := normalizeFilePath(input)

	if result != expected {
		t.Errorf("Expected '%s', got '%s'", expected, result)
	}
}

// Test generated using Keploy
func TestExtractFileNames_ValidExtensions(t *testing.T) {
	input := "Here are some files: /path/to/image1.jpg ./image2.png C:\\images\\image3.jpeg"
	expected := []string{"/path/to/image1.jpg", "./image2.png", "C:\\images\\image3.jpeg"}
	result := extractFileNames(input)

	if len(result) != len(expected) {
		t.Fatalf("Expected %d file paths, got %d", len(expected), len(result))
	}
	for i, file := range result {
		if file != expected[i] {
			t.Errorf("Expected file path '%s', got '%s'", expected[i], file)
		}
	}
}

func TestExtractFilenames(t *testing.T) {
	// Unix style paths
	input := ` some preamble 
 ./relative\ path/one.png inbetween1 ./not a valid two.jpg inbetween2 ./1.svg
/unescaped space /three.jpeg inbetween3 /valid\ path/dir/four.png "./quoted with spaces/five.JPG`
	res := extractFileNames(input)
	assert.Len(t, res, 5)
	assert.Contains(t, res[0], "one.png")
	assert.Contains(t, res[1], "two.jpg")
	assert.Contains(t, res[2], "three.jpeg")
	assert.Contains(t, res[3], "four.png")
	assert.Contains(t, res[4], "five.JPG")
	assert.NotContains(t, res[4], '"')
	assert.NotContains(t, res, "inbetween1")
	assert.NotContains(t, res, "./1.svg")

	// Windows style paths
	input = ` some preamble
 c:/users/jdoe/one.png inbetween1 c:/program files/someplace/two.jpg inbetween2 
 /absolute/nospace/three.jpeg inbetween3 /absolute/with space/four.png inbetween4
./relative\ path/five.JPG inbetween5 "./relative with/spaces/six.png inbetween6
d:\path with\spaces\seven.JPEG inbetween7 c:\users\jdoe\eight.png inbetween8 
 d:\program files\someplace\nine.png inbetween9 "E:\program files\someplace\ten.PNG some ending
`
	res = extractFileNames(input)
	assert.Len(t, res, 10)
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
}
