package filedata

import (
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNormalizePathMalformedWindowsFileURL(t *testing.T) {
	got := NormalizePath(`file://C:%5CUsers%5Cjdoe%5CPictures%5Cimg.png`)
	want := filepath.Clean(`C:\Users\jdoe\Pictures\img.png`)
	if got != want {
		t.Fatalf("path = %q, want %q", got, want)
	}
}

func TestNormalizePathTwoSlashWindowsFileURL(t *testing.T) {
	got := NormalizePath(`file://C:/Users/jdoe/Pictures/img.png`)
	want := filepath.Clean(`C:/Users/jdoe/Pictures/img.png`)
	if got != want {
		t.Fatalf("path = %q, want %q", got, want)
	}
}

func TestNormalizePathLocalhostWindowsFileURL(t *testing.T) {
	got := NormalizePath(`file://localhost/C:/Users/jdoe/Pictures/img.png`)
	want := filepath.Clean(`C:/Users/jdoe/Pictures/img.png`)
	if got != want {
		t.Fatalf("path = %q, want %q", got, want)
	}
}

func TestExtractNames(t *testing.T) {
	// Unix style paths
	input := ` some preamble
 ./relative\ path/one.png inbetween1 ./not a valid two.jpg inbetween2 ./1.svg
/unescaped space /three.jpeg inbetween3 /valid\ path/dir/four.png "./quoted with spaces/five.JPG
/unescaped space /six.webp inbetween6 /valid\ path/dir/seven.WEBP`
	res := ExtractNames(input)
	if len(res) != 7 {
		t.Fatalf("len = %d, want 7", len(res))
	}
	assertContains(t, res[0], "one.png")
	assertContains(t, res[1], "two.jpg")
	assertContains(t, res[2], "three.jpeg")
	assertContains(t, res[3], "four.png")
	assertContains(t, res[4], "five.JPG")
	assertContains(t, res[5], "six.webp")
	assertContains(t, res[6], "seven.WEBP")
	assertNotContains(t, res[4], "\"")
	for _, r := range res {
		assertNotContains(t, r, "inbetween1")
	}
	assertNotContainsSlice(t, res, "./1.svg")
}

func TestExtractNamesWindowsPaths(t *testing.T) {
	input := ` some preamble
 c:/users/jdoe/one.png inbetween1 c:/program files/someplace/two.jpg inbetween2
 /absolute/nospace/three.jpeg inbetween3 /absolute/with space/four.png inbetween4
./relative\ path/five.JPG inbetween5 "./relative with/spaces/six.png inbetween6
d:\path with\spaces\seven.JPEG inbetween7 c:\users\jdoe\eight.png inbetween8
 d:\program files\someplace\nine.png inbetween9 "E:\program files\someplace\ten.PNG
c:/users/jdoe/eleven.webp inbetween11 c:/program files/someplace/twelve.WebP inbetween12
d:\path with\spaces\thirteen.WEBP some ending
`
	res := ExtractNames(input)
	if len(res) != 13 {
		t.Fatalf("len = %d, want 13", len(res))
	}
	assertNotContainsSlice(t, res, "inbetween2")
	assertContains(t, res[0], "one.png")
	assertContains(t, res[0], "c:")
	assertContains(t, res[1], "two.jpg")
	assertContains(t, res[1], "c:")
	assertContains(t, res[2], "three.jpeg")
	assertContains(t, res[3], "four.png")
	assertContains(t, res[4], "five.JPG")
	assertContains(t, res[5], "six.png")
	assertContains(t, res[6], "seven.JPEG")
	assertContains(t, res[6], "d:")
	assertContains(t, res[7], "eight.png")
	assertContains(t, res[7], "c:")
	assertContains(t, res[8], "nine.png")
	assertContains(t, res[8], "d:")
	assertContains(t, res[9], "ten.PNG")
	assertContains(t, res[9], "E:")
	assertContains(t, res[10], "eleven.webp")
	assertContains(t, res[10], "c:")
	assertContains(t, res[11], "twelve.WebP")
	assertContains(t, res[11], "c:")
	assertContains(t, res[12], "thirteen.WEBP")
	assertContains(t, res[12], "d:")
}

func TestExtractNamesDragDropPaths(t *testing.T) {
	input := `file:///Users/jdoe/Pictures/one.png file://localhost/C:/Users/jdoe/Pictures/two.webp file:///C:/Users/jdoe/Pictures/three.jpg .\relative\four.png`
	res := ExtractNames(input)
	if len(res) != 4 {
		t.Fatalf("len = %d, want 4", len(res))
	}
	assertContains(t, res[0], "file:///Users/jdoe/Pictures/one.png")
	assertContains(t, res[1], "file://localhost/C:/Users/jdoe/Pictures/two.webp")
	assertContains(t, res[2], "file:///C:/Users/jdoe/Pictures/three.jpg")
	assertContains(t, res[3], `.\relative\four.png`)
}

func TestNormalizePathFileURL(t *testing.T) {
	got := NormalizePath("file:///C:/Users/jdoe/Pictures/img.png")
	want := filepath.FromSlash("C:/Users/jdoe/Pictures/img.png")
	if got != want {
		t.Fatalf("path = %q, want %q", got, want)
	}
}

func TestExtractRemovesQuotedFilepath(t *testing.T) {
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
	cleaned, imgs, err := Extract(input)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(imgs) != 1 {
		t.Fatalf("imgs = %d, want 1", len(imgs))
	}
	if cleaned != "before  after" {
		t.Fatalf("cleaned = %q, want %q", cleaned, "before  after")
	}
}

func TestExtractFileURL(t *testing.T) {
	dir := t.TempDir()
	fp := filepath.Join(dir, "img.png")
	data := make([]byte, 600)
	copy(data, []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
	if err := os.WriteFile(fp, data, 0o600); err != nil {
		t.Fatalf("failed to write test image: %v", err)
	}

	fileURL := (&url.URL{Scheme: "file", Path: fp}).String()
	cleaned, imgs, err := Extract("before " + fileURL + " after")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(imgs) != 1 {
		t.Fatalf("imgs = %d, want 1", len(imgs))
	}
	if cleaned != "before  after" {
		t.Fatalf("cleaned = %q, want %q", cleaned, "before  after")
	}
}

func TestExtractWAV(t *testing.T) {
	dir := t.TempDir()
	fp := filepath.Join(dir, "sample.wav")
	data := make([]byte, 600)
	copy(data[:44], []byte{
		'R', 'I', 'F', 'F',
		0x58, 0x02, 0x00, 0x00,
		'W', 'A', 'V', 'E',
		'f', 'm', 't', ' ',
		0x10, 0x00, 0x00, 0x00,
		0x01, 0x00,
		0x01, 0x00,
		0x80, 0x3e, 0x00, 0x00,
		0x00, 0x7d, 0x00, 0x00,
		0x02, 0x00,
		0x10, 0x00,
		'd', 'a', 't', 'a',
		0x34, 0x02, 0x00, 0x00,
	})
	if err := os.WriteFile(fp, data, 0o600); err != nil {
		t.Fatalf("failed to write test audio: %v", err)
	}

	input := "before " + fp + " after"
	cleaned, imgs, err := Extract(input)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(imgs) != 1 {
		t.Fatalf("imgs = %d, want 1", len(imgs))
	}
	if cleaned != "before  after" {
		t.Fatalf("cleaned = %q, want %q", cleaned, "before  after")
	}
}

func assertContains(t *testing.T, s, want string) {
	t.Helper()
	if !strings.Contains(s, want) {
		t.Fatalf("%q does not contain %q", s, want)
	}
}

func assertNotContains(t *testing.T, s, want string) {
	t.Helper()
	if strings.Contains(s, want) {
		t.Fatalf("%q unexpectedly contains %q", s, want)
	}
}

func assertNotContainsSlice(t *testing.T, ss []string, want string) {
	t.Helper()
	for _, s := range ss {
		if strings.Contains(s, want) {
			t.Fatalf("slice unexpectedly contains %q in %q", want, s)
		}
	}
}
