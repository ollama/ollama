package filedata

import (
	"path/filepath"
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
