package fileutil

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

// NormalizeFilePath unescapes shell escape sequences and URL-encoded spaces.
func NormalizeFilePath(fp string) string {
	return strings.NewReplacer(
		"\\ ", " ",
		"\\(", "(",
		"\\)", ")",
		"\\[", "[",
		"\\]", "]",
		"\\{", "{",
		"\\}", "}",
		"\\$", "$",
		"\\&", "&",
		"\\;", ";",
		"\\'", "'",
		"\\\\", "\\",
		"\\*", "*",
		"\\?", "?",
		"\\~", "~",
		"%20", " ",
	).Replace(fp)
}

// ExtractFileNames extracts file paths from input string using a regex pattern.
func ExtractFileNames(input string) []string {
	regexPattern := `(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|webp|wav)\b`
	re := regexp.MustCompile(regexPattern)
	return re.FindAllString(input, -1)
}

// GetImageData reads and validates image/audio data from a file.
func GetImageData(filePath string) ([]byte, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	buf := make([]byte, 512)
	_, err = file.Read(buf)
	if err != nil {
		return nil, err
	}

	contentType := http.DetectContentType(buf)
	allowedTypes := []string{"image/jpeg", "image/jpg", "image/png", "image/webp", "audio/wave"}
	if !slices.Contains(allowedTypes, contentType) {
		return nil, fmt.Errorf("invalid file type: %s", contentType)
	}

	info, err := file.Stat()
	if err != nil {
		return nil, err
	}

	var maxSize int64 = 100 * 1024 * 1024
	if info.Size() > maxSize {
		return nil, errors.New("file size exceeds maximum limit (100MB)")
	}

	buf = make([]byte, info.Size())
	_, err = file.Seek(0, io.SeekStart)
	if err != nil {
		return nil, err
	}

	_, err = io.ReadFull(file, buf)
	if err != nil {
		return nil, err
	}

	return buf, nil
}

// GetMediaType returns the media type label for display purposes.
func GetMediaType(ext string) string {
	switch strings.ToLower(ext) {
	case ".wav":
		return "audio"
	default:
		return "image"
	}
}
