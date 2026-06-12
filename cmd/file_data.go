package cmd

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

	"github.com/ollama/ollama/api"
)

func normalizeFilePath(fp string) string {
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
	).Replace(fp)
}

func extractFileNames(input string) []string {
	regexPattern := `(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|webp|wav)\b`
	re := regexp.MustCompile(regexPattern)

	return re.FindAllString(input, -1)
}

func extractFileData(input string) (string, []api.ImageData, error) {
	filePaths := extractFileNames(input)
	var imgs []api.ImageData

	for _, fp := range filePaths {
		nfp := normalizeFilePath(fp)
		data, err := getImageData(nfp)
		if errors.Is(err, os.ErrNotExist) {
			continue
		} else if err != nil {
			fmt.Fprintf(os.Stderr, "Couldn't process file: %q\n", err)
			return "", imgs, err
		}
		ext := strings.ToLower(filepath.Ext(nfp))
		switch ext {
		case ".wav":
			fmt.Fprintf(os.Stderr, "Added audio '%s'\n", nfp)
		default:
			fmt.Fprintf(os.Stderr, "Added image '%s'\n", nfp)
		}
		input = strings.ReplaceAll(input, "'"+nfp+"'", "")
		input = strings.ReplaceAll(input, "'"+fp+"'", "")
		input = strings.ReplaceAll(input, fp, "")
		imgs = append(imgs, data)
	}
	return strings.TrimSpace(input), imgs, nil
}

func getImageData(filePath string) ([]byte, error) {
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
	_, err = file.Seek(0, 0)
	if err != nil {
		return nil, err
	}

	_, err = io.ReadFull(file, buf)
	if err != nil {
		return nil, err
	}

	return buf, nil
}
