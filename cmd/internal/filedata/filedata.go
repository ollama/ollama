package filedata

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
)

type File struct {
	Path string
	Data api.ImageData
}

func NormalizePath(fp string) string {
	fp = strings.Trim(fp, "\"")
	fp = strings.NewReplacer(
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

	if u, err := url.Parse(fp); err == nil && strings.EqualFold(u.Scheme, "file") {
		return normalizeFileURL(u)
	} else if normalized, ok := normalizeMalformedFileURL(fp); ok {
		return normalized
	}

	return fp
}

func ExtractNames(input string) []string {
	regexPattern := `(?:file://\S+?\.(?i:jpg|jpeg|png|webp|wav)\b)|(?:(?:[a-zA-Z]:)?(?:\./|\.\\|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|webp|wav)\b)`
	re := regexp.MustCompile(regexPattern)

	return re.FindAllString(input, -1)
}

func Extract(input string) (string, []api.ImageData, error) {
	cleaned, files, err := ExtractWithFiles(input)
	if err != nil {
		return "", nil, err
	}
	data := make([]api.ImageData, 0, len(files))
	for _, file := range files {
		data = append(data, file.Data)
	}
	return cleaned, data, nil
}

func ExtractWithFiles(input string) (string, []File, error) {
	filePaths := ExtractNames(input)
	var files []File

	for _, fp := range filePaths {
		nfp := NormalizePath(fp)
		data, err := GetData(nfp)
		if errors.Is(err, os.ErrNotExist) {
			continue
		} else if err != nil {
			return "", files, fmt.Errorf("couldn't process file %q: %w", nfp, err)
		}
		input = strings.ReplaceAll(input, "'"+nfp+"'", "")
		input = strings.ReplaceAll(input, "'"+fp+"'", "")
		input = strings.ReplaceAll(input, `"`+nfp+`"`, "")
		input = strings.ReplaceAll(input, `"`+fp+`"`, "")
		input = strings.ReplaceAll(input, fp, "")
		files = append(files, File{Path: nfp, Data: data})
	}
	return strings.TrimSpace(input), files, nil
}

func GetData(filePath string) ([]byte, error) {
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

func Kind(path string) string {
	if strings.EqualFold(filepath.Ext(path), ".wav") {
		return "audio"
	}
	return "image"
}

func normalizeFileURL(u *url.URL) string {
	path := u.Path
	if unescaped, err := url.PathUnescape(path); err == nil {
		path = unescaped
	}
	host := u.Host
	if unescaped, err := url.PathUnescape(host); err == nil {
		host = unescaped
	}
	if len(host) >= 2 && host[1] == ':' && isASCIIAlpha(host[0]) {
		return filepath.Clean(filepath.FromSlash(host + path))
	}
	if len(path) >= 4 && path[0] == '/' && path[2] == ':' && isASCIIAlpha(path[1]) {
		path = path[1:]
	}
	if u.Host != "" && !strings.EqualFold(u.Host, "localhost") {
		return `\\` + u.Host + filepath.FromSlash(path)
	}
	return filepath.FromSlash(path)
}

func normalizeMalformedFileURL(raw string) (string, bool) {
	const prefix = "file://"
	if !strings.HasPrefix(strings.ToLower(raw), prefix) {
		return "", false
	}

	path := raw[len(prefix):]
	if unescaped, err := url.PathUnescape(path); err == nil {
		path = unescaped
	}
	path = strings.TrimPrefix(path, "localhost")
	if len(path) >= 3 && path[0] == '/' && path[2] == ':' && isASCIIAlpha(path[1]) {
		path = path[1:]
	}
	if len(path) >= 2 && path[1] == ':' && isASCIIAlpha(path[0]) {
		return filepath.Clean(filepath.FromSlash(path)), true
	}
	return "", false
}

func isASCIIAlpha(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}
