package server

import (
	"archive/zip"
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func createZipFile(t *testing.T, name string) *os.File {
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), "")
	if err != nil {
		t.Fatal(err)
	}

	zf := zip.NewWriter(f)
	defer zf.Close()

	zh, err := zf.CreateHeader(&zip.FileHeader{Name: name})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := io.Copy(zh, bytes.NewReader([]byte(""))); err != nil {
		t.Fatal(err)
	}

	return f
}

func TestExtractFromZipFile(t *testing.T) {
	cases := []struct {
		name   string
		expect []string
		err    error
	}{
		{
			name:   "good",
			expect: []string{"good"},
		},
		{
			name:   strings.Join([]string{"path", "..", "to", "good"}, string(os.PathSeparator)),
			expect: []string{filepath.Join("to", "good")},
		},
		{
			name:   strings.Join([]string{"path", "..", "to", "..", "good"}, string(os.PathSeparator)),
			expect: []string{"good"},
		},
		{
			name:   strings.Join([]string{"path", "to", "..", "..", "good"}, string(os.PathSeparator)),
			expect: []string{"good"},
		},
		{
			name: strings.Join([]string{"..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "bad"}, string(os.PathSeparator)),
			err:  zip.ErrInsecurePath,
		},
		{
			name: strings.Join([]string{"path", "..", "..", "to", "bad"}, string(os.PathSeparator)),
			err:  zip.ErrInsecurePath,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			f := createZipFile(t, tt.name)
			defer f.Close()

			tempDir := t.TempDir()
			if err := extractFromZipFile(tempDir, f, func(api.ProgressResponse) {}); !errors.Is(err, tt.err) {
				t.Fatal(err)
			}

			var matches []string
			if err := filepath.Walk(tempDir, func(p string, fi os.FileInfo, err error) error {
				if err != nil {
					return err
				}

				if !fi.IsDir() {
					matches = append(matches, p)
				}

				return nil
			}); err != nil {
				t.Fatal(err)
			}

			var actual []string
			for _, match := range matches {
				rel, err := filepath.Rel(tempDir, match)
				if err != nil {
					t.Error(err)
				}

				actual = append(actual, rel)
			}

			if !slices.Equal(actual, tt.expect) {
				t.Fatalf("expected %d files, got %d", len(tt.expect), len(matches))
			}
		})
	}
}
