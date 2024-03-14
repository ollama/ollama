package server

import (
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestFixBlobs(t *testing.T) {
	cases := []struct {
		path string
		want string
	}{
		{path: "sha256:1234", want: "sha256-1234"},
		{path: "sha259:5678", want: "sha259:5678"},
		{path: "sha256:abcd", want: "sha256-abcd"},
		{path: "x/y/sha256:abcd", want: "x/y/sha256-abcd"},
		{path: "x:y/sha256:abcd", want: "x:y/sha256-abcd"},
	}

	for _, c := range cases {
		t.Run(c.path, func(t *testing.T) {
			if strings.Contains(c.path, ":") && runtime.GOOS == "windows" {
				t.Skip("skipping test on windows")
			}

			rootDir := t.TempDir()
			fullPath := filepath.Join(rootDir, c.path)
			fullDir, _ := filepath.Split(fullPath)

			t.Logf("creating dir %s", fullDir)
			if err := os.MkdirAll(fullDir, 0o755); err != nil {
				t.Fatal(err)
			}

			t.Logf("writing file %s", fullPath)
			if err := os.WriteFile(fullPath, nil, 0o644); err != nil {
				t.Fatal(err)
			}

			if err := fixBlobs(rootDir); err != nil {
				t.Fatal(err)
			}

			got := slurpFiles(os.DirFS(rootDir))
			want := []string{c.want}
			if !slices.Equal(got, want) {
				t.Fatalf("got = %v, want %v", got, want)
			}
		})
	}
}

func slurpFiles(fsys fs.FS) []string {
	var sfs []string
	fn := func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		sfs = append(sfs, path)
		return nil
	}
	if err := fs.WalkDir(fsys, ".", fn); err != nil {
		panic(err)
	}
	return sfs
}
