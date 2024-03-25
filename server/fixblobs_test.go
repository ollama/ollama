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
		path []string
		want []string
	}{
		{path: []string{"sha256-1234"}, want: []string{"sha256-1234"}},
		{path: []string{"sha256:1234"}, want: []string{"sha256-1234"}},
		{path: []string{"sha259:5678"}, want: []string{"sha259:5678"}},
		{path: []string{"sha256:abcd"}, want: []string{"sha256-abcd"}},
		{path: []string{"x/y/sha256:abcd"}, want: []string{"x/y/sha256-abcd"}},
		{path: []string{"x:y/sha256:abcd"}, want: []string{"x:y/sha256-abcd"}},
		{path: []string{"x:y/sha256:abcd"}, want: []string{"x:y/sha256-abcd"}},
		{path: []string{"x:y/sha256:abcd", "sha256:1234"}, want: []string{"x:y/sha256-abcd", "sha256-1234"}},
		{path: []string{"x:y/sha256:abcd", "sha256-1234"}, want: []string{"x:y/sha256-abcd", "sha256-1234"}},
	}

	for _, tt := range cases {
		t.Run(strings.Join(tt.path, "|"), func(t *testing.T) {
			hasColon := slices.ContainsFunc(tt.path, func(s string) bool { return strings.Contains(s, ":") })
			if hasColon && runtime.GOOS == "windows" {
				t.Skip("skipping test on windows")
			}

			rootDir := t.TempDir()
			for _, path := range tt.path {
				fullPath := filepath.Join(rootDir, path)
				fullDir, _ := filepath.Split(fullPath)

				t.Logf("creating dir %s", fullDir)
				if err := os.MkdirAll(fullDir, 0o755); err != nil {
					t.Fatal(err)
				}

				t.Logf("writing file %s", fullPath)
				if err := os.WriteFile(fullPath, nil, 0o644); err != nil {
					t.Fatal(err)
				}
			}

			if err := fixBlobs(rootDir); err != nil {
				t.Fatal(err)
			}

			got := slurpFiles(os.DirFS(rootDir))

			slices.Sort(tt.want)
			slices.Sort(got)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("got = %v, want %v", got, tt.want)
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
