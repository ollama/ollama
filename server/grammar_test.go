package server

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestSurgeryOnGrape(t *testing.T) {
	testdata := filepath.Join("testdata", "grammar")
	ls, err := os.ReadDir(testdata)
	if err != nil {
		t.Fatal(err)
	}
	type testcase struct {
		filename string
		lang     string
		before   string
		after    string
		blocks   []string
	}
	var cases []testcase
	for _, f := range ls {
		if f.IsDir() || !strings.HasSuffix(f.Name(), ".txt") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(testdata, f.Name()))
		if err != nil {
			t.Fatal(err)
		}
		s := strings.Split(string(b), "\n^^^\n")
		if len(s) < 3 {
			t.Fatalf("incomplete test file %s", f.Name())
		}
		cases = append(cases, testcase{
			filename: f.Name(),
			lang:     s[0],
			before:   strings.TrimSpace(s[1]),
			after:    strings.TrimSpace(s[2]),
			blocks:   s[3:],
		})
	}
	for _, tc := range cases {
		t.Run(tc.filename, func(t *testing.T) {
			after, blocks := surgeryOnGrape(tc.before, tc.lang)
			if diff := cmp.Diff(after, tc.after); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}
			if len(blocks) != len(tc.blocks) {
				t.Fatalf("expected %d blocks, got %d", len(tc.blocks), len(blocks))
			}
			for i := range blocks {
				want := strings.TrimSpace(tc.blocks[i])
				if diff := cmp.Diff(blocks[i], want); diff != "" {
					t.Errorf("mismatch (-got +want) at block i=%d:\n%s", i, diff)
				}
			}
		})
	}
}
