package imagegen

import "testing"

// Ensure a parent directory that itself contains a valid image extension
// (e.g. "vacation.png.bak/") doesn't fracture the real path into two
// nonexistent fragments, while paths to genuinely separate images in the
// same message are still captured separately. See cmd/interactive_test.go
// for the equivalent test against the CLI's copy of this function.
func TestExtractFilenamesDoesNotFractureOnEmbeddedExtension(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  []string
	}{
		{
			name:  "extension-like directory name",
			input: "/Users/alex/vacation.png.bak/beach.jpg describe this photo",
			want:  []string{"/Users/alex/vacation.png.bak/beach.jpg"},
		},
		{
			name:  "windows drive path with embedded extension in dir name",
			input: `look at C:\Users\alex\shots.folder.png\final.jpg now`,
			want:  []string{`C:\Users\alex\shots.folder.png\final.jpg`},
		},
		{
			name:  "two genuinely separate images stay separate",
			input: "compare /Users/alex/cat.png and /Users/alex/dog.jpg please",
			want:  []string{"/Users/alex/cat.png", "/Users/alex/dog.jpg"},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := extractFileNames(c.input)
			if len(got) != len(c.want) {
				t.Fatalf("got %d matches %#v, want %d %#v", len(got), got, len(c.want), c.want)
			}
			for i := range got {
				if got[i] != c.want[i] {
					t.Errorf("match %d = %q, want %q", i, got[i], c.want[i])
				}
			}
		})
	}
}
