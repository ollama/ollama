package convert

import (
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func createTokenizerFS(t *testing.T, dir string, files map[string]io.Reader) fs.FS {
	t.Helper()

	for k, v := range files {
		if err := func() error {
			f, err := os.Create(filepath.Join(dir, k))
			if err != nil {
				return err
			}
			defer f.Close()

			if _, err := io.Copy(f, v); err != nil {
				return err
			}

			return nil
		}(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	return os.DirFS(dir)
}

func TestParseTokenizer(t *testing.T) {
	cases := []struct {
		name              string
		fsys              fs.FS
		specialTokenTypes []string
		want              *Tokenizer
	}{
		{
			name: "string chat template",
			fsys: createTokenizerFS(t, t.TempDir(), map[string]io.Reader{
				"tokenizer.json": strings.NewReader(`{}`),
				"tokenizer_config.json": strings.NewReader(`{
					"chat_template": "<default template>"
				}`),
			}),
			want: &Tokenizer{
				Vocabulary: &Vocabulary{Model: "gpt2"},
				Pre:        "default",
				Template:   "<default template>",
			},
		},
		{
			name: "list chat template",
			fsys: createTokenizerFS(t, t.TempDir(), map[string]io.Reader{
				"tokenizer.json": strings.NewReader(`{}`),
				"tokenizer_config.json": strings.NewReader(`{
					"chat_template": [
						{
							"name": "default",
							"template": "<default template>"
						},
						{
							"name": "tools",
							"template": "<tools template>"
						}
					]
				}`),
			}),
			want: &Tokenizer{
				Vocabulary: &Vocabulary{Model: "gpt2"},
				Pre:        "default",
				Template:   "<default template>",
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer, err := parseTokenizer(tt.fsys, tt.specialTokenTypes)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if diff := cmp.Diff(tt.want, tokenizer); diff != "" {
				t.Errorf("unexpected tokenizer (-want +got):\n%s", diff)
			}
		})
	}
}
