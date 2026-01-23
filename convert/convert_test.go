package convert

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"golang.org/x/exp/maps"

	"github.com/ollama/ollama/llm"
)

func convertFull(t *testing.T, fsys fs.FS) (*os.File, llm.KV, llm.Tensors) {
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), "f16")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := Convert(fsys, f); err != nil {
		t.Fatal(err)
	}

	r, err := os.Open(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })

	m, _, err := llm.DecodeGGML(r, math.MaxInt)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := r.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}

	return r, m.KV(), m.Tensors()
}

func TestMain(m *testing.M) {
	var level slog.Level
	flag.TextVar(&level, "level", slog.LevelInfo, "log level")
	flag.Parse()
	slog.SetLogLoggerLevel(level)
	os.Exit(m.Run())
}

func TestConvertFull(t *testing.T) {
	cases := []string{
		"Meta-Llama-3-8B-Instruct",
		"Mistral-7B-Instruct-v0.2",
		"Mixtral-8x7B-Instruct-v0.1",
		"gemma-2b-it",
	}

	for i := range cases {
		tt := cases[i]
		t.Run(tt, func(t *testing.T) {
			t.Parallel()

			p := filepath.Join("testdata", tt)
			if testing.Short() {
				t.Skip("skipping in short mode")
			} else if _, err := os.Stat(p); err != nil {
				t.Skipf("%s not found", p)
			}

			f, kv, tensors := convertFull(t, os.DirFS(p))
			actual := make(map[string]string)
			for k, v := range kv {
				if s, ok := v.(json.Marshaler); !ok {
					actual[k] = fmt.Sprintf("%v", v)
				} else {
					bts, err := json.Marshal(s)
					if err != nil {
						t.Fatal(err)
					}

					actual[k] = fmt.Sprintf("%x", sha256.Sum256(bts))
				}
			}

			for _, tensor := range tensors.Items {
				sha256sum := sha256.New()
				sr := io.NewSectionReader(f, int64(tensors.Offset+tensor.Offset), int64(tensor.Size()))
				if _, err := io.Copy(sha256sum, sr); err != nil {
					t.Fatal(err)
				}

				actual[tensor.Name] = hex.EncodeToString(sha256sum.Sum(nil))
			}

			expectFile, err := os.Open(filepath.Join("testdata", fmt.Sprintf("%s.json", tt)))
			if err != nil {
				t.Fatal(err)
			}

			var expect map[string]string
			if err := json.NewDecoder(expectFile).Decode(&expect); err != nil {
				t.Fatal(err)
			}

			keys := maps.Keys(expect)
			slices.Sort(keys)
			for _, k := range keys {
				if v, ok := actual[k]; !ok {
					t.Errorf("missing %s", k)
				} else if v != expect[k] {
					t.Errorf("unexpected %s: want %s, got %s", k, expect[k], v)
				}
			}
		})
	}
}
