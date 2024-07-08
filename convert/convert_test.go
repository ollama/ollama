package convert

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/stretchr/testify/assert"
	"golang.org/x/exp/maps"
)

func convertFull(t *testing.T, d string) (*os.File, llm.KV, llm.Tensors) {
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), "f16")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := Convert(d, f); err != nil {
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

			f, kv, tensors := convertFull(t, p)
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

				actual[tensor.Name] = fmt.Sprintf("%x", sha256sum.Sum(nil))
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

func TestConvertNPZ(t *testing.T) {
	cases := []string{
		"adapters.npz",
	}

	for _, fn := range cases {
		ts, err := parseNPZ(filepath.Join("testdata", fn))
		assert.NoError(t, err)
		assert.Len(t, ts, 16*2*2) // 16 layers, 2 tensors, 2 loras

		a := adapter{}

		for _, m := range ts {
			at := m.(adapterTensor)
			assert.Equal(t, filepath.Join("testdata", fn), at.path)
			assert.Equal(t, "F32", at.dtype) // only float32s supported
			assert.Len(t, at.tensorBase.shape, 2)
		}

		var ws io.WriteSeeker = &memWriter{}
		err = llm.WriteGGLA(ws, a.KV(nil), a.Tensors(ts))
		assert.NoError(t, err)

		mw := ws.(*memWriter)
		slog.Info(fmt.Sprintf("buffer len = %d", len(mw.buf)))
		assert.NotEmpty(t, mw.buf)
		rs := bytes.NewReader(mw.buf)
		ggml, _, err := llm.DecodeGGML(rs, len(mw.buf))
		assert.NoError(t, err, "decode ggml failed")
		assert.NotNil(t, ggml, "ggml was empty")

		kv := ggml.KV()
		assert.NotNil(t, kv, "lora KVs not found")

		r, ok := kv["r"]
		assert.Equal(t, true, ok, "lora rank not set")
		assert.Equal(t, uint32(8), r, "lora rank was incorrect")

		alpha, ok := kv["alpha"]
		assert.Equal(t, true, ok, "lora alpha not set")
		assert.Equal(t, uint32(160), alpha, "lora alpha value was incorrect")

		gts := ggml.Tensors()
		assert.NotNil(t, gts, "no tensors found")
		assert.Equal(t, len(ts), len(gts.Items))
	}
}

type memWriter struct {
	buf []byte
	pos int
}

func (m *memWriter) Write(p []byte) (n int, err error) {
	minCap := m.pos + len(p)
	if minCap > cap(m.buf) {
		buf2 := make([]byte, len(m.buf), minCap+len(p)) // add some extra
		copy(buf2, m.buf)
		m.buf = buf2
	}
	if minCap > len(m.buf) {
		m.buf = m.buf[:minCap]
	}
	copy(m.buf[m.pos:], p)
	m.pos += len(p)
	return len(p), nil
}

func (m *memWriter) Seek(offset int64, whence int) (int64, error) {
	newPos, offs := 0, int(offset)
	switch whence {
	case io.SeekStart:
		newPos = offs
	case io.SeekCurrent:
		newPos = m.pos + offs
	case io.SeekEnd:
		newPos = len(m.buf) + offs
	}
	if newPos < 0 {
		return 0, errors.New("negative result pos")
	}
	m.pos = newPos
	return int64(newPos), nil
}
