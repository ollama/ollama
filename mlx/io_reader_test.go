//go:build cgo

package mlx

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

type testSafetensorsReader struct {
	*bytes.Reader
	size   int64
	name   string
	closed atomic.Bool
}

func newTestSafetensorsReader(data []byte) *testSafetensorsReader {
	return &testSafetensorsReader{Reader: bytes.NewReader(data), size: int64(len(data))}
}

func (r *testSafetensorsReader) Size() int64 {
	return r.size
}

func (r *testSafetensorsReader) Name() string {
	return r.name
}

func (r *testSafetensorsReader) Close() error {
	r.closed.Store(true)
	return nil
}

func TestLoadSafetensorsReader(t *testing.T) {
	payload := []byte{3, 1, 4, 1}
	path := filepath.Join(t.TempDir(), "fixture.safetensors")
	var reader *testSafetensorsReader

	withMLXThread(t, func(mt *mlxthreadtest.T) {
		Scoped(func() {
			data := safetensorsFixture(mt, path, payload)
			reader = newTestSafetensorsReader(data)
			safetensors, err := LoadSafetensors(reader)
			if err != nil {
				mt.Fatalf("LoadSafetensors error: %v", err)
			}
			defer safetensors.Free()

			arr := safetensors.Get("tensor")
			if arr == nil {
				mt.Fatal("tensor not found")
			}
			arr32 := arr.AsType(DTypeInt32)
			Eval(arr32)
			if got, want := arr32.Ints(), []int32{3, 1, 4, 1}; !slices.Equal(got, want) {
				mt.Fatalf("values = %v, want %v", got, want)
			}
		})
	})
	if !reader.closed.Load() {
		t.Fatal("reader was not closed after loaded arrays were released")
	}
}

func TestLoadSafetensorsRejectsNilReader(t *testing.T) {
	if _, err := LoadSafetensors(nil); err == nil {
		t.Fatal("LoadSafetensors accepted a nil reader")
	}
}

func TestLoadSafetensorsReaderRejectsShortRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fixture.safetensors")
	var reader *testSafetensorsReader

	withMLXThread(t, func(mt *mlxthreadtest.T) {
		data := safetensorsFixture(mt, path, []byte{3, 1, 4, 1})
		reader = newTestSafetensorsReader(data[:4])
		reader.size = int64(len(data))
		reader.name = path
		if _, err := LoadSafetensors(reader); err == nil || !strings.Contains(err.Error(), path) {
			mt.Fatalf("LoadSafetensors error = %v, want the source filename", err)
		}
	})
	if !reader.closed.Load() {
		t.Fatal("reader was not closed after a load error")
	}
}

func safetensorsFixture(t *mlxthreadtest.T, path string, data []byte) []byte {
	t.Helper()

	arr := FromValues(data, len(data))
	Eval(arr)
	if err := SaveSafetensors(path, map[string]*Array{"tensor": arr}); err != nil {
		t.Fatalf("SaveSafetensors error: %v", err)
	}
	fixture, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read safetensors fixture: %v", err)
	}
	return fixture
}
