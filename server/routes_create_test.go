package server

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

var stream bool = false

func createBinFile(t *testing.T) string {
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), "")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := binary.Write(f, binary.LittleEndian, []byte("GGUF")); err != nil {
		t.Fatal(err)
	}

	if err := binary.Write(f, binary.LittleEndian, uint32(3)); err != nil {
		t.Fatal(err)
	}

	if err := binary.Write(f, binary.LittleEndian, uint64(0)); err != nil {
		t.Fatal(err)
	}

	if err := binary.Write(f, binary.LittleEndian, uint64(0)); err != nil {
		t.Fatal(err)
	}

	return f.Name()
}

type responseRecorder struct {
	*httptest.ResponseRecorder
	http.CloseNotifier
}

func NewRecorder() *responseRecorder {
	return &responseRecorder{
		ResponseRecorder: httptest.NewRecorder(),
	}
}

func (t *responseRecorder) CloseNotify() <-chan bool {
	return make(chan bool)
}

func createRequest(t *testing.T, fn func(*gin.Context), body any) *httptest.ResponseRecorder {
	t.Helper()

	w := NewRecorder()
	c, _ := gin.CreateTestContext(w)

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(body); err != nil {
		t.Fatal(err)
	}

	c.Request = &http.Request{
		Body: io.NopCloser(&b),
	}

	fn(c)
	return w.ResponseRecorder
}

func checkFileExists(t *testing.T, p string, expect []string) {
	t.Helper()

	actual, err := filepath.Glob(p)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(actual, expect) {
		t.Fatalf("expected slices to be equal %v", actual)
	}
}

func TestCreateFromBin(t *testing.T) {
	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)

	var s Server
	w := createRequest(t, s.CreateModelHandler, api.CreateRequest{
		Name:      "test",
		Modelfile: fmt.Sprintf("FROM %s", createBinFile(t)),
		Stream:    &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
	})

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99"),
		filepath.Join(p, "blobs", "sha256-ca239d7bd8ea90e4a5d2e6bf88f8d74a47b14336e73eb4e18bed4dd325018116"),
	})
}

func TestCreateFromModel(t *testing.T) {
	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	var s Server

	w := createRequest(t, s.CreateModelHandler, api.CreateRequest{
		Name:      "test",
		Modelfile: fmt.Sprintf("FROM %s", createBinFile(t)),
		Stream:    &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
	})

	w = createRequest(t, s.CreateModelHandler, api.CreateRequest{
		Name:      "test2",
		Modelfile: "FROM test",
		Stream:    &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test2", "latest"),
	})

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99"),
		filepath.Join(p, "blobs", "sha256-ca239d7bd8ea90e4a5d2e6bf88f8d74a47b14336e73eb4e18bed4dd325018116"),
	})
}
