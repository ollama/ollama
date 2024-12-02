package server

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
)

var stream bool = false

func createBinFile(t *testing.T, kv map[string]any, ti []ggml.Tensor) (string, string) {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", cmp.Or(os.Getenv("OLLAMA_MODELS"), t.TempDir()))

	modelDir := envconfig.Models()

	f, err := os.CreateTemp(t.TempDir(), "")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := ggml.WriteGGUF(f, kv, ti); err != nil {
		t.Fatal(err)
	}
	// Calculate sha256 of file
	if _, err := f.Seek(0, 0); err != nil {
		t.Fatal(err)
	}

	digest, _ := GetSHA256Digest(f)
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	if err := createLink(f.Name(), filepath.Join(modelDir, "blobs", fmt.Sprintf("sha256-%s", strings.TrimPrefix(digest, "sha256:")))); err != nil {
		t.Fatal(err)
	}

	return f.Name(), digest
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
	// if OLLAMA_MODELS is not set, set it to the temp directory
	t.Setenv("OLLAMA_MODELS", cmp.Or(os.Getenv("OLLAMA_MODELS"), t.TempDir()))

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
	gin.SetMode(gin.TestMode)

	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)

	var s Server

	_, digest := createBinFile(t, nil, nil)

	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:   "test",
		Files:  map[string]string{"test.gguf": digest},
		Stream: &stream,
	})

	if w.Code != http.StatusOK {
		fmt.Println(w)
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
	gin.SetMode(gin.TestMode)

	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	var s Server

	_, digest := createBinFile(t, nil, nil)

	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:   "test",
		Files:  map[string]string{"test.gguf": digest},
		Stream: &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
	})

	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:   "test2",
		From:   "test",
		Stream: &stream,
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

func TestCreateRemovesLayers(t *testing.T) {
	gin.SetMode(gin.TestMode)

	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	var s Server

	_, digest := createBinFile(t, nil, nil)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:     "test",
		Files:    map[string]string{"test.gguf": digest},
		Template: "{{ .Prompt }}",
		Stream:   &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
	})

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99"),
		filepath.Join(p, "blobs", "sha256-b507b9c2f6ca642bffcd06665ea7c91f235fd32daeefdf875a0f938db05fb315"),
		filepath.Join(p, "blobs", "sha256-bc80b03733773e0728011b2f4adf34c458b400e1aad48cb28d61170f3a2ad2d6"),
	})

	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:     "test",
		Files:    map[string]string{"test.gguf": digest},
		Template: "{{ .System }} {{ .Prompt }}",
		Stream:   &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
	})

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-8f2c2167d789c6b2302dff965160fa5029f6a24096d262c1cbb469f21a045382"),
		filepath.Join(p, "blobs", "sha256-a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99"),
		filepath.Join(p, "blobs", "sha256-fe7ac77b725cda2ccad03f88a880ecdfd7a33192d6cae08fce2c0ee1455991ed"),
	})
}

func TestCreateUnsetsSystem(t *testing.T) {
	gin.SetMode(gin.TestMode)

	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	var s Server

	_, digest := createBinFile(t, nil, nil)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:   "test",
		Files:  map[string]string{"test.gguf": digest},
		System: "Say hi!",
		Stream: &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
	})

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-8585df945d1069bc78b79bd10bb73ba07fbc29b0f5479a31a601c0d12731416e"),
		filepath.Join(p, "blobs", "sha256-a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99"),
		filepath.Join(p, "blobs", "sha256-f29e82a8284dbdf5910b1555580ff60b04238b8da9d5e51159ada67a4d0d5851"),
	})

	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:   "test",
		Files:  map[string]string{"test.gguf": digest},
		System: "",
		Stream: &stream,
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

func TestCreateMergeParameters(t *testing.T) {
	gin.SetMode(gin.TestMode)

	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	var s Server

	_, digest := createBinFile(t, nil, nil)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:  "test",
		Files: map[string]string{"test.gguf": digest},
		Parameters: map[string]any{
			"temperature": 1,
			"top_k":       10,
			"stop":        []string{"USER:", "ASSISTANT:"},
		},
		Stream: &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
	})

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-1d0ad71299d48c2fb7ae2b98e683643e771f8a5b72be34942af90d97a91c1e37"),
		filepath.Join(p, "blobs", "sha256-4a384beaf47a9cbe452dfa5ab70eea691790f3b35a832d12933a1996685bf2b6"),
		filepath.Join(p, "blobs", "sha256-a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99"),
	})

	// in order to merge parameters, the second model must be created FROM the first
	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Name: "test2",
		From: "test",
		Parameters: map[string]any{
			"temperature": 0.6,
			"top_p":       0.7,
		},
		Stream: &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test2", "latest"),
	})

	// Display contents of each blob in the directory
	blobDir := filepath.Join(p, "blobs")
	entries, err := os.ReadDir(blobDir)
	if err != nil {
		t.Fatalf("failed to read blobs directory: %v", err)
	}

	for _, entry := range entries {
		blobPath := filepath.Join(blobDir, entry.Name())
		content, err := os.ReadFile(blobPath)
		if err != nil {
			t.Fatalf("failed to read blob %s: %v", entry.Name(), err)
		}
		t.Logf("Contents of %s:\n%s", entry.Name(), string(content))
	}

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-1d0ad71299d48c2fb7ae2b98e683643e771f8a5b72be34942af90d97a91c1e37"),
		filepath.Join(p, "blobs", "sha256-4a384beaf47a9cbe452dfa5ab70eea691790f3b35a832d12933a1996685bf2b6"),
		filepath.Join(p, "blobs", "sha256-4cd9d4ba6b734d9b4cbd1e5caa60374c00722e993fce5e1e2d15a33698f71187"),
		filepath.Join(p, "blobs", "sha256-a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99"),
		filepath.Join(p, "blobs", "sha256-e29a7b3c47287a2489c895d21fe413c20f859a85d20e749492f52a838e36e1ba"),
	})

	actual, err := os.ReadFile(filepath.Join(p, "blobs", "sha256-e29a7b3c47287a2489c895d21fe413c20f859a85d20e749492f52a838e36e1ba"))
	if err != nil {
		t.Fatal(err)
	}

	expect, err := json.Marshal(map[string]any{"temperature": 0.6, "top_k": 10, "top_p": 0.7, "stop": []string{"USER:", "ASSISTANT:"}})
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(bytes.TrimSpace(expect), bytes.TrimSpace(actual)) {
		t.Errorf("expected %s, actual %s", string(expect), string(actual))
	}

	// slices are replaced
	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Name: "test2",
		From: "test",
		Parameters: map[string]any{
			"temperature": 0.6,
			"top_p":       0.7,
			"stop":        []string{"<|endoftext|>"},
		},
		Stream: &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test2", "latest"),
	})

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-12f58bb75cb3042d69a7e013ab87fb3c3c7088f50ddc62f0c77bd332f0d44d35"),
		filepath.Join(p, "blobs", "sha256-1d0ad71299d48c2fb7ae2b98e683643e771f8a5b72be34942af90d97a91c1e37"),
		filepath.Join(p, "blobs", "sha256-257aa726584f24970a4f240765e75a7169bfbe7f4966c1f04513d6b6c860583a"),
		filepath.Join(p, "blobs", "sha256-4a384beaf47a9cbe452dfa5ab70eea691790f3b35a832d12933a1996685bf2b6"),
		filepath.Join(p, "blobs", "sha256-a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99"),
	})

	actual, err = os.ReadFile(filepath.Join(p, "blobs", "sha256-12f58bb75cb3042d69a7e013ab87fb3c3c7088f50ddc62f0c77bd332f0d44d35"))
	if err != nil {
		t.Fatal(err)
	}

	expect, err = json.Marshal(map[string]any{"temperature": 0.6, "top_k": 10, "top_p": 0.7, "stop": []string{"<|endoftext|>"}})
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(bytes.TrimSpace(expect), bytes.TrimSpace(actual)) {
		t.Errorf("expected %s, actual %s", string(expect), string(actual))
	}
}

func TestCreateReplacesMessages(t *testing.T) {
	gin.SetMode(gin.TestMode)

	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	var s Server

	_, digest := createBinFile(t, nil, nil)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:  "test",
		Files: map[string]string{"test.gguf": digest},
		Messages: []api.Message{
			{
				Role:    "assistant",
				Content: "What is my purpose?",
			},
			{
				Role:    "user",
				Content: "You run tests.",
			},
			{
				Role:    "assistant",
				Content: "Oh, my god.",
			},
		},
		Stream: &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
	})

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-298baeaf6928a60cf666d88d64a1ba606feb43a2865687c39e40652e407bffc4"),
		filepath.Join(p, "blobs", "sha256-a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99"),
		filepath.Join(p, "blobs", "sha256-e0e27d47045063ccb167ae852c51d49a98eab33fabaee4633fdddf97213e40b5"),
	})

	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Name: "test2",
		From: "test",
		Messages: []api.Message{
			{
				Role:    "assistant",
				Content: "You're a test, Harry.",
			},
			{
				Role:    "user",
				Content: "I-I'm a what?",
			},
			{
				Role:    "assistant",
				Content: "A test. And a thumping good one at that, I'd wager.",
			},
		},
		Stream: &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test2", "latest"),
	})

	// Old layers will not have been pruned
	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-298baeaf6928a60cf666d88d64a1ba606feb43a2865687c39e40652e407bffc4"),
		filepath.Join(p, "blobs", "sha256-a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99"),
		filepath.Join(p, "blobs", "sha256-a60ecc9da299ec7ede453f99236e5577fd125e143689b646d9f0ddc9971bf4db"),
		filepath.Join(p, "blobs", "sha256-e0e27d47045063ccb167ae852c51d49a98eab33fabaee4633fdddf97213e40b5"),
		filepath.Join(p, "blobs", "sha256-f4e2c3690efef1b4b63ba1e1b2744ffeb6a7438a0110b86596069f6d9999c80b"),
	})

	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	f, err := os.Open(filepath.Join(p, "blobs", "sha256-a60ecc9da299ec7ede453f99236e5577fd125e143689b646d9f0ddc9971bf4db"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	var actual []message
	if err := json.NewDecoder(f).Decode(&actual); err != nil {
		t.Fatal(err)
	}

	expect := []message{
		{Role: "assistant", Content: "You're a test, Harry."},
		{Role: "user", Content: "I-I'm a what?"},
		{Role: "assistant", Content: "A test. And a thumping good one at that, I'd wager."},
	}

	if !slices.Equal(actual, expect) {
		t.Errorf("expected %s, actual %s", expect, actual)
	}
}

func TestCreateTemplateSystem(t *testing.T) {
	gin.SetMode(gin.TestMode)

	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	var s Server

	_, digest := createBinFile(t, nil, nil)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:     "test",
		Files:    map[string]string{"test.gguf": digest},
		Template: "{{ .System }} {{ .Prompt }}",
		System:   "Say bye!",
		Stream:   &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
	})

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-2b5e330885117c82f3fd75169ea323e141070a2947c11ddb9f79ee0b01c589c1"),
		filepath.Join(p, "blobs", "sha256-4c5f51faac758fecaff8db42f0b7382891a4d0c0bb885f7b86be88c814a7cc86"),
		filepath.Join(p, "blobs", "sha256-a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99"),
		filepath.Join(p, "blobs", "sha256-fe7ac77b725cda2ccad03f88a880ecdfd7a33192d6cae08fce2c0ee1455991ed"),
	})

	template, err := os.ReadFile(filepath.Join(p, "blobs", "sha256-fe7ac77b725cda2ccad03f88a880ecdfd7a33192d6cae08fce2c0ee1455991ed"))
	if err != nil {
		t.Fatal(err)
	}

	if string(template) != "{{ .System }} {{ .Prompt }}" {
		t.Errorf("expected \"{{ .System }} {{ .Prompt }}\", actual %s", template)
	}

	system, err := os.ReadFile(filepath.Join(p, "blobs", "sha256-4c5f51faac758fecaff8db42f0b7382891a4d0c0bb885f7b86be88c814a7cc86"))
	if err != nil {
		t.Fatal(err)
	}

	if string(system) != "Say bye!" {
		t.Errorf("expected \"Say bye!\", actual %s", system)
	}

	t.Run("incomplete template", func(t *testing.T) {
		_, digest := createBinFile(t, nil, nil)
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Name:     "test",
			Files:    map[string]string{"test.gguf": digest},
			Template: "{{ .Prompt",
			Stream:   &stream,
		})

		if w.Code != http.StatusBadRequest {
			t.Fatalf("expected status code 400, actual %d", w.Code)
		}
	})

	t.Run("template with unclosed if", func(t *testing.T) {
		_, digest := createBinFile(t, nil, nil)
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Name:     "test",
			Files:    map[string]string{"test.gguf": digest},
			Template: "{{ if .Prompt }}",
			Stream:   &stream,
		})

		if w.Code != http.StatusBadRequest {
			t.Fatalf("expected status code 400, actual %d", w.Code)
		}
	})

	t.Run("template with undefined function", func(t *testing.T) {
		_, digest := createBinFile(t, nil, nil)
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Name:     "test",
			Files:    map[string]string{"test.gguf": digest},
			Template: "{{ Prompt }}",
			Stream:   &stream,
		})

		if w.Code != http.StatusBadRequest {
			t.Fatalf("expected status code 400, actual %d", w.Code)
		}
	})
}

func TestCreateLicenses(t *testing.T) {
	gin.SetMode(gin.TestMode)

	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	var s Server

	_, digest := createBinFile(t, nil, nil)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:    "test",
		Files:   map[string]string{"test.gguf": digest},
		License: []string{"MIT", "Apache-2.0"},
		Stream:  &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	checkFileExists(t, filepath.Join(p, "manifests", "*", "*", "*", "*"), []string{
		filepath.Join(p, "manifests", "registry.ollama.ai", "library", "test", "latest"),
	})

	checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
		filepath.Join(p, "blobs", "sha256-2af71558e438db0b73a20beab92dc278a94e1bbe974c00c1a33e3ab62d53a608"),
		filepath.Join(p, "blobs", "sha256-79a39c37536ddee29cbadd5d5e2dcba8ed7f03e431f626ff38432c1c866bb7e2"),
		filepath.Join(p, "blobs", "sha256-a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99"),
		filepath.Join(p, "blobs", "sha256-e5dcffe836b6ec8a58e492419b550e65fb8cbdc308503979e5dacb33ac7ea3b7"),
	})

	mit, err := os.ReadFile(filepath.Join(p, "blobs", "sha256-e5dcffe836b6ec8a58e492419b550e65fb8cbdc308503979e5dacb33ac7ea3b7"))
	if err != nil {
		t.Fatal(err)
	}

	if string(mit) != "MIT" {
		t.Errorf("expected MIT, actual %s", mit)
	}

	apache, err := os.ReadFile(filepath.Join(p, "blobs", "sha256-2af71558e438db0b73a20beab92dc278a94e1bbe974c00c1a33e3ab62d53a608"))
	if err != nil {
		t.Fatal(err)
	}

	if string(apache) != "Apache-2.0" {
		t.Errorf("expected Apache-2.0, actual %s", apache)
	}
}

func TestCreateDetectTemplate(t *testing.T) {
	gin.SetMode(gin.TestMode)

	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	var s Server

	t.Run("matched", func(t *testing.T) {
		_, digest := createBinFile(t, ggml.KV{
			"tokenizer.chat_template": "{{ bos_token }}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}",
		}, nil)
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Name:   "test",
			Files:  map[string]string{"test.gguf": digest},
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Fatalf("expected status code 200, actual %d", w.Code)
		}

		checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
			filepath.Join(p, "blobs", "sha256-0d79f567714c62c048378f2107fb332dabee0135d080c302d884317da9433cc5"),
			filepath.Join(p, "blobs", "sha256-35360843d0c84fb1506952a131bbef13cd2bb4a541251f22535170c05b56e672"),
			filepath.Join(p, "blobs", "sha256-553c4a3f747b3d22a4946875f1cc8ed011c2930d83f864a0c7265f9ec0a20413"),
			filepath.Join(p, "blobs", "sha256-de3959f841e9ef6b4b6255fa41cb9e0a45da89c3066aa72bdd07a4747f848990"),
		})
	})

	t.Run("unmatched", func(t *testing.T) {
		_, digest := createBinFile(t, nil, nil)
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Name:   "test",
			Files:  map[string]string{"test.gguf": digest},
			Stream: &stream,
		})

		if w.Code != http.StatusOK {
			t.Fatalf("expected status code 200, actual %d", w.Code)
		}

		checkFileExists(t, filepath.Join(p, "blobs", "*"), []string{
			filepath.Join(p, "blobs", "sha256-a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99"),
			filepath.Join(p, "blobs", "sha256-ca239d7bd8ea90e4a5d2e6bf88f8d74a47b14336e73eb4e18bed4dd325018116"),
		})
	})
}
