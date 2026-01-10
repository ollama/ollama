package server

import (
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	gocmp "github.com/google/go-cmp/cmp"
	gocmpopts "github.com/google/go-cmp/cmp/cmpopts"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/convert"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/types/model"
)

var stream bool = false

func createBinFile(t *testing.T, kv map[string]any, ti []*ggml.Tensor) (string, string) {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", cmp.Or(os.Getenv("OLLAMA_MODELS"), t.TempDir()))

	modelDir := envconfig.Models()

	f, err := os.CreateTemp(t.TempDir(), "")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	var base convert.KV = map[string]any{"general.architecture": "test"}
	maps.Copy(base, kv)

	if err := ggml.WriteGGUF(f, base, ti); err != nil {
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

	if diff := gocmp.Diff(expect, actual, gocmpopts.SortSlices(strings.Compare), gocmpopts.EquateEmpty()); diff != "" {
		t.Errorf("file exists mismatch (-want +got):\n%s", diff)
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
		filepath.Join(p, "blobs", "sha256-6bcdb8859d417753645538d7bbfbd7ca91a3f0c191aef5379c53c05e86b669dd"),
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
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
		filepath.Join(p, "blobs", "sha256-6bcdb8859d417753645538d7bbfbd7ca91a3f0c191aef5379c53c05e86b669dd"),
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
	})
}

func TestCreateFromModelInheritsRendererParser(t *testing.T) {
	gin.SetMode(gin.TestMode)

	p := t.TempDir()
	t.Setenv("OLLAMA_MODELS", p)
	var s Server

	const (
		renderer = "custom-renderer"
		parser   = "custom-parser"
	)

	_, digest := createBinFile(t, nil, nil)

	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:     "base",
		Files:    map[string]string{"base.gguf": digest},
		Renderer: renderer,
		Parser:   parser,
		Stream:   &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:   "child",
		From:   "base",
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	manifest, err := ParseNamedManifest(model.ParseName("child"))
	if err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	if manifest.Config.Digest == "" {
		t.Fatalf("unexpected empty config digest for child manifest")
	}

	configPath, err := GetBlobsPath(manifest.Config.Digest)
	if err != nil {
		t.Fatalf("config blob path: %v", err)
	}

	cfgFile, err := os.Open(configPath)
	if err != nil {
		t.Fatalf("open config blob: %v", err)
	}
	defer cfgFile.Close()

	var cfg model.ConfigV2
	if err := json.NewDecoder(cfgFile).Decode(&cfg); err != nil {
		t.Fatalf("decode config: %v", err)
	}

	if cfg.Renderer != renderer {
		t.Fatalf("expected renderer %q, got %q", renderer, cfg.Renderer)
	}
	if cfg.Parser != parser {
		t.Fatalf("expected parser %q, got %q", parser, cfg.Parser)
	}
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
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
		filepath.Join(p, "blobs", "sha256-b507b9c2f6ca642bffcd06665ea7c91f235fd32daeefdf875a0f938db05fb315"),
		filepath.Join(p, "blobs", "sha256-f6e7e4b28e0b1d0c635f2d465bd248c5387c3e75b61a48c4374192b26d832a56"),
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
		filepath.Join(p, "blobs", "sha256-136bf7c76bac2ec09d6617885507d37829e04b41acc47687d45e512b544e893a"),
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
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
		filepath.Join(p, "blobs", "sha256-0a666d113e8e0a3d27e9c7bd136a0bdfb6241037db50729d81568451ebfdbde8"),
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
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
		filepath.Join(p, "blobs", "sha256-6bcdb8859d417753645538d7bbfbd7ca91a3f0c191aef5379c53c05e86b669dd"),
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
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
		filepath.Join(p, "blobs", "sha256-6d6e36c1f90fc7deefc33a7300aa21ad4b67c506e33ecdeddfafa98147e60bbf"),
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
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
		filepath.Join(p, "blobs", "sha256-6d6e36c1f90fc7deefc33a7300aa21ad4b67c506e33ecdeddfafa98147e60bbf"),
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
		filepath.Join(p, "blobs", "sha256-bbdce269dabe013033632238b4b2d1e02fac2f97787c5e895f4da84e09cccd5d"),
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
		filepath.Join(p, "blobs", "sha256-6d6e36c1f90fc7deefc33a7300aa21ad4b67c506e33ecdeddfafa98147e60bbf"),
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
		filepath.Join(p, "blobs", "sha256-9443591d14be23c1e33d101934d76ad03bdb0715fe0879e8b0d1819e7bb063dd"),
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
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
		filepath.Join(p, "blobs", "sha256-c84aee28f2af350596f674de51d2a802ea782653ef2930a21d48bd43d5cd5317"),
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
		filepath.Join(p, "blobs", "sha256-09cfac3e6a637e25cb41aa85c24c110dc17ba89634de7df141b564dd2da4168b"),
		filepath.Join(p, "blobs", "sha256-298baeaf6928a60cf666d88d64a1ba606feb43a2865687c39e40652e407bffc4"),
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
		filepath.Join(p, "blobs", "sha256-a60ecc9da299ec7ede453f99236e5577fd125e143689b646d9f0ddc9971bf4db"),
		filepath.Join(p, "blobs", "sha256-c84aee28f2af350596f674de51d2a802ea782653ef2930a21d48bd43d5cd5317"),
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
		filepath.Join(p, "blobs", "sha256-0a04d979734167da3b80811a1874d734697f366a689f3912589b99d2e86e7ad1"),
		filepath.Join(p, "blobs", "sha256-4c5f51faac758fecaff8db42f0b7382891a4d0c0bb885f7b86be88c814a7cc86"),
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
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

func TestCreateAndShowRemoteModel(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var s Server

	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:      "test",
		From:       "bob",
		RemoteHost: "https://ollama.com",
		Info: map[string]any{
			"capabilities":       []string{"completion", "tools", "thinking"},
			"model_family":       "gptoss",
			"context_length":     131072,
			"embedding_length":   2880,
			"quantization_level": "MXFP4",
			"parameter_size":     "20.9B",
		},
		Stream: &stream,
	})

	if w.Code != http.StatusOK {
		t.Fatalf("exected status code 200, actual %d", w.Code)
	}

	w = createRequest(t, s.ShowHandler, api.ShowRequest{Model: "test"})
	if w.Code != http.StatusOK {
		t.Fatalf("exected status code 200, actual %d", w.Code)
	}

	var resp api.ShowResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	expectedDetails := api.ModelDetails{
		ParentModel:       "",
		Format:            "",
		Family:            "gptoss",
		Families:          []string{"gptoss"},
		ParameterSize:     "20.9B",
		QuantizationLevel: "MXFP4",
	}

	if !reflect.DeepEqual(resp.Details, expectedDetails) {
		t.Errorf("model details: expected %#v, actual %#v", expectedDetails, resp.Details)
	}

	expectedCaps := []model.Capability{
		model.Capability("completion"),
		model.Capability("tools"),
		model.Capability("thinking"),
	}

	if !slices.Equal(resp.Capabilities, expectedCaps) {
		t.Errorf("capabilities: expected %#v, actual %#v", expectedCaps, resp.Capabilities)
	}

	v, ok := resp.ModelInfo["gptoss.context_length"]
	ctxlen := v.(float64)
	if !ok || int(ctxlen) != 131072 {
		t.Errorf("context len: expected %d, actual %d", 131072, int(ctxlen))
	}

	v, ok = resp.ModelInfo["gptoss.embedding_length"]
	embedlen := v.(float64)
	if !ok || int(embedlen) != 2880 {
		t.Errorf("embed len: expected %d, actual %d", 2880, int(embedlen))
	}

	fmt.Printf("resp = %#v\n", resp)
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
		filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
		filepath.Join(p, "blobs", "sha256-a762f214df0d96c9a7b82f96da98d99ceb2776c88e3ea7ffa09d1e5835516ec6"),
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
			filepath.Join(p, "blobs", "sha256-3322a0c650c758b7386ff55629d27d07c07b6c3d3515e259dc3e5598c41e9f4e"),
			filepath.Join(p, "blobs", "sha256-35360843d0c84fb1506952a131bbef13cd2bb4a541251f22535170c05b56e672"),
			filepath.Join(p, "blobs", "sha256-a56c12acca8068cb6c335e237da6643e8a802a92959a63ad5bd17828e3b5e9b0"),
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
			filepath.Join(p, "blobs", "sha256-6bcdb8859d417753645538d7bbfbd7ca91a3f0c191aef5379c53c05e86b669dd"),
			filepath.Join(p, "blobs", "sha256-89a2116c3a82d6a97f59f748d86ed4417214353fd178ee54df418fde32495fad"),
		})
	})
}

func TestDetectModelTypeFromFiles(t *testing.T) {
	t.Run("gguf file", func(t *testing.T) {
		_, digest := createBinFile(t, nil, nil)
		files := map[string]string{
			"model.gguf": digest,
		}

		modelType := detectModelTypeFromFiles(files)
		if modelType != "gguf" {
			t.Fatalf("expected model type 'gguf', got %q", modelType)
		}
	})

	t.Run("gguf file w/o extension", func(t *testing.T) {
		_, digest := createBinFile(t, nil, nil)
		files := map[string]string{
			fmt.Sprintf("%x", digest): digest,
		}

		modelType := detectModelTypeFromFiles(files)
		if modelType != "gguf" {
			t.Fatalf("expected model type 'gguf', got %q", modelType)
		}
	})

	t.Run("safetensors file", func(t *testing.T) {
		files := map[string]string{
			"model.safetensors": "sha256:abc123",
		}

		modelType := detectModelTypeFromFiles(files)
		if modelType != "safetensors" {
			t.Fatalf("expected model type 'safetensors', got %q", modelType)
		}
	})

	t.Run("unsupported file type", func(t *testing.T) {
		p := t.TempDir()
		t.Setenv("OLLAMA_MODELS", p)

		data := []byte("12345678")
		digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
		if err := os.MkdirAll(filepath.Join(p, "blobs"), 0o755); err != nil {
			t.Fatal(err)
		}

		f, err := os.Create(filepath.Join(p, "blobs", fmt.Sprintf("sha256-%s", strings.TrimPrefix(digest, "sha256:"))))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()

		if _, err := f.Write(data); err != nil {
			t.Fatal(err)
		}

		files := map[string]string{
			"model.bin": digest,
		}

		modelType := detectModelTypeFromFiles(files)
		if modelType != "" {
			t.Fatalf("expected empty model type for unsupported file, got %q", modelType)
		}
	})

	t.Run("file with less than 4 bytes", func(t *testing.T) {
		p := t.TempDir()
		t.Setenv("OLLAMA_MODELS", p)

		data := []byte("123")
		digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
		if err := os.MkdirAll(filepath.Join(p, "blobs"), 0o755); err != nil {
			t.Fatal(err)
		}

		f, err := os.Create(filepath.Join(p, "blobs", fmt.Sprintf("sha256-%s", strings.TrimPrefix(digest, "sha256:"))))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()

		if _, err := f.Write(data); err != nil {
			t.Fatal(err)
		}

		files := map[string]string{
			"noext": digest,
		}

		modelType := detectModelTypeFromFiles(files)
		if modelType != "" {
			t.Fatalf("expected empty model type for small file, got %q", modelType)
		}
	})
}
