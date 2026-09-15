package client

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPrepareRemoteSourceFilesSelectsJSONAndSafetensors(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"config.json":                 `{"architectures":["TestModel"]}`,
		"model.safetensors":           "tensor-data",
		"tokenizer.model":             "not uploaded",
		"README.md":                   "not uploaded",
		"consolidated.00.safetensors": "unsupported safetensors",
		"chat_template.jinja":         "{{ messages }}",
		"tokenizer_config.json":       `{}`,
		"special_tokens_map.json":     `{}`,
		"generation_config.json":      `{}`,
		"preprocessor_config.json":    `{}`,
		"chat_template.json":          `{}`,
		"processor_config.json":       `{}`,
		"added_tokens.json":           `{}`,
		"tokenizer.json":              `{}`,
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	got, err := prepareRemoteSourceFiles(t.Context(), dir, false)
	if err != nil {
		t.Fatal(err)
	}

	var names []string
	for _, f := range got {
		names = append(names, f.logical)
		if f.digest == "" || f.size == 0 {
			t.Fatalf("file %s digest/size = %q/%d, want populated", f.logical, f.digest, f.size)
		}
		if f.draft {
			t.Fatalf("file %s marked draft", f.logical)
		}
	}
	slices.Sort(names)
	want := []string{
		"added_tokens.json",
		"chat_template.jinja",
		"chat_template.json",
		"config.json",
		"generation_config.json",
		"model.safetensors",
		"preprocessor_config.json",
		"processor_config.json",
		"special_tokens_map.json",
		"tokenizer.json",
		"tokenizer_config.json",
	}
	if !slices.Equal(names, want) {
		t.Fatalf("uploaded files = %v, want %v", names, want)
	}
}

func TestPrepareRemoteSourceFilesUsesIndexedShardNames(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"config.json":                  `{"architectures":["TestModel"]}`,
		"model.safetensors.index.json": `{"weight_map":{"model.weight":"weights-0.safetensors"}}`,
		"weights-0.safetensors":        "tensor-data",
		"unindexed.safetensors":        "not uploaded",
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	got, err := prepareRemoteSourceFiles(t.Context(), dir, false)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, file := range got {
		names = append(names, file.logical)
	}
	want := []string{"config.json", "model.safetensors.index.json", "weights-0.safetensors"}
	if !slices.Equal(names, want) {
		t.Fatalf("uploaded files = %v, want %v", names, want)
	}
}

func TestPrepareRemoteSourceFilesCanceled(t *testing.T) {
	dir := t.TempDir()
	for name, data := range map[string]string{
		"config.json":       `{"architectures":["TestModel"]}`,
		"model.safetensors": "tensor-data",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := prepareRemoteSourceFiles(ctx, dir, false); !errors.Is(err, context.Canceled) {
		t.Fatalf("prepareRemoteSourceFiles() error = %v, want context.Canceled", err)
	}
}

func TestNewRemoteCreateRequest(t *testing.T) {
	req := newRemoteCreateRequest(CreateOptions{
		ModelName:     "example",
		Quantize:      "nvfp4",
		DraftQuantize: "mxfp8",
		Modelfile: &ModelfileConfig{
			Template:   "{{ .Prompt }}",
			System:     "system",
			Licenses:   []string{"MIT", "Apache-2.0"},
			Parser:     "mf-parser",
			Renderer:   "mf-renderer",
			Requires:   "0.20.0",
			Parameters: map[string]any{"temperature": float32(0.1)},
			Messages:   []api.Message{{Role: "user", Content: "hello"}},
		},
	}, []remoteSourceFile{
		{logical: "config.json", digest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
		{logical: "model.safetensors", digest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
		{logical: "config.json", digest: "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", draft: true},
		{logical: "model.safetensors", digest: "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", draft: true},
	})

	if req.Model != "example" {
		t.Fatalf("request model = %q, want example", req.Model)
	}
	if req.Quantize != "nvfp4" || req.DraftQuantize != "mxfp8" {
		t.Fatalf("request quantize = %q/%q, want nvfp4/mxfp8", req.Quantize, req.DraftQuantize)
	}
	if req.Parser != "mf-parser" || req.Renderer != "mf-renderer" {
		t.Fatalf("parser/renderer = %q/%q, want mf-parser/mf-renderer", req.Parser, req.Renderer)
	}
	if req.Requires != "0.20.0" {
		t.Fatalf("Requires = %q, want 0.20.0", req.Requires)
	}
	if licenses, ok := req.License.([]string); !ok || !slices.Equal(licenses, []string{"MIT", "Apache-2.0"}) {
		t.Fatalf("License = %#v, want both licenses", req.License)
	}
	if req.Files["model.safetensors"] == "" || req.DraftFiles["model.safetensors"] == "" {
		t.Fatalf("files = %v draft_files = %v, want model entries", req.Files, req.DraftFiles)
	}
	if req.Info != nil {
		t.Fatalf("Info = %#v, want source metadata derived by server", req.Info)
	}
	if len(req.Messages) != 1 || req.Messages[0].Role != "user" || req.Messages[0].Content != "hello" {
		t.Fatalf("Messages = %#v, want one user message", req.Messages)
	}
}

func TestCreateModelRemoteRejectsForceBeforeReadingSource(t *testing.T) {
	err := CreateModelRemote(t.Context(), nil, CreateOptions{Force: true, ModelDir: "missing"}, nil)
	if err == nil || !strings.Contains(err.Error(), "only supported for local") {
		t.Fatalf("CreateModelRemote() error = %v, want local-only force error", err)
	}
}

func TestUploadRemoteSourceFilesUploadsEachDigestOnce(t *testing.T) {
	var uploads atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusNotFound)
		case http.MethodPost:
			uploads.Add(1)
			w.WriteHeader(http.StatusCreated)
		default:
			t.Errorf("method = %q, want HEAD or POST", r.Method)
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer ts.Close()

	baseURL, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "shared.json")
	if err := os.WriteFile(path, []byte("shared"), 0o600); err != nil {
		t.Fatal(err)
	}
	digest := "sha256:" + strings.Repeat("0", 64)
	files := []remoteSourceFile{
		{logical: "tokenizer.json", path: path, digest: digest, size: 6},
		{logical: "tokenizer.json", path: path, digest: digest, size: 6, draft: true},
	}

	if err := uploadRemoteSourceFiles(t.Context(), api.NewClient(baseURL, ts.Client()), files, nil); err != nil {
		t.Fatal(err)
	}
	if got := uploads.Load(); got != 1 {
		t.Fatalf("uploads = %d, want 1", got)
	}
}

func TestShouldRetryUpload(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "transport error", err: errors.New("connection reset"), want: true},
		{name: "request timeout", err: api.StatusError{StatusCode: http.StatusRequestTimeout}, want: true},
		{name: "rate limited", err: api.StatusError{StatusCode: http.StatusTooManyRequests}, want: true},
		{name: "server error", err: api.StatusError{StatusCode: http.StatusInternalServerError}, want: true},
		{name: "bad request", err: api.StatusError{StatusCode: http.StatusBadRequest}},
		{name: "unauthorized", err: api.AuthorizationError{StatusCode: http.StatusUnauthorized}},
		{name: "canceled", err: context.Canceled},
		{name: "local file", err: fmt.Errorf("upload: %w", &os.PathError{Op: "read", Path: "model.safetensors", Err: errors.New("I/O error")})},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shouldRetryUpload(tt.err); got != tt.want {
				t.Fatalf("shouldRetryUpload() = %v, want %v", got, tt.want)
			}
		})
	}
}
