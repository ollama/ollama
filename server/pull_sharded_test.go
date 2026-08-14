package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestIsShardedGGUFRefusal(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{
			"hf refusal verbatim",
			errors.New(`400: {"error":"The specified tag is a sharded GGUF. Ollama does not support this yet. Please use another tag or \"latest\". Follow this issue for more info: https://github.com/ollama/ollama/issues/5245"}`),
			true,
		},
		{"case insensitive", errors.New("400: the specified tag is a Sharded GGUF"), true},
		{"unrelated 400", errors.New(`400: {"error":"The specified tag is not a valid quantization scheme."}`), false},
		{"not found", errors.New("file does not exist"), false},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := isShardedGGUFRefusal(tt.err); got != tt.want {
				t.Errorf("isShardedGGUFRefusal() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsHuggingFaceHost(t *testing.T) {
	for host, want := range map[string]bool{
		"hf.co":              true,
		"huggingface.co":     true,
		"HF.CO":              true,
		"registry.ollama.ai": false,
		"":                   false,
	} {
		if got := isHuggingFaceHost(host); got != want {
			t.Errorf("isHuggingFaceHost(%q) = %v, want %v", host, got, want)
		}
	}
}

// treeServer serves a Hugging Face style file listing.
func treeServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}))
}

const qwenTree = `[
  {"path":"README.md","size":1000},
  {"path":"Qwen3-Coder-Next-UD-TQ1_0.gguf","size":18940000000},
  {"path":"Q4_1/Qwen3-Coder-Next-Q4_1-00003-of-00003.gguf","size":340000000},
  {"path":"Q4_1/Qwen3-Coder-Next-Q4_1-00001-of-00003.gguf","size":10000000},
  {"path":"Q4_1/Qwen3-Coder-Next-Q4_1-00002-of-00003.gguf","size":49720000000},
  {"path":"Q8_0/Qwen3-Coder-Next-Q8_0-00001-of-00003.gguf","size":100},
  {"path":"Q8_0/Qwen3-Coder-Next-Q8_0-00002-of-00003.gguf","size":200},
  {"path":"Q8_0/Qwen3-Coder-Next-Q8_0-00003-of-00003.gguf","size":300}
]`

func TestHFShardSet(t *testing.T) {
	srv := treeServer(t, qwenTree)
	defer srv.Close()

	old := hfEndpointOverride
	hfEndpointOverride = srv.URL
	defer func() { hfEndpointOverride = old }()

	t.Run("orders shards and sums size", func(t *testing.T) {
		shards, total, err := hfShardSet(context.Background(), "unsloth/Qwen3-Coder-Next-GGUF", "Q4_1")
		if err != nil {
			t.Fatalf("hfShardSet() error = %v", err)
		}
		if len(shards) != 3 {
			t.Fatalf("got %d shards, want 3", len(shards))
		}
		want := []string{
			"Q4_1/Qwen3-Coder-Next-Q4_1-00001-of-00003.gguf",
			"Q4_1/Qwen3-Coder-Next-Q4_1-00002-of-00003.gguf",
			"Q4_1/Qwen3-Coder-Next-Q4_1-00003-of-00003.gguf",
		}
		for i, w := range want {
			if shards[i].Path != w {
				t.Errorf("shard[%d] = %q, want %q", i, shards[i].Path, w)
			}
		}
		if total != 340000000+10000000+49720000000 {
			t.Errorf("total = %d, unexpected", total)
		}
	})

	t.Run("tag is case insensitive", func(t *testing.T) {
		shards, _, err := hfShardSet(context.Background(), "r", "q4_1")
		if err != nil {
			t.Fatalf("hfShardSet() error = %v", err)
		}
		if len(shards) != 3 {
			t.Errorf("got %d shards, want 3", len(shards))
		}
	})

	t.Run("does not mix quants", func(t *testing.T) {
		shards, _, err := hfShardSet(context.Background(), "r", "Q8_0")
		if err != nil {
			t.Fatalf("hfShardSet() error = %v", err)
		}
		for _, s := range shards {
			if got := s.Path[:5]; got != "Q8_0/" {
				t.Errorf("leaked non-Q8_0 shard: %q", s.Path)
			}
		}
	})

	t.Run("matches root-level quant exactly", func(t *testing.T) {
		rootSrv := treeServer(t, `[
			{"path":"model-Q4_0-00001-of-00002.gguf","size":100},
			{"path":"model-Q4_0-00002-of-00002.gguf","size":200},
			{"path":"model-IQ4_0-00001-of-00002.gguf","size":300},
			{"path":"model-IQ4_0-00002-of-00002.gguf","size":400}
		]`)
		defer rootSrv.Close()

		oldEndpoint := hfEndpointOverride
		hfEndpointOverride = rootSrv.URL
		defer func() { hfEndpointOverride = oldEndpoint }()

		shards, _, err := hfShardSet(context.Background(), "r", "Q4_0")
		if err != nil {
			t.Fatalf("hfShardSet() error = %v", err)
		}
		if len(shards) != 2 {
			t.Fatalf("got %d shards, want 2", len(shards))
		}
		want := []string{
			"model-Q4_0-00001-of-00002.gguf",
			"model-Q4_0-00002-of-00002.gguf",
		}
		for i, shard := range shards {
			if shard.Path != want[i] {
				t.Errorf("shard[%d] = %q, want %q", i, shard.Path, want[i])
			}
		}
	})

	t.Run("unsharded tag is rejected", func(t *testing.T) {
		if _, _, err := hfShardSet(context.Background(), "r", "UD-TQ1_0"); err == nil {
			t.Error("expected error for a single-file tag, got nil")
		}
	})

	t.Run("unknown tag is rejected", func(t *testing.T) {
		if _, _, err := hfShardSet(context.Background(), "r", "NOPE"); err == nil {
			t.Error("expected error for unknown tag, got nil")
		}
	})
}

func TestHFShardSetFollowsPagination(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Query().Get("cursor") == "next" {
			fmt.Fprint(w, `[
				{"path":"Q4_0/model-Q4_0-00001-of-00002.gguf","size":100},
				{"path":"Q4_0/model-Q4_0-00002-of-00002.gguf","size":200}
			]`)
			return
		}

		w.Header().Set("Link", `</api/models/r/tree/main?recursive=true&cursor=next>; rel="next"`)
		fmt.Fprint(w, `[{"path":"README.md","size":1000}]`)
	}))
	defer srv.Close()

	old := hfEndpointOverride
	hfEndpointOverride = srv.URL
	defer func() { hfEndpointOverride = old }()

	shards, total, err := hfShardSet(context.Background(), "r", "Q4_0")
	if err != nil {
		t.Fatalf("hfShardSet() error = %v", err)
	}
	if len(shards) != 2 {
		t.Fatalf("got %d shards, want 2", len(shards))
	}
	if total != 300 {
		t.Errorf("total = %d, want 300", total)
	}
}
