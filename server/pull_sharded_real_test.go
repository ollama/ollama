package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/ollama/ollama/types/model"
)

// Both wordings observed from the Hugging Face registry. The message was
// reworded on 2026-08-14 to point at the `ollama create` workaround, which is
// why detection must not depend on an exact sentence.
const (
	refusalOld  = `400: {"error":"The specified tag is a sharded GGUF. Ollama does not support this yet. Please use another tag or \"latest\". Follow this issue for more info: https://github.com/ollama/ollama/issues/5245"}`
	refusalTag  = `400: {"error":"This tag is a sharded GGUF. Ollama does not yet support pulling sharded GGUF via the registry; please use a single-file quantization tag or \"latest\", or download the shards and merge them locally with ` + "`ollama create`" + ` (workaround detailed at https://github.com/ollama/ollama/issues/5245)."}`
	refusalRepo = `400: {"error":"This repository only contains sharded GGUF files. Ollama does not yet support pulling sharded GGUF via the registry; please download the shards and merge them locally with ` + "`ollama create`" + ` (workaround detailed at https://github.com/ollama/ollama/issues/5245), or use a repository with single-file quantizations."}`
)

func TestIsShardedGGUFRefusalRealWordings(t *testing.T) {
	for name, msg := range map[string]string{
		"pre-2026-08-14 wording":      refusalOld,
		"reworded, sharded tag":       refusalTag,
		"reworded, sharded-only repo": refusalRepo,
	} {
		t.Run(name, func(t *testing.T) {
			if !isShardedGGUFRefusal(errors.New(msg)) {
				t.Errorf("failed to recognise refusal: %s", msg)
			}
		})
	}

	if isShardedGGUFRefusal(errors.New(`400: {"error":"The specified tag is not a valid quantization scheme."}`)) {
		t.Error("unrelated 400 must not be treated as a sharded refusal")
	}
}

func TestIsManifestClientError(t *testing.T) {
	cases := map[string]struct {
		err  error
		want bool
	}{
		"400":                {errors.New(refusalOld), true},
		"404 as ErrNotExist": {os.ErrNotExist, true},
		"500":                {errors.New(`500: {"error":"internal"}`), false},
		"transport":          {errors.New("dial tcp: connection refused"), false},
		"nil":                {nil, false},
	}
	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			if got := isManifestClientError(tt.err); got != tt.want {
				t.Errorf("isManifestClientError() = %v, want %v", got, tt.want)
			}
		})
	}
}

// realSmolLM2Tree is the actual tree API response shape for
// owalsh/SmolLM2-135M-Instruct-GGUF-Split, a public repository containing only
// a sharded quant, trimmed to the fields this code reads.
const realSmolLM2Tree = `[
  {"type":"directory","oid":"628e439b","size":0,"path":"Q4_0"},
  {"type":"file","oid":"c5f2c12e","size":1783,"path":".gitattributes"},
  {"type":"file","oid":"e1c8dcfb","size":41746880,"path":"Q4_0/SmolLM2-135M-Instruct-Q4_0-00001-of-00003.gguf"},
  {"type":"file","oid":"f2cf445b","size":39916128,"path":"Q4_0/SmolLM2-135M-Instruct-Q4_0-00002-of-00003.gguf"},
  {"type":"file","oid":"1359bb47","size":10230400,"path":"Q4_0/SmolLM2-135M-Instruct-Q4_0-00003-of-00003.gguf"},
  {"type":"file","oid":"197cad52","size":171,"path":"README.md"}
]`

func TestShardedFallbackCandidate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(realSmolLM2Tree))
	}))
	defer srv.Close()

	old := hfEndpointOverride
	hfEndpointOverride = srv.URL
	defer func() { hfEndpointOverride = old }()

	hf := model.ParseName("hf.co/owalsh/SmolLM2-135M-Instruct-GGUF-Split:Q4_0")
	if !hf.IsValid() {
		t.Fatal("test model name did not parse")
	}

	t.Run("real repo, reworded refusal", func(t *testing.T) {
		shards, total, ok := shardedFallbackCandidate(context.Background(), hf, errors.New(refusalRepo))
		if !ok {
			t.Fatal("expected fallback to be offered")
		}
		if len(shards) != 3 {
			t.Fatalf("got %d shards, want 3", len(shards))
		}
		if total != 41746880+39916128+10230400 {
			t.Errorf("total = %d, unexpected", total)
		}
		if shards[0].Path != "Q4_0/SmolLM2-135M-Instruct-Q4_0-00001-of-00003.gguf" {
			t.Errorf("wrong first shard: %s", shards[0].Path)
		}
	})

	t.Run("wording we have never seen still works via probe", func(t *testing.T) {
		if _, _, ok := shardedFallbackCandidate(context.Background(), hf, errors.New(`400: {"error":"totally new wording nobody predicted"}`)); !ok {
			t.Error("a 4xx on a repo that does have a shard set should fall back regardless of wording")
		}
	})

	t.Run("non-4xx is not probed", func(t *testing.T) {
		if _, _, ok := shardedFallbackCandidate(context.Background(), hf, errors.New(`500: {"error":"internal"}`)); ok {
			t.Error("server errors must not trigger the fallback")
		}
	})

	t.Run("tag with no shard set falls through", func(t *testing.T) {
		other := model.ParseName("hf.co/owalsh/SmolLM2-135M-Instruct-GGUF-Split:Q8_0")
		if _, _, ok := shardedFallbackCandidate(context.Background(), other, errors.New(refusalRepo)); ok {
			t.Error("a tag with no matching shards must not fall back")
		}
	})

	t.Run("non-Hugging-Face host is never probed", func(t *testing.T) {
		ollama := model.ParseName("registry.ollama.ai/library/llama3:latest")
		if _, _, ok := shardedFallbackCandidate(context.Background(), ollama, errors.New(refusalOld)); ok {
			t.Error("only Hugging Face models should use this fallback")
		}
	})
}
