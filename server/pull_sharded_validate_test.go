package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/types/model"
)

// The create path rejects an unusable shard set only after every blob is in the
// store (#17946). On the pull path that would mean paying for the whole
// download first, so the set is validated from the listing instead. These tests
// pin that the rejection happens at discovery.
func TestShardSetValidatedBeforeDownload(t *testing.T) {
	cases := []struct {
		name    string
		tree    string
		tag     string
		wantErr string
	}{
		{
			name: "incomplete set names the missing shards",
			tree: `[
			  {"path":"Q4_0/m-Q4_0-00001-of-00003.gguf","size":100},
			  {"path":"Q4_0/m-Q4_0-00003-of-00003.gguf","size":300}
			]`,
			tag:     "Q4_0",
			wantErr: "found 2 of 3 shards, missing 00002-of-00003",
		},
		{
			name: "duplicate index is rejected",
			tree: `[
			  {"path":"Q4_0/m-Q4_0-00001-of-00002.gguf","size":100},
			  {"path":"Q4_0/a-Q4_0-00001-of-00002.gguf","size":100},
			  {"path":"Q4_0/m-Q4_0-00002-of-00002.gguf","size":200}
			]`,
			tag: "Q4_0",
			// differing prefixes are caught first, which is the same class of problem
			wantErr: "more than one shard set",
		},
		{
			name: "two shard sets under one tag are rejected",
			tree: `[
			  {"path":"Q4_0/alpha-Q4_0-00001-of-00002.gguf","size":100},
			  {"path":"Q4_0/alpha-Q4_0-00002-of-00002.gguf","size":200},
			  {"path":"Q4_0/beta-Q4_0-00001-of-00002.gguf","size":300},
			  {"path":"Q4_0/beta-Q4_0-00002-of-00002.gguf","size":400}
			]`,
			tag:     "Q4_0",
			wantErr: "more than one shard set",
		},
		{
			name: "inconsistent count between shards is rejected",
			tree: `[
			  {"path":"Q4_0/m-Q4_0-00001-of-00002.gguf","size":100},
			  {"path":"Q4_0/m-Q4_0-00002-of-00003.gguf","size":200}
			]`,
			tag:     "Q4_0",
			wantErr: "more than one shard set",
		},
		{
			name: "complete set passes",
			tree: `[
			  {"path":"Q4_0/m-Q4_0-00001-of-00002.gguf","size":100},
			  {"path":"Q4_0/m-Q4_0-00002-of-00002.gguf","size":200}
			]`,
			tag:     "Q4_0",
			wantErr: "",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(tt.tree))
			}))
			defer srv.Close()

			old := hfEndpointOverride
			hfEndpointOverride = srv.URL
			defer func() { hfEndpointOverride = old }()

			shards, _, err := hfShardSet(context.Background(), "owner/repo", tt.tag)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if len(shards) != 2 {
					t.Errorf("got %d shards, want 2", len(shards))
				}
				return
			}
			if err == nil {
				t.Fatalf("expected an error containing %q, got a set of %d shards", tt.wantErr, len(shards))
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %q, want it to contain %q", err.Error(), tt.wantErr)
			}
		})
	}
}

// An incomplete set must not reach the download loop at all: the fallback
// should decline so the registry's original error is surfaced instead.
func TestIncompleteSetDoesNotOfferFallback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`[{"path":"Q4_0/m-Q4_0-00001-of-00003.gguf","size":100}]`))
	}))
	defer srv.Close()

	old := hfEndpointOverride
	hfEndpointOverride = srv.URL
	defer func() { hfEndpointOverride = old }()

	n := model.ParseName("hf.co/owner/repo:Q4_0")
	if !n.IsValid() {
		t.Fatal("test model name did not parse")
	}
	if _, _, ok := shardedFallbackCandidate(context.Background(), n, errors.New(refusalRepo)); ok {
		t.Error("an incomplete shard set must not be offered as a fallback")
	}
}
