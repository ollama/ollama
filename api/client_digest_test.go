package api

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestBlobMethodsRejectInvalidDigest(t *testing.T) {
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	base, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	client := NewClient(base, srv.Client())
	ctx := context.Background()

	for _, digest := range []string{"", ".", "..", "../..", "sha256:../x", "a/b"} {
		if _, err := client.HeadBlob(ctx, digest); !errors.Is(err, ErrInvalidDigest) {
			t.Errorf("HeadBlob(%q) error = %v, want ErrInvalidDigest", digest, err)
		}
		if err := client.CreateBlob(ctx, digest, strings.NewReader("")); !errors.Is(err, ErrInvalidDigest) {
			t.Errorf("CreateBlob(%q) error = %v, want ErrInvalidDigest", digest, err)
		}
	}

	if hits != 0 {
		t.Fatalf("server received %d requests for invalid digests, want 0", hits)
	}
}

func TestHeadBlobStillWorks(t *testing.T) {
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	base, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	found, err := NewClient(base, srv.Client()).HeadBlob(context.Background(), "sha256:abc")
	if err != nil || !found {
		t.Fatalf("HeadBlob = (%v, %v), want (true, nil)", found, err)
	}
	if path != "/api/blobs/sha256:abc" {
		t.Fatalf("path = %q", path)
	}
}
