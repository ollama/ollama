// TODO: go:build goexperiment.synctest

package ollama

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestPullDownloadTimeout(t *testing.T) {
	rc, ctx := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		defer t.Log("upstream", r.Method, r.URL.Path)
		switch {
		case strings.HasPrefix(r.URL.Path, "/v2/library/smol/manifests/"):
			io.WriteString(w, `{
				"layers": [{"digest": "sha256:1111111111111111111111111111111111111111111111111111111111111111", "size": 3}]
			}`)
		case strings.HasPrefix(r.URL.Path, "/v2/library/smol/blobs/sha256:1111111111111111111111111111111111111111111111111111111111111111"):
			// Get headers out to client and then hang on the response
			w.WriteHeader(200)
			w.(http.Flusher).Flush()

			// Hang on the response and unblock when the client
			// gives up
			<-r.Context().Done()
		default:
			t.Fatalf("unexpected request: %s", r.URL.Path)
		}
	})
	rc.ReadTimeout = 100 * time.Millisecond

	done := make(chan error, 1)
	go func() {
		done <- rc.Pull(ctx, "http://example.com/library/smol")
	}()

	select {
	case err := <-done:
		want := context.DeadlineExceeded
		if !errors.Is(err, want) {
			t.Errorf("err = %v, want %v", err, want)
		}
	case <-time.After(3 * time.Second):
		t.Error("timeout waiting for Pull to finish")
	}
}
