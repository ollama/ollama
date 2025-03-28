//go:build goexperiment.synctest

package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/testutil"
)

// Checksum reference:
//
//	"abc" -> sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
//      "ab"  -> sha256:fb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603
//
//	"a" -> sha256:ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb
//	"b" -> sha256:3e23e8160039594a33894f6564e1b1348bbd7a0088d42c4acb73eeaed59c009d
//	"c" -> sha256:2e7d2c03a9507ae265ecf5b5356885a53393a2029d241394997265a1a25aefc6

func checkRequest(t *testing.T, req *http.Request, method, path string) {
	t.Helper()
	if got := req.URL.Path; got != path {
		t.Errorf("URL = %q, want %q", got, path)
	}
	if req.Method != method {
		t.Errorf("Method = %q, want %q", req.Method, method)
	}
}

func newRegistryClient(t *testing.T, h http.HandlerFunc) (*Registry, context.Context) {
	s := httptest.NewServer(h)
	t.Cleanup(s.Close)
	cache, err := blob.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}

	ctx := WithTrace(t.Context(), &Trace{
		Update: func(l *Layer, n int64, err error) {
			t.Log("trace:", l.Digest.Short(), n, err)
		},
	})

	rc := &Registry{
		Cache: cache,
		HTTPClient: &http.Client{Transport: &http.Transport{
			Dial: func(network, addr string) (net.Conn, error) {
				return net.Dial(network, s.Listener.Addr().String())
			},
		}},
	}
	return rc, ctx
}

func handleSimplePull(t *testing.T) http.HandlerFunc {
	var steps atomic.Int64
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch steps.Add(1) {
		case 1:
			checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
			io.WriteString(w, `{"layers":[{"size":3,"digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]}`)
		case 2:
			checkRequest(t, r, "GET", "/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			io.WriteString(w, "abc")
		default:
			t.Errorf("unexpected steps %d: %v", steps.Load(), r)
		}
	})
}

func TestPullSimple(t *testing.T) {
	c, ctx := newRegistryClient(t, handleSimplePull(t))
	if err := c.Pull(ctx, "http://o.com/library/abc"); err != nil {
		t.Fatalf("Pull: %v", err)
	}
}

func TestPullChunked(t *testing.T) {
	var steps atomic.Int64
	defer func() {
		if steps.Load() == 0 {
			t.Fatalf("expected steps")
		}
	}()
	c, ctx := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch steps.Add(1) {
		case 1:
			checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
			io.WriteString(w, `{"layers":[{"size":3,"digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]}`)
		case 2:
			checkRequest(t, r, "GET", "/v2/library/abc/chunksums/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			w.Header().Set("Content-Location", "http://blob.store/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			io.WriteString(w, "sha256:fb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603 0-1\n")
			io.WriteString(w, "sha256:2e7d2c03a9507ae265ecf5b5356885a53393a2029d241394997265a1a25aefc6 2-2\n")
		case 3, 4:
			checkRequest(t, r, "GET", "/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			switch rng := r.Header.Get("Range"); rng {
			case "bytes=0-1":
				io.WriteString(w, "ab")
			case "bytes=2-2":
				t.Logf("writing c")
				io.WriteString(w, "c")
			default:
				t.Errorf("unexpected range %q", rng)
			}
		default:
			t.Errorf("unexpected steps %d: %v", steps.Load(), r)
			http.Error(w, "unexpected steps", http.StatusInternalServerError)
		}
	})

	c.ChunkingThreshold = 1 // force chunking

	err := c.Pull(ctx, "http://o.com/library/abc")
	testutil.Check(t, err)
}

func TestPullCached(t *testing.T) {
	var totalSteps atomic.Int64 // total steps across both pulls
	h := handleSimplePull(t)
	c, ctx := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		totalSteps.Add(1)
		h(w, r)
	})

	// make initial pull
	if err := c.Pull(ctx, "http://o.com/library/abc"); err != nil {
		t.Fatalf("expected error")
	}

	// record steps taken for second pull
	before := totalSteps.Load()
	h = handleSimplePull(t) // reset handler state (not totalSteps)
	if err := c.Pull(ctx, "http://o.com/library/abc"); err != nil {
		t.Fatalf("Pull: %v", err)
	}
	after := totalSteps.Load()
	if g := after - before; g != 1 {
		t.Fatalf("got %d steps, want 1", after-before)
	}
}

func TestPullManifestError(t *testing.T) {
	c, ctx := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
		w.WriteHeader(http.StatusNotFound)
		io.WriteString(w, `{"errors":[{"code":"MANIFEST_UNKNOWN"}]}`)
	})

	err := c.Pull(ctx, "http://o.com/library/abc")
	if err == nil {
		t.Fatalf("expected error")
	}
	var got *Error
	if !errors.Is(err, ErrModelNotFound) {
		t.Fatalf("err = %v, want %v", got, ErrModelNotFound)
	}
}

func TestPullLayerError(t *testing.T) {
	c, ctx := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
		io.WriteString(w, `!`)
	})

	err := c.Pull(ctx, "http://o.com/library/abc")
	if err == nil {
		t.Fatalf("expected error")
	}
	var want *json.SyntaxError
	if !errors.As(err, &want) {
		t.Fatalf("err = %T, want %T", err, want)
	}
}

func TestPullLayerChecksumError(t *testing.T) {
	var step atomic.Int64
	c, _ := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch step.Add(1) {
		case 1:
			checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
			io.WriteString(w, `{"layers":[{"size":3,"digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]}`)
		case 2:
			checkRequest(t, r, "GET", "/v2/library/abc/chunksums/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			w.Header().Set("Content-Location", "http://blob.store/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			io.WriteString(w, "sha256:fb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603 0-1\n")
			io.WriteString(w, "sha256:2e7d2c03a9507ae265ecf5b5356885a53393a2029d241394997265a1a25aefc6 2-2\n")
		case 3:
			checkRequest(t, r, "GET", "/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			w.WriteHeader(http.StatusNotFound)
			io.WriteString(w, `{"errors":[{"code":"BLOB_UNKNOWN"}]}`)
		case 4:
			checkRequest(t, r, "GET", "/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			io.WriteString(w, "c")
		default:
			t.Errorf("unexpected steps %d: %v", step.Load(), r)
			http.Error(w, "unexpected steps", http.StatusInternalServerError)
		}
	})

	c.MaxStreams = 1
	c.ChunkingThreshold = 1 // force chunking

	var written atomic.Int64
	ctx := WithTrace(t.Context(), &Trace{
		Update: func(l *Layer, n int64, err error) {
			t.Log("trace:", l.Digest.Short(), n, err)
			written.Add(n)
		},
	})

	err := c.Pull(ctx, "http://o.com/library/abc")
	var got *Error
	if !errors.As(err, &got) || got.Code != "BLOB_UNKNOWN" {
		t.Fatalf("err = %v, want %v", err, got)
	}

	if g := written.Load(); g != 1 {
		t.Fatalf("wrote %d bytes, want 1", g)
	}
}

func TestPullChunksumStreamError(t *testing.T) {
	var step atomic.Int64
	c, ctx := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch step.Add(1) {
		case 1:
			checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
			io.WriteString(w, `{"layers":[{"size":3,"digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]}`)
		case 2:
			checkRequest(t, r, "GET", "/v2/library/abc/chunksums/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			w.Header().Set("Content-Location", "http://blob.store/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			io.WriteString(w, `sha256:fb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603 0-1`)
			io.WriteString(w, `sha256:!`) // force stream error
		case 3:
			checkRequest(t, r, "GET", "/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			if rng := r.Header.Get("Range"); rng != "bytes=0-1" {
				t.Errorf("Range = %q, want %q", rng, "bytes=0-1")
			}
			io.WriteString(w, "ab")
		default:
			t.Errorf("unexpected steps %d: %v", step.Load(), r)
			http.Error(w, "unexpected steps", http.StatusInternalServerError)
		}
	})

	c.ChunkingThreshold = 1 // force chunking

	got := c.Pull(ctx, "http://o.com/library/abc")
	if !errors.Is(got, ErrIncomplete) {
		t.Fatalf("err = %v, want %v", got, ErrIncomplete)
	}
}

type flushAfterWriter struct {
	w io.Writer
}

func (f *flushAfterWriter) Write(p []byte) (n int, err error) {
	n, err = f.w.Write(p)
	f.w.(http.Flusher).Flush() // panic if not a flusher
	return
}

func TestPullChunksumStreaming(t *testing.T) {
	csr, csw := io.Pipe()
	defer csw.Close()

	var step atomic.Int64
	c, _ := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch step.Add(1) {
		case 1:
			checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
			io.WriteString(w, `{"layers":[{"size":3,"digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]}`)
		case 2:
			checkRequest(t, r, "GET", "/v2/library/abc/chunksums/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			w.Header().Set("Content-Location", "http://blob.store/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			fw := &flushAfterWriter{w} // ensure client gets data as it arrives by aggressively flushing
			_, err := io.Copy(fw, csr)
			if err != nil {
				t.Errorf("copy: %v", err)
			}
		case 3:
			checkRequest(t, r, "GET", "/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			io.WriteString(w, "ab")
		case 4:
			checkRequest(t, r, "GET", "/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			io.WriteString(w, "c")
		default:
			t.Errorf("unexpected steps %d: %v", step.Load(), r)
			http.Error(w, "unexpected steps", http.StatusInternalServerError)
		}
	})

	c.ChunkingThreshold = 1 // force chunking

	update := make(chan int64, 1)
	ctx := WithTrace(t.Context(), &Trace{
		Update: func(l *Layer, n int64, err error) {
			t.Log("trace:", l.Digest.Short(), n, err)
			if n > 0 {
				update <- n
			}
		},
	})

	errc := make(chan error, 1)
	go func() {
		errc <- c.Pull(ctx, "http://o.com/library/abc")
	}()

	// Send first chunksum and ensure it kicks off work immediately
	io.WriteString(csw, "sha256:fb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603 0-1\n") // "ab"
	if g := <-update; g != 2 {
		t.Fatalf("got %d, want 2", g)
	}

	// now send the second chunksum and ensure it kicks off work immediately
	io.WriteString(csw, "sha256:2e7d2c03a9507ae265ecf5b5356885a53393a2029d241394997265a1a25aefc6 2-2\n") // "c"
	if g := <-update; g != 1 {
		t.Fatalf("got %d, want 1", g)
	}
	csw.Close()
	testutil.Check(t, <-errc)
}

func TestPullChunksumsCached(t *testing.T) {
	var step atomic.Int64
	c, _ := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch step.Add(1) {
		case 1:
			checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
			io.WriteString(w, `{"layers":[{"size":3,"digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]}`)
		case 2:
			checkRequest(t, r, "GET", "/v2/library/abc/chunksums/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			w.Header().Set("Content-Location", "http://blob.store/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			io.WriteString(w, "sha256:fb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603 0-1\n")
			io.WriteString(w, "sha256:2e7d2c03a9507ae265ecf5b5356885a53393a2029d241394997265a1a25aefc6 2-2\n")
		case 3, 4:
			checkRequest(t, r, "GET", "/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			switch rng := r.Header.Get("Range"); rng {
			case "bytes=0-1":
				io.WriteString(w, "ab")
			case "bytes=2-2":
				io.WriteString(w, "c")
			default:
				t.Errorf("unexpected range %q", rng)
			}
		default:
			t.Errorf("unexpected steps %d: %v", step.Load(), r)
			http.Error(w, "unexpected steps", http.StatusInternalServerError)
		}
	})

	c.MaxStreams = 1        // force serial processing of chunksums
	c.ChunkingThreshold = 1 // force chunking

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	// Cancel the pull after the first chunksum is processed, but before
	// the second chunksum is processed (which is waiting because
	// MaxStreams=1). This should cause the second chunksum to error out
	// leaving the blob incomplete.
	ctx = WithTrace(ctx, &Trace{
		Update: func(l *Layer, n int64, err error) {
			if n > 0 {
				cancel()
			}
		},
	})
	err := c.Pull(ctx, "http://o.com/library/abc")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want %v", err, context.Canceled)
	}

	_, err = c.Cache.Resolve("o.com/library/abc:latest")
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("err = %v, want nil", err)
	}

	// Reset state and pull again to ensure the blob chunks that should
	// have been cached are, and the remaining chunk was downloaded, making
	// the blob complete.
	step.Store(0)
	var written atomic.Int64
	var cached atomic.Int64
	ctx = WithTrace(t.Context(), &Trace{
		Update: func(l *Layer, n int64, err error) {
			t.Log("trace:", l.Digest.Short(), n, err)
			if errors.Is(err, ErrCached) {
				cached.Add(n)
			}
			written.Add(n)
		},
	})
	err = c.Pull(ctx, "http://o.com/library/abc")
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	_, err = c.Cache.Resolve("o.com/library/abc:latest")
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if g := written.Load(); g != 3 {
		t.Fatalf("wrote %d bytes, want 3", g)
	}
	if g := cached.Load(); g != 2 { // "ab" should have been cached
		t.Fatalf("cached %d bytes, want 3", g)
	}
}
