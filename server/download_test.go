package server

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestBlobDownloadCreatorReferencePreventsTransientCancel(t *testing.T) {
	runCtx, cancel := context.WithCancel(context.Background())
	b := &blobDownload{
		Digest:     "sha256:123456789012",
		CancelFunc: cancel,
		done:       make(chan struct{}),
	}
	b.acquire()

	waitCtx, waitCancel := context.WithCancel(context.Background())
	waitCancel()

	if err := b.Wait(waitCtx, func(api.ProgressResponse) {}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Wait() error = %v, want context.Canceled", err)
	}

	select {
	case <-runCtx.Done():
		t.Fatal("transient waiter canceled download while creator reference was held")
	default:
	}

	b.release()

	select {
	case <-runCtx.Done():
	default:
		t.Fatal("download was not canceled after creator released final reference")
	}
}

func TestDownloadBlobCreatorCancelDuringPrepareKeepsActiveWaiter(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	digest := "sha256:1111111111111111111111111111111111111111111111111111111111111111"
	blobDownloadManager.Delete(digest)
	t.Cleanup(func() {
		blobDownloadManager.Delete(digest)
		testMakeRequestDialContext = nil
	})
	defaultTransport := http.DefaultTransport
	t.Cleanup(func() {
		http.DefaultTransport = defaultTransport
	})

	headStarted := make(chan struct{})
	headCanceled := make(chan struct{})
	allowHead := make(chan struct{})
	var headStartedOnce sync.Once
	var headCanceledOnce sync.Once

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodHead && r.URL.Path == "/v2/library/test/blobs/"+digest:
			headStartedOnce.Do(func() { close(headStarted) })
			select {
			case <-allowHead:
				w.Header().Set("Content-Length", "1")
			case <-r.Context().Done():
				headCanceledOnce.Do(func() { close(headCanceled) })
			}
		case r.Method == http.MethodGet && r.URL.Path == "/v2/library/test/blobs/"+digest:
			w.Header().Set("Location", "http://direct.example.com/blob")
			w.WriteHeader(http.StatusTemporaryRedirect)
		default:
			http.NotFound(w, r)
		}
	})
	directHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/blob" {
			http.NotFound(w, r)
			return
		}

		w.Header().Set("Content-Length", "1")
		_, _ = w.Write([]byte("x"))
	})

	testMakeRequestDialContext = pipeDial(handler)
	http.DefaultTransport = &http.Transport{
		DialContext:       pipeDial(directHandler),
		DisableKeepAlives: true,
	}

	opts := downloadOpts{
		n:       model.ParseName("registry.example.com/library/test:latest"),
		digest:  digest,
		regOpts: &registryOptions{Insecure: true},
		fn:      func(api.ProgressResponse) {},
	}

	creatorCtx, creatorCancel := context.WithCancel(context.Background())
	creatorErr := make(chan error, 1)
	go func() {
		_, err := downloadBlob(creatorCtx, opts)
		creatorErr <- err
	}()

	select {
	case <-headStarted:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for creator to start blob preparation")
	}

	waiterErr := make(chan error, 1)
	go func() {
		_, err := downloadBlob(context.Background(), opts)
		waiterErr <- err
	}()

	waitForBlobDownloadReferences(t, digest, 2)
	creatorCancel()
	waitForBlobDownloadReferences(t, digest, 1)

	select {
	case <-headCanceled:
		t.Fatal("creator cancellation canceled shared preparation with an active waiter")
	default:
	}
	select {
	case err := <-waiterErr:
		t.Fatalf("waiter downloadBlob() returned before shared preparation continued: %v", err)
	default:
	}

	close(allowHead)

	if err := <-creatorErr; !errors.Is(err, context.Canceled) {
		t.Fatalf("creator downloadBlob() error = %v, want context.Canceled", err)
	}

	if err := <-waiterErr; err != nil {
		t.Fatalf("waiter downloadBlob() error = %v, want nil", err)
	}
}

func TestDownloadBlobWaitsForCanceledSharedDownloadToFinish(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	digest := "sha256:2222222222222222222222222222222222222222222222222222222222222222"
	blobDownloadManager.Delete(digest)
	t.Cleanup(func() {
		blobDownloadManager.Delete(digest)
		testMakeRequestDialContext = nil
	})
	defaultTransport := http.DefaultTransport
	t.Cleanup(func() {
		http.DefaultTransport = defaultTransport
	})

	_, oldCancel := context.WithCancel(context.Background())
	stale := &blobDownload{
		Digest:     digest,
		CancelFunc: oldCancel,
		done:       make(chan struct{}),
	}
	stale.acquire()
	blobDownloadManager.Store(digest, stale)
	stale.release()

	headStarted := make(chan struct{})
	var headStartedOnce sync.Once

	registryHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodHead && r.URL.Path == "/v2/library/test/blobs/"+digest:
			headStartedOnce.Do(func() { close(headStarted) })
			w.Header().Set("Content-Length", "1")
		case r.Method == http.MethodGet && r.URL.Path == "/v2/library/test/blobs/"+digest:
			w.Header().Set("Location", "http://direct.example.com/blob")
			w.WriteHeader(http.StatusTemporaryRedirect)
		default:
			http.NotFound(w, r)
		}
	})
	directHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/blob" {
			http.NotFound(w, r)
			return
		}

		w.Header().Set("Content-Length", "1")
		_, _ = w.Write([]byte("x"))
	})

	testMakeRequestDialContext = pipeDial(registryHandler)
	http.DefaultTransport = &http.Transport{
		DialContext:       pipeDial(directHandler),
		DisableKeepAlives: true,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	errCh := make(chan error, 1)
	cacheHitCh := make(chan bool, 1)
	go func() {
		cacheHit, err := downloadBlob(ctx, downloadOpts{
			n:       model.ParseName("registry.example.com/library/test:latest"),
			digest:  digest,
			regOpts: &registryOptions{Insecure: true},
			fn:      func(api.ProgressResponse) {},
		})
		cacheHitCh <- cacheHit
		errCh <- err
	}()

	select {
	case <-headStarted:
		t.Fatal("replacement download started before canceled shared download finished")
	case <-time.After(100 * time.Millisecond):
	}

	close(stale.done)

	select {
	case <-headStarted:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for replacement download to start")
	}

	err := <-errCh
	if err != nil {
		t.Fatalf("downloadBlob() error = %v, want nil", err)
	}
	if cacheHit := <-cacheHitCh; cacheHit {
		t.Fatal("downloadBlob() cacheHit = true, want false")
	}
}

func TestBlobDownloadRunCanceledContextDoesNotCreatePartialFile(t *testing.T) {
	name := filepath.Join(t.TempDir(), "sha256-333333333333")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	download := &blobDownload{
		Name:   name,
		Digest: "sha256:333333333333",
		done:   make(chan struct{}),
	}

	download.Run(ctx, &url.URL{}, &registryOptions{})

	if !errors.Is(download.err, context.Canceled) {
		t.Fatalf("Run() error = %v, want context.Canceled", download.err)
	}
	if _, err := os.Stat(name + "-partial"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("Run() created partial file after cancellation: %v", err)
	}
}

func waitForBlobDownloadReferences(t *testing.T, digest string, want int32) {
	t.Helper()

	deadline := time.After(time.Second)
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()

	var got int32
	var found bool
	for {
		if data, ok := blobDownloadManager.Load(digest); ok {
			found = true
			got = data.(*blobDownload).references.Load()
			if got == want {
				return
			}
		} else {
			found = false
			got = 0
		}

		select {
		case <-deadline:
			t.Fatalf("timed out waiting for blob references = %d; found=%t got=%d", want, found, got)
		case <-ticker.C:
		}
	}
}

func pipeDial(handler http.Handler) func(context.Context, string, string) (net.Conn, error) {
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		client, server := net.Pipe()
		go func() {
			_ = http.Serve(&singleConnListener{conn: server}, handler)
		}()
		return client, nil
	}
}

type singleConnListener struct {
	mu   sync.Mutex
	conn net.Conn
}

func (l *singleConnListener) Accept() (net.Conn, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.conn == nil {
		return nil, io.EOF
	}

	conn := l.conn
	l.conn = nil
	return conn, nil
}

func (l *singleConnListener) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.conn == nil {
		return nil
	}

	err := l.conn.Close()
	l.conn = nil
	return err
}

func (l *singleConnListener) Addr() net.Addr {
	return pipeAddr("pipe")
}

type pipeAddr string

func (p pipeAddr) Network() string {
	return string(p)
}

func (p pipeAddr) String() string {
	return string(p)
}
