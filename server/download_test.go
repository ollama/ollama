package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/transfer"
)

func TestDownloadChunkRedirects(t *testing.T) {
	for _, tc := range []struct {
		name     string
		location string
		insecure bool
		wantErr  string
		wantHits int32
	}{
		{name: "public", location: "https://8.8.8.8/data", wantHits: 2},
		{name: "private", location: "https://127.0.0.1/data", wantErr: "not allowed", wantHits: 1},
		{name: "downgrade", location: "http://8.8.8.8/data", wantErr: "not allowed", wantHits: 1},
		{name: "lookup", wantErr: "not allowed"},
		{name: "insecure", insecure: true, wantHits: 2},
		{name: "loop", location: "https://8.8.4.4/start", wantErr: "stopped after 10 redirects", wantHits: 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var hits atomic.Int32
			var location string
			server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				if r.URL.Path == "/start" {
					http.Redirect(w, r, location, http.StatusTemporaryRedirect)
					return
				}
				w.Write([]byte("x"))
			}))
			t.Cleanup(server.Close)
			location = tc.location
			if location == "" {
				location = server.URL + "/data"
			}
			client := transfer.NewRedirectClient("https://registry.example", tc.insecure)
			t.Cleanup(client.CloseIdleConnections)
			tr := server.Client().Transport.(*http.Transport).Clone()
			client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
				if err := transfer.ValidateRedirectTarget(req.Context(), req.URL, "https://registry.example", tc.insecure); err != nil {
					return nil, err
				}
				return tr.RoundTrip(req)
			})
			tr.TLSClientConfig.ServerName = "127.0.0.1"
			dial := tr.DialContext
			tr.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
				// Route only the public fixture addresses to the test server.
				if addr == "8.8.4.4:443" || addr == "8.8.8.8:443" || addr == "8.8.8.8:80" {
					return new(net.Dialer).DialContext(ctx, network, server.Listener.Addr().String())
				}
				return dial(ctx, network, addr)
			}
			requestURL, _ := url.Parse("https://8.8.4.4/start")
			if tc.name == "lookup" {
				requestURL, _ = url.Parse(strings.Replace(server.URL, "127.0.0.1", "localhost", 1) + "/data")
			}
			download := &blobDownload{Name: filepath.Join(t.TempDir(), "blob")}
			part := &blobDownloadPart{Size: 1, blobDownload: download}
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			var body bytes.Buffer
			err := download.downloadChunk(ctx, client, requestURL, &body, part)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("downloadChunk() error = %v, want %q", err, tc.wantErr)
				}
			} else if err != nil || body.String() != "x" {
				t.Fatalf("downloadChunk() = %q, %v, want x, nil", body.String(), err)
			}
			if got := hits.Load(); got != tc.wantHits {
				t.Errorf("requests = %d, want %d", got, tc.wantHits)
			}
		})
	}
}

func TestDownloadDiscoveryRedirectOrigin(t *testing.T) {
	t.Run("same origin", func(t *testing.T) {
		var hits atomic.Int32
		registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			hits.Add(1)
			if r.URL.Path == "/start" {
				http.Redirect(w, r, "/data", http.StatusTemporaryRedirect)
				return
			}
			fmt.Fprint(w, "ok")
		}))
		defer registry.Close()
		base, _ := url.Parse(registry.URL)
		client := &http.Client{CheckRedirect: downloadDiscoveryRedirect(base)}
		resp, err := client.Get(registry.URL + "/start")
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK || hits.Load() != 2 {
			t.Fatalf("same-origin redirect: status %d, hits %d", resp.StatusCode, hits.Load())
		}
	})

	t.Run("different port", func(t *testing.T) {
		var targetHits atomic.Int32
		target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			targetHits.Add(1)
		}))
		defer target.Close()
		registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, target.URL+"/private", http.StatusFound)
		}))
		defer registry.Close()
		base, _ := url.Parse(registry.URL)
		client := &http.Client{CheckRedirect: downloadDiscoveryRedirect(base)}
		resp, err := client.Get(registry.URL + "/start")
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusFound || targetHits.Load() != 0 {
			t.Fatalf("different-port redirect: status %d, target hits %d", resp.StatusCode, targetHits.Load())
		}
	})

	t.Run("https downgrade", func(t *testing.T) {
		var registryURL string
		registry := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			loc := strings.Replace(registryURL, "https://", "http://", 1) + "/data"
			http.Redirect(w, r, loc, http.StatusTemporaryRedirect)
		}))
		defer registry.Close()
		registryURL = registry.URL
		base, _ := url.Parse(registry.URL)
		client := registry.Client()
		client.CheckRedirect = downloadDiscoveryRedirect(base)
		resp, err := client.Get(registry.URL + "/start")
		if err == nil {
			resp.Body.Close()
		}
		if err == nil || !strings.Contains(err.Error(), "not allowed") {
			t.Fatalf("HTTPS downgrade was followed: %v", err)
		}
	})
}

func TestDownloadRedirect(t *testing.T) {
	const body = "x"
	wantDigest := fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(body)))
	var hits atomic.Int32
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Write([]byte(body))
	}))
	t.Cleanup(cdn.Close)
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, cdn.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(registry.Close)
	requestURL, _ := url.Parse(strings.Replace(registry.URL, "127.0.0.1", "localhost", 1))
	for _, insecure := range []bool{false, true} {
		hits.Store(0)
		download := &blobDownload{
			Name: filepath.Join(t.TempDir(), "blob"), Total: int64(len(body)),
			Digest: wantDigest,
		}
		if err := download.newPart(0, int64(len(body))); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		err := download.run(ctx, requestURL, &registryOptions{Insecure: insecure})
		cancel()
		if insecure {
			if err != nil || hits.Load() != 1 {
				t.Fatalf("insecure download: error = %v, requests = %d, want nil, 1", err, hits.Load())
			}
		} else if err == nil || !strings.Contains(err.Error(), "not allowed") || hits.Load() != 0 {
			t.Fatalf("download: error = %v, requests = %d, want rejection before request", err, hits.Load())
		}
	}
}

func TestHashCompletedParts(t *testing.T) {
	const data = "abcdefghijklmnopqrstuvwxyz"
	want := sha256.Sum256([]byte(data))

	dir := t.TempDir()
	partial := filepath.Join(dir, "blob-partial")
	if err := os.WriteFile(partial, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(partial)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })

	download := &blobDownload{
		Name: filepath.Join(dir, "blob"),
		Parts: []*blobDownloadPart{
			{N: 0, Offset: 0, Size: 10},
			{N: 1, Offset: 10, Size: 10},
			{N: 2, Offset: 20, Size: 6},
		},
	}
	for _, part := range download.Parts {
		part.blobDownload = download
		part.Completed.Store(part.Size)
	}

	got, err := download.hashCompletedParts(t.Context(), f)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want[:]) {
		t.Fatalf("digest = %x, want %x", got, want)
	}
}

func TestHashCompletedPartsResumePrefix(t *testing.T) {
	const data = "0123456789abcdef"
	want := sha256.Sum256([]byte(data))

	dir := t.TempDir()
	partial := filepath.Join(dir, "blob-partial")
	if err := os.WriteFile(partial, []byte(data[:8]), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(partial, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })

	download := &blobDownload{
		Name: filepath.Join(dir, "blob"),
		Parts: []*blobDownloadPart{
			{N: 0, Offset: 0, Size: 8},
			{N: 1, Offset: 8, Size: 8},
		},
	}
	for _, part := range download.Parts {
		part.blobDownload = download
	}
	download.Parts[0].Completed.Store(8)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		sum, err := download.hashCompletedParts(t.Context(), f)
		if err != nil {
			t.Errorf("hashCompletedParts: %v", err)
			return
		}
		if !bytes.Equal(sum, want[:]) {
			t.Errorf("digest = %x, want %x", sum, want)
		}
	}()

	time.Sleep(20 * time.Millisecond)
	if _, err := f.WriteAt([]byte(data[8:]), 8); err != nil {
		t.Fatal(err)
	}
	download.Parts[1].Completed.Store(8)
	wg.Wait()
}

func TestDownloadRunVerifiesDigest(t *testing.T) {
	const data = "hello-ollama-blob"
	wantDigest := fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(data)))

	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprint(len(data)))
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write([]byte(data))
	}))
	t.Cleanup(cdn.Close)
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, cdn.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(registry.Close)
	requestURL, err := url.Parse(strings.Replace(registry.URL, "127.0.0.1", "localhost", 1))
	if err != nil {
		t.Fatal(err)
	}

	t.Run("match", func(t *testing.T) {
		name := filepath.Join(t.TempDir(), "blob")
		download := &blobDownload{Name: name, Digest: wantDigest, Total: int64(len(data))}
		if err := download.newPart(0, int64(len(data))); err != nil {
			t.Fatal(err)
		}
		if err := download.run(t.Context(), requestURL, &registryOptions{Insecure: true}); err != nil {
			t.Fatalf("run() error = %v", err)
		}
		got, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != data {
			t.Fatalf("blob = %q, want %q", got, data)
		}
	})

	t.Run("mismatch", func(t *testing.T) {
		name := filepath.Join(t.TempDir(), "blob")
		download := &blobDownload{
			Name:   name,
			Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000",
			Total:  int64(len(data)),
		}
		if err := download.newPart(0, int64(len(data))); err != nil {
			t.Fatal(err)
		}
		err := download.run(t.Context(), requestURL, &registryOptions{Insecure: true})
		if !errors.Is(err, errDigestMismatch) {
			t.Fatalf("run() error = %v, want %v", err, errDigestMismatch)
		}
		if _, err := os.Stat(name); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("blob path exists after mismatch: %v", err)
		}
		if _, err := os.Stat(name + "-partial"); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("partial file remains after mismatch: %v", err)
		}
	})
}

func BenchmarkDownloadChunkCompletion(b *testing.B) {
	data := make([]byte, 1024*1024)
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprint(len(data)))
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(data)
	}))
	b.Cleanup(server.Close)

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		b.Fatal(err)
	}
	downloadPath := filepath.Join(b.TempDir(), "blob")

	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		download := &blobDownload{Name: downloadPath, Digest: digest}
		part := &blobDownloadPart{Size: int64(len(data)), blobDownload: download}
		if err := download.downloadChunk(b.Context(), server.Client(), requestURL, io.Discard, part); err != nil {
			b.Fatal(err)
		}
	}
}

func TestDownloadChunkReturnsWhenTransferCompletes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1")
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write([]byte{0})
	}))
	t.Cleanup(server.Close)

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}

	download := &blobDownload{
		Name:   filepath.Join(t.TempDir(), "blob"),
		Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000",
	}
	part := &blobDownloadPart{Size: 1, blobDownload: download}
	ctx, cancel := context.WithTimeout(t.Context(), 250*time.Millisecond)
	defer cancel()

	if err := download.downloadChunk(ctx, server.Client(), requestURL, io.Discard, part); err != nil {
		t.Fatalf("downloadChunk() error = %v, want nil", err)
	}
}

func TestDownloadChunkDetectsStallBeforeFirstByte(t *testing.T) {
	requestStarted := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(requestStarted)
		w.Header().Set("Content-Length", "1")
		w.WriteHeader(http.StatusPartialContent)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(server.Close)

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}

	download := &blobDownload{Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000"}
	part := &blobDownloadPart{Size: 1, blobDownload: download}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()

	originalStallTimeout := downloadStallTimeout
	downloadStallTimeout = 50 * time.Millisecond
	t.Cleanup(func() {
		downloadStallTimeout = originalStallTimeout
	})

	started := time.Now()
	err = download.downloadChunk(ctx, server.Client(), requestURL, io.Discard, part)
	elapsed := time.Since(started)

	select {
	case <-requestStarted:
	default:
		t.Fatal("download request did not start")
	}
	if !errors.Is(err, errPartStalled) {
		t.Fatalf("downloadChunk() error = %v after %v, want %v", err, elapsed, errPartStalled)
	}
	if elapsed >= 5*downloadStallTimeout {
		t.Fatalf("downloadChunk() detected the stall after %v, want less than %v", elapsed, 5*downloadStallTimeout)
	}
}
