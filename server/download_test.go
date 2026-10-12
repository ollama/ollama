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
	"path/filepath"
	"strings"
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
		{name: "public", location: "https://203.0.113.2/data", wantHits: 2},
		{name: "private", location: "https://127.0.0.1/data", wantErr: "not allowed", wantHits: 1},
		{name: "downgrade", location: "http://203.0.113.2/data", wantErr: "not allowed", wantHits: 1},
		{name: "lookup", wantErr: "not allowed"},
		{name: "insecure", insecure: true, wantHits: 2},
		{name: "loop", location: "https://203.0.113.1/start", wantErr: "stopped after 10 redirects", wantHits: 10},
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
			tr := client.Transport.(*http.Transport)
			tr.TLSClientConfig = server.Client().Transport.(*http.Transport).TLSClientConfig.Clone()
			tr.TLSClientConfig.ServerName = "127.0.0.1"
			dial := tr.DialContext
			tr.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
				// Route only the public fixture addresses to the test server.
				if addr == "203.0.113.1:443" || addr == "203.0.113.2:443" || addr == "203.0.113.2:80" {
					return new(net.Dialer).DialContext(ctx, network, server.Listener.Addr().String())
				}
				return dial(ctx, network, addr)
			}
			requestURL, _ := url.Parse("https://203.0.113.1/start")
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

func TestDownloadRedirect(t *testing.T) {
	var hits atomic.Int32
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Write([]byte("x"))
	}))
	t.Cleanup(cdn.Close)
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, cdn.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(registry.Close)
	requestURL, _ := url.Parse(strings.Replace(registry.URL, "127.0.0.1", "localhost", 1))
	for _, insecure := range []bool{false, true} {
		download := &blobDownload{
			Name: filepath.Join(t.TempDir(), "blob"), Total: 1,
			Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000",
		}
		if err := download.newPart(0, 1); err != nil {
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
