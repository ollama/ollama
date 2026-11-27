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
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/transfer"
)

func TestDownloadRangeFallback(t *testing.T) {
	data := []byte("abcdefghijkl")
	for _, tc := range []struct {
		name           string
		supportsRanges bool
		completed      []int64
	}{
		{name: "partial responses", supportsRanges: true, completed: []int64{0, 0}},
		{name: "resumed partial responses", supportsRanges: true, completed: []int64{2, 3}},
		{name: "ignored ranges", completed: []int64{0, 0}},
		{name: "ignored resumed ranges", completed: []int64{2, 3}},
		{name: "whole file response", completed: []int64{0}},
		{name: "ignored resumed whole file", completed: []int64{5}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var fullRequests atomic.Int32
			cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Range") == fmt.Sprintf("bytes=0-%d", len(data)-1) {
					fullRequests.Add(1)
				}
				if tc.supportsRanges {
					http.ServeContent(w, r, "blob", time.Time{}, bytes.NewReader(data))
					return
				}
				_, _ = w.Write(data)
			}))
			t.Cleanup(cdn.Close)
			registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Redirect(w, r, cdn.URL, http.StatusTemporaryRedirect)
			}))
			t.Cleanup(registry.Close)
			requestURL, err := url.Parse(registry.URL)
			if err != nil {
				t.Fatal(err)
			}
			download := &blobDownload{
				Name:   filepath.Join(t.TempDir(), "blob"),
				Digest: fmt.Sprintf("sha256:%x", sha256.Sum256(data)),
				Total:  int64(len(data)),
			}
			initial := make([]byte, len(data))
			size := download.Total / int64(len(tc.completed))
			for i, completed := range tc.completed {
				offset := int64(i) * size
				if err := download.newPart(offset, size); err != nil {
					t.Fatal(err)
				}
				part := download.Parts[i]
				part.Completed.Store(completed)
				download.Completed.Add(completed)
				if err := download.writePart(part.Name(), part); err != nil {
					t.Fatal(err)
				}
				copy(initial[offset:offset+completed], data[offset:offset+completed])
			}
			if err := os.WriteFile(download.Name+"-partial", initial, 0o644); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			if err := download.run(ctx, requestURL, &registryOptions{Insecure: true}); err != nil {
				t.Fatal(err)
			}
			got, err := os.ReadFile(download.Name)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, data) {
				t.Errorf("downloaded %q, want %q", got, data)
			}
			if got := download.Completed.Load(); got != download.Total {
				t.Errorf("completed bytes = %d, want %d", got, download.Total)
			}
			if !tc.supportsRanges && fullRequests.Load() != 1 {
				t.Errorf("whole file requests = %d, want 1", fullRequests.Load())
			}
			partials, err := filepath.Glob(download.Name + "-partial*")
			if err != nil || len(partials) != 0 {
				t.Errorf("partial files = %v, %v, want none", partials, err)
			}
		})
	}
}

func TestDownloadRangeFallbackCancellation(t *testing.T) {
	data := []byte("abcdefghijkl")
	var interrupt atomic.Bool
	interrupt.Store(true)
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprint(len(data)))
		if r.Header.Get("Range") == "bytes=0-11" && interrupt.Load() {
			_, _ = w.Write(data[:3])
			w.(http.Flusher).Flush()
			<-r.Context().Done()
			return
		}
		_, _ = w.Write(data)
	}))
	t.Cleanup(cdn.Close)
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, cdn.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(registry.Close)
	requestURL, err := url.Parse(registry.URL)
	if err != nil {
		t.Fatal(err)
	}
	download := &blobDownload{
		Name:   filepath.Join(t.TempDir(), "blob"),
		Digest: fmt.Sprintf("sha256:%x", sha256.Sum256(data)),
		Total:  int64(len(data)),
	}
	for _, offset := range []int64{0, 6} {
		if err := download.newPart(offset, 6); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		done <- download.run(ctx, requestURL, &registryOptions{Insecure: true})
	}()
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for download.Completed.Load() == 0 && ctx.Err() == nil {
		select {
		case <-ticker.C:
		case <-ctx.Done():
		}
	}
	if download.Completed.Load() == 0 {
		cancel()
		<-done
		t.Fatal("fallback did not start writing")
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("interrupted download = %v, want cancellation", err)
	}
	if _, err := os.Stat(download.Name); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("unfinished download was published: %v", err)
	}

	interrupt.Store(false)
	resumed := &blobDownload{Name: download.Name, Digest: download.Digest}
	opts := &registryOptions{Insecure: true}
	if err := resumed.Prepare(t.Context(), requestURL, opts); err != nil {
		t.Fatal(err)
	}
	if len(resumed.Parts) != 1 || resumed.Completed.Load() != 3 {
		t.Fatalf("resumed %d parts and %d bytes, want 1 part and 3 bytes", len(resumed.Parts), resumed.Completed.Load())
	}
	if err := resumed.run(t.Context(), requestURL, opts); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(resumed.Name)
	if err != nil || !bytes.Equal(got, data) {
		t.Fatalf("resumed download = %q, %v, want %q", got, err, data)
	}
}

func TestDownloadChunkRejectsHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	}))
	t.Cleanup(server.Close)
	requestURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	download := &blobDownload{Name: filepath.Join(t.TempDir(), "blob"), Total: 3}
	part := &blobDownloadPart{Size: 3, blobDownload: download}
	var body bytes.Buffer
	err = download.downloadChunk(t.Context(), server.Client(), requestURL, &body, part)
	if err == nil || !strings.Contains(err.Error(), "503") {
		t.Fatalf("download error = %v, want HTTP 503", err)
	}
	if body.Len() != 0 || download.Completed.Load() != 0 || part.Completed.Load() != 0 {
		t.Fatal("HTTP error body was saved as blob data")
	}
}

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
			download := &blobDownload{Name: filepath.Join(t.TempDir(), "blob"), Total: 1}
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
