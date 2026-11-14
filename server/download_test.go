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

	"github.com/ollama/ollama/api"
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

func TestDownloadChunkStatusErrors(t *testing.T) {
	for _, tc := range []struct {
		name            string
		status          int
		body            string
		offset          int64
		wantStatusCode  int
		wantErrorSubstr string
	}{
		{
			name:            "403 forbidden with html body",
			status:          http.StatusForbidden,
			body:            "<html><body>Blocked by corporate proxy</body></html>",
			wantStatusCode:  http.StatusForbidden,
			wantErrorSubstr: "Blocked by corporate proxy",
		},
		{
			name:            "401 unauthorized with json body",
			status:          http.StatusUnauthorized,
			body:            `{"error": "unauthorized access"}`,
			wantStatusCode:  http.StatusUnauthorized,
			wantErrorSubstr: "unauthorized access",
		},
		{
			name:            "404 not found",
			status:          http.StatusNotFound,
			body:            "blob not found",
			wantStatusCode:  http.StatusNotFound,
			wantErrorSubstr: "404",
		},
		{
			name:            "502 bad gateway",
			status:          http.StatusBadGateway,
			body:            "bad gateway",
			wantStatusCode:  http.StatusBadGateway,
			wantErrorSubstr: "502",
		},
		{
			name:            "200 ok for non-zero range",
			status:          http.StatusOK,
			body:            "some data",
			offset:          100,
			wantStatusCode:  http.StatusOK,
			wantErrorSubstr: "range request not supported",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			t.Cleanup(server.Close)

			requestURL, err := url.Parse(server.URL)
			if err != nil {
				t.Fatal(err)
			}

			download := &blobDownload{
				Name:   filepath.Join(t.TempDir(), "blob"),
				Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000",
				Total:  200,
			}
			part := &blobDownloadPart{
				Offset:       tc.offset,
				Size:         100,
				blobDownload: download,
			}

			var buf bytes.Buffer
			ctx, cancel := context.WithTimeout(t.Context(), time.Second)
			defer cancel()

			err = download.downloadChunk(ctx, server.Client(), requestURL, &buf, part)
			if err == nil {
				t.Fatalf("downloadChunk() expected error, got nil")
			}

			var statusErr api.StatusError
			if !errors.As(err, &statusErr) {
				t.Fatalf("downloadChunk() error = %T (%v), want api.StatusError", err, err)
			}
			if statusErr.StatusCode != tc.wantStatusCode {
				t.Errorf("StatusCode = %d, want %d", statusErr.StatusCode, tc.wantStatusCode)
			}
			if !strings.Contains(err.Error(), tc.wantErrorSubstr) {
				t.Errorf("error %q does not contain %q", err.Error(), tc.wantErrorSubstr)
			}
			if buf.Len() != 0 {
				t.Errorf("downloadChunk() wrote %d bytes to writer on error, want 0", buf.Len())
			}
			if part.Completed.Load() != 0 {
				t.Errorf("part.Completed = %d, want 0", part.Completed.Load())
			}
		})
	}
}

func TestIsRetryable(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		{name: "400", err: api.StatusError{StatusCode: http.StatusBadRequest}, want: false},
		{name: "401", err: api.StatusError{StatusCode: http.StatusUnauthorized}, want: false},
		{name: "403", err: api.StatusError{StatusCode: http.StatusForbidden}, want: false},
		{name: "404", err: api.StatusError{StatusCode: http.StatusNotFound}, want: false},
		{name: "408", err: api.StatusError{StatusCode: http.StatusRequestTimeout}, want: true},
		{name: "429", err: api.StatusError{StatusCode: http.StatusTooManyRequests}, want: true},
		{name: "500", err: api.StatusError{StatusCode: http.StatusInternalServerError}, want: true},
		{name: "502", err: api.StatusError{StatusCode: http.StatusBadGateway}, want: true},
		{name: "503", err: api.StatusError{StatusCode: http.StatusServiceUnavailable}, want: true},
		{name: "generic io error", err: io.ErrUnexpectedEOF, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := isRetryable(tc.err); got != tc.want {
				t.Errorf("isRetryable(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func TestBlobDownloadRun403ForbiddenAbortsWithoutRetry(t *testing.T) {
	var blobHits atomic.Int32
	blobServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		blobHits.Add(1)
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte("Blocked by proxy"))
	}))
	t.Cleanup(blobServer.Close)

	registryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, blobServer.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(registryServer.Close)

	registryURL, err := url.Parse(registryServer.URL)
	if err != nil {
		t.Fatal(err)
	}

	downloadPath := filepath.Join(t.TempDir(), "blob")
	download := &blobDownload{
		Name:   downloadPath,
		Digest: "sha256:1234567890123456789012345678901234567890123456789012345678901234",
		Total:  100,
	}
	part := &blobDownloadPart{
		N:            0,
		Offset:       0,
		Size:         100,
		blobDownload: download,
	}
	download.Parts = []*blobDownloadPart{part}

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	start := time.Now()
	err = download.run(ctx, registryURL, &registryOptions{Insecure: true})
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if errors.Is(err, errMaxRetriesExceeded) {
		t.Fatalf("expected immediate failure without retries, got errMaxRetriesExceeded: %v", err)
	}

	var statusErr api.StatusError
	if !errors.As(err, &statusErr) {
		t.Fatalf("expected api.StatusError, got %T: %v", err, err)
	}
	if statusErr.StatusCode != http.StatusForbidden {
		t.Errorf("StatusCode = %d, want %d", statusErr.StatusCode, http.StatusForbidden)
	}
	if !strings.Contains(err.Error(), "Blocked by proxy") {
		t.Errorf("error %q does not contain proxy message", err.Error())
	}
	if got := blobHits.Load(); got != 1 {
		t.Errorf("blob download attempts = %d, want 1 (should not retry 403)", got)
	}
	if elapsed > 2*time.Second {
		t.Errorf("download.run took %v, want < 2s", elapsed)
	}
}
