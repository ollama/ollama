package transfer

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type countedDownloadBody struct {
	io.Reader
	read   int
	closed bool
}

func (b *countedDownloadBody) Read(p []byte) (int, error) {
	n, err := b.Reader.Read(p)
	b.read += n
	return n, err
}

func (b *countedDownloadBody) Close() error {
	b.closed = true
	return nil
}

func TestDownloadReadsSuccessfulResponseOnce(t *testing.T) {
	for _, status := range []int{http.StatusOK, http.StatusTemporaryRedirect, http.StatusUnauthorized} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			blob, data := createTestBlob(t, t.TempDir(), 4096)
			var bodies []*countedDownloadBody
			var controlBody *countedDownloadBody
			requests := 0
			client := &http.Client{
				CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
				Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
					requests++
					if r.Method != http.MethodGet {
						t.Errorf("method = %s, want GET", r.Method)
					}
					resp := &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Request: r}
					if requests == 1 && status != http.StatusOK {
						controlBody = &countedDownloadBody{Reader: bytes.NewReader([]byte("control"))}
						resp.StatusCode = status
						resp.Header.Set("Location", "/blob")
						resp.Header.Set("WWW-Authenticate", `Bearer realm="https://registry.example/token"`)
						resp.Body = controlBody
						return resp, nil
					}
					if status == http.StatusUnauthorized && r.Header.Get("Authorization") != "Bearer refreshed" {
						t.Errorf("Authorization = %q, want refreshed bearer token", r.Header.Get("Authorization"))
					}
					body := &countedDownloadBody{Reader: bytes.NewReader(data)}
					bodies = append(bodies, body)
					resp.Body = body
					return resp, nil
				}),
			}
			dir := t.TempDir()
			if err := Download(t.Context(), DownloadOptions{
				Blobs: []Blob{blob}, BaseURL: "https://registry.example", DestDir: dir, Client: client,
				GetToken: func(context.Context, AuthChallenge) (string, error) { return "refreshed", nil },
			}); err != nil {
				t.Fatal(err)
			}
			verifyBlob(t, dir, blob, data)
			read := 0
			for _, body := range bodies {
				read += body.read
				if !body.closed {
					t.Error("successful response body was not closed")
				}
			}
			if len(bodies) != 1 || read != len(data) {
				t.Errorf("consumed %d bytes from %d successful responses, want %d bytes from one", read, len(bodies), len(data))
			}
			if controlBody != nil && (!controlBody.closed || controlBody.read != len("control")) {
				t.Error("redirect/auth response body was not drained and closed")
			}
		})
	}
}

type stalledDownloadBody func([]byte) (int, error)

func (b stalledDownloadBody) Read(p []byte) (int, error) { return b(p) }

func (stalledDownloadBody) Close() error { return nil }

func TestDownloadInitialResponseStall(t *testing.T) {
	blob, _ := createTestBlob(t, t.TempDir(), 4096)
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		body := stalledDownloadBody(func([]byte) (int, error) {
			<-r.Context().Done()
			return 0, context.Cause(r.Context())
		})
		return &http.Response{StatusCode: http.StatusOK, Body: body, Header: make(http.Header)}, nil
	})}
	d := &downloader{
		client: client, baseURL: "https://registry.example", destDir: t.TempDir(),
		stallTimeout: time.Millisecond, progress: newProgressTracker(blob.Size, nil), speeds: &speedTracker{},
	}
	// The deadline only bounds a broken watchdog; the assertion checks its cause.
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	if _, err := d.downloadOnce(ctx, blob); !errors.Is(err, errStalled) {
		t.Fatalf("downloadOnce() = %v, want %v", err, errStalled)
	}
}

func TestDownloadClosesResponseOnSaveError(t *testing.T) {
	blob, data := createTestBlob(t, t.TempDir(), 4096)
	body := &countedDownloadBody{Reader: bytes.NewReader(data)}
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: body, Header: make(http.Header)}, nil
	})}
	// A regular file cannot contain the downloaded blob.
	destDir := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(destDir, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	d := &downloader{client: client, baseURL: "https://registry.example", destDir: destDir}
	if _, err := d.downloadOnce(t.Context(), blob); err == nil {
		t.Fatal("downloadOnce() succeeded with a file as destination directory")
	}
	if !body.closed || body.read != 0 {
		t.Errorf("response closed = %v, bytes read = %d; want closed without reading the payload", body.closed, body.read)
	}
}
