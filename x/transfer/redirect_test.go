package transfer

import (
	"context"
	"errors"
	"fmt"
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
)

// SSRF regression tests for CVE-2026-85180: a registry can answer a blob
// request with a redirect to an internal address. By default the transfer
// package must refuse to fetch it; AllowPrivateHosts opts out for trusted
// LAN/local registries.

func TestDownloadRedirectToPrivateHostDenied(t *testing.T) {
	blob, _ := createTestBlob(t, t.TempDir(), 1024)

	var internalHit atomic.Bool
	internal := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		internalHit.Store(true)
	}))
	defer internal.Close()

	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, internal.URL+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer registry.Close()

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: registry.URL,
		DestDir: t.TempDir(),
	})
	if !errors.Is(err, errRedirectNotAllowed) {
		t.Fatalf("expected errRedirectNotAllowed, got %v", err)
	}
	if internalHit.Load() {
		t.Error("internal host received a request despite redirect rejection")
	}
}

func TestDownloadRedirectToPrivateHostAllowedWithOptIn(t *testing.T) {
	cdnDir := t.TempDir()
	blob, data := createTestBlob(t, cdnDir, 1024)

	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := filepath.Join(cdnDir, digestToPath(filepath.Base(r.URL.Path)))
		data, err := os.ReadFile(path)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.Write(data)
	}))
	defer cdn.Close()

	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, cdn.URL+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer registry.Close()

	clientDir := t.TempDir()
	err := Download(context.Background(), DownloadOptions{
		Blobs:             []Blob{blob},
		BaseURL:           registry.URL,
		DestDir:           clientDir,
		AllowPrivateHosts: true,
	})
	if err != nil {
		t.Fatalf("download with AllowPrivateHosts failed: %v", err)
	}
	verifyBlob(t, clientDir, blob, data)
}

func TestDownloadRedirectMetadataEndpoint(t *testing.T) {
	blob, _ := createTestBlob(t, t.TempDir(), 1024)

	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://169.254.169.254/latest/meta-data/", http.StatusFound)
	}))
	defer registry.Close()

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: registry.URL,
		DestDir: t.TempDir(),
	})
	if !errors.Is(err, errRedirectNotAllowed) {
		t.Fatalf("expected errRedirectNotAllowed, got %v", err)
	}
}

func TestUploadRedirectToPrivateHostDenied(t *testing.T) {
	clientDir := t.TempDir()
	blob, _ := createTestBlob(t, clientDir, 1024)

	var internalHit atomic.Bool
	internal := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		internalHit.Store(true)
	}))
	defer internal.Close()

	// Minimal registry: HEAD 404, POST hands an absolute session URL on the
	// internal host — simulating a hostile Docker-Upload-Location.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)
		case http.MethodPost:
			w.Header().Set("Location", internal.URL+"/v2/library/_/blobs/uploads/1")
			w.WriteHeader(http.StatusAccepted)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
	})
	if !errors.Is(err, errRedirectNotAllowed) {
		t.Fatalf("expected errRedirectNotAllowed, got %v", err)
	}
	if internalHit.Load() {
		t.Error("internal host received a request despite rejection")
	}
}

func TestUploadPatchRedirectToPrivateHostDenied(t *testing.T) {
	clientDir := t.TempDir()
	blob, _ := createTestBlob(t, clientDir, 1024)

	var internalHit atomic.Bool
	internal := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		internalHit.Store(true)
	}))
	defer internal.Close()

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)
		case http.MethodPost:
			w.Header().Set("Location", serverURL+"/v2/library/_/blobs/uploads/1")
			w.WriteHeader(http.StatusAccepted)
		case http.MethodPatch:
			// 307 the part body at the internal host
			w.Header().Set("Docker-Upload-Location", r.URL.Path)
			http.Redirect(w, r, internal.URL+r.URL.Path, http.StatusTemporaryRedirect)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: server.URL,
		SrcDir:  clientDir,
	})
	if !errors.Is(err, errRedirectNotAllowed) {
		t.Fatalf("expected errRedirectNotAllowed, got %v", err)
	}
	if internalHit.Load() {
		t.Error("internal host received a request despite rejection")
	}
}

func TestCheckedDialerRejectsRebind(t *testing.T) {
	// A hostname resolving to a private address must be refused at dial
	// time even when the URL-level validation step is bypassed or raced
	// (DNS rebinding). localhost deterministically resolves to loopback.
	dial := checkedDialer(&net.Dialer{Timeout: 2 * time.Second}, false)
	conn, err := dial(context.Background(), "tcp", "localhost:1")
	if !strings.Contains(fmt.Sprint(err), "not allowed") {
		t.Errorf("expected dns rejection, got conn=%v err=%v", conn, err)
	}

	// With the escape hatch, the same dial is permitted (then fails
	// normally on connection refused for port 1).
	dial = checkedDialer(&net.Dialer{Timeout: 2 * time.Second}, true)
	conn, err = dial(context.Background(), "tcp", "localhost:1")
	if err != nil && strings.Contains(fmt.Sprint(err), "not allowed") {
		t.Errorf("allowPrivate dial was still rejected: %v", err)
	}
	if conn != nil {
		conn.Close()
	}

	// IP literals are checked without DNS.
	dial = checkedDialer(&net.Dialer{Timeout: 2 * time.Second}, false)
	if _, err := dial(context.Background(), "tcp", "169.254.169.254:80"); !strings.Contains(fmt.Sprint(err), "not allowed") {
		t.Errorf("metadata IP dial not rejected: %v", err)
	}
}

func TestCheckedClientExemptsBaseHost(t *testing.T) {
	// The registry base host is caller-directed, so it must be exempt from
	// the public-IP dial check even when the base URL has no explicit port —
	// otherwise a private registry without AllowPrivateHosts breaks on
	// default ports (regression: dial addr carries ":443", baseHost didn't).
	for _, addr := range []string{"127.0.0.1:443", "localhost:443"} {
		tr := checkedClient("https://127.0.0.1", false).Transport.(*http.Transport)
		if addr == "localhost:443" {
			tr = checkedClient("https://localhost", false).Transport.(*http.Transport)
		}
		_, err := tr.DialContext(t.Context(), "tcp", addr)
		if strings.Contains(fmt.Sprint(err), "not allowed") {
			t.Errorf("base host %s was validated at dial time: %v", addr, err)
		}
		// A real dial is expected to fail with connection refused here (or
		// succeed if something listens) — only the policy rejection is a bug.
	}
}

func TestValidateRedirectTarget(t *testing.T) {
	ctx := context.Background()
	okPublic, _ := url.Parse("https://203.0.113.10/blob")
	httpsBase := "https://registry.example.com"
	httpBase := "http://192.168.1.10:5000"

	cases := []struct {
		name         string
		raw          string
		base         string
		allowPrivate bool
		wantErr      bool
	}{
		{"public https allowed", "https://203.0.113.10/blob", httpsBase, false, false},
		{"metadata denied", "http://169.254.169.254/latest/meta-data", httpsBase, false, true},
		{"loopback denied", "http://127.0.0.1:11434/api/tags", httpsBase, false, true},
		{"rfc1918 denied", "http://10.0.0.5/internal", httpsBase, false, true},
		{"cgnat denied", "http://100.64.1.1/internal", httpsBase, false, true},
		{"alibaba metadata denied", "http://100.100.2.148/latest/meta-data", httpsBase, false, true},
		{"cgnat 4-in-6 denied", "http://[::ffff:100.64.1.1]/internal", httpsBase, false, true},
		{"benchmarking net denied", "http://198.18.0.1/bench", httpsBase, false, true},
		{"ipv6 link-local denied", "http://[fe80::1]/x", httpsBase, false, true},
		{"localhost hostname denied", "http://localhost/x", httpsBase, false, true},
		{"bad scheme denied", "file:///etc/passwd", httpsBase, false, true},
		{"https registry downgrade to http denied", "http://203.0.113.10/blob", httpsBase, false, true},
		{"unresolvable name fails closed", "https://no-such-host.invalid/blob", httpsBase, false, true},
		{"http registry may redirect to public http", "http://203.0.113.10/blob", httpBase, false, false},
		{"private allowed with opt-in", "http://192.168.1.20/cdn/blob", httpsBase, true, false},
		{"metadata allowed with opt-in", "http://169.254.169.254/latest", httpsBase, true, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			u, _ := url.Parse(tc.raw)
			err := validateRedirectTarget(ctx, u, tc.base, tc.allowPrivate)
			if tc.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}

	// The public-IP literal case must pass too (no DNS involved).
	if err := validateRedirectTarget(ctx, okPublic, httpsBase, false); err != nil {
		t.Errorf("public https target rejected: %v", err)
	}
}

// Same-host HTTPS→HTTP redirects must be rejected even though the host is
// unchanged — otherwise a registry can strip TLS and rebinding DNS steers
// the cleartext request at an internal address.
func TestDownloadSameHostDowngradeRedirectDenied(t *testing.T) {
	blob, _ := createTestBlob(t, t.TempDir(), 1024)

	var serverURL string
	registry := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Redirect to the same host:port, but cleartext.
		u, _ := url.Parse(serverURL)
		u.Scheme = "http"
		http.Redirect(w, r, u.String()+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer registry.Close()
	serverURL = registry.URL

	// The checked dialer wouldn't trust the test cert, so use the server's
	// own TLS config; but mirror checkedClient's no-auto-follow policy so
	// resolve() sees the 307 and the redirect policy is under test.
	client := registry.Client()
	client.CheckRedirect = func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}

	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: registry.URL,
		DestDir: t.TempDir(),
		Client:  client,
	})
	if !errors.Is(err, errRedirectNotAllowed) {
		t.Fatalf("expected errRedirectNotAllowed, got %v", err)
	}
}

func TestUploadSameHostDowngradeSessionURLDenied(t *testing.T) {
	clientDir := t.TempDir()
	blob, _ := createTestBlob(t, clientDir, 1024)

	var serverURL string
	registry := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			http.NotFound(w, r)
		case http.MethodPost:
			// Hand back a same-host session URL, but cleartext.
			u, _ := url.Parse(serverURL)
			u.Scheme = "http"
			w.Header().Set("Location", u.String()+"/v2/library/_/blobs/uploads/1")
			w.WriteHeader(http.StatusAccepted)
		default:
			http.NotFound(w, r)
		}
	}))
	defer registry.Close()
	serverURL = registry.URL

	err := Upload(context.Background(), UploadOptions{
		Blobs:   []Blob{blob},
		BaseURL: registry.URL,
		SrcDir:  clientDir,
		Client:  registry.Client(),
	})
	if !errors.Is(err, errRedirectNotAllowed) {
		t.Fatalf("expected errRedirectNotAllowed, got %v", err)
	}
}

func TestValidateRedirectScheme(t *testing.T) {
	cases := []struct {
		name    string
		raw     string
		base    string
		wantErr bool
	}{
		{"https to same-host https ok", "https://registry.example.com/v2/x", "https://registry.example.com", false},
		{"https to same-host http denied", "http://registry.example.com/v2/x", "https://registry.example.com", true},
		{"http registry to same-host http ok", "http://192.168.1.10:5000/v2/x", "http://192.168.1.10:5000", false},
		{"bad scheme denied", "file:///etc/passwd", "https://registry.example.com", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			u, _ := url.Parse(tc.raw)
			err := validateRedirectScheme(u, tc.base)
			if tc.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
