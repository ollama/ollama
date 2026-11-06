package transfer

import (
	"context"
	"crypto/tls"
	"crypto/x509"
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
)

// SSRF regression tests: a registry can answer a blob request with a
// redirect to an internal address. By default the transfer package must
// refuse to fetch it; AllowPrivateHosts opts out for trusted LAN/local
// registries.

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
		tr := checkedClient("https://127.0.0.1", false).Transport.(*checkedTransport).direct
		if addr == "localhost:443" {
			tr = checkedClient("https://localhost", false).Transport.(*checkedTransport).direct
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
	okPublic, _ := url.Parse("https://8.8.8.8/blob")
	httpsBase := "https://registry.example.com"
	httpBase := "http://192.168.1.10:5000"

	cases := []struct {
		name         string
		raw          string
		base         string
		allowPrivate bool
		wantErr      bool
	}{
		{"public https allowed", "https://8.8.8.8/blob", httpsBase, false, false},
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
		{"https registry downgrade to http denied", "http://8.8.8.8/blob", httpsBase, false, true},
		{"unresolvable name fails closed", "https://no-such-host.invalid/blob", httpsBase, false, true},
		{"http registry may redirect to public http", "http://8.8.8.8/blob", httpBase, false, false},
		{"private allowed with opt-in", "http://192.168.1.20/cdn/blob", httpBase, true, false},
		{"metadata allowed with opt-in", "http://169.254.169.254/latest", httpBase, true, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			u, _ := url.Parse(tc.raw)
			err := ValidateRedirectTarget(ctx, u, tc.base, tc.allowPrivate)
			if tc.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}

	// The public-IP literal case must pass too (no DNS involved).
	if err := ValidateRedirectTarget(ctx, okPublic, httpsBase, false); err != nil {
		t.Errorf("public https target rejected: %v", err)
	}
}

func TestIsPublicIPRejectsSpecialUseRanges(t *testing.T) {
	for _, tc := range []struct {
		address string
		public  bool
	}{
		{"8.8.8.8", true},
		{"2606:4700:4700::1111", true},
		{"0.0.0.1", false},
		{"192.0.0.8", false},
		{"192.0.2.1", false},
		{"198.51.100.1", false},
		{"203.0.113.1", false},
		{"240.0.0.1", false},
		{"::ffff:0.0.0.1", false},
		{"64:ff9b::a9fe:a9fe", false}, // NAT64 can translate to link-local IPv4
		{"100::1", false},
		{"2001::1", false},
		{"2001:db8::1", false},
		{"2002::1", false},
		{"3fff::1", false},
	} {
		t.Run(tc.address, func(t *testing.T) {
			if got := isPublicIP(net.ParseIP(tc.address)); got != tc.public {
				t.Errorf("isPublicIP(%s) = %t, want %t", tc.address, got, tc.public)
			}
		})
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
			err := ValidateRedirectScheme(u, tc.base)
			if tc.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

// withProxy points proxyFromEnvironment at proxy for the duration of the test.
func withProxy(t *testing.T, proxy *url.URL) {
	t.Helper()
	orig := proxyFromEnvironment
	proxyFromEnvironment = func(*http.Request) (*url.URL, error) { return proxy, nil }
	t.Cleanup(func() { proxyFromEnvironment = orig })
}

func TestRedirectClientUsesProxy(t *testing.T) {
	// Behind a proxy the CDN host is unreachable directly; the request
	// must be handed to the proxy instead of dialed. .invalid never
	// resolves locally, so success proves the request went via the proxy.
	var gotHost atomic.Value
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHost.Store(r.URL.Host)
		fmt.Fprint(w, "blob")
	}))
	defer proxy.Close()
	proxyURL, _ := url.Parse(proxy.URL)
	withProxy(t, proxyURL)

	client := NewRedirectClient("http://registry.example", false)
	resp, err := client.Get("http://blobs.example.invalid/sha256-abc")
	if err != nil {
		t.Fatalf("download through proxy failed: %v", err)
	}
	defer resp.Body.Close()
	if got := gotHost.Load(); got != "blobs.example.invalid" {
		t.Errorf("proxy saw host %v, want blobs.example.invalid", got)
	}
}

func TestConcurrentProxyAndDirectRequests(t *testing.T) {
	var hits atomic.Int32
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		fmt.Fprint(w, "ok")
	}))
	defer proxy.Close()
	proxyURL, _ := url.Parse(proxy.URL)
	original := proxyFromEnvironment
	proxyFromEnvironment = func(req *http.Request) (*url.URL, error) {
		if req.URL.Hostname() == "cdn.example.invalid" {
			return proxyURL, nil
		}
		return nil, nil
	}
	t.Cleanup(func() { proxyFromEnvironment = original })
	client := checkedClient("http://registry.example", false)
	defer client.CloseIdleConnections()
	const requests = 32
	errCh := make(chan error, requests*2)
	for range requests {
		go func() {
			resp, err := client.Get("http://cdn.example.invalid/blob")
			if err == nil {
				resp.Body.Close()
			}
			errCh <- err
		}()
		go func() {
			resp, err := client.Get(proxy.URL + "/private")
			if resp != nil {
				resp.Body.Close()
			}
			if !errors.Is(err, errRedirectNotAllowed) {
				errCh <- fmt.Errorf("direct request to private proxy address: %v", err)
				return
			}
			errCh <- nil
		}()
	}
	for range requests * 2 {
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
	}
	if got := hits.Load(); got != requests {
		t.Fatalf("proxy received %d requests, want %d", got, requests)
	}
}

func TestValidateRedirectTargetBehindProxy(t *testing.T) {
	// When the proxy will carry the request, redirect validation must not
	// consult local DNS at all: resolution outcomes are registry-influenced
	// and must not be usable as a validation oracle. Targets that will be
	// dialed directly keep the local resolution requirement.
	unresolvable, _ := url.Parse("https://blobs.example.invalid/x")
	private, _ := url.Parse("https://localhost/x")

	// No proxy: unresolvable fails closed, private is rejected.
	if err := ValidateRedirectTarget(t.Context(), unresolvable, "https://registry.example", false); err == nil {
		t.Error("unresolvable target allowed without a proxy")
	}
	if err := ValidateRedirectTarget(t.Context(), private, "https://registry.example", false); err == nil {
		t.Error("private target allowed without a proxy")
	}

	proxyURL, _ := url.Parse("http://127.0.0.1:3128")
	withProxy(t, proxyURL)

	// Proxy configured: an unresolvable name is deferred to the proxy, and
	// an explicit private IP literal is still rejected — it needs no
	// resolution, so the proxy changes nothing.
	if err := ValidateRedirectTarget(t.Context(), unresolvable, "https://registry.example", false); err != nil {
		t.Errorf("proxy-routed target rejected: %v", err)
	}
	privateIP, _ := url.Parse("https://169.254.169.254/latest/meta-data")
	if err := ValidateRedirectTarget(t.Context(), privateIP, "https://registry.example", false); err == nil {
		t.Error("private IP literal allowed behind a proxy")
	}
}

func TestValidateRedirectTargetDefaultRegistryHostsSkipLocalDNS(t *testing.T) {
	// The default registry and its CDN hostnames are trusted without a
	// local DNS lookup, so a proxied (or DNS-less) host still pulls. This
	// must not relax checks for any other hostname, nor for IP literals.
	for _, raw := range []string{
		"https://dd20bb891979d25aebc8bec07b2b3bbc.r2.cloudflarestorage.com/x",
		"https://registry.ollama.com/x",
		"https://ollama.com/x",
		"https://cas-bridge.xethub.hf.co/x",
	} {
		u, _ := url.Parse(raw)
		if err := ValidateRedirectTarget(t.Context(), u, "https://registry.ollama.com", false); err != nil {
			t.Errorf("%s: unexpected rejection: %v", raw, err)
		}
	}

	// A lookalike host that merely ends in the same string is not allowed.
	lookalike, _ := url.Parse("https://evilregistryollama.com/x")
	if err := ValidateRedirectTarget(t.Context(), lookalike, "https://registry.ollama.com", false); err == nil {
		t.Error("lookalike host allowed without DNS resolution")
	}

	// An unrecognized hostname still fails closed with no local DNS answer.
	other, _ := url.Parse("https://no-such-host.invalid/x")
	if err := ValidateRedirectTarget(t.Context(), other, "https://registry.ollama.com", false); err == nil {
		t.Error("unrecognized host allowed without DNS resolution")
	}
}

// TestDownloadThroughProxyCONNECT verifies that behind a proxy, a registry
// 307 to an https CDN URL that cannot resolve locally still completes the
// pull: discovery redirect and blob bytes both traverse the proxy via CONNECT.
func TestDownloadThroughProxyCONNECT(t *testing.T) {
	blob, data := createTestBlob(t, t.TempDir(), 1024)

	cdn := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(data)
	}))
	defer cdn.Close()

	// CONNECT proxy: tunnel to the real CDN listener, record what was asked;
	// serve the http registry hop as a plain forward proxy.
	var connectHost atomic.Value
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodConnect {
			resp, err := http.DefaultTransport.RoundTrip(r)
			if err != nil {
				w.WriteHeader(http.StatusBadGateway)
				return
			}
			defer resp.Body.Close()
			for k := range resp.Header {
				w.Header()[k] = resp.Header[k]
			}
			w.WriteHeader(resp.StatusCode)
			io.Copy(w, resp.Body)
			return
		}
		connectHost.Store(r.Host)
		client, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		upstream, err := net.Dial("tcp", cdn.Listener.Addr().String())
		if err != nil {
			client.Close()
			t.Error(err)
			return
		}
		fmt.Fprint(client, "HTTP/1.1 200 Connection Established\r\n\r\n")
		go func() { io.Copy(upstream, client); upstream.Close() }()
		go func() { io.Copy(client, upstream); client.Close() }()
	}))
	defer proxy.Close()
	proxyURL, _ := url.Parse(proxy.URL)
	withProxy(t, proxyURL)

	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "https://cdn.example.com/"+digestToPath(blob.Digest), http.StatusTemporaryRedirect)
	}))
	defer registry.Close()

	client := checkedClient(registry.URL, false)
	defer client.CloseIdleConnections()
	// Trust the CDN test cert on the cached proxy transport.
	pool := x509.NewCertPool()
	pool.AddCert(cdn.Certificate())
	tr := client.Transport.(*checkedTransport).proxyTransport(proxyTransportKey{proxy: proxyURL.String()}, proxyURL)
	tr.TLSClientConfig = &tls.Config{RootCAs: pool}

	dest := t.TempDir()
	if err := Download(t.Context(), DownloadOptions{Blobs: []Blob{blob}, BaseURL: registry.URL, DestDir: dest, Client: client}); err != nil {
		t.Fatalf("Download through proxy: %v", err)
	}
	verifyBlob(t, dest, blob, data)
	if connectHost.Load() != "cdn.example.com:443" {
		t.Errorf("CONNECT = %v, want cdn.example.com:443", connectHost.Load())
	}
}

func TestUploadPatchContinuationRejectsPrivateHost(t *testing.T) {
	var privateHits atomic.Int32
	private := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		privateHits.Add(1)
	}))
	defer private.Close()
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Docker-Upload-Location", private.URL+"/next")
		w.WriteHeader(http.StatusAccepted)
	}))
	defer registry.Close()
	f, err := os.CreateTemp(t.TempDir(), "blob")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.WriteString("blob"); err != nil {
		t.Fatal(err)
	}
	session, _ := url.Parse(registry.URL + "/upload")
	u := &uploader{
		client: checkedClient(registry.URL, false), baseURL: registry.URL,
		progress: newProgressTracker(4, nil), token: "secret",
	}
	defer u.client.CloseIdleConnections()
	_, _, _, err = u.uploadOnePart(t.Context(), session, &uploadPart{size: 4}, f)
	if !errors.Is(err, errRedirectNotAllowed) {
		t.Fatalf("private continuation allowed: %v", err)
	}
	if privateHits.Load() != 0 {
		t.Fatal("private endpoint received an upload request")
	}
}
