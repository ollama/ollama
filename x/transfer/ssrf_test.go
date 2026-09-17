package transfer

import (
	"context"
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

// TestDownloadBlocksRedirectToLoopback is the SSRF regression test for
// issue #17041. A registry redirects the blob GET to a loopback address on a
// different hostname than the one the download started from. The download
// must fail and the loopback target must never receive a request.
func TestDownloadBlocksRedirectToLoopback(t *testing.T) {
	serverDir := t.TempDir()
	blob, _ := createTestBlob(t, serverDir, 1024)

	// Loopback target that serves valid blob content and counts hits.
	// Without redirect validation the downloader happily follows the
	// redirect here and the download succeeds.
	var hits atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		digest := filepath.Base(r.URL.Path)
		blobData, err := os.ReadFile(filepath.Join(serverDir, digestToPath(digest)))
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(blobData)))
		w.Write(blobData)
	}))
	defer target.Close()

	targetPort := mustURLPort(t, target.URL)

	// Registry redirects to 127.0.0.1, a different hostname than the
	// "localhost" the download was started with.
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://127.0.0.1:"+targetPort+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer registry.Close()
	registryPort := mustURLPort(t, registry.URL)
	clientDir := t.TempDir()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errc := make(chan error, 1)
	go func() {
		errc <- Download(ctx, DownloadOptions{
			Blobs:   []Blob{blob},
			BaseURL: "http://localhost:" + registryPort,
			DestDir: clientDir,
		})
	}()

	// Give the download enough time to reach the loopback target if the
	// redirect were followed, then cancel to stop the retry loop.
	time.Sleep(2 * time.Second)
	cancel()

	err := <-errc
	if err == nil {
		t.Errorf("download succeeded via a redirect to a loopback address, want error")
	}
	if got := hits.Load(); got != 0 {
		t.Errorf("loopback target received %d requests, want 0", got)
	}
}

// TestDownloadBlocksRedirectToLinkLocal verifies that a redirect to the
// link-local metadata address is rejected before any request is sent to it.
func TestDownloadBlocksRedirectToLinkLocal(t *testing.T) {
	serverDir := t.TempDir()
	blob, _ := createTestBlob(t, serverDir, 1024)

	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://169.254.169.254/latest/meta-data/"+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer registry.Close()
	registryPort := mustURLPort(t, registry.URL)
	clientDir := t.TempDir()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errc := make(chan error, 1)
	go func() {
		errc <- Download(ctx, DownloadOptions{
			Blobs:   []Blob{blob},
			BaseURL: "http://localhost:" + registryPort,
			DestDir: clientDir,
		})
	}()

	time.Sleep(2 * time.Second)
	cancel()

	if err := <-errc; err == nil {
		t.Fatal("expected download to fail when redirected to a link-local address")
	}
}

// TestDownloadBlocksRedirectToPrivateIP verifies that redirects to RFC1918
// addresses are rejected.
func TestDownloadBlocksRedirectToPrivateIP(t *testing.T) {
	serverDir := t.TempDir()
	blob, _ := createTestBlob(t, serverDir, 1024)

	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://192.168.1.1"+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer registry.Close()
	registryPort := mustURLPort(t, registry.URL)
	clientDir := t.TempDir()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errc := make(chan error, 1)
	go func() {
		errc <- Download(ctx, DownloadOptions{
			Blobs:   []Blob{blob},
			BaseURL: "http://localhost:" + registryPort,
			DestDir: clientDir,
		})
	}()

	time.Sleep(2 * time.Second)
	cancel()

	if err := <-errc; err == nil {
		t.Fatal("expected download to fail when redirected to a private address")
	}
}

// TestDownloadWithSameHostRedirect verifies that same-host redirects remain
// allowed: private registries that redirect within themselves are a
// legitimate configuration.
func TestDownloadWithSameHostRedirect(t *testing.T) {
	serverDir := t.TempDir()
	blob, data := createTestBlob(t, serverDir, 1024)

	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/cdn") {
			digest := filepath.Base(r.URL.Path)
			blobData, err := os.ReadFile(filepath.Join(serverDir, digestToPath(digest)))
			if err != nil {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Length", fmt.Sprintf("%d", len(blobData)))
			w.Write(blobData)
			return
		}
		http.Redirect(w, r, "/cdn"+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer registry.Close()

	clientDir := t.TempDir()
	err := Download(context.Background(), DownloadOptions{
		Blobs:   []Blob{blob},
		BaseURL: registry.URL,
		DestDir: clientDir,
	})
	if err != nil {
		t.Fatalf("Download with same-host redirect failed: %v", err)
	}

	verifyBlob(t, clientDir, blob, data)
}

// TestDownloadBlocksRedirectToLoopbackHostname verifies that a redirect to a
// hostname (rather than an IP literal) that resolves to a loopback address is
// also rejected.
func TestDownloadBlocksRedirectToLoopbackHostname(t *testing.T) {
	serverDir := t.TempDir()
	blob, _ := createTestBlob(t, serverDir, 1024)

	var hits atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	targetPort := mustURLPort(t, target.URL)

	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// "localhost" differs from the "127.0.0.1" hostname the download
		// started with and resolves to a loopback address.
		http.Redirect(w, r, "http://localhost:"+targetPort+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer registry.Close()
	registryPort := mustURLPort(t, registry.URL)
	clientDir := t.TempDir()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errc := make(chan error, 1)
	go func() {
		errc <- Download(ctx, DownloadOptions{
			Blobs:   []Blob{blob},
			BaseURL: "http://127.0.0.1:" + registryPort,
			DestDir: clientDir,
		})
	}()

	time.Sleep(2 * time.Second)
	cancel()

	err := <-errc
	if err == nil {
		t.Errorf("download succeeded via a redirect to a loopback hostname, want error")
	}
	if got := hits.Load(); got != 0 {
		t.Errorf("loopback target received %d requests, want 0", got)
	}
}

// TestDownloadBlocksRedirectToUnresolvableHost verifies fail-closed behavior:
// a redirect to a hostname that cannot be resolved is rejected rather than
// followed.
func TestDownloadBlocksRedirectToUnresolvableHost(t *testing.T) {
	serverDir := t.TempDir()
	blob, _ := createTestBlob(t, serverDir, 1024)

	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// .invalid is a reserved TLD and never resolves.
		http.Redirect(w, r, "http://nonexistent.invalid"+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer registry.Close()
	registryPort := mustURLPort(t, registry.URL)
	clientDir := t.TempDir()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	errc := make(chan error, 1)
	go func() {
		errc <- Download(ctx, DownloadOptions{
			Blobs:   []Blob{blob},
			BaseURL: "http://localhost:" + registryPort,
			DestDir: clientDir,
		})
	}()

	time.Sleep(2 * time.Second)
	cancel()

	if err := <-errc; err == nil {
		t.Fatal("expected download to fail when redirected to an unresolvable host")
	}
}

// TestRedirectGuardPreservesDefaultBehavior verifies that clients without an
// existing CheckRedirect keep following allowed (same-hostname) redirects.
func TestRedirectGuardPreservesDefaultBehavior(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/blob") {
			w.WriteHeader(http.StatusOK)
			return
		}
		http.Redirect(w, r, "/blob"+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer server.Close()

	// httptest.Server.Client() has no CheckRedirect, so allowed redirects
	// should still be followed automatically.
	client := redirectGuard(server.Client())
	resp, err := client.Get(server.URL + "/v2/library/_/blobs/sha256:abc")
	if err != nil {
		t.Fatalf("GET with allowed redirect failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("unexpected status after allowed redirect: %d", resp.StatusCode)
	}
}

// TestValidateRedirectHost covers the address classification rules directly,
// including addresses not reachable through the download tests.
func TestValidateRedirectHost(t *testing.T) {
	tests := []struct {
		name    string
		rawURL  string
		wantErr bool
	}{
		{name: "public IP literal", rawURL: "http://8.8.8.8/x", wantErr: false},
		{name: "loopback IPv4 literal", rawURL: "http://127.0.0.1/x", wantErr: true},
		{name: "link-local IPv4 literal", rawURL: "http://169.254.169.254/latest/meta-data/", wantErr: true},
		{name: "private IPv4 literal", rawURL: "http://10.0.0.1/x", wantErr: true},
		{name: "multicast IPv4 literal", rawURL: "http://224.0.0.1/x", wantErr: true},
		{name: "unspecified IPv4 literal", rawURL: "http://0.0.0.0/x", wantErr: true},
		{name: "loopback IPv6 literal", rawURL: "http://[::1]/x", wantErr: true},
		{name: "link-local IPv6 literal", rawURL: "http://[fe80::1]/x", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u, err := url.Parse(tt.rawURL)
			if err != nil {
				t.Fatal(err)
			}
			err = validateRedirectHost(context.Background(), u)
			if (err != nil) != tt.wantErr {
				t.Fatalf("validateRedirectHost(%q) error = %v, wantErr %v", tt.rawURL, err, tt.wantErr)
			}
		})
	}
}

// TestValidateRedirectHostAllowsPublicHostname covers the branch where a
// hostname resolves and every address is publicly routable. The default
// resolver is stubbed with a fake DNS server that answers A/AAAA queries with
// public addresses, so the test does not touch the real network.
func TestValidateRedirectHostAllowsPublicHostname(t *testing.T) {
	pc, err := net.ListenPacket("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer pc.Close()

	go func() {
		buf := make([]byte, 512)
		for {
			n, addr, err := pc.ReadFrom(buf)
			if err != nil {
				return
			}
			if resp := fakeDNSResponse(buf[:n]); resp != nil {
				pc.WriteTo(resp, addr)
			}
		}
	}()

	prev := net.DefaultResolver
	t.Cleanup(func() { net.DefaultResolver = prev })
	net.DefaultResolver = &net.Resolver{
		PreferGo: true,
		Dial: func(ctx context.Context, network, address string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, network, pc.LocalAddr().String())
		},
	}

	u, err := url.Parse("http://cdn.example.com/blob")
	if err != nil {
		t.Fatal(err)
	}
	if err := validateRedirectHost(context.Background(), u); err != nil {
		t.Fatalf("validateRedirectHost(public hostname) = %v, want nil", err)
	}
}

// fakeDNSResponse encodes a minimal DNS response that echoes the query's
// question and answers it with a public address (8.8.8.8 for A queries,
// 2001:4860:4860::8888 for AAAA queries).
func fakeDNSResponse(query []byte) []byte {
	if len(query) < 12 {
		return nil
	}
	// Find the end of the question section: labels until a zero length
	// byte, then QTYPE and QCLASS (4 bytes).
	end := 12
	for end < len(query) && query[end] != 0 {
		end += int(query[end]) + 1
	}
	if end >= len(query) {
		return nil
	}
	end += 1 + 4 // zero length byte + QTYPE + QCLASS
	if end > len(query) {
		return nil
	}
	qtype := query[end-4 : end-2]

	resp := make([]byte, 0, end+16)
	resp = append(resp, query[0:2]...)          // ID
	resp = append(resp, 0x81, 0x80)             // QR|RD|RA, RCODE 0
	resp = append(resp, query[4:6]...)          // QDCOUNT
	resp = append(resp, 0x00, 0x01)             // ANCOUNT = 1
	resp = append(resp, 0x00, 0x00)             // NSCOUNT = 0
	resp = append(resp, 0x00, 0x00)             // ARCOUNT = 0
	resp = append(resp, query[12:end]...)       // question, verbatim
	resp = append(resp, 0xC0, 0x0C)             // answer name: pointer to offset 12
	resp = append(resp, qtype...)               // answer type matches question
	resp = append(resp, 0x00, 0x01)             // class IN
	resp = append(resp, 0x00, 0x00, 0x00, 0x3C) // TTL 60
	if qtype[0] == 0 && qtype[1] == 1 {         // A
		resp = append(resp, 0x00, 0x04, 8, 8, 8, 8)
	} else { // AAAA
		resp = append(resp, 0x00, 0x10,
			0x20, 0x01, 0x48, 0x60, 0x48, 0x60, 0, 0, 0, 0, 0, 0, 0, 0, 0x88, 0x88)
	}
	return resp
}

func mustURLPort(t *testing.T, rawURL string) string {
	t.Helper()
	u, err := url.Parse(rawURL)
	if err != nil {
		t.Fatal(err)
	}
	return u.Port()
}
