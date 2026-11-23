package transfer

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

var errRedirectNotAllowed = errors.New("redirect target not allowed")

const maxProxyTransports = 128

// lookupRedirectIPs is replaceable by tests to exercise proxy-only DNS.
var lookupRedirectIPs = net.DefaultResolver.LookupIP

func publicRedirectIPs(ctx context.Context, host string) ([]net.IP, error) {
	if ip := net.ParseIP(host); ip != nil {
		if !isPublicIP(ip) {
			return nil, fmt.Errorf("%w: %s is not a public address", errRedirectNotAllowed, ip)
		}
		return []net.IP{ip}, nil
	}
	ips, err := lookupRedirectIPs(ctx, "ip", host)
	if err != nil {
		return nil, fmt.Errorf("%w: resolving %s: %w", errRedirectNotAllowed, host, err)
	}
	if len(ips) == 0 {
		return nil, fmt.Errorf("%w: %s has no addresses", errRedirectNotAllowed, host)
	}
	for _, ip := range ips {
		if !isPublicIP(ip) {
			return nil, fmt.Errorf("%w: %s resolves to non-public %s", errRedirectNotAllowed, host, ip)
		}
	}
	return ips, nil
}

// net.IP.IsGlobalUnicast includes special-use addresses that are not
// globally reachable. Keep the redirect allowlist conservative. See the
// IANA IPv4/IPv6 special-purpose registries.
var blockedIPv4Nets = []*net.IPNet{
	parseIPNet("0.0.0.0/8"),
	parseIPNet("100.64.0.0/10"),   // shared address space, including cloud metadata
	parseIPNet("192.0.0.0/24"),    // protocol assignments
	parseIPNet("192.0.2.0/24"),    // documentation
	parseIPNet("192.88.99.0/24"),  // deprecated 6to4 relay
	parseIPNet("198.18.0.0/15"),   // benchmarking
	parseIPNet("198.51.100.0/24"), // documentation
	parseIPNet("203.0.113.0/24"),  // documentation
	parseIPNet("240.0.0.0/4"),     // reserved
}

// Current IPv6 global unicast allocations are within 2000::/3. Deny the
// special-purpose ranges in that block too, including transition prefixes.
var (
	publicIPv6Net   = parseIPNet("2000::/3")
	blockedIPv6Nets = []*net.IPNet{
		parseIPNet("2001::/23"),
		parseIPNet("2001:db8::/32"),
		parseIPNet("2002::/16"),
		parseIPNet("3ffe::/16"),
		parseIPNet("3fff::/20"),
	}
)

func parseIPNet(s string) *net.IPNet {
	_, n, err := net.ParseCIDR(s)
	if err != nil {
		panic(err)
	}
	return n
}

// isPublicIP reports whether ip is a globally routed address.
func isPublicIP(ip net.IP) bool {
	if !ip.IsGlobalUnicast() ||
		ip.IsLoopback() ||
		ip.IsPrivate() ||
		ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() ||
		ip.IsUnspecified() {
		return false
	}
	// To4() unwraps 4-in-6 mapped addresses so mapped special-use ranges
	// are blocked too.
	if v4 := ip.To4(); v4 != nil {
		for _, n := range blockedIPv4Nets {
			if n.Contains(v4) {
				return false
			}
		}
		return true
	}
	if !publicIPv6Net.Contains(ip) {
		return false
	}
	for _, n := range blockedIPv6Nets {
		if n.Contains(ip) {
			return false
		}
	}
	return true
}

// ValidateRedirectScheme rejects redirects that downgrade an https session
// to cleartext http, even when the target host is unchanged — a hostile or
// compromised registry must not be able to strip TLS off follow-up requests.
// It applies even under allowPrivate (the --insecure opt-in), which relaxes
// address checks but never scheme checks.
func ValidateRedirectScheme(loc *url.URL, baseURL string) error {
	if loc == nil {
		return fmt.Errorf("%w: missing Location", errRedirectNotAllowed)
	}
	if loc.Scheme != "https" && loc.Scheme != "http" {
		return fmt.Errorf("%w: scheme %q", errRedirectNotAllowed, loc.Scheme)
	}
	// http is only acceptable when the registry itself was already reached
	// over plain http (i.e. the caller opted into an insecure registry).
	base, _ := url.Parse(baseURL)
	if loc.Scheme == "http" && base != nil && base.Scheme == "https" {
		return fmt.Errorf("%w: https registry redirecting to http", errRedirectNotAllowed)
	}
	return nil
}

// allowedRedirectHosts are the hosts the default registry and our CDN
// actually redirect blobs to. A hostname redirect target matching one of
// these is trusted without a local DNS lookup, so proxied hosts with no
// external resolver still pull. Any other hostname still resolves locally
// and must be public.
var allowedRedirectHosts = []string{
	"ollama.com",
	"ollama.ai",
	"hf.co",
	"huggingface.co",
	"cloudflarestorage.com",
}

func isAllowedRedirectHost(host string) bool {
	for _, h := range allowedRedirectHosts {
		if strings.EqualFold(host, h) || strings.HasSuffix(strings.ToLower(host), "."+h) {
			return true
		}
	}
	return false
}

// ValidateRedirectTarget checks a registry-supplied target. When a proxy
// carries the request, or the target is a default-registry host we redirect
// to anyway, name resolution is deferred — the proxy enforces its own access
// rules and no local DNS is needed. Otherwise the target must resolve
// locally to public addresses. Explicit private IP literals are always
// rejected unless allowPrivate is set.
func ValidateRedirectTarget(ctx context.Context, loc *url.URL, baseURL string, allowPrivate bool) error {
	if loc == nil {
		return fmt.Errorf("%w: missing Location", errRedirectNotAllowed)
	}
	if err := ValidateRedirectScheme(loc, baseURL); err != nil {
		return err
	}
	if allowPrivate {
		return nil
	}
	host := loc.Hostname()
	if host == "" {
		return fmt.Errorf("%w: missing host", errRedirectNotAllowed)
	}
	// An IP literal needs no resolution; check it directly.
	if net.ParseIP(host) != nil {
		_, err := publicRedirectIPs(ctx, host)
		return err
	}
	// A hostname the default registry redirects to needs no local DNS.
	if isAllowedRedirectHost(host) {
		return nil
	}
	// Behind a proxy, the proxy resolves the name; defer to its rules.
	if u, err := proxyFromEnvironment(&http.Request{URL: loc}); err != nil {
		return err
	} else if u != nil {
		return nil
	}
	_, err := publicRedirectIPs(ctx, host)
	return err
}

// lookupDialIPs resolves a host for checkedDialer; tests replace it to
// control the address order.
var lookupDialIPs = func(ctx context.Context, r *net.Resolver, host string) ([]net.IP, error) {
	return r.LookupIP(ctx, "ip", host)
}

// partialDialDeadline returns the deadline for one dial attempt when
// addrsRemaining addresses share the overall deadline. Like net.Dialer, each
// attempt gets an even share (but at least 2s) so one unreachable address
// can't use up the whole budget.
func partialDialDeadline(now, deadline time.Time, addrsRemaining int) time.Time {
	if deadline.IsZero() {
		return deadline
	}
	timeRemaining := deadline.Sub(now)
	if timeRemaining <= 0 {
		return deadline
	}
	timeout := timeRemaining / time.Duration(addrsRemaining)
	const saneMinimum = 2 * time.Second
	if timeout < saneMinimum {
		timeout = min(timeRemaining, saneMinimum)
	}
	return now.Add(timeout)
}

// dialFirst dials each address in order and returns the first connection
// that succeeds, or the first error if none do. Callers pass only validated
// addresses, so falling back never reaches an unchecked IP.
func dialFirst(ctx context.Context, d *net.Dialer, network string, ips []net.IP, port string) (net.Conn, error) {
	deadline, _ := ctx.Deadline()
	if d.Timeout > 0 {
		if t := time.Now().Add(d.Timeout); deadline.IsZero() || t.Before(deadline) {
			deadline = t
		}
	}

	var firstErr error
	for i, ip := range ips {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		dialCtx, cancel := ctx, context.CancelFunc(func() {})
		if pd := partialDialDeadline(time.Now(), deadline, len(ips)-i); !pd.IsZero() {
			dialCtx, cancel = context.WithDeadline(ctx, pd)
		}
		// Canceling after a successful dial doesn't affect the connection.
		conn, err := d.DialContext(dialCtx, network, net.JoinHostPort(ip.String(), port))
		cancel()
		if err == nil {
			return conn, nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	return nil, firstErr
}

// checkedDialer resolves, validates, and dials fixed IPs so DNS rebinding
// can't swap a private address in after validation. Every resolved address
// must pass validation; they are then tried in order so an unreachable
// first address (e.g. IPv6 without a route) falls back to the next.
func checkedDialer(d *net.Dialer, allowPrivate bool) func(ctx context.Context, network, addr string) (net.Conn, error) {
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		host, port, err := net.SplitHostPort(addr)
		if err != nil {
			return nil, err
		}

		ips := []net.IP{net.ParseIP(host)}
		if ips[0] == nil {
			ips, err = lookupDialIPs(ctx, d.Resolver, host)
			if err != nil {
				return nil, err
			}
			if len(ips) == 0 {
				return nil, fmt.Errorf("no addresses for %s", host)
			}
			for _, ip := range ips {
				if !allowPrivate && !isPublicIP(ip) {
					return nil, fmt.Errorf("%w: %s resolves to non-public %s", errRedirectNotAllowed, host, ip)
				}
			}
		} else if !allowPrivate && !isPublicIP(ips[0]) {
			return nil, fmt.Errorf("%w: %s is not a public address", errRedirectNotAllowed, ips[0])
		}

		conn, err := dialFirst(ctx, d, network, ips, port)
		if err != nil {
			return nil, err
		}
		if tc, ok := conn.(*net.TCPConn); ok {
			tc.SetKeepAlive(true)
			tc.SetKeepAlivePeriod(3 * time.Minute)
		}
		return conn, nil
	}
}

// proxyFromEnvironment is http.ProxyFromEnvironment; tests replace it.
var proxyFromEnvironment = http.ProxyFromEnvironment

// NewRedirectClient returns a client for download URLs supplied by a registry.
func NewRedirectClient(baseURL string, allowPrivate bool) *http.Client {
	// Redirect destinations do not get the registry host's exemption.
	client := newCheckedClient(baseURL, allowPrivate, false)
	client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if len(via) >= 10 {
			return errors.New("stopped after 10 redirects")
		}
		if err := ValidateRedirectScheme(req.URL, baseURL); err != nil {
			return err
		}
		return ValidateRedirectTarget(req.Context(), req.URL, baseURL, allowPrivate)
	}
	return client
}

// checkedClient exempts the exact registry origin selected by the caller.
func checkedClient(baseURL string, allowPrivate bool) *http.Client {
	return newCheckedClient(baseURL, allowPrivate, true)
}

func newCheckedClient(baseURL string, allowPrivate, exemptBase bool) *http.Client {
	var base *url.URL
	if exemptBase {
		base, _ = url.Parse(baseURL)
	}
	dialer := &net.Dialer{Timeout: 30 * time.Second, KeepAlive: 3 * time.Minute}
	rt := &checkedTransport{
		baseURL: baseURL, base: base, allowPrivate: allowPrivate,
		proxyTransports: make(map[proxyTransportKey]*http.Transport),
	}
	rt.direct = newTransferTransport(nil, func(ctx context.Context, network, addr string) (net.Conn, error) {
		if base != nil && sameOriginAddr(base, addr) {
			return dialer.DialContext(ctx, network, addr)
		}
		return checkedDialer(dialer, allowPrivate)(ctx, network, addr)
	})
	return &http.Client{
		Transport: rt,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

func sameOriginAddr(base *url.URL, addr string) bool {
	host, port, err := net.SplitHostPort(addr)
	if err != nil || !strings.EqualFold(host, base.Hostname()) {
		return false
	}
	basePort := base.Port()
	if basePort == "" {
		if base.Scheme == "https" {
			basePort = "443"
		} else {
			basePort = "80"
		}
	}
	return port == basePort
}

func sameOrigin(a, b *url.URL) bool {
	if a == nil || b == nil || a.Scheme != b.Scheme {
		return false
	}
	port := func(u *url.URL) string {
		if p := u.Port(); p != "" {
			return p
		}
		if u.Scheme == "https" {
			return "443"
		}
		return "80"
	}
	return strings.EqualFold(a.Hostname(), b.Hostname()) && port(a) == port(b)
}

func newTransferTransport(proxy *url.URL, dial func(context.Context, string, string) (net.Conn, error)) *http.Transport {
	return &http.Transport{
		Proxy: http.ProxyURL(proxy), DialContext: dial,
		MaxIdleConns: 100, MaxIdleConnsPerHost: 100,
		IdleConnTimeout: 90 * time.Second, ForceAttemptHTTP2: true,
	}
}

// checkedTransport binds proxy selection and its dialer to one request. A
// shared mutable proxy address is unsafe when direct and proxied requests
// run concurrently.
type checkedTransport struct {
	baseURL         string
	base            *url.URL
	allowPrivate    bool
	direct          *http.Transport
	mu              sync.Mutex
	proxyTransports map[proxyTransportKey]*http.Transport
}

type proxyTransportKey struct {
	proxy string
}

func (t *checkedTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if !sameOrigin(req.URL, t.base) {
		if err := ValidateRedirectTarget(req.Context(), req.URL, t.baseURL, t.allowPrivate); err != nil {
			return nil, err
		}
	}
	proxy, err := proxyFromEnvironment(req)
	if err != nil {
		return nil, err
	}
	if proxy == nil {
		return t.direct.RoundTrip(req)
	}
	key := proxyTransportKey{proxy: proxy.String()}
	return t.proxyTransport(key, proxy).RoundTrip(req)
}

func (t *checkedTransport) proxyTransport(key proxyTransportKey, proxy *url.URL) *http.Transport {
	t.mu.Lock()
	defer t.mu.Unlock()
	if tr := t.proxyTransports[key]; tr != nil {
		return tr
	}
	if len(t.proxyTransports) >= maxProxyTransports {
		for oldKey, old := range t.proxyTransports {
			old.CloseIdleConnections()
			delete(t.proxyTransports, oldKey)
			break
		}
	}
	d := &net.Dialer{Timeout: 30 * time.Second, KeepAlive: 3 * time.Minute}
	tr := newTransferTransport(proxy, d.DialContext)
	t.proxyTransports[key] = tr
	return tr
}

func (t *checkedTransport) CloseIdleConnections() {
	t.direct.CloseIdleConnections()
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, tr := range t.proxyTransports {
		tr.CloseIdleConnections()
	}
}
