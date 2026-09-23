package transfer

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

var errRedirectNotAllowed = errors.New("redirect target not allowed")

// blockedIPv4Nets are globally-unicast-but-not-public IPv4 ranges that
// net.IP's helpers don't classify: CGNAT (RFC 6598), which carries real
// internal services like Alibaba Cloud metadata (100.100.2.148), and the
// benchmarking block (RFC 2544).
var blockedIPv4Nets = []*net.IPNet{parseIPv4Net("100.64.0.0/10"), parseIPv4Net("198.18.0.0/15")}

func parseIPv4Net(s string) *net.IPNet {
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
	// To4() unwraps 4-in-6 mapped addresses so mapped CGNAT is blocked too.
	if v4 := ip.To4(); v4 != nil {
		for _, n := range blockedIPv4Nets {
			if n.Contains(v4) {
				return false
			}
		}
	}
	return true
}

// validateRedirectScheme rejects redirects that downgrade an https session
// to cleartext http, even when the target host is unchanged — a hostile or
// compromised registry must not be able to strip TLS off follow-up requests.
// It applies even under allowPrivate (the --insecure opt-in), which relaxes
// address checks but never scheme checks.
func validateRedirectScheme(loc *url.URL, baseURL string) error {
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

// validateRedirectTarget rejects redirect targets that aren't public HTTPS
// endpoints, unless allowPrivate is set.
func validateRedirectTarget(ctx context.Context, loc *url.URL, baseURL string, allowPrivate bool) error {
	if loc == nil {
		return fmt.Errorf("%w: missing Location", errRedirectNotAllowed)
	}
	if allowPrivate {
		return nil
	}
	if err := validateRedirectScheme(loc, baseURL); err != nil {
		return err
	}
	host := loc.Hostname()
	if ip := net.ParseIP(host); ip != nil {
		if !isPublicIP(ip) {
			return fmt.Errorf("%w: %s is not a public address", errRedirectNotAllowed, ip)
		}
		return nil
	}

	ips, err := net.DefaultResolver.LookupIP(ctx, "ip", host)
	if err != nil {
		return fmt.Errorf("%w: resolving %s: %w", errRedirectNotAllowed, host, err)
	}
	if len(ips) == 0 {
		return fmt.Errorf("%w: %s has no addresses", errRedirectNotAllowed, host)
	}
	for _, ip := range ips {
		if !isPublicIP(ip) {
			return fmt.Errorf("%w: %s resolves to non-public %s", errRedirectNotAllowed, host, ip)
		}
	}
	return nil
}

// checkedDialer resolves, validates, and dials a pinned IP so DNS rebinding
// can't swap a private address in after validation.
func checkedDialer(d *net.Dialer, allowPrivate bool) func(ctx context.Context, network, addr string) (net.Conn, error) {
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		host, port, err := net.SplitHostPort(addr)
		if err != nil {
			return nil, err
		}

		dialHost := host
		if ip := net.ParseIP(host); ip == nil {
			ips, err := d.Resolver.LookupIP(ctx, "ip", host)
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
			dialHost = ips[0].String()
		} else if !allowPrivate && !isPublicIP(ip) {
			return nil, fmt.Errorf("%w: %s is not a public address", errRedirectNotAllowed, ip)
		}

		conn, err := d.DialContext(ctx, network, net.JoinHostPort(dialHost, port))
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

// checkedClient returns an HTTP client using checkedDialer; the registry
// base host is exempt since the caller explicitly directed traffic at it.
func checkedClient(baseURL string, allowPrivate bool) *http.Client {
	var baseHostname string
	if b, err := url.Parse(baseURL); err == nil {
		baseHostname = b.Hostname()
	}
	return &http.Client{
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 100,
			IdleConnTimeout:     90 * time.Second,
			// Custom DialContext disables HTTP/2 auto-configuration;
			// ForceAttemptHTTP2 opts back in.
			ForceAttemptHTTP2: true,
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				host, _, err := net.SplitHostPort(addr)
				if err == nil && baseHostname != "" && strings.EqualFold(host, baseHostname) {
					return new(net.Dialer).DialContext(ctx, network, addr)
				}
				return checkedDialer(&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 3 * time.Minute}, allowPrivate)(ctx, network, addr)
			},
		},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}
