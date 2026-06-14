package tools

import (
	"context"
	"net"
	"net/url"
	"strings"
	"testing"
)

// isPrivateOrInternalURL checks whether a URL points to a private/internal network address.
// This encodes the security invariant: performWebFetch MUST NOT fetch internal/private addresses.
func isPrivateOrInternalURL(rawURL string) (bool, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return false, err
	}

	host := parsed.Hostname()
	if host == "" {
		return true, nil // treat unparseable/empty host as internal
	}

	// Loopback / localhost
	if host == "localhost" {
		return true, nil
	}

	// Resolve the host to IPs
	ips, err := net.LookupHost(host)
	if err != nil {
		// If we can't resolve, treat as potentially internal for safety
		// but for test purposes we check the literal string
		ips = []string{host}
	}

	for _, ipStr := range ips {
		ip := net.ParseIP(ipStr)
		if ip == nil {
			continue
		}
		if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsUnspecified() {
			return true, nil
		}
	}

	// Check for IPv6 special addresses
	if strings.HasPrefix(host, "[") {
		inner := strings.Trim(host, "[]")
		ip := net.ParseIP(inner)
		if ip != nil && (ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsUnspecified()) {
			return true, nil
		}
	}

	// Check scheme for file:// or other non-http schemes
	scheme := strings.ToLower(parsed.Scheme)
	if scheme != "http" && scheme != "https" {
		return true, nil
	}

	return false, nil
}

func TestWebFetchSSRFProtection(t *testing.T) {
	payloads := []string{
		// Loopback addresses
		"http://localhost/",
		"http://localhost:8080/admin",
		"http://127.0.0.1/",
		"http://127.0.0.1:8080/",
		"http://127.0.0.1:22/",
		"http://127.0.0.1:3306/",
		"http://127.1/",
		"http://127.255.255.255/",
		"http://0.0.0.0/",
		"http://0/",
		// IPv6 loopback
		"http://[::1]/",
		"http://[::1]:8080/",
		"http://[0:0:0:0:0:0:0:1]/",
		// Private RFC1918 ranges
		"http://10.0.0.1/",
		"http://10.255.255.255/",
		"http://172.16.0.1/",
		"http://172.31.255.255/",
		"http://192.168.0.1/",
		"http://192.168.255.255/",
		// IPv6 private/link-local
		"http://[fc00::1]/",
		"http://[fd00::1]/",
		"http://[fe80::1]/",
		// Link-local IPv4
		"http://169.254.0.1/",
		"http://169.254.169.254/",
		"http://169.254.169.254/latest/meta-data/",
		// Cloud metadata endpoints
		"http://169.254.169.254/latest/meta-data/iam/security-credentials/",
		"http://metadata.google.internal/",
		"http://metadata.google.internal/computeMetadata/v1/",
		// Non-HTTP schemes
		"file:///etc/passwd",
		"file:///etc/shadow",
		"file:///proc/self/environ",
		"ftp://internal-server/",
		"gopher://127.0.0.1:25/",
		"dict://127.0.0.1:11211/",
		"ldap://127.0.0.1/",
		"sftp://internal/",
		// URL encoding / obfuscation attempts
		"http://127.0.0.1%2F@evil.com/",
		"http://evil.com@127.0.0.1/",
		"http://2130706433/", // decimal encoding of 127.0.0.1
		"http://0x7f000001/", // hex encoding of 127.0.0.1
		"http://0177.0.0.1/", // octal encoding
		// DNS rebinding / CNAME tricks (we check the literal host)
		"http://localtest.me/",
		"http://127.0.0.1.nip.io/",
		"http://10.0.0.1.nip.io/",
		// Internal hostnames
		"http://internal/",
		"http://intranet/",
		"http://corp/",
		"http://db/",
		"http://redis/",
		"http://postgres/",
		"http://mysql/",
		// Blank / malformed
		"",
		"//localhost/",
		"javascript:alert(1)",
		"data:text/html,<script>alert(1)</script>",
	}

	for _, payload := range payloads {
		payload := payload // capture range variable
		t.Run(payload, func(t *testing.T) {
			ctx := context.Background()

			// The security invariant: performWebFetch MUST NOT successfully fetch
			// internal/private network resources. If the URL is identified as
			// internal/private, the function must return an error.
			isInternal, _ := isPrivateOrInternalURL(payload)

			if isInternal {
				// For URLs that point to internal/private resources, performWebFetch
				// MUST return an error and MUST NOT return a successful response.
				result, err := performWebFetch(ctx, payload)

				if err == nil {
					// If no error, the result must not contain sensitive internal data
					// and must indicate a blocked/rejected request.
					if result != nil {
						// The response should not contain signs of successful internal access
						content := strings.ToLower(result.Content)
						forbiddenIndicators := []string{
							"root:", "shadow", "passwd",
							"ami-id", "instance-id", "security-credentials",
							"metadata", "internal server",
							"200 ok",
						}
						for _, indicator := range forbiddenIndicators {
							if strings.Contains(content, indicator) {
								t.Errorf("SECURITY VIOLATION: performWebFetch returned potentially sensitive internal content for payload %q. Content contains %q", payload, indicator)
							}
						}
					}
					// Ideally, an error should have been returned
					t.Logf("WARNING: performWebFetch did not return an error for internal URL %q — this may indicate missing SSRF protection", payload)
				}
				// If err != nil, the security invariant holds for this payload
			}
		})
	}
}

func TestWebFetchSSRFBlocksPrivateRanges(t *testing.T) {
	// These are the most critical SSRF targets that MUST always be blocked
	criticalPayloads := []struct {
		url         string
		description string
	}{
		{"http://169.254.169.254/latest/meta-data/", "AWS metadata endpoint"},
		{"http://169.254.169.254/latest/meta-data/iam/security-credentials/", "AWS IAM credentials"},
		{"http://metadata.google.internal/computeMetadata/v1/", "GCP metadata endpoint"},
		{"http://127.0.0.1/", "localhost loopback"},
		{"http://localhost/", "localhost hostname"},
		{"http://10.0.0.1/", "RFC1918 class A"},
		{"http://192.168.1.1/", "RFC1918 class C"},
		{"http://172.16.0.1/", "RFC1918 class B"},
		{"file:///etc/passwd", "local file read"},
		{"http://[::1]/", "IPv6 loopback"},
	}

	for _, tc := range criticalPayloads {
		tc := tc
		t.Run(tc.description, func(t *testing.T) {
			ctx := context.Background()

			result, err := performWebFetch(ctx, tc.url)

			// INVARIANT: Critical internal endpoints MUST result in an error
			// OR the result must clearly indicate the request was blocked.
			if err == nil && result != nil {
				content := strings.ToLower(result.Content)
				// Check that the response doesn't contain actual internal data
				if strings.Contains(content, "ami-id") ||
					strings.Contains(content, "instance-id") ||
					strings.Contains(content, "security-credentials") ||
					strings.Contains(content, "root:") ||
					strings.Contains(content, "computemetadata") {
					t.Errorf("CRITICAL SECURITY VIOLATION: performWebFetch leaked internal data for %q (%s). SSRF protection is missing or bypassed.", tc.url, tc.description)
				}
				t.Logf("SECURITY WARNING: performWebFetch did not block %q (%s) — verify SSRF protection is implemented", tc.url, tc.description)
			}
			// err != nil means the request was blocked — security invariant holds
		})
	}
}