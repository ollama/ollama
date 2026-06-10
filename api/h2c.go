package api

import (
	"context"
	"crypto/tls"
	"net"
	"net/http"
	"time"

	"golang.org/x/net/http2"
)

// H2CTransport returns an http2.Transport configured for plaintext h2c
// (HTTP/2 cleartext) with the DialTLSContext override (required to avoid
// "server gave HTTP response to HTTPS client" on local gRPC listeners).
// Includes keepalive settings (G7) for long-lived agent/stream connections.
// Used for both dedicated gRPC port (11435) and SAMEPORT (cmux) cases.
func H2CTransport() *http2.Transport {
	return &http2.Transport{
		AllowHTTP: true,
		DialTLSContext: func(ctx context.Context, network, addr string, _ *tls.Config) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, network, addr)
		},
		// Phase 2c + 4a: keepalive pings for resilience (Triton/vLLM-like client hardening).
		ReadIdleTimeout: 30 * time.Second,
		PingTimeout:     15 * time.Second,
	}
}

// H2CClient returns an http.Client wrapping H2CTransport().
// Central source of truth for all gRPC/Connect h2c clients (NewGRPCClient,
// integration tests, quality scripts, etc.). No globals, small, reusable.
func H2CClient() *http.Client {
	return &http.Client{
		Transport: H2CTransport(),
	}
}
