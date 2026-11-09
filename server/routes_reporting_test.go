package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestReportingPassthrough(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_NO_CLOUD", "false")
	testHome := t.TempDir()
	setTestHome(t, testHome)
	writeTestOllamaPrivateKey(t, testHome)

	originalBaseURL, originalSigningHost := cloudProxyBaseURL, cloudProxySigningHost
	cloudProxySigningHost = "127.0.0.1"
	t.Cleanup(func() {
		cloudProxyBaseURL, cloudProxySigningHost = originalBaseURL, originalSigningHost
	})

	s := &Server{}
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}

	for _, tt := range []struct {
		name       string
		path       string
		query      string
		malformed  bool
		status     int
		body       string
		retryAfter string
	}{
		{
			name:   "balance",
			path:   "/api/balance",
			status: http.StatusOK,
			body:   `{"purchased":{"balance_usd":25},"future_field":true}`,
		},
		{
			name:   "usage",
			path:   "/api/usage",
			query:  "range=24h&scope=team",
			status: http.StatusOK,
			body:   `{"totals":{"request_count":2},"future_field":true}`,
		},
		{
			name:       "upstream error",
			path:       "/api/balance",
			status:     http.StatusTooManyRequests,
			body:       `{"error":"rate limit exceeded"}`,
			retryAfter: "60",
		},
		{
			name:      "malformed query",
			path:      "/api/usage",
			query:     "range=24h;",
			malformed: true,
			status:    http.StatusBadRequest,
			body:      `{"error":"upstream rejected malformed query"}`,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			requests := make(chan *http.Request, 1)
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests <- r.Clone(t.Context())
				w.Header().Set("Content-Type", "application/json")
				if tt.retryAfter != "" {
					w.Header().Set("Retry-After", tt.retryAfter)
				}
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer upstream.Close()
			cloudProxyBaseURL = upstream.URL

			req := httptest.NewRequestWithContext(t.Context(), http.MethodGet, tt.path+"?ts=old&"+tt.query, nil)
			req.Header.Set("Authorization", "Bearer caller-credential")
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			if w.Code != tt.status || w.Body.String() != tt.body {
				t.Fatalf("response = %d %s, want %d %s", w.Code, w.Body, tt.status, tt.body)
			}
			for header, want := range map[string]string{
				"Content-Type":  "application/json",
				"Cache-Control": "private, no-store",
				"Retry-After":   tt.retryAfter,
			} {
				if got := w.Header().Get(header); got != want {
					t.Errorf("%s = %q, want %q", header, got, want)
				}
			}

			select {
			case forwarded := <-requests:
				if forwarded.Method != http.MethodGet || forwarded.URL.Path != tt.path {
					t.Fatalf("forwarded request = %s %s", forwarded.Method, forwarded.URL.Path)
				}
				query := forwarded.URL.Query()
				if ts := query.Get("ts"); ts == "" || ts == "old" {
					t.Errorf("expected a fresh signing timestamp, got %q", ts)
				}
				if tt.malformed {
					_, original, ok := strings.Cut(forwarded.URL.RawQuery, "&")
					if !ok || original != req.URL.RawQuery {
						t.Errorf("original query was changed: forwarded %q, original %q", forwarded.URL.RawQuery, req.URL.RawQuery)
					}
				} else {
					query.Del("ts")
					if got := query.Encode(); got != tt.query {
						t.Errorf("forwarded query = %q, want %q", got, tt.query)
					}
				}
				verifySignedOllamaRequest(t, forwarded)
			default:
				t.Fatal("reporting request was not forwarded")
			}
		})
	}
}

func TestReportingLocalErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	originalSignRequest := cloudProxySignRequest
	t.Cleanup(func() {
		cloudProxySignRequest = originalSignRequest
	})

	s := &Server{}
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}

	for _, endpoint := range []string{"balance", "usage"} {
		t.Run(endpoint, func(t *testing.T) {
			for _, tt := range []struct {
				name      string
				method    string
				noCloud   string
				status    int
				signCalls int
			}{
				{
					name: "cloud disabled", method: http.MethodGet, noCloud: "true",
					status: http.StatusForbidden,
				},
				{
					name: "signing failed", method: http.MethodGet, noCloud: "false",
					status: http.StatusUnauthorized, signCalls: 1,
				},
				{
					name: "method not allowed", method: http.MethodPost, noCloud: "false",
					status: http.StatusMethodNotAllowed,
				},
			} {
				t.Run(tt.name, func(t *testing.T) {
					t.Setenv("OLLAMA_NO_CLOUD", tt.noCloud)
					signCalls := 0
					cloudProxySignRequest = func(_ context.Context, req *http.Request) error {
						signCalls++
						if got := req.Header.Get("Authorization"); got != "" {
							t.Errorf("caller credentials reached signer: %q", got)
						}
						return errors.New("signing failed")
					}

					req := httptest.NewRequestWithContext(t.Context(), tt.method, "/api/"+endpoint, nil)
					req.Header.Set("Authorization", "Bearer caller-credential")
					w := httptest.NewRecorder()
					router.ServeHTTP(w, req)
					if w.Code != tt.status {
						t.Fatalf("status = %d, want %d (%s)", w.Code, tt.status, w.Body)
					}
					if tt.method == http.MethodGet && w.Header().Get("Cache-Control") != "private, no-store" {
						t.Error("reporting error response must not be cached")
					}
					if signCalls != tt.signCalls {
						t.Errorf("signer called %d times, want %d", signCalls, tt.signCalls)
					}
				})
			}
		})
	}
}
