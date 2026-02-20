package server

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestMatchOrigin(t *testing.T) {
	tests := []struct {
		name     string
		origin   string
		pattern  string
		expected bool
	}{
		// Exact matches
		{
			name:     "exact match",
			origin:   "http://localhost",
			pattern:  "http://localhost",
			expected: true,
		},
		{
			name:     "exact match with port",
			origin:   "http://localhost:8080",
			pattern:  "http://localhost:8080",
			expected: true,
		},
		{
			name:     "exact mismatch",
			origin:   "http://localhost",
			pattern:  "http://example.com",
			expected: false,
		},

		// Wildcard patterns
		{
			name:     "wildcard all",
			origin:   "http://example.com",
			pattern:  "*",
			expected: true,
		},
		{
			name:     "wildcard port match",
			origin:   "http://localhost:8080",
			pattern:  "http://localhost:*",
			expected: true,
		},
		{
			name:     "wildcard port match different port",
			origin:   "http://localhost:3000",
			pattern:  "http://localhost:*",
			expected: true,
		},
		{
			name:     "wildcard port no match wrong host",
			origin:   "http://example.com:8080",
			pattern:  "http://localhost:*",
			expected: false,
		},
		{
			name:     "wildcard subdomain",
			origin:   "http://api.example.com",
			pattern:  "http://*.example.com",
			expected: true,
		},
		{
			name:     "wildcard subdomain no match",
			origin:   "http://api.other.com",
			pattern:  "http://*.example.com",
			expected: false,
		},

		// Protocol patterns
		{
			name:     "app protocol wildcard",
			origin:   "app://myapp",
			pattern:  "app://*",
			expected: true,
		},
		{
			name:     "tauri protocol wildcard",
			origin:   "tauri://localhost",
			pattern:  "tauri://*",
			expected: true,
		},
		{
			name:     "vscode-webview wildcard",
			origin:   "vscode-webview://extension-id",
			pattern:  "vscode-webview://*",
			expected: true,
		},

		// Edge cases
		{
			name:     "empty origin",
			origin:   "",
			pattern:  "http://localhost",
			expected: false,
		},
		{
			name:     "https vs http",
			origin:   "https://localhost",
			pattern:  "http://localhost",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := matchOrigin(tt.origin, tt.pattern)
			if result != tt.expected {
				t.Errorf("matchOrigin(%q, %q) = %v, want %v", tt.origin, tt.pattern, result, tt.expected)
			}
		})
	}
}

func TestCorsResponseWriter(t *testing.T) {
	allowedOrigins := []string{
		"http://localhost",
		"http://localhost:*",
		"http://127.0.0.1:*",
		"app://*",
	}

	tests := []struct {
		name           string
		requestOrigin  string
		expectedHeader string
	}{
		{
			name:           "allowed origin localhost",
			requestOrigin:  "http://localhost",
			expectedHeader: "http://localhost",
		},
		{
			name:           "allowed origin with port",
			requestOrigin:  "http://localhost:8080",
			expectedHeader: "http://localhost:8080",
		},
		{
			name:           "allowed origin 127.0.0.1",
			requestOrigin:  "http://127.0.0.1:3000",
			expectedHeader: "http://127.0.0.1:3000",
		},
		{
			name:           "allowed app protocol",
			requestOrigin:  "app://myapp",
			expectedHeader: "app://myapp",
		},
		{
			name:           "disallowed origin",
			requestOrigin:  "http://evil.com",
			expectedHeader: "",
		},
		{
			name:           "no origin header",
			requestOrigin:  "",
			expectedHeader: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a request with the Origin header
			req := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
			if tt.requestOrigin != "" {
				req.Header.Set("Origin", tt.requestOrigin)
			}

			// Create a response recorder
			rec := httptest.NewRecorder()

			// Wrap with corsResponseWriter
			cw := &corsResponseWriter{
				ResponseWriter: rec,
				request:        req,
				allowedOrigins: allowedOrigins,
			}

			// Simulate a redirect response (301)
			cw.WriteHeader(http.StatusMovedPermanently)

			// Check the Access-Control-Allow-Origin header
			result := rec.Header().Get("Access-Control-Allow-Origin")
			if result != tt.expectedHeader {
				t.Errorf("Access-Control-Allow-Origin = %q, want %q", result, tt.expectedHeader)
			}

			// Check that Vary header is set when CORS is applied
			if tt.expectedHeader != "" {
				vary := rec.Header().Get("Vary")
				if vary != "Origin" {
					t.Errorf("Vary header = %q, want %q", vary, "Origin")
				}
			}
		})
	}
}

func TestCorsWrapper(t *testing.T) {
	allowedOrigins := []string{
		"http://localhost:*",
	}

	// Create a handler that returns a 301 redirect
	redirectHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", "/api/tags")
		w.WriteHeader(http.StatusMovedPermanently)
	})

	// Wrap with corsWrapper
	wrapped := corsWrapper(redirectHandler, allowedOrigins)

	// Test with allowed origin
	t.Run("redirect with allowed origin", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "//api/tags", nil)
		req.Header.Set("Origin", "http://localhost:8080")

		rec := httptest.NewRecorder()
		wrapped.ServeHTTP(rec, req)

		// Check status code
		if rec.Code != http.StatusMovedPermanently {
			t.Errorf("status code = %d, want %d", rec.Code, http.StatusMovedPermanently)
		}

		// Check CORS header is present
		corsHeader := rec.Header().Get("Access-Control-Allow-Origin")
		if corsHeader != "http://localhost:8080" {
			t.Errorf("Access-Control-Allow-Origin = %q, want %q", corsHeader, "http://localhost:8080")
		}

		// Check Location header is preserved
		location := rec.Header().Get("Location")
		if location != "/api/tags" {
			t.Errorf("Location = %q, want %q", location, "/api/tags")
		}
	})

	// Test with disallowed origin
	t.Run("redirect with disallowed origin", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "//api/tags", nil)
		req.Header.Set("Origin", "http://evil.com")

		rec := httptest.NewRecorder()
		wrapped.ServeHTTP(rec, req)

		// CORS header should NOT be present
		corsHeader := rec.Header().Get("Access-Control-Allow-Origin")
		if corsHeader != "" {
			t.Errorf("Access-Control-Allow-Origin = %q, want empty", corsHeader)
		}
	})
}

func TestCorsWrapperDoesNotOverwriteExistingHeaders(t *testing.T) {
	allowedOrigins := []string{
		"http://localhost:*",
	}

	// Create a handler that sets CORS headers itself (like gin-contrib/cors would)
	handlerWithCors := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "http://custom-origin.com")
		w.WriteHeader(http.StatusOK)
	})

	// Wrap with corsWrapper
	wrapped := corsWrapper(handlerWithCors, allowedOrigins)

	req := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	req.Header.Set("Origin", "http://localhost:8080")

	rec := httptest.NewRecorder()
	wrapped.ServeHTTP(rec, req)

	// The existing CORS header should be preserved (not overwritten)
	corsHeader := rec.Header().Get("Access-Control-Allow-Origin")
	if corsHeader != "http://custom-origin.com" {
		t.Errorf("Access-Control-Allow-Origin = %q, want %q (should preserve existing)", corsHeader, "http://custom-origin.com")
	}
}
