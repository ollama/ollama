package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/version"
)

type webExperimentalUpstreamCapture struct {
	path   string
	body   string
	header http.Header
}

func newWebExperimentalUpstream(t *testing.T, responseBody string) (*httptest.Server, *webExperimentalUpstreamCapture) {
	t.Helper()

	capture := &webExperimentalUpstreamCapture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload, _ := io.ReadAll(r.Body)
		capture.path = r.URL.Path
		capture.body = string(payload)
		capture.header = r.Header.Clone()

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(responseBody))
	}))

	return srv, capture
}

func TestExperimentalWebEndpointsPassthrough(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	tests := []struct {
		name         string
		localPath    string
		upstreamPath string
		requestBody  string
		responseBody string
		assertBody   string
	}{
		{
			name:         "web_search",
			localPath:    "/api/experimental/web_search",
			upstreamPath: "/api/web_search",
			requestBody:  `{"query":"what is ollama?","max_results":3}`,
			responseBody: `{"results":[{"title":"Ollama","url":"https://ollama.com","content":"Cloud models are now available"}]}`,
			assertBody:   `"query":"what is ollama?"`,
		},
		{
			name:         "web_fetch",
			localPath:    "/api/experimental/web_fetch",
			upstreamPath: "/api/web_fetch",
			requestBody:  `{"url":"https://ollama.com"}`,
			responseBody: `{"title":"Ollama","content":"Cloud models are now available","links":["https://ollama.com/"]}`,
			assertBody:   `"url":"https://ollama.com"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			upstream, capture := newWebExperimentalUpstream(t, tt.responseBody)
			defer upstream.Close()

			original := cloudProxyBaseURL
			cloudProxyBaseURL = upstream.URL
			t.Cleanup(func() { cloudProxyBaseURL = original })

			s := &Server{}
			router, err := s.GenerateRoutes(nil)
			if err != nil {
				t.Fatal(err)
			}
			local := httptest.NewServer(router)
			defer local.Close()

			req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+tt.localPath, bytes.NewBufferString(tt.requestBody))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer should-forward")
			req.Header.Set("X-Test-Header", "web-experimental")
			req.Header.Set(cloudProxyClientVersionHeader, "should-be-overwritten")

			resp, err := local.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()

			body, _ := io.ReadAll(resp.Body)
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("expected status 200, got %d (%s)", resp.StatusCode, string(body))
			}
			if capture.path != tt.upstreamPath {
				t.Fatalf("expected upstream path %q, got %q", tt.upstreamPath, capture.path)
			}
			if !bytes.Contains([]byte(capture.body), []byte(tt.assertBody)) {
				t.Fatalf("expected upstream body to contain %q, got %q", tt.assertBody, capture.body)
			}
			if got := capture.header.Get("Authorization"); got != "Bearer should-forward" {
				t.Fatalf("expected forwarded Authorization header, got %q", got)
			}
			if got := capture.header.Get("X-Test-Header"); got != "web-experimental" {
				t.Fatalf("expected forwarded X-Test-Header=web-experimental, got %q", got)
			}
			if got := capture.header.Get(cloudProxyClientVersionHeader); got != version.Version {
				t.Fatalf("expected %s=%q, got %q", cloudProxyClientVersionHeader, version.Version, got)
			}
		})
	}
}

func TestExperimentalWebEndpointsMissingBody(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}

	local := httptest.NewServer(router)
	defer local.Close()

	tests := []string{
		"/api/experimental/web_search",
		"/api/experimental/web_fetch",
	}

	for _, path := range tests {
		t.Run(path, func(t *testing.T) {
			req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+path, nil)
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")

			resp, err := local.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()

			body, _ := io.ReadAll(resp.Body)
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("expected status 400, got %d (%s)", resp.StatusCode, string(body))
			}
			if string(body) != `{"error":"missing request body"}` {
				t.Fatalf("unexpected response body: %s", string(body))
			}
		})
	}
}

func TestExperimentalWebEndpointsCloudDisabled(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "1")

	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}

	local := httptest.NewServer(router)
	defer local.Close()

	tests := []struct {
		name      string
		path      string
		request   string
		operation string
	}{
		{
			name:      "web_search",
			path:      "/api/experimental/web_search",
			request:   `{"query":"latest ollama release"}`,
			operation: cloudErrWebSearchUnavailable,
		},
		{
			name:      "web_fetch",
			path:      "/api/experimental/web_fetch",
			request:   `{"url":"https://ollama.com"}`,
			operation: cloudErrWebFetchUnavailable,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+tt.path, bytes.NewBufferString(tt.request))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")

			resp, err := local.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()

			body, _ := io.ReadAll(resp.Body)
			if resp.StatusCode != http.StatusForbidden {
				t.Fatalf("expected status 403, got %d (%s)", resp.StatusCode, string(body))
			}

			var got map[string]string
			if err := json.Unmarshal(body, &got); err != nil {
				t.Fatalf("expected json error body, got: %q", string(body))
			}
			if got["error"] != internalcloud.DisabledError(tt.operation) {
				t.Fatalf("unexpected error message: %q", got["error"])
			}
		})
	}
}

func TestExperimentalWebEndpointSigningFailureReturnsUnauthorized(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	origSignRequest := cloudProxySignRequest
	origSigninURL := cloudProxySigninURL
	cloudProxySignRequest = func(context.Context, *http.Request) error {
		return errors.New("ssh: no key found")
	}
	cloudProxySigninURL = func() (string, error) {
		return "https://ollama.com/signin/example", nil
	}
	t.Cleanup(func() {
		cloudProxySignRequest = origSignRequest
		cloudProxySigninURL = origSigninURL
	})

	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	defer local.Close()

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/experimental/web_search", bytes.NewBufferString(`{"query":"hello"}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected status 401, got %d (%s)", resp.StatusCode, string(body))
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("expected json error body, got: %q", string(body))
	}
	if got["error"] != "unauthorized" {
		t.Fatalf("unexpected error message: %v", got["error"])
	}
	if got["signin_url"] != "https://ollama.com/signin/example" {
		t.Fatalf("unexpected signin_url: %v", got["signin_url"])
	}
}

func TestExperimentalWebEndpointSigningFailureWithoutSigninURL(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	origSignRequest := cloudProxySignRequest
	origSigninURL := cloudProxySigninURL
	cloudProxySignRequest = func(context.Context, *http.Request) error {
		return errors.New("ssh: no key found")
	}
	cloudProxySigninURL = func() (string, error) {
		return "", errors.New("key missing")
	}
	t.Cleanup(func() {
		cloudProxySignRequest = origSignRequest
		cloudProxySigninURL = origSigninURL
	})

	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	defer local.Close()

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/experimental/web_fetch", bytes.NewBufferString(`{"url":"https://ollama.com"}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected status 401, got %d (%s)", resp.StatusCode, string(body))
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("expected json error body, got: %q", string(body))
	}
	if got["error"] != "unauthorized" {
		t.Fatalf("unexpected error message: %v", got["error"])
	}
	if _, ok := got["signin_url"]; ok {
		t.Fatalf("did not expect signin_url when helper fails, got %v", got["signin_url"])
	}
}
