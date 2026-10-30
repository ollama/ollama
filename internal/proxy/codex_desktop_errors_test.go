package proxy

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestCodexDesktopPreservesUpstreamErrors(t *testing.T) {
	for _, tt := range []struct {
		name        string
		status      int
		contentType string
		body        string
	}{
		{"sign in", http.StatusUnauthorized, "application/json", `{"error":{"message":"Please sign in to Ollama.","type":"authentication_error","code":"authentication_error"},"signin_url":"https://ollama.com/connect?key=test-key"}`},
		{"subscription", http.StatusPaymentRequired, "application/json", `{"error":{"message":"this model requires a subscription or extra usage","code":"subscription_required"},"request_id":"keep-me"}`},
		{"rate limit", http.StatusTooManyRequests, "application/json", `{"error":{"message":"too many requests","code":"rate_limit_exceeded"}}`},
		{"stream failure", http.StatusOK, "text/event-stream", "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"authentication_error\",\"message\":\"Please sign in to Ollama.\"},\"signin_url\":\"https://ollama.com/connect?key=test-key\"}}\n\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", tt.contentType)
				w.Header().Set("Content-Length", strconv.Itoa(len(tt.body)))
				w.Header().Set("X-Request-ID", "keep-me")
				w.WriteHeader(tt.status)
				_, _ = io.WriteString(w, tt.body)
			}))
			defer upstream.Close()
			handler := newTestCodexDesktop(t, upstream.URL, upstream.URL, writeCatalog(t, "test:cloud"))
			endpoint := httptest.NewServer(handler)
			defer endpoint.Close()
			for _, model := range []string{"test:cloud", "gpt-native"} {
				req, err := http.NewRequest(http.MethodPost, endpoint.URL+CodexDesktopPathPrefix+"/v1/responses", strings.NewReader(fmt.Sprintf(`{"model":%q,"stream":true}`, model)))
				if err != nil {
					t.Fatal(err)
				}
				req.Header.Set("Authorization", "Bearer test-chatgpt")
				req.Header.Set("ChatGPT-Account-ID", "test-account")
				resp, err := endpoint.Client().Do(req)
				if err != nil {
					t.Fatal(err)
				}
				body, err := io.ReadAll(resp.Body)
				resp.Body.Close()
				if err != nil {
					t.Fatal(err)
				}
				if resp.StatusCode != tt.status || resp.Header.Get("Content-Type") != tt.contentType || resp.Header.Get("X-Request-ID") != "keep-me" {
					t.Fatalf("%s status/headers changed: %d %v", model, resp.StatusCode, resp.Header)
				}
				if string(body) != tt.body {
					t.Fatalf("%s body = %q, want %q", model, body, tt.body)
				}
			}
		})
	}
}

func TestCodexDesktopDoesNotBufferSuccessfulStream(t *testing.T) {
	first := "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n"
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, first)
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer upstream.Close()
	defer close(release)
	handler := newTestCodexDesktop(t, upstream.URL, upstream.URL, writeCatalog(t, "test:cloud"))
	endpoint := httptest.NewServer(handler)
	defer endpoint.Close()
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Post(endpoint.URL+CodexDesktopPathPrefix+"/v1/responses", "application/json", strings.NewReader(`{"model":"test:cloud","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got := make([]byte, len(first))
	if _, err := io.ReadFull(resp.Body, got); err != nil {
		t.Fatal(err)
	}
	if string(got) != first {
		t.Fatalf("first event = %q, want %q", got, first)
	}
}
