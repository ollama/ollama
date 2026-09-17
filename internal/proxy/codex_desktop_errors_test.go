package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"testing/iotest"
	"time"
)

const (
	testCodexSubscriptionError = "this model requires a subscription or extra usage, upgrade for access at https://ollama.com/upgrade or add extra usage at https://ollama.com/settings (ref: test-reference)"
	testCodexSubscriptionCopy  = "This model requires a subscription or extra usage credits. Please upgrade at https://ollama.com/upgrade or add extra usage at https://ollama.com/settings to use this model."
	testCodexSignInCopy        = "This model requires an Ollama account. Please sign in to Ollama to use this model."
)

func TestCodexDesktopAccessErrorCopy(t *testing.T) {
	for _, tt := range []struct {
		name        string
		status      int
		contentType string
		body        string
		want        string
	}{
		{
			name: "subscription JSON", status: http.StatusForbidden, contentType: "application/json",
			body: fmt.Sprintf(`{"error":{"message":%q,"type":"permission_error","code":"subscription_required"},"request_id":"keep-me"}`, testCodexSubscriptionError),
			want: testCodexSubscriptionCopy,
		},
		{
			name: "unauthorized JSON", status: http.StatusUnauthorized, contentType: "application/json",
			body: `{"error":"unauthorized","signin_url":"https://ollama.com/connect?name=test&key=test-key"}`,
			want: testCodexSignInCopy,
		},
		{
			name: "unauthorized JSON with embedded sign-in URL", status: http.StatusUnauthorized, contentType: "application/json",
			body: `{"error":{"message":"Sign in: https://ollama.com/connect?name=test&key=test-key","code":"authentication_error","signin_url":"https://ollama.com/connect?name=test&key=test-key"}}`,
			want: testCodexSignInCopy,
		},
		{
			name: "subscription stream", status: http.StatusOK, contentType: "text/event-stream; charset=utf-8",
			body: fmt.Sprintf("event: response.failed\ndata: {\"type\":\"response.failed\",\"sequence_number\":4,\"response\":{\"id\":\"resp_test\",\"status\":\"failed\",\"error\":{\"code\":\"api_error\",\"message\":%q}}}\n\n", testCodexSubscriptionError),
			want: testCodexSubscriptionCopy,
		},
		{
			name: "unauthorized stream", status: http.StatusOK, contentType: "text/event-stream",
			body: "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"code\":\"authentication_error\",\"message\":\"Sign in: https://ollama.com/connect?name=test&key=test-key\"},\"signin_url\":\"https://ollama.com/connect?name=test&key=test-key\"}}\n\n",
			want: testCodexSignInCopy,
		},
		{
			name: "nested SSE error", status: http.StatusOK, contentType: "text/event-stream",
			body: "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"unauthorized\",\"signin_url\":\"https://ollama.com/connect?name=test&key=test-key\"}}\n\n",
			want: testCodexSignInCopy,
		},
		{
			name: "multiline stream with CRLF", status: http.StatusOK, contentType: "text/event-stream",
			body: "id: keep-me\r\nevent: error\r\ndata: {\"type\":\"error\",\r\ndata: \"code\":\"authentication_error\",\"message\":\"unauthorized\",\"signin_url\":\"https://ollama.com/connect?name=test&key=test-key\"}\r\n\r\n",
			want: testCodexSignInCopy,
		},
		{
			name: "unrelated forbidden error", status: http.StatusForbidden, contentType: "application/json",
			body: `{"error":{"message":"cloud is disabled","code":"cloud_disabled"}}`,
		},
		{
			name: "rate limit", status: http.StatusTooManyRequests, contentType: "application/json",
			body: `{"error":{"message":"too many requests","code":"rate_limit_exceeded"}}`,
		},
		{
			name: "successful JSON", status: http.StatusOK, contentType: "application/json",
			body: `{"status":"completed","output":[{"text":"unauthorized"}]}`,
		},
		{
			name: "successful stream mentioning error text", status: http.StatusOK, contentType: "text/event-stream",
			body: fmt.Sprintf("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":%q}\n\n: heartbeat\n\ndata: [DONE]\n\n", testCodexSubscriptionError),
		},
		{
			name: "malformed error", status: http.StatusForbidden, contentType: "application/json",
			body: `{"error":`,
		},
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
				if resp.StatusCode != tt.status || resp.Header.Get("X-Request-ID") != "keep-me" {
					t.Errorf("status/headers changed: %d %v", resp.StatusCode, resp.Header)
				}
				if model == "gpt-native" || tt.want == "" {
					if string(body) != tt.body {
						t.Errorf("%s response changed: %s", model, body)
					}
					continue
				}
				if !strings.Contains(string(body), tt.want) {
					t.Errorf("body = %s, want copy %q", body, tt.want)
				}
				if tt.want == testCodexSignInCopy {
					for _, unwanted := range []string{"signin_url", "https://ollama.com/connect", "ollama://connect", "test-key"} {
						if strings.Contains(string(body), unwanted) {
							t.Errorf("device sign-in link is still visible: %s", body)
						}
					}
				}
				if tt.want == testCodexSubscriptionCopy && strings.Contains(string(body), "test-reference") {
					t.Errorf("old subscription message is still visible: %s", body)
				}
				if strings.Contains(tt.contentType, "event-stream") {
					if !strings.Contains(string(body), "event:") {
						t.Errorf("SSE framing was lost: %s", body)
					}
				} else if !json.Valid(body) {
					t.Errorf("invalid JSON response: %s", body)
				}
			}
		})
	}
}

func TestCodexDesktopAccessErrorCopySignInMessage(t *testing.T) {
	handler := newTestCodexDesktop(t, "http://localhost", "http://localhost", "unused")
	body, changed := handler.rewriteAccessErrorJSON([]byte(`{"error":"unauthorized"}`), http.StatusUnauthorized)
	want, err := json.Marshal(map[string]string{"error": testCodexSignInCopy})
	if err != nil {
		t.Fatal(err)
	}
	if !changed || !bytes.Equal(body, want) {
		t.Fatalf("error = %s, want only the short sign-in message %s", body, want)
	}
}

func TestCodexDesktopAccessErrorCopyDoesNotLogSignInSecret(t *testing.T) {
	var logs bytes.Buffer
	handler := newTestCodexDesktop(t, "http://localhost", "http://localhost", "unused")
	handler.logger = slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
	body := []byte(`{"error":{"message":"Sign in: https://ollama.com/connect?name=test&key=test-key","code":"authentication_error","signin_url":"https://ollama.com/connect?name=test&key=test-key"}}`)

	if _, changed := handler.rewriteAccessErrorJSON(body, http.StatusUnauthorized); !changed {
		t.Fatal("sign-in error was not rewritten")
	}
	for _, secret := range []string{"test-key", "https://ollama.com/connect"} {
		if strings.Contains(logs.String(), secret) {
			t.Fatalf("debug log contains sign-in secret %q: %s", secret, logs.String())
		}
	}
}

func TestCodexAccessErrorStreamFragmentedFrames(t *testing.T) {
	body := "event: error\ndata: {\"type\":\"error\",\"code\":\"authentication_error\",\"message\":\"unauthorized\"}\n\n"
	handler := newTestCodexDesktop(t, "http://localhost", "http://localhost", "unused")
	upstream := io.NopCloser(iotest.OneByteReader(strings.NewReader(body)))
	stream := &codexAccessErrorStream{
		ReadCloser: upstream,
		reader:     bufio.NewReader(upstream),
		rewrite:    func(body []byte) ([]byte, bool) { return handler.rewriteAccessErrorJSON(body, http.StatusOK) },
		limit:      defaultMaxBodyBytes,
	}
	defer stream.Close()
	got, err := io.ReadAll(stream)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(got), testCodexSignInCopy) || !bytes.HasSuffix(got, []byte("\n\n")) {
		t.Fatalf("fragmented event = %s, want rewritten message and intact SSE framing", got)
	}
}

func TestCodexAccessErrorStreamOversizedFramePassesThrough(t *testing.T) {
	body := "data: " + strings.Repeat("x", 8192) + "\n\n: heartbeat\n\n"
	upstream := io.NopCloser(strings.NewReader(body))
	stream := &codexAccessErrorStream{
		ReadCloser: upstream,
		reader:     bufio.NewReader(upstream),
		rewrite: func([]byte) ([]byte, bool) {
			t.Error("oversized frame should not be rewritten")
			return nil, false
		},
		limit: 4096,
	}
	defer stream.Close()
	got, err := io.ReadAll(stream)
	if err != nil || string(got) != body {
		t.Fatalf("large event was not passed through unchanged: length=%d, error=%v", len(got), err)
	}
}

func TestCodexDesktopAccessErrorCopyDoesNotBufferSuccessfulStream(t *testing.T) {
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
