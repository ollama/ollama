package server

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/middleware"
)

func TestRemoteChatAnthropicStreamError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "0")
	t.Setenv("OLLAMA_AUTH", "0")
	t.Setenv("OLLAMA_REMOTES", "127.0.0.1")

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("unexpected upstream path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprintln(w, `{"model":"test","message":{"role":"assistant","content":"Hel"},"done":false}`)
		w.(http.Flusher).Flush()
		fmt.Fprintln(w, `{"error":"model runner has unexpectedly stopped"}`)
	}))
	defer upstream.Close()

	s := Server{}
	created := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:      "test-remote",
		RemoteHost: upstream.URL,
		From:       "test",
		Info:       map[string]any{"capabilities": []string{"completion"}},
		Stream:     &stream,
	})
	if created.Code != http.StatusOK {
		t.Fatalf("create remote model: status %d: %s", created.Code, created.Body.String())
	}

	for _, webSearch := range []bool{false, true} {
		t.Run(fmt.Sprintf("web_search=%t", webSearch), func(t *testing.T) {
			router := gin.New()
			router.POST("/v1/messages", middleware.AnthropicMessagesMiddleware(), s.ChatHandler)
			tools := ""
			if webSearch {
				tools = `,"tools":[{"type":"web_search_20250305","name":"web_search"}]`
			}
			body := `{"model":"test-remote","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"Hi"}]` + tools + `}`
			req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if resp.Code != http.StatusOK {
				t.Fatalf("expected committed status 200, got %d: %s", resp.Code, resp.Body.String())
			}
			if got := resp.Result().Header.Get("Content-Type"); got != "text/event-stream" {
				t.Errorf("expected SSE, got Content-Type %q", got)
			}
			wire := resp.Body.String()
			if !strings.Contains(wire, `"text":"Hel"`) {
				t.Errorf("missing partial text: %s", wire)
			}
			wantError := "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"model runner has unexpectedly stopped\"}}\n\n"
			if !strings.HasSuffix(wire, wantError) || strings.Count(wire, "event: error\n") != 1 {
				t.Errorf("expected one terminal error event: %s", wire)
			}
			if strings.Contains(wire, "event: message_stop\n") {
				t.Error("failed stream reported successful completion")
			}
		})
	}
}

func TestRemoteInferenceCancellation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "0")
	t.Setenv("OLLAMA_AUTH", "0")
	t.Setenv("OLLAMA_REMOTES", "127.0.0.1")

	s := Server{}
	for _, endpoint := range []struct {
		name     string
		handler  gin.HandlerFunc
		request  string
		response string
	}{
		{"generate", s.GenerateHandler, `"prompt":"hi"`, `{"response":"hello","done":false}`},
		{"chat", s.ChatHandler, `"messages":[{"role":"user","content":"hi"}]`, `{"message":{"role":"assistant","content":"hello"},"done":false}`},
	} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", endpoint.name, streaming), func(t *testing.T) {
				upstreamStarted := make(chan struct{})
				upstreamCanceled := make(chan struct{})
				releaseUpstream := make(chan struct{})
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path != "/api/"+endpoint.name {
						t.Errorf("unexpected upstream path: %s", r.URL.Path)
					}
					_, _ = io.Copy(io.Discard, r.Body)
					close(upstreamStarted)
					if streaming {
						w.Header().Set("Content-Type", "application/x-ndjson")
						fmt.Fprintln(w, endpoint.response)
						w.(http.Flusher).Flush()
					}
					select {
					case <-r.Context().Done():
						close(upstreamCanceled)
					case <-releaseUpstream:
					}
				}))
				var local *httptest.Server
				t.Cleanup(func() {
					// Unblock both servers even when cancellation fails.
					close(releaseUpstream)
					if local != nil {
						local.Close()
					}
					upstream.Close()
				})

				w := createRequest(t, s.CreateHandler, api.CreateRequest{
					Model:      "test-remote",
					RemoteHost: upstream.URL,
					From:       "test",
					Info:       map[string]any{"capabilities": []string{"completion"}},
					Stream:     &stream,
				})
				if w.Code != http.StatusOK {
					t.Fatalf("create remote model: status %d: %s", w.Code, w.Body.String())
				}

				handlerDone := make(chan struct{})
				router := gin.New()
				router.POST("/api/"+endpoint.name, func(c *gin.Context) {
					defer close(handlerDone)
					endpoint.handler(c)
				})
				local = httptest.NewServer(router)
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				body := fmt.Sprintf(`{"model":"test-remote","stream":%t,%s}`, streaming, endpoint.request)
				req, err := http.NewRequestWithContext(ctx, http.MethodPost, local.URL+"/api/"+endpoint.name, strings.NewReader(body))
				if err != nil {
					t.Fatal(err)
				}
				req.Header.Set("Content-Type", "application/json")
				firstChunk := make(chan error, 1)
				clientDone := make(chan error, 1)
				go func() {
					resp, err := local.Client().Do(req)
					if err == nil {
						defer resp.Body.Close()
						reader := bufio.NewReader(resp.Body)
						if streaming {
							_, err = reader.ReadString('\n')
							firstChunk <- err
						}
						if err == nil {
							_, err = io.Copy(io.Discard, reader)
						}
					}
					clientDone <- err
				}()

				select {
				case <-upstreamStarted:
				case err := <-clientDone:
					t.Fatalf("request ended before reaching upstream: %v", err)
				case <-time.After(3 * time.Second):
					t.Fatal("request did not reach upstream")
				}
				if streaming {
					select {
					case err := <-firstChunk:
						if err != nil {
							t.Fatalf("read first chunk: %v", err)
						}
					case <-time.After(3 * time.Second):
						t.Fatal("first chunk was not forwarded")
					}
				}
				cancel()
				select {
				case <-upstreamCanceled:
				case <-time.After(3 * time.Second):
					t.Fatal("client cancellation did not cancel the upstream request")
				}
				select {
				case <-handlerDone:
				case <-time.After(3 * time.Second):
					t.Fatal("remote inference handler did not return")
				}
				select {
				case err := <-clientDone:
					if !errors.Is(err, context.Canceled) {
						t.Fatalf("client error = %v, want context.Canceled", err)
					}
				case <-time.After(3 * time.Second):
					t.Fatal("client request did not return")
				}
			})
		}
	}
}
