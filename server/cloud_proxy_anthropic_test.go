package server

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
)

const claudeAcceptEncoding = "gzip, deflate, br, zstd"

func TestCloudAnthropicWebSearchCompression(t *testing.T) {
	for _, encoding := range []string{"identity", "gzip"} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", encoding, streaming), func(t *testing.T) {
				var chatCalls, searchCalls atomic.Int32
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					switch r.URL.Path {
					case "/api/show":
						fmt.Fprint(w, `{}`)
					case "/api/web_search":
						searchCalls.Add(1)
						var req api.WebSearchRequest
						if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
							t.Error(err)
						}
						if req.Query != "Ollama" {
							t.Errorf("search query = %q, want Ollama", req.Query)
						}
						fmt.Fprint(w, `{"results":[{"title":"Test result","url":"https://example.com","content":"A local fixture."}]}`)
					case "/api/chat":
						chatCalls.Add(1)
						if got := r.Header.Get("Accept-Encoding"); got != "gzip" {
							t.Errorf("unexpected compression negotiation %q", got)
						}
						var req api.ChatRequest
						if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
							t.Error(err)
						}
						if req.Model != "test-model" {
							t.Errorf("upstream model = %q", req.Model)
						}
						payload := `{"model":"test-model","message":{"role":"assistant","content":"A local answer."},"done":true,"done_reason":"stop"}`
						if len(req.Messages) == 1 {
							payload = `{"model":"test-model","message":{"role":"assistant","tool_calls":[{"function":{"name":"web_search","arguments":{"query":"Ollama"}}}]},"done":true,"done_reason":"stop"}`
							if streaming {
								// Coalesced records followed by a final record without a newline.
								payload = "{\"message\":{\"role\":\"assistant\"},\"done\":false}\n" + payload
							}
						} else {
							last := req.Messages[len(req.Messages)-1]
							if last.Role != "tool" || !strings.Contains(last.Content, "https://example.com") || !strings.Contains(last.Content, "A local fixture.") {
								t.Errorf("follow-up request is missing the search result: %+v", last)
							}
						}
						w.Header().Set("Content-Type", "application/x-ndjson")
						data := []byte(payload)
						if encoding == "gzip" {
							data = gzipCloudFixture(t, data)
							w.Header().Set("Content-Encoding", "gzip")
						}
						w.Header().Set("Content-Length", fmt.Sprint(len(data)))
						_, _ = w.Write(data)
					default:
						t.Errorf("unexpected upstream path %s", r.URL.Path)
						w.WriteHeader(http.StatusNotFound)
					}
				}))
				defer upstream.Close()
				local := cloudAnthropicServer(t, upstream.URL)
				req := cloudAnthropicRequest(t, local.URL, streaming)
				resp, err := local.Client().Do(req)
				if err != nil {
					t.Fatal(err)
				}
				defer resp.Body.Close()
				data, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("status=%d headers=%v body=%s", resp.StatusCode, resp.Header, data)
				if resp.StatusCode != http.StatusOK || len(data) == 0 {
					t.Fatalf("expected nonempty success, status=%d body=%q", resp.StatusCode, data)
				}
				if got := resp.Header.Get("Content-Encoding"); got != "" {
					t.Errorf("transformed response has Content-Encoding %q", got)
				}
				if resp.ContentLength >= 0 && resp.ContentLength != int64(len(data)) {
					t.Errorf("transformed response has stale Content-Length %d, body length %d", resp.ContentLength, len(data))
				}
				if streaming {
					if resp.Header.Get("Content-Type") != "text/event-stream" {
						t.Errorf("expected SSE content type")
					}
					for _, event := range []string{"message_start", "content_block_start", "content_block_stop", "message_delta", "message_stop"} {
						if !bytes.Contains(data, []byte("event: "+event+"\n")) {
							t.Errorf("missing event %s", event)
						}
					}
					for _, frame := range strings.Split(strings.TrimSpace(string(data)), "\n\n") {
						lines := strings.Split(frame, "\n")
						if len(lines) != 2 || !strings.HasPrefix(lines[0], "event: ") || !strings.HasPrefix(lines[1], "data: ") {
							t.Fatalf("invalid SSE frame: %q", frame)
						}
						var event struct {
							Type string `json:"type"`
						}
						if err := json.Unmarshal([]byte(strings.TrimPrefix(lines[1], "data: ")), &event); err != nil || event.Type != strings.TrimPrefix(lines[0], "event: ") {
							t.Fatalf("invalid SSE data: %q", frame)
						}
					}
					if strings.Count(string(data), "event: message_start\n") != 1 || strings.Count(string(data), "event: message_stop\n") != 1 || strings.Contains(string(data), "event: error\n") {
						t.Errorf("invalid SSE lifecycle: %s", data)
					}
				} else {
					if resp.Header.Get("Content-Type") != "application/json" {
						t.Errorf("expected JSON content type")
					}
					var message anthropic.MessagesResponse
					if err := json.Unmarshal(data, &message); err != nil {
						t.Fatal(err)
					}
					if message.Type != "message" || message.StopReason != "end_turn" || len(message.Content) != 3 {
						t.Errorf("unexpected message: %+v", message)
					}
				}
				for _, want := range []string{"server_tool_use", "web_search_tool_result", "A local answer."} {
					if !bytes.Contains(data, []byte(want)) {
						t.Errorf("missing %q in response", want)
					}
				}
				if chatCalls.Load() != 2 || searchCalls.Load() != 1 {
					t.Errorf("chat calls=%d search calls=%d, want 2 and 1", chatCalls.Load(), searchCalls.Load())
				}
			})
		}
	}
}

func TestCloudAnthropicInvalidResponse(t *testing.T) {
	final := []byte(`{"message":{"role":"assistant","content":"hello"},"done":true}`)
	truncated := gzipCloudFixture(t, final)
	corrupt := bytes.Clone(truncated)
	corrupt[len(corrupt)-8] ^= 0xff
	for _, tc := range []struct {
		name, encoding string
		data           []byte
	}{
		{"empty", "", nil},
		{"malformed JSON", "", []byte(`{"message":`)},
		{"malformed gzip", "gzip", []byte("not a gzip stream")},
		{"truncated gzip", "gzip", truncated[:len(truncated)-4]},
		{"corrupt gzip", "gzip", corrupt},
		{"trailing garbage", "", append(bytes.Clone(final), []byte("garbage")...)},
		{"unsupported encoding", "br", []byte("encoded data")},
		{"error payload", "", []byte(`{"error":"upstream failed"}`)},
	} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tc.name, stream), func(t *testing.T) {
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path == "/api/show" {
						fmt.Fprint(w, `{}`)
						return
					}
					w.Header().Set("Content-Type", "application/x-ndjson")
					w.Header().Set("Content-Encoding", tc.encoding)
					_, _ = w.Write(tc.data)
				}))
				defer upstream.Close()
				local := cloudAnthropicServer(t, upstream.URL)
				req := cloudAnthropicRequest(t, local.URL, stream)
				resp, err := local.Client().Do(req)
				if err != nil {
					t.Fatal(err)
				}
				defer resp.Body.Close()
				data, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("status=%d headers=%v body=%s", resp.StatusCode, resp.Header, data)
				var response anthropic.ErrorResponse
				if err := json.Unmarshal(data, &response); err != nil {
					t.Fatal(err)
				}
				if resp.StatusCode != http.StatusBadGateway || response.Type != "error" || response.Error.Type != "api_error" || response.Error.Message == "" {
					t.Errorf("expected an Anthropic 502 error, got %d %s", resp.StatusCode, data)
				}
				if resp.Header.Get("Content-Type") != "application/json" || resp.Header.Get("Content-Encoding") != "" {
					t.Errorf("unexpected error headers: %v", resp.Header)
				}
			})
		}
	}
}

func TestCloudAnthropicCancellation(t *testing.T) {
	for _, afterOutput := range []bool{false, true} {
		t.Run(fmt.Sprintf("after_output=%t", afterOutput), func(t *testing.T) {
			started, canceled, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/api/show" {
					fmt.Fprint(w, `{}`)
					return
				}
				w.Header().Set("Content-Type", "application/x-ndjson")
				w.Header().Set("Content-Encoding", "gzip")
				compressor := gzip.NewWriter(w)
				defer compressor.Close()
				if afterOutput {
					fmt.Fprintln(compressor, `{"message":{"role":"assistant","content":"hello"},"done":false}`)
				} else {
					fmt.Fprint(compressor, `{"message":`)
				}
				compressor.Flush()
				w.(http.Flusher).Flush()
				close(started)
				select {
				case <-r.Context().Done():
					close(canceled)
				case <-release:
				}
			}))
			t.Cleanup(upstream.Close)
			local := cloudAnthropicServer(t, upstream.URL)
			t.Cleanup(func() { close(release) })
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			done, firstEvent := make(chan error, 1), make(chan string, 1)
			req := cloudAnthropicRequest(t, local.URL, true).WithContext(ctx)
			go func() {
				resp, err := local.Client().Do(req)
				if err == nil {
					defer resp.Body.Close()
					reader := bufio.NewReader(resp.Body)
					line, readErr := reader.ReadString('\n')
					firstEvent <- line
					err = readErr
					if err == nil {
						_, err = io.Copy(io.Discard, reader)
					}
				}
				done <- err
			}()
			select {
			case <-started:
			case <-time.After(5 * time.Second):
				t.Fatal("upstream did not start")
			}
			if afterOutput {
				select {
				case line := <-firstEvent:
					if line != "event: message_start\n" {
						t.Fatalf("unexpected first output: %q", line)
					}
				case <-time.After(5 * time.Second):
					t.Fatal("first event was not flushed")
				}
			}
			cancel()
			select {
			case <-canceled:
			case <-time.After(5 * time.Second):
				t.Fatal("cancellation did not reach cloud upstream")
			}
			select {
			case err := <-done:
				if err == nil {
					t.Error("canceled request succeeded")
				}
			case <-time.After(5 * time.Second):
				t.Fatal("client did not stop")
			}
		})
	}
}

func TestCloudAnthropicStreamFailure(t *testing.T) {
	for _, tc := range []struct {
		name, encoding, tail string
	}{
		{"malformed/identity", "identity", `{"message":`},
		{"malformed/gzip", "gzip", `{"message":`},
		{"missing completion", "identity", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			release := make(chan struct{})
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/api/show" {
					fmt.Fprint(w, `{}`)
					return
				}
				w.Header().Set("Content-Type", "application/x-ndjson")
				var writer io.Writer = w
				var compressor *gzip.Writer
				if tc.encoding == "gzip" {
					w.Header().Set("Content-Encoding", "gzip")
					compressor = gzip.NewWriter(w)
					defer compressor.Close()
					writer = compressor
				}
				// Split within a JSON token; transport writes are not record boundaries.
				fmt.Fprint(writer, `{"message":{"role":"assistant","content":"hel`)
				if compressor != nil {
					compressor.Flush()
				}
				w.(http.Flusher).Flush()
				fmt.Fprint(writer, "lo\"},\"done\":false}\n")
				if compressor != nil {
					compressor.Flush()
				}
				w.(http.Flusher).Flush()
				select {
				case <-release:
				case <-r.Context().Done():
					return
				}
				fmt.Fprint(writer, tc.tail)
			}))
			defer upstream.Close()
			local := cloudAnthropicServer(t, upstream.URL)
			req := cloudAnthropicRequest(t, local.URL, true)
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			resp, err := local.Client().Do(req.WithContext(ctx))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			reader := bufio.NewReader(resp.Body)
			first, err := reader.ReadString('\n')
			close(release)
			if err != nil || first != "event: message_start\n" {
				t.Fatalf("first event=%q err=%v", first, err)
			}
			rest, err := io.ReadAll(reader)
			if err != nil {
				t.Fatal(err)
			}
			data := first + string(rest)
			t.Logf("status=%d body=%s", resp.StatusCode, data)
			if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "text/event-stream" {
				t.Errorf("unexpected response: %d %v", resp.StatusCode, resp.Header)
			}
			if !strings.Contains(data, `"text":"hello"`) || strings.Count(data, "event: error\n") != 1 || strings.Contains(data, "event: message_stop\n") {
				t.Errorf("expected partial text followed by one terminal error: %s", data)
			}
		})
	}
}

func TestCloudAnthropicRawCompression(t *testing.T) {
	payload := gzipCloudFixture(t, []byte(`{"type":"message","content":[{"type":"text","text":"raw"}]}`))
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" || r.Header.Get("Accept-Encoding") != claudeAcceptEncoding {
			t.Errorf("unexpected raw request: path=%s encoding=%q", r.URL.Path, r.Header.Get("Accept-Encoding"))
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Encoding", "gzip")
		w.Header().Set("Content-Length", fmt.Sprint(len(payload)))
		w.Header().Set("X-Upstream", "preserved")
		_, _ = w.Write(payload)
	}))
	defer upstream.Close()
	local := cloudAnthropicServer(t, upstream.URL)
	req, err := http.NewRequest(http.MethodPost, local.URL+"/v1/messages", strings.NewReader(`{"model":"test-model:cloud","max_tokens":100,"messages":[{"role":"user","content":"Hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept-Encoding", claudeAcceptEncoding)
	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, payload) || resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Encoding") != "gzip" || resp.Header.Get("Content-Type") != "application/json" || resp.Header.Get("X-Upstream") != "preserved" {
		t.Errorf("raw response changed: status=%d headers=%v body=%x", resp.StatusCode, resp.Header, data)
	}
}

func cloudAnthropicRequest(t *testing.T, baseURL string, stream bool) *http.Request {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, baseURL+"/v1/messages", strings.NewReader(fmt.Sprintf(`{"model":"test-model:cloud","max_tokens":100,"stream":%t,"messages":[{"role":"user","content":"Search Ollama"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}`, stream)))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept-Encoding", claudeAcceptEncoding)
	return req
}

func gzipCloudFixture(t *testing.T, data []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	w := gzip.NewWriter(&buf)
	if _, err := w.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func cloudAnthropicServer(t *testing.T, upstreamURL string) *httptest.Server {
	t.Helper()
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "0")
	t.Setenv("OLLAMA_AUTH", "0")
	withCloudProxyBaseURL(t, upstreamURL)
	originalSearchEndpoint := anthropic.WebSearchEndpoint
	anthropic.WebSearchEndpoint = upstreamURL + "/api/web_search"
	t.Cleanup(func() { anthropic.WebSearchEndpoint = originalSearchEndpoint })
	s := &Server{}
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	t.Cleanup(local.Close)
	t.Setenv("OLLAMA_HOST", local.URL)
	return local
}
