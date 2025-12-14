package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestClientFromEnvironment(t *testing.T) {
	type testCase struct {
		value  string
		expect string
		err    error
	}

	testCases := map[string]*testCase{
		"empty":                      {value: "", expect: "http://127.0.0.1:11434"},
		"only address":               {value: "1.2.3.4", expect: "http://1.2.3.4:11434"},
		"only port":                  {value: ":1234", expect: "http://:1234"},
		"address and port":           {value: "1.2.3.4:1234", expect: "http://1.2.3.4:1234"},
		"scheme http and address":    {value: "http://1.2.3.4", expect: "http://1.2.3.4:80"},
		"scheme https and address":   {value: "https://1.2.3.4", expect: "https://1.2.3.4:443"},
		"scheme, address, and port":  {value: "https://1.2.3.4:1234", expect: "https://1.2.3.4:1234"},
		"hostname":                   {value: "example.com", expect: "http://example.com:11434"},
		"hostname and port":          {value: "example.com:1234", expect: "http://example.com:1234"},
		"scheme http and hostname":   {value: "http://example.com", expect: "http://example.com:80"},
		"scheme https and hostname":  {value: "https://example.com", expect: "https://example.com:443"},
		"scheme, hostname, and port": {value: "https://example.com:1234", expect: "https://example.com:1234"},
		"trailing slash":             {value: "example.com/", expect: "http://example.com:11434"},
		"trailing slash port":        {value: "example.com:1234/", expect: "http://example.com:1234"},
	}

	for k, v := range testCases {
		t.Run(k, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", v.value)

			client, err := ClientFromEnvironment()
			if err != v.err {
				t.Fatalf("expected %s, got %s", v.err, err)
			}

			if client.base.String() != v.expect {
				t.Fatalf("expected %s, got %s", v.expect, client.base.String())
			}
		})
	}
}

// testError represents an internal error type with status code and message
// this is used since the error response from the server is not a standard error struct
type testError struct {
	message    string
	statusCode int
	raw        bool // if true, write message as-is instead of JSON encoding
}

func (e testError) Error() string {
	return e.message
}

func TestClientStream(t *testing.T) {
	testCases := []struct {
		name      string
		responses []any
		wantErr   string
	}{
		{
			name: "immediate error response",
			responses: []any{
				testError{
					message:    "test error message",
					statusCode: http.StatusBadRequest,
				},
			},
			wantErr: "test error message",
		},
		{
			name: "error after successful chunks, ok response",
			responses: []any{
				ChatResponse{Message: Message{Content: "partial response 1"}},
				ChatResponse{Message: Message{Content: "partial response 2"}},
				testError{
					message:    "mid-stream error",
					statusCode: http.StatusOK,
				},
			},
			wantErr: "mid-stream error",
		},
		{
			name: "http status error takes precedence over general error",
			responses: []any{
				testError{
					message:    "custom error message",
					statusCode: http.StatusInternalServerError,
				},
			},
			wantErr: "500",
		},
		{
			name: "successful stream completion",
			responses: []any{
				ChatResponse{Message: Message{Content: "chunk 1"}},
				ChatResponse{Message: Message{Content: "chunk 2"}},
				ChatResponse{
					Message:    Message{Content: "final chunk"},
					Done:       true,
					DoneReason: "stop",
				},
			},
		},
		{
			name: "plain text error response",
			responses: []any{
				"internal server error",
			},
			wantErr: "internal server error",
		},
		{
			name: "HTML error page",
			responses: []any{
				"<html><body>404 Not Found</body></html>",
			},
			wantErr: "404 Not Found",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				flusher, ok := w.(http.Flusher)
				if !ok {
					t.Fatal("expected http.Flusher")
				}

				w.Header().Set("Content-Type", "application/x-ndjson")

				for _, resp := range tc.responses {
					if errResp, ok := resp.(testError); ok {
						w.WriteHeader(errResp.statusCode)
						err := json.NewEncoder(w).Encode(map[string]string{
							"error": errResp.message,
						})
						if err != nil {
							t.Fatal("failed to encode error response:", err)
						}
						return
					}

					if str, ok := resp.(string); ok {
						fmt.Fprintln(w, str)
						flusher.Flush()
						continue
					}

					if err := json.NewEncoder(w).Encode(resp); err != nil {
						t.Fatalf("failed to encode response: %v", err)
					}
					flusher.Flush()
				}
			}))
			defer ts.Close()

			client := NewClient(&url.URL{Scheme: "http", Host: ts.Listener.Addr().String()}, http.DefaultClient)

			var receivedChunks []ChatResponse
			err := client.stream(t.Context(), http.MethodPost, "/v1/chat", nil, func(chunk []byte) error {
				var resp ChatResponse
				if err := json.Unmarshal(chunk, &resp); err != nil {
					return fmt.Errorf("failed to unmarshal chunk: %w", err)
				}
				receivedChunks = append(receivedChunks, resp)
				return nil
			})

			if tc.wantErr != "" {
				if err == nil {
					t.Fatal("expected error but got nil")
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("expected error containing %q, got %v", tc.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestClientDo(t *testing.T) {
	testCases := []struct {
		name           string
		response       any
		wantErr        string
		wantStatusCode int
	}{
		{
			name: "immediate error response",
			response: testError{
				message:    "test error message",
				statusCode: http.StatusBadRequest,
			},
			wantErr:        "test error message",
			wantStatusCode: http.StatusBadRequest,
		},
		{
			name: "server error response",
			response: testError{
				message:    "internal error",
				statusCode: http.StatusInternalServerError,
			},
			wantErr:        "internal error",
			wantStatusCode: http.StatusInternalServerError,
		},
		{
			name: "successful response",
			response: struct {
				ID      string `json:"id"`
				Success bool   `json:"success"`
			}{
				ID:      "msg_123",
				Success: true,
			},
		},
		{
			name: "plain text error response",
			response: testError{
				message:    "internal server error",
				statusCode: http.StatusInternalServerError,
				raw:        true,
			},
			wantErr:        "internal server error",
			wantStatusCode: http.StatusInternalServerError,
		},
		{
			name: "HTML error page",
			response: testError{
				message:    "<html><body>404 Not Found</body></html>",
				statusCode: http.StatusNotFound,
				raw:        true,
			},
			wantErr:        "<html><body>404 Not Found</body></html>",
			wantStatusCode: http.StatusNotFound,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if errResp, ok := tc.response.(testError); ok {
					w.WriteHeader(errResp.statusCode)
					if !errResp.raw {
						err := json.NewEncoder(w).Encode(map[string]string{
							"error": errResp.message,
						})
						if err != nil {
							t.Fatal("failed to encode error response:", err)
						}
					} else {
						// Write raw message (simulates non-JSON error responses)
						fmt.Fprint(w, errResp.message)
					}
					return
				}

				w.Header().Set("Content-Type", "application/json")
				if err := json.NewEncoder(w).Encode(tc.response); err != nil {
					t.Fatalf("failed to encode response: %v", err)
				}
			}))
			defer ts.Close()

			client := NewClient(&url.URL{Scheme: "http", Host: ts.Listener.Addr().String()}, http.DefaultClient)

			var resp struct {
				ID      string `json:"id"`
				Success bool   `json:"success"`
			}
			err := client.do(t.Context(), http.MethodPost, "/v1/messages", nil, &resp)

			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("got nil, want error %q", tc.wantErr)
				}
				if err.Error() != tc.wantErr {
					t.Errorf("error message mismatch: got %q, want %q", err.Error(), tc.wantErr)
				}
				if tc.wantStatusCode != 0 {
					if statusErr, ok := err.(StatusError); ok {
						if statusErr.StatusCode != tc.wantStatusCode {
							t.Errorf("status code mismatch: got %d, want %d", statusErr.StatusCode, tc.wantStatusCode)
						}
					} else {
						t.Errorf("expected StatusError, got %T", err)
					}
				}
				return
			}

			if err != nil {
				t.Fatalf("got error %q, want nil", err)
			}

			if expectedResp, ok := tc.response.(struct {
				ID      string `json:"id"`
				Success bool   `json:"success"`
			}); ok {
				if resp.ID != expectedResp.ID {
					t.Errorf("response ID mismatch: got %q, want %q", resp.ID, expectedResp.ID)
				}
				if resp.Success != expectedResp.Success {
					t.Errorf("response Success mismatch: got %v, want %v", resp.Success, expectedResp.Success)
				}
			}
		})
	}
}

func TestClientTranscribe(t *testing.T) {
	tests := []struct {
		name       string
		response   TranscribeResponse
		statusCode int
		wantErr    bool
	}{
		{
			name: "successful transcription",
			response: TranscribeResponse{
				Model:    "whisper:base",
				Text:     "Hello world",
				Language: "en",
				Duration: 2.5,
				Task:     "transcribe",
				Done:     true,
			},
			statusCode: http.StatusOK,
			wantErr:    false,
		},
		{
			name:       "server error",
			response:   TranscribeResponse{},
			statusCode: http.StatusInternalServerError,
			wantErr:    true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost {
					t.Errorf("expected POST, got %s", r.Method)
				}
				if r.URL.Path != "/api/transcribe" {
					t.Errorf("expected /api/transcribe, got %s", r.URL.Path)
				}

				w.WriteHeader(tc.statusCode)
				if tc.statusCode == http.StatusOK {
					json.NewEncoder(w).Encode(tc.response)
				} else {
					json.NewEncoder(w).Encode(map[string]string{"error": "server error"})
				}
			}))
			defer ts.Close()

			u, _ := url.Parse(ts.URL)
			client := NewClient(u, http.DefaultClient)

			req := &TranscribeRequest{
				Model: "whisper:base",
				Audio: []byte("fake audio data"),
			}

			resp, err := client.Transcribe(context.Background(), req)

			if tc.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if resp.Text != tc.response.Text {
				t.Errorf("text mismatch: got %q, want %q", resp.Text, tc.response.Text)
			}
			if resp.Language != tc.response.Language {
				t.Errorf("language mismatch: got %q, want %q", resp.Language, tc.response.Language)
			}
		})
	}
}

func TestClientTranslate(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/api/translate" {
			t.Errorf("expected /api/translate, got %s", r.URL.Path)
		}

		// Verify translate flag is set
		var req TranscribeRequest
		json.NewDecoder(r.Body).Decode(&req)
		if !req.Translate {
			t.Error("expected translate=true")
		}

		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(TranscribeResponse{
			Model:    "whisper:base",
			Text:     "Hello world",
			Language: "fr",
			Task:     "translate",
			Done:     true,
		})
	}))
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	client := NewClient(u, http.DefaultClient)

	req := &TranscribeRequest{
		Model: "whisper:base",
		Audio: []byte("fake audio data"),
	}

	resp, err := client.Translate(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Task != "translate" {
		t.Errorf("task mismatch: got %q, want %q", resp.Task, "translate")
	}
}

func TestClientTranscribeStream(t *testing.T) {
	segments := []TranscribeStreamResponse{
		{Segment: &TranscribeSegment{ID: 0, Start: 0.0, End: 2.5, Text: "Hello"}, Done: false},
		{Segment: &TranscribeSegment{ID: 1, Start: 2.5, End: 5.0, Text: "world"}, Done: false},
		{PartialText: "Hello world", Done: true},
	}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/api/transcribe" {
			t.Errorf("expected /api/transcribe, got %s", r.URL.Path)
		}

		// Verify stream flag is set
		var req TranscribeRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Stream == nil || !*req.Stream {
			t.Error("expected stream=true")
		}

		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)

		for _, seg := range segments {
			data, _ := json.Marshal(seg)
			w.Write(data)
			w.Write([]byte("\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	client := NewClient(u, http.DefaultClient)

	req := &TranscribeRequest{
		Model: "whisper:base",
		Audio: []byte("fake audio data"),
	}

	var received []TranscribeStreamResponse
	err := client.TranscribeStream(context.Background(), req, func(resp TranscribeStreamResponse) error {
		received = append(received, resp)
		return nil
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(received) != len(segments) {
		t.Errorf("received %d segments, want %d", len(received), len(segments))
	}

	// Verify last segment is done
	if len(received) > 0 && !received[len(received)-1].Done {
		t.Error("expected last segment to have Done=true")
	}
}

func TestClientTranscribeRejectsStreamingRequest(t *testing.T) {
	u, _ := url.Parse("http://localhost:11434")
	client := NewClient(u, http.DefaultClient)

	stream := true
	req := &TranscribeRequest{
		Model:  "whisper:base",
		Audio:  []byte("fake audio data"),
		Stream: &stream,
	}

	_, err := client.Transcribe(context.Background(), req)
	if err == nil {
		t.Error("expected error for streaming request with Transcribe()")
	}
}
