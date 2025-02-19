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

					if err := json.NewEncoder(w).Encode(resp); err != nil {
						t.Fatalf("failed to encode response: %v", err)
					}
					flusher.Flush()
				}
			}))
			defer ts.Close()

			client := NewClient(&url.URL{Scheme: "http", Host: ts.Listener.Addr().String()}, http.DefaultClient)

			var receivedChunks []ChatResponse
			err := client.stream(context.Background(), http.MethodPost, "/v1/chat", nil, func(chunk []byte) error {
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
		name     string
		response any
		wantErr  string
	}{
		{
			name: "immediate error response",
			response: testError{
				message:    "test error message",
				statusCode: http.StatusBadRequest,
			},
			wantErr: "test error message",
		},
		{
			name: "server error response",
			response: testError{
				message:    "internal error",
				statusCode: http.StatusInternalServerError,
			},
			wantErr: "internal error",
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
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if errResp, ok := tc.response.(testError); ok {
					w.WriteHeader(errResp.statusCode)
					err := json.NewEncoder(w).Encode(map[string]string{
						"error": errResp.message,
					})
					if err != nil {
						t.Fatal("failed to encode error response:", err)
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
			err := client.do(context.Background(), http.MethodPost, "/v1/messages", nil, &resp)

			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("got nil, want error %q", tc.wantErr)
				}
				if err.Error() != tc.wantErr {
					t.Errorf("error message mismatch: got %q, want %q", err.Error(), tc.wantErr)
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
