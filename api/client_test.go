package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
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

func TestStream(t *testing.T) {
	tests := []struct {
		name           string
		serverResponse []string
		statusCode     int
		expectedError  error
	}{
		{
			name: "unknown key error",
			serverResponse: []string{
				`{"error":"unauthorized access","code":"unknown_key","data":{"key":"test-key"}}`,
			},
			statusCode: http.StatusUnauthorized,
			expectedError: &ErrUnknownOllamaKey{
				Message: "unauthorized access",
				Key:     "test-key",
			},
		},
		{
			name: "general error message",
			serverResponse: []string{
				`{"error":"something went wrong"}`,
			},
			statusCode:    http.StatusInternalServerError,
			expectedError: fmt.Errorf("something went wrong"),
		},
		{
			name: "malformed json response",
			serverResponse: []string{
				`{invalid-json`,
			},
			statusCode:    http.StatusOK,
			expectedError: fmt.Errorf("unmarshal: invalid character 'i' looking for beginning of object key string"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/x-ndjson")
				w.WriteHeader(tt.statusCode)
				for _, resp := range tt.serverResponse {
					fmt.Fprintln(w, resp)
				}
			}))
			defer server.Close()

			baseURL, err := url.Parse(server.URL)
			if err != nil {
				t.Fatalf("failed to parse server URL: %v", err)
			}

			client := &Client{
				http: server.Client(),
				base: baseURL,
			}

			var responses [][]byte
			err = client.stream(context.Background(), "POST", "/test", "test", func(bts []byte) error {
				responses = append(responses, bts)
				return nil
			})

			// Error checking
			if tt.expectedError == nil {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}

			if err == nil {
				t.Fatalf("expected error %v, got nil", tt.expectedError)
			}

			// Check for specific error types
			var unknownKeyErr ErrUnknownOllamaKey
			if errors.As(tt.expectedError, &unknownKeyErr) {
				var gotErr ErrUnknownOllamaKey
				if !errors.As(err, &gotErr) {
					t.Fatalf("expected ErrUnknownOllamaKey, got %T", err)
				}
				if unknownKeyErr.Key != gotErr.Key {
					t.Errorf("expected key %q, got %q", unknownKeyErr.Key, gotErr.Key)
				}
				if unknownKeyErr.Message != gotErr.Message {
					t.Errorf("expected message %q, got %q", unknownKeyErr.Message, gotErr.Message)
				}
				return
			}

			var statusErr StatusError
			if errors.As(tt.expectedError, &statusErr) {
				var gotErr StatusError
				if !errors.As(err, &gotErr) {
					t.Fatalf("expected StatusError, got %T", err)
				}
				if statusErr.StatusCode != gotErr.StatusCode {
					t.Errorf("expected status code %d, got %d", statusErr.StatusCode, gotErr.StatusCode)
				}
				if statusErr.ErrorMessage != gotErr.ErrorMessage {
					t.Errorf("expected error message %q, got %q", statusErr.ErrorMessage, gotErr.ErrorMessage)
				}
				return
			}

			// For other errors, compare error strings
			if err.Error() != tt.expectedError.Error() {
				t.Errorf("expected error %q, got %q", tt.expectedError, err)
			}
		})
	}
}
