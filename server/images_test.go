package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
)

func TestMakeRequestWithRetry(t *testing.T) {
	authServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		err := json.NewEncoder(w).Encode(map[string]string{
			"token": "test-token",
		})
		if err != nil {
			t.Errorf("failed to encode response: %v", err)
		}
	}))
	defer authServer.Close()

	tests := []struct {
		name          string
		serverHandler http.HandlerFunc
		method        string
		body          string
		wantErr       error
		wantStatus    int
	}{
		{
			name: "successful request",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				if _, err := w.Write([]byte("success")); err != nil {
					t.Errorf("failed to write response: %v", err)
				}
			},
			method:     http.MethodGet,
			wantStatus: http.StatusOK,
		},
		{
			name: "not found error",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusNotFound)
			},
			method:  http.MethodGet,
			wantErr: os.ErrNotExist,
		},
		{
			name: "request with body retry",
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") == "" {
					w.Header().Set("WWW-Authenticate", `Bearer realm="`+authServer.URL+`"`)
					w.WriteHeader(http.StatusUnauthorized)
					return
				}
				buf := new(bytes.Buffer)
				if _, err := buf.ReadFrom(r.Body); err != nil {
					t.Errorf("failed to read request body: %v", err)
				}
				if buf.String() != `{"key": "value"}` {
					t.Errorf("body not preserved on retry, got %s", buf.String())
				}
				w.WriteHeader(http.StatusOK)
			},
			method:     http.MethodPost,
			body:       `{"key": "value"}`,
			wantStatus: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(tt.serverHandler)
			defer server.Close()

			requestURL, _ := url.Parse(server.URL)
			var body io.ReadSeeker
			if tt.body != "" {
				body = strings.NewReader(tt.body)
			}

			regOpts := &registryOptions{
				Insecure: true,
			}

			resp, err := makeRequestWithRetry(context.Background(), tt.method, requestURL, nil, body, regOpts)

			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Errorf("got error %v, want %v", err, tt.wantErr)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if resp.StatusCode != tt.wantStatus {
				t.Errorf("got status %d, want %d", resp.StatusCode, tt.wantStatus)
			}

			resp.Body.Close()
		})
	}
}
