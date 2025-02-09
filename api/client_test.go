package api

import (
    "testing"
    "context"
    "net/http"
    "net/http/httptest"
    "net/url"
    "errors"
    "time"
    "strings"
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

// Test generated using Keploy
func TestDo_SuccessfulResponse(t *testing.T) {
    ctx := context.Background()
    mockResponse := `{"key": "value"}`
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(mockResponse))
    }))
    defer server.Close()

    baseURL, _ := url.Parse(server.URL)
    client := NewClient(baseURL, server.Client())

    var result map[string]string
    err := client.do(ctx, http.MethodGet, "/test-path", nil, &result)
    if err != nil {
        t.Fatalf("expected no error, got %v", err)
    }

    if result["key"] != "value" {
        t.Fatalf("expected 'value', got %v", result["key"])
    }
}


// Test generated using Keploy
func TestDo_ErrorResponse(t *testing.T) {
    ctx := context.Background()
    mockResponse := `{"error": "something went wrong"}`
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusInternalServerError)
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(mockResponse))
    }))
    defer server.Close()

    baseURL, _ := url.Parse(server.URL)
    client := NewClient(baseURL, server.Client())

    err := client.do(ctx, http.MethodGet, "/test-path", nil, nil)
    if err == nil {
        t.Fatalf("expected an error, got nil")
    }

    var statusErr StatusError
    if !errors.As(err, &statusErr) {
        t.Fatalf("expected StatusError, got %T", err)
    }

    if statusErr.StatusCode != http.StatusInternalServerError {
        t.Fatalf("expected status code 500, got %d", statusErr.StatusCode)
    }
}


// Test generated using Keploy
func TestDo_RequestTimeout(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
    defer cancel()

    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        time.Sleep(200 * time.Millisecond)
        w.WriteHeader(http.StatusOK)
        w.Write([]byte(`{"key": "value"}`))
    }))
    defer server.Close()

    baseURL, _ := url.Parse(server.URL)
    httpClient := server.Client()
    client := NewClient(baseURL, httpClient)

    var result map[string]string
    err := client.do(ctx, http.MethodGet, "/test-timeout", nil, &result)
    if err == nil {
        t.Fatalf("expected an error due to timeout, got nil")
    }
    if !errors.Is(err, context.DeadlineExceeded) {
        t.Fatalf("expected context deadline exceeded error, got %v", err)
    }
}


// Test generated using Keploy
func TestDo_UnexpectedContentType(t *testing.T) {
    ctx := context.Background()
    mockResponse := `Not JSON data`
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Header().Set("Content-Type", "text/plain")
        w.Write([]byte(mockResponse))
    }))
    defer server.Close()

    baseURL, _ := url.Parse(server.URL)
    client := NewClient(baseURL, server.Client())

    var result map[string]string
    err := client.do(ctx, http.MethodGet, "/test-unexpected-content-type", nil, &result)
    if err == nil {
        t.Fatalf("expected an error due to unexpected Content-Type, got nil")
    }
    if !strings.Contains(err.Error(), "invalid character") {
        t.Fatalf("expected JSON parsing error, got %v", err)
    }
}

