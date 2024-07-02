package api

import (
	"net/http"
	"net/url"
	"testing"

	"github.com/ollama/ollama/envconfig"
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
			envconfig.LoadConfig()

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

// Test function
func TestIsLocal(t *testing.T) {
	type test struct {
		client *Client
		want   bool
		err    error
	}

	tests := map[string]test{
		"localhost": {
			client: func() *Client {
				baseURL, _ := url.Parse("http://localhost:1234")
				return &Client{base: baseURL, http: &http.Client{}}
			}(),
			want: true,
			err:  nil,
		},
		"127.0.0.1": {
			client: func() *Client {
				baseURL, _ := url.Parse("http://127.0.0.1:1234")
				return &Client{base: baseURL, http: &http.Client{}}
			}(),
			want: true,
			err:  nil,
		},
		"example.com": {
			client: func() *Client {
				baseURL, _ := url.Parse("http://example.com:1111")
				return &Client{base: baseURL, http: &http.Client{}}
			}(),
			want: false,
			err:  nil,
		},
		"8.8.8.8": {
			client: func() *Client {
				baseURL, _ := url.Parse("http://8.8.8.8:1234")
				return &Client{base: baseURL, http: &http.Client{}}
			}(),
			want: false,
			err:  nil,
		},
		"empty host with port": {
			client: func() *Client {
				baseURL, _ := url.Parse("http://:1234")
				return &Client{base: baseURL, http: &http.Client{}}
			}(),
			want: true,
			err:  nil,
		},
		"empty host without port": {
			client: func() *Client {
				baseURL, _ := url.Parse("http://")
				return &Client{base: baseURL, http: &http.Client{}}
			}(),
			want: true,
			err:  nil,
		},
		"remote host without port": {
			client: func() *Client {
				baseURL, _ := url.Parse("http://example.com")
				return &Client{base: baseURL, http: &http.Client{}}
			}(),
			want: false,
			err:  nil,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			got := tc.client.IsLocal()
			if got != tc.want {
				t.Errorf("test %s failed: got %v, want %v", name, got, tc.want)
			}
		})
	}
}
