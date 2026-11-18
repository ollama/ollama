package cmd

import (
	"bytes"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

type cloudListTransport func(*http.Request) (*http.Response, error)

func (f cloudListTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestCloudList(t *testing.T) {
	t.Cleanup(envconfig.ReloadServerConfig)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:1")
	t.Setenv("OLLAMA_AUTH", "1")
	envconfig.ReloadServerConfig()

	for _, tt := range []struct {
		name    string
		body    string
		status  int
		noCloud string
		wantErr string
	}{
		{
			name:   "direct discovery without daemon or credentials",
			body:   `{"models":[{"name":"beta","digest":"short","size":1000000000},{"model":"alpha:8b","digest":"1234567890abcdef","size":1000000000}]}`,
			status: http.StatusOK,
		},
		{
			name:    "upstream failure",
			status:  http.StatusServiceUnavailable,
			body:    `{"error":"catalog unavailable"}`,
			wantErr: "catalog unavailable",
		},
		{
			name:    "cloud disabled",
			noCloud: "true",
			wantErr: "ollama cloud is disabled",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_NO_CLOUD", tt.noCloud)
			oldClient := http.DefaultClient
			http.DefaultClient = &http.Client{Transport: cloudListTransport(func(req *http.Request) (*http.Response, error) {
				if tt.noCloud == "true" {
					t.Fatal("cloud-disabled listing made a network request")
				}
				if req.Method != http.MethodGet || req.URL.String() != "https://ollama.com/api/tags" {
					t.Errorf("unexpected request: %s %s", req.Method, req.URL)
				}
				if req.Header.Get("Authorization") != "" {
					t.Error("public discovery must not send credentials")
				}
				return &http.Response{
					StatusCode: tt.status,
					Status:     http.StatusText(tt.status),
					Body:       io.NopCloser(strings.NewReader(tt.body)),
				}, nil
			})}
			t.Cleanup(func() { http.DefaultClient = oldClient })

			var output bytes.Buffer
			cmd := NewCLI()
			cmd.SetOut(&output)
			cmd.SetErr(io.Discard)
			cmd.SetArgs([]string{"ls", "--cloud"})
			err := cmd.ExecuteContext(t.Context())
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			want := "NAME ID SIZE MODIFIED alpha:8b:cloud 1234567890ab - Never beta:cloud short - Never"
			if got := strings.Join(strings.Fields(output.String()), " "); got != want {
				t.Errorf("output = %q, want %q", got, want)
			}
		})
	}
}
