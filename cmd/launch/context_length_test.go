package launch

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func testLauncherClientWithStatus(t *testing.T, contextLength int) (*launcherClient, *int) {
	t.Helper()

	statusCalls := 0
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/api/status" {
			return &http.Response{
				StatusCode: http.StatusNotFound,
				Body:       io.NopCloser(strings.NewReader(`{"error":"not found"}`)),
				Header:     make(http.Header),
			}, nil
		}
		statusCalls++
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(fmt.Sprintf(`{"cloud":{"disabled":false,"source":"none"},"context_length":%d}`, contextLength))),
			Header:     make(http.Header),
		}, nil
	})}

	u, err := url.Parse("http://ollama.test")
	if err != nil {
		t.Fatal(err)
	}
	return &launcherClient{apiClient: api.NewClient(u, client)}, &statusCalls
}

func captureContextWarningStderr(t *testing.T, fn func()) string {
	t.Helper()

	old := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stderr = w
	defer func() {
		os.Stderr = old
		r.Close()
	}()

	fn()

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestFormatContextLength(t *testing.T) {
	tests := map[int]string{
		32 * 1024: "32k",
		64 * 1024: "64k",
		100_000:   "100k",
		100_001:   "100001",
	}
	for tokens, want := range tests {
		if got := formatContextLength(tokens); got != want {
			t.Fatalf("formatContextLength(%d) = %q, want %q", tokens, got, want)
		}
	}
}
