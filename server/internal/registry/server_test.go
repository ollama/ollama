package registry

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/client/ollama"
	"github.com/ollama/ollama/server/internal/testutil"
)

type panicTransport struct{}

func (t *panicTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	panic("unexpected RoundTrip call")
}

var panicOnRoundTrip = &http.Client{Transport: &panicTransport{}}

// bytesResetter is an interface for types that can be reset and return a byte
// slice, only. This is to prevent inadvertent use of bytes.Buffer.Read/Write
// etc for the purpose of checking logs.
type bytesResetter interface {
	Bytes() []byte
	Reset()
}

func newTestServer(t *testing.T) *Local {
	t.Helper()
	dir := t.TempDir()
	err := os.CopyFS(dir, os.DirFS("testdata/models"))
	if err != nil {
		t.Fatal(err)
	}
	c, err := blob.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	rc := &ollama.Registry{
		HTTPClient: panicOnRoundTrip,
	}
	l := &Local{
		Cache:  c,
		Client: rc,
		Logger: testutil.Slogger(t),
	}
	return l
}

func (s *Local) send(t *testing.T, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequestWithContext(t.Context(), method, path, strings.NewReader(body))
	return s.sendRequest(t, req)
}

func (s *Local) sendRequest(t *testing.T, req *http.Request) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)
	return w
}

type invalidReader struct{}

func (r *invalidReader) Read(p []byte) (int, error) {
	return 0, os.ErrInvalid
}

// captureLogs is a helper to capture logs from the server. It returns a
// shallow copy of the server with a new logger and a bytesResetter for the
// logs.
func captureLogs(t *testing.T, s *Local) (*Local, bytesResetter) {
	t.Helper()
	log, logs := testutil.SlogBuffer()
	l := *s // shallow copy
	l.Logger = log
	return &l, logs
}

func TestServerDelete(t *testing.T) {
	check := testutil.Checker(t)

	s := newTestServer(t)

	_, err := s.Client.ResolveLocal(s.Cache, "smol")
	check(err)

	got := s.send(t, "DELETE", "/api/delete", `{"model": "smol"}`)
	if got.Code != 200 {
		t.Fatalf("Code = %d; want 200", got.Code)
	}

	_, err = s.Client.ResolveLocal(s.Cache, "smol")
	if err == nil {
		t.Fatal("expected smol to have been deleted")
	}

	got = s.send(t, "DELETE", "/api/delete", `!`)
	checkErrorResponse(t, got, 400, "bad_request", "invalid character '!' looking for beginning of value")

	got = s.send(t, "GET", "/api/delete", `{"model": "smol"}`)
	checkErrorResponse(t, got, 405, "method_not_allowed", "method not allowed")

	got = s.send(t, "DELETE", "/api/delete", ``)
	checkErrorResponse(t, got, 400, "bad_request", "empty request body")

	got = s.send(t, "DELETE", "/api/delete", `{"model": "://"}`)
	checkErrorResponse(t, got, 400, "bad_request", "invalid or missing name")

	got = s.send(t, "DELETE", "/unknown_path", `{}`) // valid body
	checkErrorResponse(t, got, 404, "not_found", "not found")

	s, logs := captureLogs(t, s)
	req := httptest.NewRequestWithContext(t.Context(), "DELETE", "/api/delete", &invalidReader{})
	got = s.sendRequest(t, req)
	checkErrorResponse(t, got, 500, "internal_error", "internal server error")
	ok, err := regexp.Match(`ERROR.*error="invalid argument"`, logs.Bytes())
	check(err)
	if !ok {
		t.Logf("logs:\n%s", logs)
		t.Fatalf("expected log to contain ERROR with invalid argument")
	}
}

func TestServerUnknownPath(t *testing.T) {
	s := newTestServer(t)
	got := s.send(t, "DELETE", "/api/unknown", `{}`)
	checkErrorResponse(t, got, 404, "not_found", "not found")
}

func checkErrorResponse(t *testing.T, got *httptest.ResponseRecorder, status int, code, msg string) {
	t.Helper()

	var printedBody bool
	errorf := func(format string, args ...any) {
		t.Helper()
		if !printedBody {
			t.Logf("BODY:\n%s", got.Body.String())
			printedBody = true
		}
		t.Errorf(format, args...)
	}

	if got.Code != status {
		errorf("Code = %d; want %d", got.Code, status)
	}

	// unmarshal the error as *ollama.Error (proving *serverError is an *ollama.Error)
	var e *ollama.Error
	if err := json.Unmarshal(got.Body.Bytes(), &e); err != nil {
		errorf("unmarshal error: %v", err)
		t.FailNow()
	}
	if e.Code != code {
		errorf("Code = %q; want %q", e.Code, code)
	}
	if !strings.Contains(e.Message, msg) {
		errorf("Message = %q; want to contain %q", e.Message, msg)
	}
}
