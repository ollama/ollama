package registry

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strings"
	"sync"
	"testing"

	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/client/ollama"
	"github.com/ollama/ollama/server/internal/testutil"
	"golang.org/x/tools/txtar"

	_ "embed"
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

func newTestServer(t *testing.T, upstreamRegistry http.HandlerFunc) *Local {
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

	client := panicOnRoundTrip
	if upstreamRegistry != nil {
		s := httptest.NewTLSServer(upstreamRegistry)
		t.Cleanup(s.Close)
		tr := s.Client().Transport.(*http.Transport).Clone()
		tr.DialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "tcp", s.Listener.Addr().String())
		}
		client = &http.Client{Transport: tr}
	}

	rc := &ollama.Registry{
		Cache:      c,
		HTTPClient: client,
		Mask:       "example.com/library/_:latest",
	}

	l := &Local{
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

	s := newTestServer(t, nil)

	_, err := s.Client.ResolveLocal("smol")
	check(err)

	got := s.send(t, "DELETE", "/api/delete", `{"model": "smol"}`)
	if got.Code != 200 {
		t.Fatalf("Code = %d; want 200", got.Code)
	}

	_, err = s.Client.ResolveLocal("smol")
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

//go:embed testdata/registry.txt
var registryTXT []byte

var registryFS = sync.OnceValue(func() fs.FS {
	// Txtar gets hung up on \r\n line endings, so we need to convert them
	// to \n when parsing the txtar on Windows.
	data := bytes.ReplaceAll(registryTXT, []byte("\r\n"), []byte("\n"))
	a := txtar.Parse(data)
	fmt.Printf("%q\n", a.Comment)
	fsys, err := txtar.FS(a)
	if err != nil {
		panic(err)
	}
	return fsys
})

func TestServerPull(t *testing.T) {
	modelsHandler := http.FileServerFS(registryFS())
	s := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v2/library/BOOM/manifests/latest":
			w.WriteHeader(999)
			io.WriteString(w, `{"error": "boom"}`)
		case "/v2/library/unknown/manifests/latest":
			w.WriteHeader(404)
			io.WriteString(w, `{"errors": [{"code": "MANIFEST_UNKNOWN", "message": "manifest unknown"}]}`)
		default:
			t.Logf("serving file: %s", r.URL.Path)
			modelsHandler.ServeHTTP(w, r)
		}
	})

	checkResponse := func(got *httptest.ResponseRecorder, wantlines string) {
		t.Helper()

		if got.Code != 200 {
			t.Fatalf("Code = %d; want 200", got.Code)
		}
		gotlines := got.Body.String()
		t.Logf("got:\n%s", gotlines)
		for want := range strings.Lines(wantlines) {
			want = strings.TrimSpace(want)
			want, unwanted := strings.CutPrefix(want, "!")
			want = strings.TrimSpace(want)
			if !unwanted && !strings.Contains(gotlines, want) {
				t.Fatalf("! missing %q in body", want)
			}
			if unwanted && strings.Contains(gotlines, want) {
				t.Fatalf("! unexpected %q in body", want)
			}
		}
	}

	got := s.send(t, "POST", "/api/pull", `{"model": "BOOM"}`)
	checkResponse(got, `
		{"status":"pulling manifest"}
		{"status":"error: request error https://example.com/v2/library/BOOM/manifests/latest: registry responded with status 999: boom"}
	`)

	got = s.send(t, "POST", "/api/pull", `{"model": "smol"}`)
	checkResponse(got, `
		{"status":"pulling manifest"}
		{"status":"pulling","digest":"sha256:68e0ec597aee59d35f8dc44942d7b17d471ade10d3aca07a5bb7177713950312","total":5}
		{"status":"pulling","digest":"sha256:ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356","total":3}
		{"status":"pulling","digest":"sha256:68e0ec597aee59d35f8dc44942d7b17d471ade10d3aca07a5bb7177713950312","total":5,"completed":5}
		{"status":"pulling","digest":"sha256:ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356","total":3,"completed":3}
		{"status":"verifying layers"}
		{"status":"writing manifest"}
		{"status":"success"}
	`)

	got = s.send(t, "POST", "/api/pull", `{"model": "unknown"}`)
	checkResponse(got, `
		{"status":"pulling manifest"}
		{"status":"error: model \"unknown\" not found"}
	`)

	got = s.send(t, "DELETE", "/api/pull", `{"model": "smol"}`)
	checkErrorResponse(t, got, 405, "method_not_allowed", "method not allowed")

	got = s.send(t, "POST", "/api/pull", `!`)
	checkErrorResponse(t, got, 400, "bad_request", "invalid character '!' looking for beginning of value")

	got = s.send(t, "POST", "/api/pull", ``)
	checkErrorResponse(t, got, 400, "bad_request", "empty request body")

	got = s.send(t, "POST", "/api/pull", `{"model": "://"}`)
	checkResponse(got, `
		{"status":"pulling manifest"}
		{"status":"error: invalid or missing name: \"\""}

		!verifying
		!writing
		!success
	`)
}

func TestServerUnknownPath(t *testing.T) {
	s := newTestServer(t, nil)
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
