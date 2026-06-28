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
	ctx := ollama.WithTrace(t.Context(), &ollama.Trace{
		Update: func(l *ollama.Layer, n int64, err error) {
			t.Logf("update: %s %d %v", l.Digest, n, err)
		},
	})
	req := httptest.NewRequestWithContext(ctx, method, path, strings.NewReader(body))
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

func newLoggedRequest(t *testing.T, method, path string, body io.Reader, contentLength int64) *http.Request {
	t.Helper()
	ctx := ollama.WithTrace(t.Context(), &ollama.Trace{
		Update: func(*ollama.Layer, int64, error) {},
	})
	req := httptest.NewRequestWithContext(ctx, method, path, body)
	req.RemoteAddr = "203.0.113.10:4567"
	req.Proto = "HTTP/2.0"
	req.ProtoMajor = 2
	req.ProtoMinor = 0
	req.ContentLength = contentLength
	return req
}

func requireLogContains(t *testing.T, logs bytesResetter, parts ...string) string {
	t.Helper()
	got := string(logs.Bytes())
	for _, part := range parts {
		if !strings.Contains(got, part) {
			t.Fatalf("missing %q in logs:\n%s", part, got)
		}
	}
	return got
}

func requireNoLogs(t *testing.T, logs bytesResetter) {
	t.Helper()
	if got := strings.TrimSpace(string(logs.Bytes())); got != "" {
		t.Fatalf("expected no logs, got:\n%s", got)
	}
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

func TestServerLogging(t *testing.T) {
	t.Run("info handled request", func(t *testing.T) {
		s := newTestServer(t, nil)
		s, logs := captureLogs(t, s)

		body := `{"model":"smol"}`
		req := newLoggedRequest(t, "DELETE", "/api/delete?trace=info", strings.NewReader(body), int64(len(body)))
		got := s.sendRequest(t, req)
		if got.Code != 200 {
			t.Fatalf("Code = %d; want 200", got.Code)
		}

		logLine := requireLogContains(t, logs,
			"level=INFO",
			"msg=http",
			"status=200",
			"method=DELETE",
			"path=/api/delete",
			fmt.Sprintf("content-length=%d", len(body)),
			"remote=",
			"203.0.113.10:4567",
			"proto=HTTP/2.0",
			"query=",
			"trace=info",
		)
		if strings.Contains(logLine, `error="`) {
			t.Fatalf("expected info log without error attr, got:\n%s", logLine)
		}
	})

	t.Run("warn handled request", func(t *testing.T) {
		s := newTestServer(t, nil)
		s, logs := captureLogs(t, s)

		body := `{"model":"smol"}`
		req := newLoggedRequest(t, "GET", "/api/delete?trace=warn", strings.NewReader(body), int64(len(body)))
		got := s.sendRequest(t, req)
		checkErrorResponse(t, got, 405, "method_not_allowed", "method not allowed")

		requireLogContains(t, logs,
			"level=WARN",
			"msg=http",
			"status=405",
			"method=GET",
			"path=/api/delete",
			fmt.Sprintf("content-length=%d", len(body)),
			"remote=",
			"203.0.113.10:4567",
			"proto=HTTP/2.0",
			"query=",
			"trace=warn",
			`error="method not allowed"`,
		)
	})

	t.Run("error handled request", func(t *testing.T) {
		s := newTestServer(t, nil)
		s, logs := captureLogs(t, s)

		req := newLoggedRequest(t, "DELETE", "/api/delete?trace=error", &invalidReader{}, 7)
		got := s.sendRequest(t, req)
		checkErrorResponse(t, got, 500, "internal_error", "internal server error")

		requireLogContains(t, logs,
			"level=ERROR",
			"msg=http",
			"status=500",
			"method=DELETE",
			"path=/api/delete",
			"content-length=7",
			"remote=",
			"203.0.113.10:4567",
			"proto=HTTP/2.0",
			"query=",
			"trace=error",
			`error="invalid argument"`,
		)
	})

	t.Run("proxied fallback request does not log", func(t *testing.T) {
		s := newTestServer(t, nil)
		s, logs := captureLogs(t, s)
		s.Fallback = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNoContent)
		})

		body := `{}`
		req := newLoggedRequest(t, "DELETE", "/api/unknown?trace=proxy", strings.NewReader(body), int64(len(body)))
		got := s.sendRequest(t, req)
		if got.Code != http.StatusNoContent {
			t.Fatalf("Code = %d; want %d", got.Code, http.StatusNoContent)
		}

		requireNoLogs(t, logs)
	})
}

//go:embed testdata/registry.txt
var registryTXT []byte

var registryFS = sync.OnceValue(func() fs.FS {
	// Txtar gets hung up on \r\n line endings, so we need to convert them
	// to \n when parsing the txtar on Windows.
	data := bytes.ReplaceAll(registryTXT, []byte("\r\n"), []byte("\n"))
	a := txtar.Parse(data)
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
			t.Logf("serving blob: %s", r.URL.Path)
			modelsHandler.ServeHTTP(w, r)
		}
	})

	checkResponse := func(got *httptest.ResponseRecorder, wantlines string) {
		t.Helper()
		if got.Code != 200 {
			t.Errorf("Code = %d; want 200", got.Code)
		}
		gotlines := got.Body.String()
		if strings.TrimSpace(gotlines) == "" {
			gotlines = "<empty>"
		}
		t.Logf("got:\n%s", gotlines)
		for want := range strings.Lines(wantlines) {
			want = strings.TrimSpace(want)
			want, unwanted := strings.CutPrefix(want, "!")
			want = strings.TrimSpace(want)
			if !unwanted && !strings.Contains(gotlines, want) {
				t.Errorf("\t! missing %q in body", want)
			}
			if unwanted && strings.Contains(gotlines, want) {
				t.Errorf("\t! unexpected %q in body", want)
			}
		}
	}

	got := s.send(t, "POST", "/api/pull", `{"model": "smol"}`)
	checkResponse(got, `
		{"status":"pulling manifest"}
		{"digest":"sha256:68e0ec597aee59d35f8dc44942d7b17d471ade10d3aca07a5bb7177713950312","total":5,"completed":5}
		{"status":"verifying sha256 digest"}
		{"status":"writing manifest"}
		{"status":"success"}
	`)

	got = s.send(t, "POST", "/api/pull", `{"model": "unknown"}`)
	checkResponse(got, `
		{"code":"not_found","error":"model \"unknown\" not found"}
	`)

	got = s.send(t, "DELETE", "/api/pull", `{"model": "smol"}`)
	checkErrorResponse(t, got, 405, "method_not_allowed", "method not allowed")

	got = s.send(t, "POST", "/api/pull", `!`)
	checkErrorResponse(t, got, 400, "bad_request", "invalid character '!' looking for beginning of value")

	got = s.send(t, "POST", "/api/pull", ``)
	checkErrorResponse(t, got, 400, "bad_request", "empty request body")

	got = s.send(t, "POST", "/api/pull", `{"model": "://"}`)
	checkResponse(got, `
		{"code":"bad_request","error":"invalid or missing name: \"\""}
	`)

	// Non-streaming pulls
	got = s.send(t, "POST", "/api/pull", `{"model": "://", "stream": false}`)
	checkErrorResponse(t, got, 400, "bad_request", "invalid or missing name")
	got = s.send(t, "POST", "/api/pull", `{"model": "smol", "stream": false}`)
	checkResponse(got, `
		{"status":"success"}
		!digest
		!total
		!completed
	`)
	got = s.send(t, "POST", "/api/pull", `{"model": "unknown", "stream": false}`)
	checkErrorResponse(t, got, 404, "not_found", "model not found")
}

func TestServerUnknownPath(t *testing.T) {
	s := newTestServer(t, nil)
	got := s.send(t, "DELETE", "/api/unknown", `{}`)
	checkErrorResponse(t, got, 404, "not_found", "not found")

	var fellback bool
	s.Fallback = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fellback = true
	})
	got = s.send(t, "DELETE", "/api/unknown", `{}`)
	if !fellback {
		t.Fatal("expected Fallback to be called")
	}
	if got.Code != 200 {
		t.Fatalf("Code = %d; want 200", got.Code)
	}
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
