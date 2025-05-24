package ollama

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/testutil"
)

func ExampleRegistry_cancelOnFirstError() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	ctx = WithTrace(ctx, &Trace{
		Update: func(l *Layer, n int64, err error) {
			if err != nil {
				// Discontinue pulling layers if there is an
				// error instead of continuing to pull more
				// data.
				cancel()
			}
		},
	})

	var r Registry
	if err := r.Pull(ctx, "model"); err != nil {
		// panic for demo purposes
		panic(err)
	}
}

func TestManifestMarshalJSON(t *testing.T) {
	// All manifests should contain an "empty" config object.
	var m Manifest
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(data, []byte(`"config":{"digest":"sha256:`)) {
		t.Error("expected manifest to contain empty config")
		t.Fatalf("got:\n%s", string(data))
	}
}

var errRoundTrip = errors.New("forced roundtrip error")

type recordRoundTripper http.HandlerFunc

func (rr recordRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	w := httptest.NewRecorder()
	rr(w, req)
	if w.Code == 499 {
		return nil, errRoundTrip
	}
	resp := w.Result()
	// For some reason, Response.Request is not set by httptest.NewRecorder, so we
	// set it manually.
	resp.Request = req
	return w.Result(), nil
}

// newClient constructs a cache with predefined manifests for testing. The manifests are:
//
//	empty:         no data
//	zero:          no layers
//	single:        one layer with the contents "exists"
//	multiple:      two layers with the contents "exists" and "here"
//	notfound:      a layer that does not exist in the cache
//	null:          one null layer (e.g. [null])
//	sizemismatch:  one valid layer, and one with a size mismatch (file size is less than the reported size)
//	invalid:       a layer with invalid JSON data
//
// Tests that want to ensure the client does not communicate with the upstream
// registry should pass a nil handler, which will cause a panic if
// communication is attempted.
//
// To simulate a network error, pass a handler that returns a 499 status code.
func newClient(t *testing.T, upstreamRegistry http.HandlerFunc) (*Registry, *blob.DiskCache) {
	t.Helper()

	c, err := blob.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}

	mklayer := func(data string) *Layer {
		return &Layer{
			Digest: importBytes(t, c, data),
			Size:   int64(len(data)),
		}
	}

	r := &Registry{
		Cache: c,
		HTTPClient: &http.Client{
			Transport: recordRoundTripper(upstreamRegistry),
		},
	}

	link := func(name string, manifest string) {
		n, err := r.parseName(name)
		if err != nil {
			panic(err)
		}
		d, err := c.Import(bytes.NewReader([]byte(manifest)), int64(len(manifest)))
		if err != nil {
			panic(err)
		}
		if err := c.Link(n.String(), d); err != nil {
			panic(err)
		}
	}

	commit := func(name string, layers ...*Layer) {
		t.Helper()
		data, err := json.Marshal(&Manifest{Layers: layers})
		if err != nil {
			t.Fatal(err)
		}
		link(name, string(data))
	}

	link("empty", "")
	commit("zero")
	commit("single", mklayer("exists"))
	commit("multiple", mklayer("exists"), mklayer("present"))
	commit("notfound", &Layer{Digest: blob.DigestFromBytes("notfound"), Size: int64(len("notfound"))})
	commit("null", nil)
	commit("sizemismatch", mklayer("exists"), &Layer{Digest: blob.DigestFromBytes("present"), Size: 499})
	link("invalid", "!!!!!")

	return r, c
}

func okHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func checkErrCode(t *testing.T, err error, status int, code string) {
	t.Helper()
	var e *Error
	if !errors.As(err, &e) || e.status != status || e.Code != code {
		t.Errorf("err = %v; want %v %v", err, status, code)
	}
}

func importBytes(t *testing.T, c *blob.DiskCache, data string) blob.Digest {
	d, err := c.Import(strings.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func withTraceUnexpected(ctx context.Context) (context.Context, *Trace) {
	t := &Trace{Update: func(*Layer, int64, error) { panic("unexpected") }}
	return WithTrace(ctx, t), t
}

func TestPushZero(t *testing.T) {
	rc, _ := newClient(t, okHandler)
	err := rc.Push(t.Context(), "empty", nil)
	if !errors.Is(err, ErrManifestInvalid) {
		t.Errorf("err = %v; want %v", err, ErrManifestInvalid)
	}
}

func TestPushSingle(t *testing.T) {
	rc, _ := newClient(t, okHandler)
	err := rc.Push(t.Context(), "single", nil)
	testutil.Check(t, err)
}

func TestPushMultiple(t *testing.T) {
	rc, _ := newClient(t, okHandler)
	err := rc.Push(t.Context(), "multiple", nil)
	testutil.Check(t, err)
}

func TestPushNotFound(t *testing.T) {
	rc, _ := newClient(t, func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("unexpected request: %v", r)
	})
	err := rc.Push(t.Context(), "notfound", nil)
	if !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("err = %v; want %v", err, fs.ErrNotExist)
	}
}

func TestPushNullLayer(t *testing.T) {
	rc, _ := newClient(t, nil)
	err := rc.Push(t.Context(), "null", nil)
	if err == nil || !strings.Contains(err.Error(), "invalid manifest") {
		t.Errorf("err = %v; want invalid manifest", err)
	}
}

func TestPushSizeMismatch(t *testing.T) {
	rc, _ := newClient(t, nil)
	ctx, _ := withTraceUnexpected(t.Context())
	got := rc.Push(ctx, "sizemismatch", nil)
	if got == nil || !strings.Contains(got.Error(), "size mismatch") {
		t.Errorf("err = %v; want size mismatch", got)
	}
}

func TestPushInvalid(t *testing.T) {
	rc, _ := newClient(t, nil)
	err := rc.Push(t.Context(), "invalid", nil)
	if err == nil || !strings.Contains(err.Error(), "invalid manifest") {
		t.Errorf("err = %v; want invalid manifest", err)
	}
}

func TestPushExistsAtRemote(t *testing.T) {
	var pushed bool
	rc, _ := newClient(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/uploads/") {
			if !pushed {
				// First push. Return an uploadURL.
				pushed = true
				w.Header().Set("Location", "http://blob.store/blobs/123")
				return
			}
			w.WriteHeader(http.StatusAccepted)
			return
		}

		io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusOK)
	})

	rc.MaxStreams = 1 // prevent concurrent uploads

	var errs []error
	ctx := WithTrace(t.Context(), &Trace{
		Update: func(_ *Layer, n int64, err error) {
			// uploading one at a time so no need to lock
			errs = append(errs, err)
		},
	})

	check := testutil.Checker(t)

	err := rc.Push(ctx, "single", nil)
	check(err)

	if !errors.Is(errors.Join(errs...), nil) {
		t.Errorf("errs = %v; want %v", errs, []error{ErrCached})
	}

	err = rc.Push(ctx, "single", nil)
	check(err)
}

func TestPushRemoteError(t *testing.T) {
	rc, _ := newClient(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/blobs/") {
			w.WriteHeader(500)
			io.WriteString(w, `{"errors":[{"code":"blob_error"}]}`)
			return
		}
	})
	got := rc.Push(t.Context(), "single", nil)
	checkErrCode(t, got, 500, "blob_error")
}

func TestPushLocationError(t *testing.T) {
	rc, _ := newClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", ":///x")
		w.WriteHeader(http.StatusAccepted)
	})
	got := rc.Push(t.Context(), "single", nil)
	wantContains := "invalid upload URL"
	if got == nil || !strings.Contains(got.Error(), wantContains) {
		t.Errorf("err = %v; want to contain %v", got, wantContains)
	}
}

func TestPushUploadRoundtripError(t *testing.T) {
	rc, _ := newClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Host == "blob.store" {
			w.WriteHeader(499) // force RoundTrip error on upload
			return
		}
		w.Header().Set("Location", "http://blob.store/blobs/123")
	})
	got := rc.Push(t.Context(), "single", nil)
	if !errors.Is(got, errRoundTrip) {
		t.Errorf("got = %v; want %v", got, errRoundTrip)
	}
}

func TestPushUploadFileOpenError(t *testing.T) {
	rc, c := newClient(t, okHandler)
	ctx := WithTrace(t.Context(), &Trace{
		Update: func(l *Layer, _ int64, err error) {
			// Remove the file just before it is opened for upload,
			// but after the initial Stat that happens before the
			// upload starts
			os.Remove(c.GetFile(l.Digest))
		},
	})
	got := rc.Push(ctx, "single", nil)
	if !errors.Is(got, fs.ErrNotExist) {
		t.Errorf("got = %v; want fs.ErrNotExist", got)
	}
}

func TestPushCommitRoundtripError(t *testing.T) {
	rc, _ := newClient(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/blobs/") {
			panic("unexpected")
		}
		w.WriteHeader(499) // force RoundTrip error
	})
	err := rc.Push(t.Context(), "zero", nil)
	if !errors.Is(err, errRoundTrip) {
		t.Errorf("err = %v; want %v", err, errRoundTrip)
	}
}

func TestRegistryPullInvalidName(t *testing.T) {
	rc, _ := newRegistryClient(t, nil)
	err := rc.Pull(t.Context(), "://")
	if !errors.Is(err, ErrNameInvalid) {
		t.Errorf("err = %v; want %v", err, ErrNameInvalid)
	}
}

func TestRegistryPullInvalidManifest(t *testing.T) {
	cases := []string{
		"",
		"null",
		"!!!",
		`{"layers":[]}`,
	}

	for _, resp := range cases {
		rc, _ := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
			io.WriteString(w, resp)
		})
		err := rc.Pull(t.Context(), "http://example.com/a/b")
		if !errors.Is(err, ErrManifestInvalid) {
			t.Errorf("err = %v; want invalid manifest", err)
		}
	}
}

func TestRegistryResolveByDigest(t *testing.T) {
	check := testutil.Checker(t)

	exists := blob.DigestFromBytes("exists")
	rc, _ := newClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v2/alice/palace/blobs/"+exists.String() {
			w.WriteHeader(499) // should not hit manifest endpoint
		}
		fmt.Fprintf(w, `{"layers":[{"digest":%q,"size":5}]}`, exists)
	})

	_, err := rc.Resolve(t.Context(), "alice/palace@"+exists.String())
	check(err)
}

func TestInsecureSkipVerify(t *testing.T) {
	exists := blob.DigestFromBytes("exists")

	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"layers":[{"digest":%q,"size":5}]}`, exists)
	}))
	defer s.Close()

	const name = "library/insecure"

	var rc Registry
	url := fmt.Sprintf("https://%s/%s", s.Listener.Addr(), name)
	_, err := rc.Resolve(t.Context(), url)
	if err == nil || !strings.Contains(err.Error(), "failed to verify") {
		t.Errorf("err = %v; want cert verification failure", err)
	}

	url = fmt.Sprintf("https+insecure://%s/%s", s.Listener.Addr(), name)
	_, err = rc.Resolve(t.Context(), url)
	testutil.Check(t, err)
}

func TestErrorUnmarshal(t *testing.T) {
	cases := []struct {
		name    string
		data    string
		want    *Error
		wantErr bool
	}{
		{
			name:    "errors empty",
			data:    `{"errors":[]}`,
			wantErr: true,
		},
		{
			name:    "errors empty",
			data:    `{"errors":[]}`,
			wantErr: true,
		},
		{
			name: "errors single",
			data: `{"errors":[{"code":"blob_unknown"}]}`,
			want: &Error{Code: "blob_unknown", Message: ""},
		},
		{
			name: "errors multiple",
			data: `{"errors":[{"code":"blob_unknown"},{"code":"blob_error"}]}`,
			want: &Error{Code: "blob_unknown", Message: ""},
		},
		{
			name:    "error empty",
			data:    `{"error":""}`,
			wantErr: true,
		},
		{
			name:    "error very empty",
			data:    `{}`,
			wantErr: true,
		},
		{
			name: "error message",
			data: `{"error":"message", "code":"code"}`,
			want: &Error{Code: "code", Message: "message"},
		},
		{
			name:    "invalid value",
			data:    `{"error": 1}`,
			wantErr: true,
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			var got Error
			err := json.Unmarshal([]byte(tt.data), &got)
			if err != nil {
				if tt.wantErr {
					return
				}
				t.Errorf("Unmarshal() error = %v", err)
				// fallthrough and check got
			}
			if tt.want == nil {
				tt.want = &Error{}
			}
			if !reflect.DeepEqual(got, *tt.want) {
				t.Errorf("got = %v; want %v", got, *tt.want)
			}
		})
	}
}

// TestParseNameErrors tests that parseName returns errors messages with enough
// detail for users to debug naming issues they may encounter. Previous to this
// test, the error messages were not very helpful and each problem was reported
// as the same message.
//
// It is only for testing error messages, not that all invalids and valids are
// covered. Those are in other tests for names.Name and blob.Digest.
func TestParseNameExtendedErrors(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want string
	}{}

	var r Registry
	for _, tt := range cases {
		_, _, _, err := r.parseNameExtended(tt.name)
		if !errors.Is(err, tt.err) {
			t.Errorf("[%s]: err = %v; want %v", tt.name, err, tt.err)
		}
		if err != nil && !strings.Contains(err.Error(), tt.want) {
			t.Errorf("[%s]: err =\n\t%v\nwant\n\t%v", tt.name, err, tt.want)
		}
	}
}

func TestParseNameExtended(t *testing.T) {
	cases := []struct {
		in     string
		scheme string
		name   string
		digest string
		err    string
	}{
		{in: "http://m", scheme: "http", name: "m"},
		{in: "https+insecure://m", scheme: "https+insecure", name: "m"},
		{in: "http+insecure://m", err: "unsupported scheme"},

		{in: "http://m@sha256:1111111111111111111111111111111111111111111111111111111111111111", scheme: "http", name: "m", digest: "sha256:1111111111111111111111111111111111111111111111111111111111111111"},

		{in: "", err: "invalid or missing name"},
		{in: "m", scheme: "https", name: "m"},
		{in: "://", err: "invalid or missing name"},
		{in: "@sha256:deadbeef", err: "invalid digest"},
		{in: "@sha256:deadbeef@sha256:deadbeef", err: "invalid digest"},
	}
	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			var r Registry
			scheme, n, digest, err := r.parseNameExtended(tt.in)
			if err != nil {
				if tt.err == "" {
					t.Errorf("err = %v; want nil", err)
				} else if !strings.Contains(err.Error(), tt.err) {
					t.Errorf("err = %v; want %q", err, tt.err)
				}
			} else if tt.err != "" {
				t.Errorf("err = nil; want %q", tt.err)
			}
			if err == nil && !n.IsFullyQualified() {
				t.Errorf("name = %q; want fully qualified", n)
			}

			if scheme != tt.scheme {
				t.Errorf("scheme = %q; want %q", scheme, tt.scheme)
			}

			// smoke-test name is superset of tt.name
			if !strings.Contains(n.String(), tt.name) {
				t.Errorf("name = %q; want %q", n, tt.name)
			}

			tt.digest = cmp.Or(tt.digest, (&blob.Digest{}).String())
			if digest.String() != tt.digest {
				t.Errorf("digest = %q; want %q", digest, tt.digest)
			}
		})
	}
}

func TestUnlink(t *testing.T) {
	t.Run("found by name", func(t *testing.T) {
		check := testutil.Checker(t)

		rc, _ := newRegistryClient(t, nil)
		// make a blob and link it
		d := blob.DigestFromBytes("{}")
		err := blob.PutBytes(rc.Cache, d, "{}")
		check(err)
		err = rc.Cache.Link("registry.ollama.ai/library/single:latest", d)
		check(err)

		// confirm linked
		_, err = rc.ResolveLocal("single")
		check(err)

		// unlink
		_, err = rc.Unlink("single")
		check(err)

		// confirm unlinked
		_, err = rc.ResolveLocal("single")
		if !errors.Is(err, fs.ErrNotExist) {
			t.Errorf("err = %v; want fs.ErrNotExist", err)
		}
	})
	t.Run("not found by name", func(t *testing.T) {
		rc, _ := newRegistryClient(t, nil)
		ok, err := rc.Unlink("manifestNotFound")
		if err != nil {
			t.Fatal(err)
		}
		if ok {
			t.Error("expected not found")
		}
	})
}

// Many tests from here out, in this file are based on a single blob, "abc",
// with the checksum of its sha256 hash. The checksum is:
//
//	"abc" -> sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
//
// Using the literal value instead of a constant with fmt.Xprintf calls proved
// to be the most readable and maintainable approach. The sum is consistently
// used in the tests and unique so searches do not yield false positives.

func checkRequest(t *testing.T, req *http.Request, method, path string) {
	t.Helper()
	if got := req.URL.Path; got != path {
		t.Errorf("URL = %q, want %q", got, path)
	}
	if req.Method != method {
		t.Errorf("Method = %q, want %q", req.Method, method)
	}
}

func newRegistryClient(t *testing.T, upstream http.HandlerFunc) (*Registry, context.Context) {
	s := httptest.NewServer(upstream)
	t.Cleanup(s.Close)
	cache, err := blob.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}

	ctx := WithTrace(t.Context(), &Trace{
		Update: func(l *Layer, n int64, err error) {
			t.Log("trace:", l.Digest.Short(), n, err)
		},
	})

	rc := &Registry{
		Cache: cache,
		HTTPClient: &http.Client{Transport: &http.Transport{
			Dial: func(network, addr string) (net.Conn, error) {
				return net.Dial(network, s.Listener.Addr().String())
			},
		}},
	}
	return rc, ctx
}

func TestPullChunked(t *testing.T) {
	var steps atomic.Int64
	c, ctx := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch steps.Add(1) {
		case 1:
			checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
			io.WriteString(w, `{"layers":[{"size":3,"digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]}`)
		case 2:
			checkRequest(t, r, "GET", "/v2/library/abc/chunksums/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			w.Header().Set("Content-Location", "http://blob.store/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			fmt.Fprintf(w, "%s 0-1\n", blob.DigestFromBytes("ab"))
			fmt.Fprintf(w, "%s 2-2\n", blob.DigestFromBytes("c"))
		case 3, 4:
			checkRequest(t, r, "GET", "/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			switch rng := r.Header.Get("Range"); rng {
			case "bytes=0-1":
				io.WriteString(w, "ab")
			case "bytes=2-2":
				t.Logf("writing c")
				io.WriteString(w, "c")
			default:
				t.Errorf("unexpected range %q", rng)
			}
		default:
			t.Errorf("unexpected steps %d: %v", steps.Load(), r)
			http.Error(w, "unexpected steps", http.StatusInternalServerError)
		}
	})

	c.ChunkingThreshold = 1 // force chunking

	err := c.Pull(ctx, "http://o.com/library/abc")
	testutil.Check(t, err)

	_, err = c.Cache.Resolve("o.com/library/abc:latest")
	testutil.Check(t, err)

	if g := steps.Load(); g != 4 {
		t.Fatalf("got %d steps, want 4", g)
	}
}

func TestPullCached(t *testing.T) {
	c, ctx := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
		io.WriteString(w, `{"layers":[{"size":3,"digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]}`)
	})

	check := testutil.Checker(t)

	// Premeptively cache the blob
	d, err := blob.ParseDigest("sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
	check(err)
	err = blob.PutBytes(c.Cache, d, []byte("abc"))
	check(err)

	// Pull only the manifest, which should be enough to resolve the cached blob
	err = c.Pull(ctx, "http://o.com/library/abc")
	check(err)
}

func TestPullManifestError(t *testing.T) {
	c, ctx := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
		w.WriteHeader(http.StatusNotFound)
		io.WriteString(w, `{"errors":[{"code":"MANIFEST_UNKNOWN"}]}`)
	})

	err := c.Pull(ctx, "http://o.com/library/abc")
	if err == nil {
		t.Fatalf("expected error")
	}
	var got *Error
	if !errors.Is(err, ErrModelNotFound) {
		t.Fatalf("err = %v, want %v", got, ErrModelNotFound)
	}
}

func TestPullLayerError(t *testing.T) {
	c, ctx := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
		io.WriteString(w, `!`)
	})

	err := c.Pull(ctx, "http://o.com/library/abc")
	if err == nil {
		t.Fatalf("expected error")
	}
	var want *json.SyntaxError
	if !errors.As(err, &want) {
		t.Fatalf("err = %T, want %T", err, want)
	}
}

func TestPullLayerChecksumError(t *testing.T) {
	var step atomic.Int64
	c, _ := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch step.Add(1) {
		case 1:
			checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
			io.WriteString(w, `{"layers":[{"size":3,"digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]}`)
		case 2:
			checkRequest(t, r, "GET", "/v2/library/abc/chunksums/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			w.Header().Set("Content-Location", "http://blob.store/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			fmt.Fprintf(w, "%s 0-1\n", blob.DigestFromBytes("ab"))
			fmt.Fprintf(w, "%s 2-2\n", blob.DigestFromBytes("c"))
		case 3:
			w.WriteHeader(http.StatusNotFound)
			io.WriteString(w, `{"errors":[{"code":"BLOB_UNKNOWN"}]}`)
		case 4:
			io.WriteString(w, "c")
		default:
			t.Errorf("unexpected steps %d: %v", step.Load(), r)
			http.Error(w, "unexpected steps", http.StatusInternalServerError)
		}
	})

	c.MaxStreams = 1
	c.ChunkingThreshold = 1 // force chunking

	var written atomic.Int64
	ctx := WithTrace(t.Context(), &Trace{
		Update: func(l *Layer, n int64, err error) {
			t.Log("trace:", l.Digest.Short(), n, err)
			written.Add(n)
		},
	})

	err := c.Pull(ctx, "http://o.com/library/abc")
	var got *Error
	if !errors.As(err, &got) || got.Code != "BLOB_UNKNOWN" {
		t.Fatalf("err = %v, want %v", err, got)
	}

	if g := written.Load(); g != 1 {
		t.Fatalf("wrote %d bytes, want 1", g)
	}
}

func TestPullChunksumStreamError(t *testing.T) {
	var step atomic.Int64
	c, ctx := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch step.Add(1) {
		case 1:
			checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
			io.WriteString(w, `{"layers":[{"size":3,"digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]}`)
		case 2:
			w.Header().Set("Content-Location", "http://blob.store/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

			// Write one valid chunksum and one invalid chunksum
			fmt.Fprintf(w, "%s 0-1\n", blob.DigestFromBytes("ab")) // valid
			fmt.Fprint(w, "sha256:!")                              // invalid
		case 3:
			io.WriteString(w, "ab")
		default:
			t.Errorf("unexpected steps %d: %v", step.Load(), r)
			http.Error(w, "unexpected steps", http.StatusInternalServerError)
		}
	})

	c.ChunkingThreshold = 1 // force chunking

	got := c.Pull(ctx, "http://o.com/library/abc")
	if !errors.Is(got, ErrIncomplete) {
		t.Fatalf("err = %v, want %v", got, ErrIncomplete)
	}
}

type flushAfterWriter struct {
	w io.Writer
}

func (f *flushAfterWriter) Write(p []byte) (n int, err error) {
	n, err = f.w.Write(p)
	f.w.(http.Flusher).Flush() // panic if not a flusher
	return
}

func TestPullChunksumStreaming(t *testing.T) {
	csr, csw := io.Pipe()
	defer csw.Close()

	var step atomic.Int64
	c, _ := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch step.Add(1) {
		case 1:
			checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
			io.WriteString(w, `{"layers":[{"size":3,"digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]}`)
		case 2:
			w.Header().Set("Content-Location", "http://blob.store/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			fw := &flushAfterWriter{w} // ensure client gets data as it arrives by aggressively flushing
			_, err := io.Copy(fw, csr)
			if err != nil {
				t.Errorf("copy: %v", err)
			}
		case 3:
			io.WriteString(w, "ab")
		case 4:
			io.WriteString(w, "c")
		default:
			t.Errorf("unexpected steps %d: %v", step.Load(), r)
			http.Error(w, "unexpected steps", http.StatusInternalServerError)
		}
	})

	c.ChunkingThreshold = 1 // force chunking

	update := make(chan int64, 1)
	ctx := WithTrace(t.Context(), &Trace{
		Update: func(l *Layer, n int64, err error) {
			t.Log("trace:", l.Digest.Short(), n, err)
			if n > 0 {
				update <- n
			}
		},
	})

	errc := make(chan error, 1)
	go func() {
		errc <- c.Pull(ctx, "http://o.com/library/abc")
	}()

	// Send first chunksum and ensure it kicks off work immediately
	fmt.Fprintf(csw, "%s 0-1\n", blob.DigestFromBytes("ab"))
	if g := <-update; g != 2 {
		t.Fatalf("got %d, want 2", g)
	}

	// now send the second chunksum and ensure it kicks off work immediately
	fmt.Fprintf(csw, "%s 2-2\n", blob.DigestFromBytes("c"))
	if g := <-update; g != 3 {
		t.Fatalf("got %d, want 3", g)
	}
	csw.Close()
	testutil.Check(t, <-errc)
}

func TestPullChunksumsCached(t *testing.T) {
	var step atomic.Int64
	c, _ := newRegistryClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch step.Add(1) {
		case 1:
			checkRequest(t, r, "GET", "/v2/library/abc/manifests/latest")
			io.WriteString(w, `{"layers":[{"size":3,"digest":"sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]}`)
		case 2:
			w.Header().Set("Content-Location", "http://blob.store/v2/library/abc/blobs/sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
			fmt.Fprintf(w, "%s 0-1\n", blob.DigestFromBytes("ab"))
			fmt.Fprintf(w, "%s 2-2\n", blob.DigestFromBytes("c"))
		case 3, 4:
			switch rng := r.Header.Get("Range"); rng {
			case "bytes=0-1":
				io.WriteString(w, "ab")
			case "bytes=2-2":
				io.WriteString(w, "c")
			default:
				t.Errorf("unexpected range %q", rng)
			}
		default:
			t.Errorf("unexpected steps %d: %v", step.Load(), r)
			http.Error(w, "unexpected steps", http.StatusInternalServerError)
		}
	})

	c.MaxStreams = 1        // force serial processing of chunksums
	c.ChunkingThreshold = 1 // force chunking

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	// Cancel the pull after the first chunksum is processed, but before
	// the second chunksum is processed (which is waiting because
	// MaxStreams=1). This should cause the second chunksum to error out
	// leaving the blob incomplete.
	ctx = WithTrace(ctx, &Trace{
		Update: func(l *Layer, n int64, err error) {
			if n > 0 {
				cancel()
			}
		},
	})
	err := c.Pull(ctx, "http://o.com/library/abc")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want %v", err, context.Canceled)
	}

	_, err = c.Cache.Resolve("o.com/library/abc:latest")
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("err = %v, want nil", err)
	}

	// Reset state and pull again to ensure the blob chunks that should
	// have been cached are, and the remaining chunk was downloaded, making
	// the blob complete.
	step.Store(0)
	var written atomic.Int64
	var cached atomic.Int64
	ctx = WithTrace(t.Context(), &Trace{
		Update: func(l *Layer, n int64, err error) {
			t.Log("trace:", l.Digest.Short(), n, err)
			if errors.Is(err, ErrCached) {
				cached.Add(n)
			}
			written.Add(n)
		},
	})

	check := testutil.Checker(t)

	err = c.Pull(ctx, "http://o.com/library/abc")
	check(err)

	_, err = c.Cache.Resolve("o.com/library/abc:latest")
	check(err)

	if g := written.Load(); g != 5 {
		t.Fatalf("wrote %d bytes, want 3", g)
	}
	if g := cached.Load(); g != 2 { // "ab" should have been cached
		t.Fatalf("cached %d bytes, want 5", g)
	}
}
