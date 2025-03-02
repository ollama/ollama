// Package registry provides an http.Handler for handling local Ollama API
// requests for performing tasks related to the ollama.com model registry and
// the local disk cache.
package registry

import (
	"cmp"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"

	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/client/ollama"
)

// Local is an http.Handler for handling local Ollama API requests for
// performing tasks related to the ollama.com model registry combined with the
// local disk cache.
//
// It is not concern of Local, or this package, to handle model creation, which
// proceeds any registry operations for models it produces.
//
// NOTE: The package built for dealing with model creation should use
// [DefaultCache] to access the blob store and not attempt to read or write
// directly to the blob disk cache.
type Local struct {
	Client *ollama.Registry // required
	Cache  *blob.DiskCache  // required
	Logger *slog.Logger     // required

	// Fallback, if set, is used to handle requests that are not handled by
	// this handler.
	Fallback http.Handler
}

// serverError is like ollama.Error, but with a Status field for the HTTP
// response code. We want to avoid adding that field to ollama.Error because it
// would always be 0 to clients (we don't want to leak the status code in
// errors), and so it would be confusing to have a field that is always 0.
type serverError struct {
	Status int `json:"-"`

	// TODO(bmizerany): Decide if we want to keep this and maybe
	// bring back later.
	Code string `json:"code"`

	Message string `json:"error"`
}

func (e serverError) Error() string {
	return e.Message
}

// Common API errors
var (
	errMethodNotAllowed = &serverError{405, "method_not_allowed", "method not allowed"}
	errNotFound         = &serverError{404, "not_found", "not found"}
	errInternalError    = &serverError{500, "internal_error", "internal server error"}
)

type statusCodeRecorder struct {
	_status int // use status() to get the status code
	http.ResponseWriter
}

func (r *statusCodeRecorder) WriteHeader(status int) {
	if r._status == 0 {
		r._status = status
	}
	r.ResponseWriter.WriteHeader(status)
}

var (
	_ http.ResponseWriter = (*statusCodeRecorder)(nil)
	_ http.CloseNotifier  = (*statusCodeRecorder)(nil)
	_ http.Flusher        = (*statusCodeRecorder)(nil)
)

// CloseNotify implements the http.CloseNotifier interface, for Gin. Remove with Gin.
//
// It panics if the underlying ResponseWriter is not a CloseNotifier.
func (r *statusCodeRecorder) CloseNotify() <-chan bool {
	return r.ResponseWriter.(http.CloseNotifier).CloseNotify()
}

// Flush implements the http.Flusher interface, for Gin. Remove with Gin.
//
// It panics if the underlying ResponseWriter is not a Flusher.
func (r *statusCodeRecorder) Flush() {
	r.ResponseWriter.(http.Flusher).Flush()
}

func (r *statusCodeRecorder) status() int {
	return cmp.Or(r._status, 200)
}

func (s *Local) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	rec := &statusCodeRecorder{ResponseWriter: w}
	s.serveHTTP(rec, r)
}

func (s *Local) serveHTTP(rec *statusCodeRecorder, r *http.Request) {
	var errattr slog.Attr
	proxied, err := func() (bool, error) {
		switch r.URL.Path {
		case "/api/delete":
			return false, s.handleDelete(rec, r)
		default:
			if s.Fallback != nil {
				s.Fallback.ServeHTTP(rec, r)
				return true, nil
			}
			return false, errNotFound
		}
	}()
	if err != nil {
		// We always log the error, so fill in the error log attribute
		errattr = slog.String("error", err.Error())

		var e *serverError
		switch {
		case errors.As(err, &e):
		case errors.Is(err, ollama.ErrNameInvalid):
			e = &serverError{400, "bad_request", err.Error()}
		default:
			e = errInternalError
		}

		data, err := json.Marshal(e)
		if err != nil {
			// unreachable
			panic(err)
		}
		rec.Header().Set("Content-Type", "application/json")
		rec.WriteHeader(e.Status)
		rec.Write(data)

		// fallthrough to log
	}

	if !proxied {
		// we're only responsible for logging if we handled the request
		var level slog.Level
		if rec.status() >= 500 {
			level = slog.LevelError
		} else if rec.status() >= 400 {
			level = slog.LevelWarn
		}

		s.Logger.LogAttrs(r.Context(), level, "http",
			errattr, // report first in line to make it easy to find

			// TODO(bmizerany): Write a test to ensure that we are logging
			// all of this correctly. That also goes for the level+error
			// logic above.
			slog.Int("status", rec.status()),
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.Int64("content-length", r.ContentLength),
			slog.String("remote", r.RemoteAddr),
			slog.String("proto", r.Proto),
			slog.String("query", r.URL.RawQuery),
		)
	}
}

type params struct {
	DeprecatedName string `json:"name"`  // Use [params.model]
	Model          string `json:"model"` // Use [params.model]

	// AllowNonTLS is a flag that indicates a client using HTTP
	// is doing so, deliberately.
	//
	// Deprecated: This field is ignored and only present for this
	// deprecation message. It should be removed in a future release.
	//
	// Users can just use http or https+insecure to show intent to
	// communicate they want to do insecure things, without awkward and
	// confusing flags such as this.
	AllowNonTLS bool `json:"insecure"`

	// ProgressStream is a flag that indicates the client is expecting a stream of
	// progress updates.
	ProgressStream bool `json:"stream"`
}

// model returns the model name for both old and new API requests.
func (p params) model() string {
	return cmp.Or(p.Model, p.DeprecatedName)
}

func (s *Local) handleDelete(_ http.ResponseWriter, r *http.Request) error {
	if r.Method != "DELETE" {
		return errMethodNotAllowed
	}
	p, err := decodeUserJSON[*params](r.Body)
	if err != nil {
		return err
	}
	ok, err := s.Client.Unlink(s.Cache, p.model())
	if err != nil {
		return err
	}
	if !ok {
		return &serverError{404, "not_found", "model not found"}
	}
	return nil
}

func decodeUserJSON[T any](r io.Reader) (T, error) {
	var v T
	err := json.NewDecoder(r).Decode(&v)
	if err == nil {
		return v, nil
	}
	var zero T

	// Not sure why, but I can't seem to be able to use:
	//
	//   errors.As(err, &json.UnmarshalTypeError{})
	//
	// This is working fine in stdlib, so I'm not sure what rules changed
	// and why this no longer works here. So, we do it the verbose way.
	var a *json.UnmarshalTypeError
	var b *json.SyntaxError
	if errors.As(err, &a) || errors.As(err, &b) {
		err = &serverError{Status: 400, Message: err.Error(), Code: "bad_request"}
	}
	if errors.Is(err, io.EOF) {
		err = &serverError{Status: 400, Message: "empty request body", Code: "bad_request"}
	}
	return zero, err
}
