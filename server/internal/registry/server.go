// Package registry implements an http.Handler for handling local Ollama API
// model management requests. See [Local] for details.
package registry

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/client/ollama"
	"github.com/ollama/ollama/server/internal/internal/backoff"
)

// Local implements an http.Handler for handling local Ollama API model
// management requests, such as pushing, pulling, and deleting models.
//
// It can be arranged for all unknown requests to be passed through to a
// fallback handler, if one is provided.
type Local struct {
	Client *ollama.Registry // required
	Logger *slog.Logger     // required

	// Fallback, if set, is used to handle requests that are not handled by
	// this handler.
	Fallback http.Handler

	// Prune, if set, is called to prune the local disk cache after a model
	// is deleted.
	Prune func() error // optional
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
	errModelNotFound    = &serverError{404, "not_found", "model not found"}
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
		case "/api/pull":
			return false, s.handlePull(rec, r)
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
	// DeprecatedName is the name of the model to push, pull, or delete,
	// but is deprecated. New clients should use [Model] instead.
	//
	// Use [model()] to get the model name for both old and new API requests.
	DeprecatedName string `json:"name"`

	// Model is the name of the model to push, pull, or delete.
	//
	// Use [model()] to get the model name for both old and new API requests.
	Model string `json:"model"`

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

	// Stream, if true, will make the server send progress updates in a
	// streaming of JSON objects. If false, the server will send a single
	// JSON object with the final status as "success", or an error object
	// if an error occurred.
	//
	// Unfortunately, this API was designed to be a bit awkward. Stream is
	// defined to default to true if not present, so we need a way to check
	// if the client decisively set it to false. So, we use a pointer to a
	// bool. Gross.
	//
	// Use [stream()] to get the correct value for this field.
	Stream *bool `json:"stream"`
}

// model returns the model name for both old and new API requests.
func (p params) model() string {
	return cmp.Or(p.Model, p.DeprecatedName)
}

func (p params) stream() bool {
	if p.Stream == nil {
		return true
	}
	return *p.Stream
}

func (s *Local) handleDelete(_ http.ResponseWriter, r *http.Request) error {
	if r.Method != "DELETE" {
		return errMethodNotAllowed
	}
	p, err := decodeUserJSON[*params](r.Body)
	if err != nil {
		return err
	}
	ok, err := s.Client.Unlink(p.model())
	if err != nil {
		return err
	}
	if !ok {
		return errModelNotFound
	}
	if s.Prune != nil {
		return s.Prune()
	}
	return nil
}

type progressUpdateJSON struct {
	Error     string      `json:"error,omitempty,omitzero"`
	Status    string      `json:"status,omitempty,omitzero"`
	Digest    blob.Digest `json:"digest,omitempty,omitzero"`
	Total     int64       `json:"total,omitempty,omitzero"`
	Completed int64       `json:"completed,omitempty,omitzero"`
}

func (s *Local) handlePull(w http.ResponseWriter, r *http.Request) error {
	if r.Method != "POST" {
		return errMethodNotAllowed
	}

	p, err := decodeUserJSON[*params](r.Body)
	if err != nil {
		return err
	}

	enc := json.NewEncoder(w)
	if !p.stream() {
		if err := s.Client.Pull(r.Context(), p.model()); err != nil {
			if errors.Is(err, ollama.ErrModelNotFound) {
				return errModelNotFound
			}
			return err
		}
		enc.Encode(progressUpdateJSON{Status: "success"})
		return nil
	}

	var mu sync.Mutex
	var progress []progressUpdateJSON
	flushProgress := func() {
		mu.Lock()
		progress := slices.Clone(progress) // make a copy and release lock before encoding to the wire
		mu.Unlock()
		for _, p := range progress {
			enc.Encode(p)
		}
		fl, _ := w.(http.Flusher)
		if fl != nil {
			fl.Flush()
		}
	}

	t := time.NewTicker(1<<63 - 1) // "unstarted" timer
	start := sync.OnceFunc(func() {
		flushProgress() // flush initial state
		t.Reset(100 * time.Millisecond)
	})
	ctx := ollama.WithTrace(r.Context(), &ollama.Trace{
		Update: func(l *ollama.Layer, n int64, err error) {
			if err != nil && !errors.Is(err, ollama.ErrCached) {
				s.Logger.Error("pulling", "model", p.model(), "error", err)
				return
			}

			func() {
				mu.Lock()
				defer mu.Unlock()
				for i, p := range progress {
					if p.Digest == l.Digest {
						progress[i].Completed = n
						return
					}
				}
				progress = append(progress, progressUpdateJSON{
					Digest: l.Digest,
					Total:  l.Size,
				})
			}()

			// Block flushing progress updates until every
			// layer is accounted for. Clients depend on a
			// complete model size to calculate progress
			// correctly; if they use an incomplete total,
			// progress indicators would erratically jump
			// as new layers are registered.
			start()
		},
	})

	done := make(chan error, 1)
	go func() (err error) {
		defer func() { done <- err }()
		for _, err := range backoff.Loop(ctx, 3*time.Second) {
			if err != nil {
				return err
			}
			err := s.Client.Pull(ctx, p.model())
			if canRetry(err) {
				continue
			}
			return err
		}
		return nil
	}()

	enc.Encode(progressUpdateJSON{Status: "pulling manifest"})
	for {
		select {
		case <-t.C:
			flushProgress()
		case err := <-done:
			flushProgress()
			if err != nil {
				if errors.Is(err, ollama.ErrModelNotFound) {
					return &serverError{
						Status:  404,
						Code:    "not_found",
						Message: fmt.Sprintf("model %q not found", p.model()),
					}
				} else {
					return err
				}
			}

			// Emulate old client pull progress (for now):
			enc.Encode(progressUpdateJSON{Status: "verifying sha256 digest"})
			enc.Encode(progressUpdateJSON{Status: "writing manifest"})
			enc.Encode(progressUpdateJSON{Status: "success"})
			return nil
		}
	}
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

func canRetry(err error) bool {
	if err == nil {
		return false
	}
	var oe *ollama.Error
	if errors.As(err, &oe) {
		return oe.Temporary()
	}
	s := err.Error()
	return cmp.Or(
		errors.Is(err, context.DeadlineExceeded),
		strings.Contains(s, "unreachable"),
		strings.Contains(s, "no route to host"),
		strings.Contains(s, "connection reset by peer"),
	)
}
