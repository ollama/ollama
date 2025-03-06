package ollama

import (
	"context"
)

// Trace is a set of functions that are called to report progress during blob
// downloads and uploads.
//
// Use [WithTrace] to attach a Trace to a context for use with [Registry.Push]
// and [Registry.Pull].
type Trace struct {
	// Update is called during [Registry.Push] and [Registry.Pull] to
	// report the progress of blob uploads and downloads.
	//
	// The n argument is the number of bytes transferred so far, and err is
	// any error that has occurred. If n == 0, and err is nil, the download
	// or upload has just started. If err is [ErrCached], the download or
	// upload has been skipped because the blob is already present in the
	// local cache or remote registry, respectively. Otherwise, if err is
	// non-nil, the download or upload has failed. When l.Size == n, and
	// err is nil, the download or upload has completed.
	//
	// A function assigned must be safe for concurrent use. The function is
	// called synchronously and so should not block or take long to run.
	Update func(_ *Layer, n int64, _ error)
}

func (t *Trace) update(l *Layer, n int64, err error) {
	if t.Update != nil {
		t.Update(l, n, err)
	}
}

type traceKey struct{}

// WithTrace returns a context derived from ctx that uses t to report trace
// events.
func WithTrace(ctx context.Context, t *Trace) context.Context {
	return context.WithValue(ctx, traceKey{}, t)
}

var emptyTrace = &Trace{}

// traceFromContext returns the Trace associated with ctx, or an empty Trace if
// none is found.
//
// It never returns nil.
func traceFromContext(ctx context.Context) *Trace {
	t, _ := ctx.Value(traceKey{}).(*Trace)
	if t == nil {
		return emptyTrace
	}
	return t
}
