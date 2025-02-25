package ollama

import (
	"context"
)

// Trace is a set of functions that are called to report progress during blob
// downloads and uploads.
type Trace struct {
	// Update is called during [Registry.Push] and [Registry.Pull] to
	// report the progress of blob uploads and downloads.
	//
	// It is called once at the beginning of the download with a zero n and
	// then once per read operation with the number of bytes read so far,
	// and an error if any.
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
