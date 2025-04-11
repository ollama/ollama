package ollama

import (
	"context"
)

// Trace reports progress during model transfers.
// Attach using [WithTrace] for use with [Registry.Push] and [Registry.Pull].
type Trace struct {
	// Update reports transfer progress with different states:
	// When n=0 and err=nil: transfer started
	// When err=[ErrCached]: layer already exists, skipped
	// When err!=nil: transfer failed
	// When n=l.Size and err=nil: transfer completed
	// Otherwise: n bytes transferred so far
	//
	// Must be safe for concurrent use and non-blocking.
	Update func(_ *Layer, n int64, _ error)
}

func (t *Trace) update(l *Layer, n int64, err error) {
	if t.Update != nil {
		t.Update(l, n, err)
	}
}

type traceKey struct{}

// WithTrace adds a trace to the context for transfer progress reporting.
func WithTrace(ctx context.Context, t *Trace) context.Context {
	old := traceFromContext(ctx)
	if old == t {
		// No change, return the original context. This also prevents
		// infinite recursion below, if the caller passes the same
		// Trace.
		return ctx
	}

	// Create a new Trace that wraps the old one, if any. If we used the
	// same pointer t, we end up with a recursive structure.
	composed := &Trace{
		Update: func(l *Layer, n int64, err error) {
			if old != nil {
				old.update(l, n, err)
			}
			t.update(l, n, err)
		},
	}
	return context.WithValue(ctx, traceKey{}, composed)
}

var emptyTrace = &Trace{}

// traceFromContext extracts the Trace from ctx or returns an empty non-nil Trace.
func traceFromContext(ctx context.Context) *Trace {
	t, _ := ctx.Value(traceKey{}).(*Trace)
	if t == nil {
		return emptyTrace
	}
	return t
}
