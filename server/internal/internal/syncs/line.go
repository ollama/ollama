// Package syncs provides synchronization primitives.
package syncs

import (
	"cmp"
	"io"
	"sync"
)

var closedChan = func() chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}()

// Ticket represents a ticket in a sequence of tickets. The zero value is
// invalid. Use [Line.Take] to get a valid ticket.
//
// A Ticket is not safe for concurrent use.
type Ticket struct {
	ahead chan struct{} // ticket ahead of this one
	ch    chan struct{}
}

// Ready returns a channel that is closed when the ticket before this one is
// done.
//
// It is incorrect to wait on Ready after the ticket is done.
func (t *Ticket) Ready() chan struct{} {
	return cmp.Or(t.ahead, closedChan)
}

// Done signals that this ticket is done and that the next ticket in line can
// proceed.
//
// The first call to [Done] unblocks the ticket after it, if any. Subsequent
// calls are no-ops.
func (t *Ticket) Done() {
	if t.ch != nil {
		close(t.ch)
	}
	t.ch = nil
}

// Line is an ordered sequence of tickets waiting for their turn to proceed.
//
// To get a ticket use [Line.Take].
// To signal that a ticket is done use [Ticket.Done].
// To wait your turn use [Ticket.Ready].
//
// A Line is not safe for concurrent use.
type Line struct {
	last chan struct{} // last ticket in line
}

func (q *Line) Take() *Ticket {
	t := &Ticket{
		ahead: q.last,
		ch:    make(chan struct{}),
	}
	q.last = t.ch
	return t
}

// RelayReader implements an [io.WriterTo] that yields the passed
// writer to its [WriteTo] method each [io.WriteCloser] taken from [Take], in
// the order they are taken. Each [io.WriteCloser] blocks until the previous
// one is closed, or a call to [RelayReader.CloseWithError] is made.
//
// The zero value is invalid. Use [NewWriteToLine] to get a valid RelayReader.
//
// It is not safe for concurrent use.
type RelayReader struct {
	line Line
	t    *Ticket
	w    io.Writer
	n    int64

	mu       sync.Mutex
	err      error         // set by CloseWithError
	closedCh chan struct{} // closed if err is set
}

var (
	_ io.Closer   = (*RelayReader)(nil)
	_ io.WriterTo = (*RelayReader)(nil)
	_ io.Reader   = (*RelayReader)(nil)
)

func NewRelayReader() *RelayReader {
	var q RelayReader
	q.closedCh = make(chan struct{})
	q.t = q.line.Take()
	return &q
}

// CloseWithError terminates the line, unblocking any writer waiting for its
// turn with the error, or [io.EOF] if err is nil. It is safe to call
// [CloseWithError] multiple times and across multiple goroutines.
//
// If the line is already closed, [CloseWithError] is a no-op.
//
// It never returns an error.
func (q *RelayReader) CloseWithError(err error) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.err == nil {
		q.err = cmp.Or(q.err, err, io.EOF)
		close(q.closedCh)
	}
	return nil
}

// Close closes the line. Any writer waiting for its turn will be unblocked
// with an [io.ErrClosedPipe] error.
//
// It never returns an error.
func (q *RelayReader) Close() error {
	return q.CloseWithError(nil)
}

func (q *RelayReader) closed() <-chan struct{} {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.closedCh
}

func (q *RelayReader) Read(p []byte) (int, error) {
	panic("RelayReader.Read is for show only; use WriteTo")
}

// WriteTo yields the writer w to the first writer in line and blocks until the
// first call to [Close].
//
// It is safe to call [Take] concurrently with [WriteTo].
func (q *RelayReader) WriteTo(dst io.Writer) (int64, error) {
	select {
	case <-q.closed():
		return 0, io.ErrClosedPipe
	default:
	}

	// We have a destination writer; let the relay begin.
	q.w = dst
	q.t.Done()
	<-q.closed()
	return q.n, nil
}

// Take returns a writer that will be passed to the next writer in line.
//
// It is not safe for use across multiple goroutines.
func (q *RelayReader) Take() io.WriteCloser {
	return &relayWriter{q: q, t: q.line.Take()}
}

type relayWriter struct {
	q     *RelayReader
	t     *Ticket
	ready bool
}

var _ io.StringWriter = (*relayWriter)(nil)

// Write writes to the writer passed to [RelayReader.WriteTo] as soon as the
// writer is ready. It returns io.ErrClosedPipe if the line is closed before
// the writer is ready.
func (w *relayWriter) Write(p []byte) (int, error) {
	if !w.awaitTurn() {
		return 0, w.q.err
	}
	n, err := w.q.w.Write(p)
	w.q.n += int64(n)
	return n, err
}

func (w *relayWriter) WriteString(s string) (int, error) {
	if !w.awaitTurn() {
		return 0, w.q.err
	}
	return io.WriteString(w.q.w, s)
}

// Close signals that the writer is done, unblocking the next writer in line.
func (w *relayWriter) Close() error {
	w.t.Done()
	return nil
}

func (t *relayWriter) awaitTurn() (ok bool) {
	if t.ready {
		return true
	}
	select {
	case <-t.t.Ready():
		t.ready = true
		return true
	case <-t.q.closed():
		return false
	}
}
